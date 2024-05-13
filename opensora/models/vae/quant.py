from typing import List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from torch import distributed as tdist, nn as nn
from torch.nn import functional as F

# from tats.modules import dist
import torch.distributed as dist



# this file only defines the VectorQuantizer2 used in VQVAE
__all__ = ['VectorQuantizer2',]


class VectorQuantizer2(nn.Module):
    # VQGAN originally use beta=1.0, never tried 0.25; SD seems using 0.25
    def __init__(
        self, vocab_size, Cvae, using_znorm, beta: float = 0.25,
        default_qresi_counts=0, v_patch_nums=None, quant_resi=0.5, share_quant_resi=4,  # share_quant_resi: args.qsr
    ):
        super().__init__()
        self.vocab_size: int = vocab_size
        self.Cvae: int = Cvae
        self.using_znorm: bool = using_znorm
        self.v_patch_nums: Tuple[int] = v_patch_nums
        
        self.quant_resi_ratio = quant_resi
        if share_quant_resi == 0:   # non-shared: \phi_{1 to K} for K scales
            self.quant_resi = PhiNonShared([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(default_qresi_counts or len(self.v_patch_nums))])
        elif share_quant_resi == 1: # fully shared: only a single \phi for K scales
            self.quant_resi = PhiShared(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity())
        else:                       # partially shared: \phi_{1 to share_quant_resi} for K scales
            self.quant_resi = PhiPartiallyShared(nn.ModuleList([(Phi(Cvae, quant_resi) if abs(quant_resi) > 1e-6 else nn.Identity()) for _ in range(share_quant_resi)]))
        
        self.register_buffer('ema_vocab_hit_SV', torch.full((len(self.v_patch_nums), self.vocab_size), fill_value=0.0))
        self.record_hit = 0
        
        self.beta: float = beta
        self.embedding = nn.Embedding(self.vocab_size, self.Cvae)
        
        # only used for progressive training of VAR (not supported yet, will be tested and supported in the future)
        self.prog_si = -1   # progressive training: not supported yet, prog_si always -1
    
    def eini(self, eini):
        if eini > 0: nn.init.trunc_normal_(self.embedding.weight.data, std=eini)
        elif eini < 0: self.embedding.weight.data.uniform_(-abs(eini) / self.vocab_size, abs(eini) / self.vocab_size)
    
    def extra_repr(self) -> str:
        return f'{self.v_patch_nums}, znorm={self.using_znorm}, beta={self.beta}  |  S={len(self.v_patch_nums)}, quant_resi={self.quant_resi_ratio}'
    
    def idxBl_to_latent(self, ms_idx_Bl: torch.Tensor) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl.shape[0]
        ms_h_BCThw = []
        #print ("idx_Bl: ", idx_Bl.shape)
        T = ms_idx_Bl.shape[1]
        #l = idx_Bl.shape[2]
        #pn = round(l ** 0.5)
        pn = ms_idx_Bl.shape[2] # shape[3]
        latent = self.embedding(ms_idx_Bl).permute(0, 4, 1, 2, 3).view(B, self.Cvae, T, pn, pn)
        return latent #self.embed_to_fhat(ms_h_BCThw=ms_h_BCThw, all_to_max_scale=same_shape, last_one=last_one)


    # ===================== `forward` is only used in VAE training =====================
    def forward(self, f_BCThw: torch.Tensor, ret_usages=False) -> Tuple[torch.Tensor, List[float], torch.Tensor]:
        dtype = f_BCThw.dtype
        if dtype != torch.float32: f_BChw = f_BCThw.float()
        B, C, T, H, W = f_BCThw.shape
        f_no_grad = f_BCThw.detach()
        
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        with torch.cuda.amp.autocast(enabled=False):
            mean_vq_loss: torch.Tensor = 0.0
            vocab_hit_V = torch.zeros(self.vocab_size, dtype=torch.float, device=f_BCThw.device)
            SN = len(self.v_patch_nums)
            idx_BThw_list: List[torch.Tensor] = []
            #print ("self.v_patch_nums: ", self.v_patch_nums)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                # find the nearest embedding
                if self.using_znorm:
                    rest_NC = F.interpolate(f_rest, size=(T, pn, pn), mode='trilinear', align_corners=False).permute(0, 2, 3, 4, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 4, 1).reshape(-1, C)
                    rest_NC = F.normalize(rest_NC, dim=-1)
                    idx_NT = torch.argmax(rest_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
                else:
                    rest_NC = F.interpolate(f_rest, size=(T, pn, pn), mode='trilinear', align_corners=False).permute(0, 2, 3, 4, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 4, 1).reshape(-1, C)
                    #print ("rest_NC: ", rest_NC.shape) # [24, 32] -> [96, 32] -> ... -> [6144, 32]
                    d_no_grad = torch.sum(rest_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                    #print ("d_no_grad: ", d_no_grad.shape) # [24, 4096] -> [96, 4096] -> ... -> [6144, 4096]
                    d_no_grad.addmm_(rest_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*T*h*w, vocab_size)
                    #print ("d_no_grad 2: ", d_no_grad.shape) # [24, 4096] -> [96, 4096] -> ... -> [6144, 4096]
                    idx_NT = torch.argmin(d_no_grad, dim=1)
                    #print ("idx_NT: ", idx_NT.shape) # [24] -> [96] -> ... -> [6144]
                    #print ("")
                
                hit_V = idx_NT.bincount(minlength=self.vocab_size).float()
                if self.training:
                    if dist.initialized(): handler = tdist.all_reduce(hit_V, async_op=True)
                
                # calc loss
                # import pdb; pdb.set_trace()
                idx_BThw = idx_NT.view(B, T, pn, pn) # [6, 4, pn, pn]
                h_BCThw = F.interpolate(self.embedding(idx_BThw).permute(0, 4, 1, 2, 3), size=(T, H, W), mode='trilinear', align_corners=False).contiguous() if (si != SN-1) else self.embedding(idx_BThw).permute(0, 4, 1, 2, 3).contiguous()
                h_BCThw = self.quant_resi[si/(SN-1)](h_BCThw)
                f_hat = f_hat + h_BCThw
                f_rest -= h_BCThw
                idx_BThw_list.append(idx_BThw)
                
                if self.training and dist.initialized():
                    handler.wait()
                    if self.record_hit == 0: self.ema_vocab_hit_SV[si].copy_(hit_V)
                    elif self.record_hit < 100: self.ema_vocab_hit_SV[si].mul_(0.9).add_(hit_V.mul(0.1))
                    else: self.ema_vocab_hit_SV[si].mul_(0.99).add_(hit_V.mul(0.01))
                    self.record_hit += 1
                vocab_hit_V.add_(hit_V)
                mean_vq_loss += F.mse_loss(f_hat.data, f_BCThw).mul_(self.beta) + F.mse_loss(f_hat, f_no_grad)
            
            mean_vq_loss *= 1. / SN
            f_hat = (f_hat.data - f_no_grad).add_(f_BCThw)
        
        margin = tdist.get_world_size() * (f_BCThw.numel() / f_BCThw.shape[1]) / self.vocab_size * 0.08
        # margin = pn*pn / 100
        if ret_usages: 
            usages = [(self.ema_vocab_hit_SV[si] >= margin).float().mean().item() * 100 for si, pn in enumerate(self.v_patch_nums)]
        else: 
            usages = None
        return f_hat, usages, mean_vq_loss, idx_BThw_list
    # ===================== `forward` is only used in VAE training =====================

    def idxBl_to_f_hat(self, ms_idx_Bl: List[torch.Tensor], same_shape: bool, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        B = ms_idx_Bl[0].shape[0]
        ms_h_BCThw = []
        for idx_Bl in ms_idx_Bl:
            #print ("idx_Bl: ", idx_Bl.shape)
            T = idx_Bl.shape[1]
            #l = idx_Bl.shape[2]
            #pn = round(l ** 0.5)
            pn = idx_Bl.shape[2] # shape[3]
            ms_h_BCThw.append(self.embedding(idx_Bl).permute(0, 4, 1, 2, 3).view(B, self.Cvae, T, pn, pn))
        return self.embed_to_fhat(ms_h_BCThw=ms_h_BCThw, all_to_max_scale=same_shape, last_one=last_one)

    def embed_to_fhat(self, ms_h_BCThw: List[torch.Tensor], all_to_max_scale=True, last_one=False) -> Union[List[torch.Tensor], torch.Tensor]:
        ls_f_hat_BCThw = []
        B = ms_h_BCThw[0].shape[0]
        T = ms_h_BCThw[0].shape[2]
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        if all_to_max_scale:
            f_hat = ms_h_BCThw[0].new_zeros(B, self.Cvae, T, H, W, dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                h_BCThw = ms_h_BCThw[si]
                if si < len(self.v_patch_nums) - 1:
                    h_BCThw = F.interpolate(h_BCThw, size=(T, H, W), mode='trilinear', align_corners=False)
                h_BCThw = self.quant_resi[si/(SN-1)](h_BCThw)
                #print ("h_BCThw: ", h_BCThw.shape)
                f_hat.add_(h_BCThw)
                if last_one: ls_f_hat_BCThw = f_hat
                else: ls_f_hat_BCThw.append(f_hat.clone())
        else:
            # WARNING: this is not the case in VQ-VAE training (where we'll interpolate every token map to the max scale), so it may cause some training-inference inconsistency
            # WARNING: this should only be used for experimental visualization
            f_hat = ms_h_BCThw[0].new_zeros(B, self.Cvae, T, self.v_patch_nums[0], self.v_patch_nums[0], dtype=torch.float32)
            for si, pn in enumerate(self.v_patch_nums): # from small to large
                f_hat = F.interpolate(f_hat, size=(T, pn, pn), mode='trilinear', align_corners=False)
                h_BCThw = self.quant_resi[si/(SN-1)](ms_h_BCThw[si])
                f_hat.add_(h_BCThw)
                if last_one: ls_f_hat_BCThw = f_hat
                else: ls_f_hat_BCThw.append(f_hat)
        
        return ls_f_hat_BCThw
    
    def f_to_idxBl_or_fhat(self, f_BCThw: torch.Tensor, to_fhat: bool, v_patch_nums: Optional[Sequence[Union[int, Tuple[int, int]]]] = None) -> List[torch.Tensor]:  # z_BChw is the feature from inp_img_no_grad
        B, C, T, H, W = f_BCThw.shape
        f_no_grad = f_BCThw.detach()
        f_rest = f_no_grad.clone()
        f_hat = torch.zeros_like(f_rest)
        
        f_hat_or_idx_Bl: List[torch.Tensor] = []
        
        patch_hws = [(pn, pn) if isinstance(pn, int) else (pn[0], pn[1]) for pn in v_patch_nums]    # from small to large
        assert patch_hws[-1][0] == H and patch_hws[-1][1] == W #, f'{patch_hws[-1]=} != ({H=}, {W=})'
        
        SN = len(patch_hws)
        for si, (ph, pw) in enumerate(patch_hws): # from small to large
            if 0 <= self.prog_si < si: break    # progressive training: not supported yet, prog_si always -1
            # find the nearest embedding
            z_NC = F.interpolate(f_rest, size=(T, ph, pw), mode='trilinear', align_corners=False).permute(0, 2, 3, 4, 1).reshape(-1, C) if (si != SN-1) else f_rest.permute(0, 2, 3, 4, 1).reshape(-1, C)
            if self.using_znorm:
                z_NC = F.normalize(z_NC, dim=-1)
                idx_NT = torch.argmax(z_NC @ F.normalize(self.embedding.weight.data.T, dim=0), dim=1)
            else:
                d_no_grad = torch.sum(z_NC.square(), dim=1, keepdim=True) + torch.sum(self.embedding.weight.data.square(), dim=1, keepdim=False)
                d_no_grad.addmm_(z_NC, self.embedding.weight.data.T, alpha=-2, beta=1)  # (B*h*w, vocab_size)
                idx_NT = torch.argmin(d_no_grad, dim=1)
            
            idx_BThw = idx_NT.view(B, T, ph, pw)
            h_BCThw = F.interpolate(self.embedding(idx_BThw).permute(0, 4, 1, 2, 3), size=(T, H, W), mode='trilinear', align_corners=False).contiguous() if (si != SN-1) else self.embedding(idx_Bhw).permute(0, 4, 1, 2, 3).contiguous()
            h_BCThw = self.quant_resi[si/(SN-1)](h_BCThw)
            f_hat.add_(h_BCThw)
            f_rest.sub_(h_BCThw)
            f_hat_or_idx_Bl.append(f_hat.clone() if to_fhat else idx_NT.reshape(B, T, ph*pw))
        
        return f_hat_or_idx_Bl
    

    # ===================== idxBl_to_var_input: only used in VAR training, for getting teacher-forcing input =====================
    def idxBl_to_var_input(self, gt_ms_idx_Bl: List[torch.Tensor]) -> torch.Tensor:
        next_scales = []
        B = gt_ms_idx_Bl[0].shape[0]
        C = self.Cvae
        H = W = self.v_patch_nums[-1]
        SN = len(self.v_patch_nums)
        with torch.cuda.amp.autocast(enabled=False):
            f_hat = gt_ms_idx_Bl[0].new_zeros(B, C, H, W, dtype=torch.float32)
            pn_next: int = self.v_patch_nums[0]
            for si in range(SN-1):
                if self.prog_si == 0 or (0 <= self.prog_si-1 < si): break   # progressive training: not supported yet, prog_si always -1
                h_BChw = F.interpolate(self.embedding(gt_ms_idx_Bl[si]).transpose_(1, 2).view(B, C, pn_next, pn_next), size=(H, W), mode='bicubic')
                f_hat.add_(self.quant_resi[si/(SN-1)](h_BChw))
                pn_next = self.v_patch_nums[si+1]
                next_scales.append(F.interpolate(f_hat, size=(pn_next, pn_next), mode='area').view(B, C, -1).transpose(1, 2))
        return torch.cat(next_scales, dim=1) if len(next_scales) else None    # cat BlCs to BLC, this should be float32
    
    # ===================== get_next_autoregressive_input: only used in VAR inference, for getting next step's input =====================
    def get_next_autoregressive_input(self, si: int, SN: int, f_hat: torch.Tensor, h_BChw: torch.Tensor) -> Tuple[Optional[torch.Tensor], torch.Tensor]: # only used in VAR inference
        HW = self.v_patch_nums[-1]
        if si != SN-1:
            h = self.quant_resi[si/(SN-1)](F.interpolate(h_BChw, size=(HW, HW), mode='bicubic'))     # conv after upsample
            f_hat.add_(h)
            return f_hat, F.interpolate(f_hat, size=(self.v_patch_nums[si+1], self.v_patch_nums[si+1]), mode='area')
        else:
            h = self.quant_resi[si/(SN-1)](h_BChw)
            f_hat.add_(h)
            return f_hat, f_hat


class Phi(nn.Conv3d):
    def __init__(self, embed_dim, quant_resi):
        ks = 3
        super().__init__(in_channels=embed_dim, out_channels=embed_dim, kernel_size=(1, ks, ks), stride=1, padding=(0, ks//2, ks//2))
        self.resi_ratio = abs(quant_resi)
    
    def forward(self, h_BCThw):
        return h_BCThw.mul(1-self.resi_ratio) + super().forward(h_BCThw).mul_(self.resi_ratio)


class PhiShared(nn.Module):
    def __init__(self, qresi: Phi):
        super().__init__()
        self.qresi: Phi = qresi
    
    def __getitem__(self, _) -> Phi:
        return self.qresi


class PhiPartiallyShared(nn.Module):
    def __init__(self, qresi_ls: nn.ModuleList):
        super().__init__()
        self.qresi_ls = qresi_ls
        K = len(qresi_ls)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return self.qresi_ls[np.argmin(np.abs(self.ticks - at_from_0_to_1)).item()]
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'


class PhiNonShared(nn.ModuleList):
    def __init__(self, qresi: List):
        super().__init__(qresi)
        # self.qresi = qresi
        K = len(qresi)
        self.ticks = np.linspace(1/3/K, 1-1/3/K, K) if K == 4 else np.linspace(1/2/K, 1-1/2/K, K)
    
    def __getitem__(self, at_from_0_to_1: float) -> Phi:
        return super().__getitem__(np.argmin(np.abs(self.ticks - at_from_0_to_1)).item())
    
    def extra_repr(self) -> str:
        return f'ticks={self.ticks}'
