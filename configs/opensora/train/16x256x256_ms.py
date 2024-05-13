# Define dataset
dataset = dict(
    type="VideoTextDataset",
    data_path=None,
    num_frames=16,
    frame_interval=3,
    image_size=(128, 128),
)

# Define acceleration
num_workers = 4
dtype = "bf16"
grad_checkpoint = True
plugin = "zero2"
sp_size = 1

# Define model

model = dict(
    type="STDiT-XL/2",
    space_scale=0.5,
    time_scale=1.0,
    from_pretrained="PixArt-XL-2-512x512.pth", #None, #
    enable_flashattn=True,
    enable_layernorm_kernel=True,

)
vae = dict(
    type="VQGANMS",
    from_pretrained="/home/v-zongyili/Open-Sora/last.ckpt",
)
text_encoder = dict(
    type="t5",
    from_pretrained="DeepFloyd/t5-v1_1-xxl",
    model_max_length=120,
    shardformer=True,
)
scheduler = dict(
    type="iddpm",
    timestep_respacing="",
)

# Others
seed = 42
outputs = "outputs"
wandb = False

epochs = 100
log_every = 1
ckpt_every = 5000
load = None

batch_size = 32
lr = 2e-5 #1e-4 #
grad_clip = 1.0
