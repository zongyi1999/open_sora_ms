source activate pytorch
CODE_ROOT=$PWD
# DATAPATH='./ucf101_train.csv'
DATAPATH='/valleblob/v-zongyili/data/ucf101/ucf101_train.csv'
gpus=$1
nvcc -V
pip list
pip install imageio-ffmpeg
pip install pytorch_lightning
pip install imageio
# pip install -r requirements.txt
# python generate_data.py
# PYTHONPATH=$CODE_ROOT torchrun --nnodes=1 --nproc_per_node=8 scripts/train.py configs/latte/train/16x256x256_v100.py  --data-path ${DATAPATH} /scratch/Open-sora/outputs/047-STDiT-XL-2/epoch64-global_step2000
# PYTHONPATH=$CODE_ROOT torchrun --nnodes=1  --master-port=26857 --nproc_per_node=4 scripts/train.py configs/latte/train/16x256x256_v100.py  --data-path ${DATAPATH} 
#--master-port=5361 

# PYTHONPATH=$CODE_ROOT colossalai run --nproc_per_node 8 --hostfile hostfile scripts/train_ms.py configs/opensora/train/16x256x256_ms_v100.py   --data-path ${DATAPATH}   --ckpt-path pretrained_models/OpenSora-v1-HQ-16x256x256.pth
PYTHONPATH=$CODE_ROOT torchrun --nnodes=1 --nproc_per_node=8 scripts/train_ms.py configs/opensora/train/16x256x256_ms_v100.py   --data-path ${DATAPATH}   --ckpt-path pretrained_models/OpenSora-v1-HQ-16x256x256.pth
# torchrun --nnodes=1--nproc_per_node=8 scripts/train_ms.py configs/opensora/train/16x256x256_ms.py --data-path /home/v-zongyili/Open-Sora/ucf101_train.csv --ckpt-path OpenSora-v1-HQ-16x256x256.pth
#--master_port --master_port 63536 