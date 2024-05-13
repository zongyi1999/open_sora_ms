python -m tools.datasets.convert video /home/v-zongyili/Open-Sora/00000 --output data/meta.csv
python -m tools.datasets.datautil data/meta.csv --info --fmin 1
python -m tools.scene_cut.scene_detect data/meta_info_fmin1.csv
python -m tools.scene_cut.cut ${ROOT_META}/meta_info_fmin1_timestamp.csv --save_dir ${ROOT_CLIPS}

docker run -ti --gpus all -v /home/v-zongyili/Open-Sora:/data opensora
