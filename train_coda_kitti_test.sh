name=cylinder_asym_networks
gpuid=7

CUDA_VISIBLE_DEVICES=${gpuid}  python -u train_coda_small_kitti.py \
2>&1 | tee logs_dir/${name}_logs_tee.txt