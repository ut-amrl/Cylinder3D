name=cylinder_asym_networks
gpuid=1

CUDA_VISIBLE_DEVICES=${gpuid}  python -u test_coda_kitti.py \
2>&1 | tee logs_dir/${name}_logs_tee.txt