name=cylinder_asym_networks
gpuid=5

CUDA_VISIBLE_DEVICES=${gpuid}  python -u train_cylinder_asym.py --config_path config/semantickitti.yaml \
2>&1 | tee logs_dir/${name}_logs_tee.txt