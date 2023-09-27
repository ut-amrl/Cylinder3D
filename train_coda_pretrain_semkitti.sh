name=cylinder_asym_networks
gpuid=0
LEARNING_RATE="0_1"

CUDA_VISIBLE_DEVICES=${gpuid}  python -u train_cylinder_asym.py --config_path config/coda-semantickitti-finetune-lr-${LEARNING_RATE}.yaml \
2>&1 | tee logs_dir/${name}_logs_tee.txt