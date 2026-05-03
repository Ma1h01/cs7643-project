#!/bin/bash
#SBATCH --job-name=sd_fullft
#SBATCH --output=/home/hice1/xkong43/scratch/outputs/logs/%x-%j.out
#SBATCH --error=/home/hice1/xkong43/scratch/outputs/logs/%x-%j.err
#SBATCH --account=coc
#SBATCH --partition=coc-gpu
#SBATCH --qos=coc-ice
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00

set -e
module load anaconda3
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sd_full_ft

export HF_HOME=/home/hice1/xkong43/scratch/.cache/huggingface
export TRANSFORMERS_CACHE=/home/hice1/xkong43/scratch/.cache/huggingface
export TORCH_HOME=/home/hice1/xkong43/scratch/.cache/torch

DATA_ROOT=/home/hice1/xkong43/scratch/datasets/eldenring_dataset/train
OUTPUT_DIR=/home/hice1/xkong43/scratch/outputs/full_ft_sd15

mkdir -p "$OUTPUT_DIR"
mkdir -p /home/hice1/xkong43/scratch/outputs/logs

cd /storage/ice1/4/9/xkong43/diffusers/examples/text_to_image

echo "===== JOB START ====="
date
hostname
which python
python --version
nvidia-smi

accelerate launch --mixed_precision="fp16" train_text_to_image.py \
  --pretrained_model_name_or_path="stable-diffusion-v1-5/stable-diffusion-v1-5" \
  --train_data_dir="$DATA_ROOT" \
  --use_ema \
  --resolution=384 \
  --center_crop \
  --random_flip \
  --train_batch_size=1 \
  --gradient_accumulation_steps=1 \
  --dataloader_num_workers=0 \
  --gradient_checkpointing \
  --learning_rate=1e-5 \
  --lr_scheduler="constant" \
  --lr_warmup_steps=0 \
  --max_grad_norm=1 \
  --checkpointing_steps=500 \
  --max_train_steps=4000 \
  --output_dir="$OUTPUT_DIR"

echo "===== JOB END ====="
date
