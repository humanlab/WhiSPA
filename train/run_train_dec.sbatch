#!/bin/bash
#SBATCH --job-name=whispa-train-dec
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
#SBATCH --partition=gpu
#SBATCH --gres=gpu:8
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=0

set -euo pipefail

# module load cuda/12.3
# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate whispa

if [[ -z "${CHECKPOINT_DIR:-}" ]]; then
  echo "ERROR: CHECKPOINT_DIR is not set" >&2
  exit 1
fi

export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes 8 train/train.py \
  --config_yaml train/configs/train_dec.yaml \
  --stage train_dec