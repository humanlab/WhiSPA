#!/bin/bash
#SBATCH --job-name=whispa-train-enc
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

# Teacher embedding keys (override via environment if different)
AUDIO_EMB_KEY="${AUDIO_EMB_KEY:-vox_3072_audio}"
TEXT_EMB_KEY="${TEXT_EMB_KEY:-qwen_2560_text}"
PSYCH_EMB_KEY="${PSYCH_EMB_KEY:-qwen_2560_affect}"

export TOKENIZERS_PARALLELISM=false

accelerate launch --num_processes 8 train/train.py \
  --config_yaml train/configs/train_enc.yaml \
  --stage train_enc \
  --audio_emb_key "${AUDIO_EMB_KEY}" \
  --text_emb_key "${TEXT_EMB_KEY}" \
  --psych_emb_key "${PSYCH_EMB_KEY}"


