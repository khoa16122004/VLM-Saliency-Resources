#!/bin/bash
#SBATCH --job-name=SALIENCY_GUIDED
#SBATCH --output=logs_SALIENCY_GUIDED/mps_%j.out
#SBATCH --error=logs_SALIENCY_GUIDED/mps_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=mps:a100:2
#SBATCH --mem=4G
#SBATCH --time=72:00:00

set -euo pipefail

REQUIRED_VRAM=12000

# =========================================================
# PREPARE ENVIRONMENT
# =========================================================
module clear -f
source /home/elo/miniconda3/etc/profile.d/conda.sh
conda activate bcos_attack

echo "ENV: $CONDA_DEFAULT_ENV"
echo "PREFIX: $CONDA_PREFIX"
which python
python -c "import sys; print(sys.executable)"

mkdir -p logs_SALIENCY_GUIDED

unset CUDA_VISIBLE_DEVICES
CHECK_OUT=$(/usr/local/bin/gpu_check.sh $REQUIRED_VRAM $SLURM_JOB_ID)
EXIT_CODE=$?

if [ $EXIT_CODE -eq 10 ]; then
    echo "$CHECK_OUT"
    exit 0
elif [ $EXIT_CODE -eq 11 ]; then
    echo "$CHECK_OUT"
    exit 1
fi

BEST_GPU=$CHECK_OUT
echo "Job $SLURM_JOB_ID starts on GPU: $BEST_GPU"

export CUDA_MPS_PIPE_DIRECTORY=/tmp/nvidia-mps-job$SLURM_JOB_ID
export CUDA_MPS_LOG_DIRECTORY=/tmp/nvidia-mps-log-job$SLURM_JOB_ID

rm -rf "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"

export CUDA_VISIBLE_DEVICES=$BEST_GPU

# =========================================================
# RUN CLASSIFICATION
# =========================================================
MODEL_NAME=${MODEL_NAME:-"clip"}
PRETRAINED=${PRETRAINED:-"ViT-B/32"}
DATASET_NAME=${DATASET_NAME:-"imagenet"}
BATCH_SIZE=${BATCH_SIZE:-64}
CLASSIFICATION_METHOD=${CLASSIFICATION_METHOD:-"simple"}
RIGHT_DIR=${RIGHT_DIR:-"correct_classified_samples"}

echo "Running classify.py with:"
echo "  model_name=$MODEL_NAME"
echo "  pretrained=$PRETRAINED"
echo "  dataset_name=$DATASET_NAME"
echo "  batch_size=$BATCH_SIZE"
echo "  classification_method=$CLASSIFICATION_METHOD"
echo "  right_dir=$RIGHT_DIR"

python classify.py \
    --model_name "$MODEL_NAME" \
    --pretrained "$PRETRAINED" \
    --dataset_name "$DATASET_NAME" \
    --batch_size "$BATCH_SIZE" \
    --classification_method "$CLASSIFICATION_METHOD" \
    --right_dir "$RIGHT_DIR"

