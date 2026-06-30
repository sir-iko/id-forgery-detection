#!/bin/bash
#SBATCH --job-name=idfd-robust
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=results/slurm_robust_%j.out

module load python/miniforge/25.3.0-3
condastart
conda activate idfd

python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

for ckpt in resnet50_baseline densenet121_baseline vit_baseline; do
    echo "===================="
    echo "CHECKPOINT: ${ckpt} - CLEAN"
    echo "===================="
    python src/evaluate.py --split test --checkpoint checkpoints/${ckpt}_best.pt --num-workers 4

    echo "===================="
    echo "CHECKPOINT: ${ckpt} - JPEG80 BLUR0.5"
    echo "===================="
    python src/evaluate.py --split test --checkpoint checkpoints/${ckpt}_best.pt --num-workers 4 --jpeg-quality 80 --blur-sigma 0.5

    echo "===================="
    echo "CHECKPOINT: ${ckpt} - JPEG60 BLUR1.0"
    echo "===================="
    python src/evaluate.py --split test --checkpoint checkpoints/${ckpt}_best.pt --num-workers 4 --jpeg-quality 60 --blur-sigma 1.0

    echo "===================="
    echo "CHECKPOINT: ${ckpt} - JPEG40 BLUR1.5"
    echo "===================="
    python src/evaluate.py --split test --checkpoint checkpoints/${ckpt}_best.pt --num-workers 4 --jpeg-quality 40 --blur-sigma 1.5

done

echo "SWEEP COMPLETE"
