#!/bin/bash
#SBATCH --job-name=idfd-gradcam
#SBATCH --partition=gpu
#SBATCH --gres=gpu:tesla:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err

module load python/miniforge/25.3.0-3
condastart
conda activate idfd

python -c "import torch; print('cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else '')"

python -m src.gradcam
