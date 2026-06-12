#!/bin/bash

#SBATCH --job-name=idfd-vit-e35

#SBATCH --partition=gpu

#SBATCH --gres=gpu:tesla:1

#SBATCH --cpus-per-task=4

#SBATCH --mem=32G

#SBATCH --time=04:00:00

#SBATCH --output=results/slurm_%j.out

#SBATCH --error=results/slurm_%j.err



# --- Activate conda env (Viper login/compute nodes) ---

module load python/miniforge/25.3.0-3

condastart

conda activate idfd



# --- Diagnostics: confirm the GPU is visible and torch sees CUDA ---

nvidia-smi

python -c "import torch; print('torch', torch.__version__, 'cuda', torch.cuda.is_available(), torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'no-cuda')"



# Run from the submit directory so the `src` package imports correctly

cd "$SLURM_SUBMIT_DIR"

python -m src.train --config configs/vit.yaml --epochs 35 --patience 7

