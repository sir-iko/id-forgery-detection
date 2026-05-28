#!/bin/bash
#SBATCH --job-name=idfd-smoke
#SBATCH --partition=gpu            # CHECK: confirm the GPU partition name on Viper
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00            # request only what you need (Viper max is 5 days)
#SBATCH --output=results/slurm_%j.out
#SBATCH --error=results/slurm_%j.err

# --- Modules: confirm exact names with `module avail` ---
# module load cuda                 # match the CUDA your PyTorch build expects

# --- Activate conda env ---
source ~/miniconda3/etc/profile.d/conda.sh
conda activate idfd

nvidia-smi

# Run from the submit directory so the `src` package imports correctly
cd "$SLURM_SUBMIT_DIR"
python -m src.train --config configs/default.yaml
