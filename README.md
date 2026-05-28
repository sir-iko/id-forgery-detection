# Identity Document Forgery Detection

Deep learning framework to classify real vs. forged identity documents, with
explainability (Grad-CAM) and a calibration experiment. MSc project, 2026.

## Layout

```
configs/        run configuration (YAML)
src/            source package
  config.py     load YAML config
  data.py       dataset (TODO: MIDV-2020 / DocTamper / FantasyID)
  models.py     model factory (ResNet50, DenseNet121, ViT)
  train.py      training entrypoint (currently a GPU smoke test)
  utils/        seeding + lightweight run logger
scripts/        SLURM submission scripts for Viper
data/           datasets (gitignored)
results/        runs, checkpoints, logs (gitignored)
notebooks/      exploration
```

## Setup on Viper (University of Hull HPC)

You do all of the steps below on a Viper LOGIN node. You never train on the
login node: training goes to the GPU queue via SLURM (steps 6 to 8).

### 1. Connect
Connect to the campus network or the university VPN, then SSH into Viper.

### 2. Install Miniconda in your home directory (no admin needed)
```bash
cd ~
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3
source ~/miniconda3/etc/profile.d/conda.sh
conda init bash
```
(If Viper already provides an Anaconda module you prefer, run `module avail`
to find it and `module load` it instead.)

### 3. Get the code onto Viper
```bash
# Option A: clone your GitHub repo (recommended: code is then backed up off Viper)
git clone https://github.com/<you>/id-forgery-detection.git
cd id-forgery-detection

# Option B: copy this skeleton up from your laptop
# scp -r id-forgery-detection <user>@viper.hull.ac.uk:~/
```

### 4. Create the environment
```bash
conda env create -f environment.yml
conda activate idfd
```

### 5. Install PyTorch to match Viper's CUDA
Check the CUDA version the GPU nodes expose (do this inside an interactive GPU
session, step 6, then run `nvidia-smi` and read the top-right CUDA version).
Then install the matching build, for example for CUDA 12.1:
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```
Verify, then lock your exact versions so the project is reproducible:
```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available())"
conda env export > environment.lock.yml
git add environment.lock.yml && git commit -m "Lock environment"
```

### 6. Test interactively before submitting batch jobs
```bash
# Request a short interactive session on the GPU queue (confirm the partition
# name with `sinfo`; 'gpu' is shown here as the likely name).
srun --partition=gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=00:30:00 --pty bash
conda activate idfd
python -m src.train --config configs/default.yaml   # runs the smoke test
exit
```

### 7. Cache pretrained weights on the login node
Compute nodes may have no internet. On the login node, trigger the torchvision
download once so the weights are cached in ~/.cache/torch:
```bash
python -c "from torchvision import models; models.resnet50(weights='DEFAULT')"
```

### 8. Submit a batch job
```bash
sbatch scripts/slurm_train.sh
squeue -u $USER          # watch your job
cat results/slurm_<jobid>.out
```

### View training curves
```bash
tensorboard --logdir results --port 6006
# then SSH-tunnel port 6006 from your laptop to view in a browser
```

## Reproducibility checklist
- Seeds set centrally (src/utils/seed.py).
- One config file per run, snapshotted into the run folder.
- Exact versions locked in environment.lock.yml (committed).
- Code in git/GitHub; data and results stay on Viper (not committed).
- Back up important results OFF Viper, which is not backed up.
