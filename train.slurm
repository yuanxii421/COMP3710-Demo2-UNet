#!/bin/bash

#SBATCH --job-name=unet_oasis
#SBATCH --partition=comp3710
#SBATCH --account=comp3710
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --time=08:00:00
#SBATCH --output=unet_%j.out
#SBATCH --error=unet_%j.err

source ~/.bashrc
conda activate comp3710-unet

cd ~/COMP3710-Demo2-UNet

echo "=== Job started ==="
date
hostname

echo "=== GPU ==="
nvidia-smi

echo "=== PyTorch CUDA check ==="
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None')"

echo "=== Start training ==="
python train.py

echo "=== Job finished ==="
date
