#!/bin/bash

#SBATCH --partition=gpu-v100
#SBATCH -N1
#SBATCH -n3
#SBATCH --gres=gpu:1
#SBATCH --time=4:0:0
#SBATCH --mem=16G

export PATH=/home/roberto.medeirosdeso/software/miniconda3/bin:$PATH 
source  activate tf2gpu 

python /home/roberto.medeirosdeso/Cluster-example/train_unet_jpeg.py
