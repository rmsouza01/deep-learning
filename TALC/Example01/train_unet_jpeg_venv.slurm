#!/bin/bash

#SBATCH --partition=gpu
#SBATCH -N1
#SBATCH -n2
#SBATCH --gres=gpu:1
#SBATCH --time=2:0:0
#SBATCH --mem=16G

export PATH=/work/TALC/enel645/software/miniconda3/bin:$PATH 
source  activate tf2gpu 

python /home/roberto.medeirosdeso/TALC-cluster-example/train_unet_jpeg.py
