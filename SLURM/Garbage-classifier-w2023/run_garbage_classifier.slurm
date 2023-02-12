#!/bin/bash

####### Reserve computing resources #############
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --time=06:00:00
#SBATCH --mem=32G
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=1

module load python/anaconda3-2018.12

source activate pytorch-2023

python /work/TALC/enel645_2023w/Garbage-classification/garbage_main.py --best_model_path /work/TALC/enel645_2023w/Garbage-classification/garbage_net.pth --images_path /work/TALC/enel645_2023w/Garbage-classification/ --transfer_learning True

conda deactivate
