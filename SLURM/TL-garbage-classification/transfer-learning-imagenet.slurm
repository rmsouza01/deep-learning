#!/bin/bash

#SBATCH --time=24:00:00
#SBATCH --output=%x.out
#SBATCH --cpus-per-task=3
#SBATCH --gres=gpu:1
#SBATCH --mem=32G

export PATH=/work/TALC/enel645_2022w/Software/miniconda3/bin:$PATH
source activate enel645

python /home/roberto.medeirosdeso/TL-garbage-classification/transfer_learning_imagenet.py
conda deactivate
