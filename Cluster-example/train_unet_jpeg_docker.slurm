#!/bin/bash

#SBATCH --partition=gpu-v100
#SBATCH -N1
#SBATCH -n3
#SBATCH --gres=gpu:1
#SBATCH --time=4:0:0
#SBATCH --mem=16G


# Singularity may use TMPDIR for some things.
#TMPDIR=/scratch/${SLURM_JOB_ID}
TMPDIR=/home/roberto.medeirosdeso/${SLURM_JOB_ID}
mkdir $TMPDIR
export TMPDIR

# Define a location to serve as the parent of various cache directories
# This directory is supposed to be for temporary runtime files

SINGULARITY_LOCALCACHEDIR=$TMPDIR/slocalcache
mkdir $SINGULARITY_LOCALCACHEDIR
export SINGULARITY_LOCALCACHEDIR
echo "Defined SINGULARITY_LOCALCACHEDIR as $SINGULARITY_LOCALCACHEDIR"

# This directory is for "caching layers and singularity hub images"

SINGULARITY_CACHEDIR=$TMPDIR/scache
mkdir $SINGULARITY_CACHEDIR
export SINGULARITY_CACHEDIR
echo "Defined SINGULARITY_CACHEDIR as $SINGULARITY_CACHEDIR"

# This is referenced in Singularity code
SINGULARITY_CLEANUPDIR=$TMPDIR/scleanup
mkdir $SINGULARITY_CLEANUPDIR
export SINGULARITY_CLEANUPDIR
echo "Defined SINGULARITY_CLEANUPDIR as $SINGULARITY_CLEANUPDIR"

which singularity
singularity exec --nv docker://tensorflow/tensorflow:latest-gpu-py3 /home/roberto.medeirosdeso/Cluster-example/train_unet_jpeg.sh 

rm -r $TMPDIR
