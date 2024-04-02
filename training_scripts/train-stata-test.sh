#!/bin/bash
#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:a40  # number of GPUs
#SBATCH --mem=128G  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 72:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=28  # number of cpus to use - there are 32 on each node.

set -e # fail fast

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# ====================
# Activate Anaconda environment
# ====================
source /home/${USER}/miniconda3/bin/activate hf
echo "hf activated"
# ====================
# RSYNC data from /home/ to /disk/scratch/
# ====================
export SCRATCH_HOME=/disk/scratch/${USER}
echo SCRATCH_HOME
export DATA_HOME=${PWD}/../clean_csvs/stata
export DATA_SCRATCH=${SCRATCH_HOME}/data
export MODEL_HOME=${PWD}/../models/mt5-large
export MODEL_SCRATCH=${SCRATCH_HOME}/mt5-large
echo "about to mkdir"
mkdir -p ${SCRATCH_HOME}/data
mkdir -p ${SCRATCH_HOME}/mt5-large
echo "mkdir done"
rsync --archive --update --compress --progress ${DATA_HOME}/ ${DATA_SCRATCH}
rsync --archive --update --compress --progress ${MODEL_HOME}/ ${MODEL_SCRATCH}

# ====================
# Run training. Here we use src/train-mt5-small.py
# ====================
echo "Creating directory to save model weights"
export OUTPUT_DIR=${SCRATCH_HOME}/output-stata
mkdir -p ${OUTPUT_DIR}

# torchrun --nproc_per_node 3 train-stata-new.py

python train-stata-new.py

# ====================
# RSYNC data from /disk/scratch/ to /home/. This moves everything we want back onto the distributed file system
# ====================
OUTPUT_HOME=${PWD}/../finetuned-models
mkdir -p ${OUTPUT_HOME}
rsync --archive --update --compress --progress ${OUTPUT_DIR} ${OUTPUT_HOME}

# ====================
# Finally we cleanup after ourselves by deleting what we created on /disk/scratch/
# ====================
# rm -rf ${OUTPUT_DIR}

echo "Job ${SLURM_JOB_ID} is done!"
