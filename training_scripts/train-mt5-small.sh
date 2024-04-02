#!/bin/bash
#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:a40:1  # number of GPUs
#SBATCH --mem=32000  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 12:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=26  # number of cpus to use - there are 32 on each node.

set -e # fail fast

dt=$(date '+%d_%m_%y_%H_%M');
echo "I am job ${SLURM_JOB_ID}"
echo "I'm running on ${SLURM_JOB_NODELIST}"
echo "Job started at ${dt}"

# ====================
# Activate Anaconda environment
# ====================
source /home/${USER}/miniconda3/bin/activate hons

# ====================
# RSYNC data from /home/ to /disk/scratch/
# ====================
export SCRATCH_HOME=/disk/scratch/${USER}
export DATA_HOME=${PWD}/../clean_csvs
export DATA_SCRATCH=${SCRATCH_HOME}/data
export MODEL_HOME=${PWD}/../models/mt5-small
export MODEL_SCRATCH=${SCRATCH_HOME}/mt5-small
mkdir -p ${SCRATCH_HOME}/data
mkdir -p ${SCRATCH_HOME}/mt5-small
rsync --archive --update --compress --progress ${DATA_HOME}/ ${DATA_SCRATCH}
rsync --archive --update --compress --progress ${MODEL_HOME}/ ${MODEL_SCRATCH}

# ====================
# Run training. Here we use src/train-mt5-small.py
# ====================
echo "Creating directory to save model weights"
export OUTPUT_DIR=${SCRATCH_HOME}/output
mkdir -p ${OUTPUT_DIR}

# This script runs the training
python train-mt5-small.py \
	--data_path=${DATA_SCRATCH} \
	--model_path=${MODEL_SCRATCH} \
	--output_dir=${OUTPUT_DIR}

# ====================
# RSYNC data from /disk/scratch/ to /home/. This moves everything we want back onto the distributed file system
# ====================
OUTPUT_HOME=${PWD}/../finetuned-models
mkdir -p ${OUTPUT_HOME}
rsync --archive --update --compress --progress ${OUTPUT_DIR} ${OUTPUT_HOME}

# ====================
# Finally we cleanup after ourselves by deleting what we created on /disk/scratch/
# ====================
rm -rf ${OUTPUT_DIR}


echo "Job ${SLURM_JOB_ID} is done!"