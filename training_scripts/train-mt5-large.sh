#!/bin/bash
#SBATCH -o /home/%u/slogs/sl_%A.out
#SBATCH -e /home/%u/slogs/sl_%A.out
#SBATCH -N 1	  # nodes requested
#SBATCH -n 1	  # tasks requested
#SBATCH --gres=gpu:a40  # number of GPUs
#SBATCH --mem=128000  # memory in Mb
#SBATCH --partition=PGR-Standard
#SBATCH -t 12:00:00  # time requested in hour:minute:seconds
#SBATCH --cpus-per-task=30  # number of cpus to use - there are 32 on each node.

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
export DATA_HOME=${PWD}/../clean_csvs
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
export OUTPUT_DIR=${SCRATCH_HOME}/output
mkdir -p ${OUTPUT_DIR}

# This script runs the training
# python train-mt5-small.py \
# 	--data_path=${DATA_SCRATCH} \
# 	--model_path=${MODEL_SCRATCH} \
# 	--output_dir=${OUTPUT_DIR}

python train-mt5-small.py \
    --model_name_or_path ${MODEL_SCRATCH} \
    --do_train \
    --do_eval \
    --train_file ${DATA_SCRATCH}/train.csv \
    --validation_file ${DATA_SCRATCH}/dev.csv \
    --test_file ${DATA_SCRATCH}/test.csv \
    --text_column linearized_input \
    --summary_column target \
    --output_dir ${OUTPUT_DIR} \
    --overwrite_output_dir True \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --predict_with_generate True \
    --learning_rate 0.001 \
    --eval_accumulation_steps 1 \
    --max_source_length 512 \
    --max_target_length 512 \
    --load_best_model_at_end True \
    --evaluation_strategy steps \
    --save_total_limit 1 \
    --lr_scheduler_type constant \
    --num_train_epochs 5 \
    --eval_steps 100 \
    --logging_steps 100 \
    --save_steps 100 \
    --weight_decay 0.001
    # --save_strategy no
    # --weight_decay 0.01 \
    # --do_predict True \

# python -m torch.distributed.launch \
#     --nproc_per_node 2 train-mt5-small.py \
#       --model_name_or_path ${MODEL_SCRATCH} \
#       --do_train True \
#       --do_eval True \
#       --train_file ${DATA_SCRATCH}/train.csv \
#       --validation_file ${DATA_SCRATCH}/dev.csv \
#       --test_file ${DATA_SCRATCH}/test.csv \
#       --text_column linearized_input \
#       --summary_column target \
#       --output_dir ${DATA_SCRATCH}/output \
#       --per_device_train_batch_size=8 \
#       --per_device_eval_batch_size=8 \
#       --predict_with_generate True

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
