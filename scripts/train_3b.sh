#!/bin/bash
# SDiaReward 3B model training script
#
# Usage:
#   bash scripts/train_3b.sh
#
# Prerequisites:
#   - Set MODEL_NAME_OR_PATH to your Qwen2.5-Omni-3B checkpoint path
#   - Set DATASET_NAME to your preference dataset path
#   - Adjust CUDA_VISIBLE_DEVICES and NUM_GPUS for your hardware

export CUDA_VISIBLE_DEVICES=0,1,2,3

# Fix multiprocessing deadlock issues with librosa/soundfile
export OMP_NUM_THREADS=1

# Logging directory
BASE_LOG_DIR="./logs/sdialog_reward"

# Configuration - modify these paths for your setup
MODEL_NAME_OR_PATH="ckpt/Qwen2.5-Omni-3B"
DATASET_TYPE="datasets"
POOLING_TYPE="mean_center"
DATASET_NAME="data/merged_preference_dataset"
OUTPUT_DIR="ckpt/SDiaReward-3B_${POOLING_TYPE}"

# DeepSpeed config (options: ds_config_zero2.json, ds_config_zero3.json, ds_config_zero3_offload.json)
DS_CONFIG="deepspeed_configs/ds_config_zero2.json"

# Number of GPUs
NUM_GPUS=4

# Create logging directory
RUN_NAME="run_3B_${POOLING_TYPE}"
LOG_DIR="${BASE_LOG_DIR}/${RUN_NAME}"
mkdir -p ${LOG_DIR}

echo "=================================================="
echo "Starting DeepSpeed Training Run"
echo "Using config: ${DS_CONFIG}"
echo "Number of GPUs: ${NUM_GPUS}"
echo "Logging to: ${LOG_DIR}"
echo "=================================================="

deepspeed --master_port=29506 train.py\
    --model_name_or_path ${MODEL_NAME_OR_PATH} \
    --dataset_type ${DATASET_TYPE} \
    --dataset_name ${DATASET_NAME} \
    --output_dir ${OUTPUT_DIR} \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --num_train_epochs 1 \
    --max_grad_norm 1.0 \
    --lr_scheduler_type cosine \
    --learning_rate 2e-5 \
    --weight_decay 0.05 \
    --warmup_ratio 0.15 \
    --eval_steps 50 \
    --save_steps 50 \
    --load_best_model_at_end True \
    --eval_strategy steps \
    --max_length 2048 \
    --save_strategy steps \
    --save_total_limit 20 \
    --metric_for_best_model eval_loss \
    --greater_is_better False \
    --load_best_model_at_end True \
    --dataset_test_split 'validation' \
    --dataset_train_split 'train' \
    --dataset_num_proc 8 \
    --dataloader_num_workers 4 \
    --logging_steps 1 \
    --report_to tensorboard \
    --logging_dir "${LOG_DIR}" \
    --bf16 True \
    --tf32 True \
    --remove_unused_columns False \
    --ddp_find_unused_parameters True \
    --dataloader_persistent_workers True \
    --gradient_checkpointing True \
    --deepspeed ${DS_CONFIG} \
    --attn_implementation flash_attention_2 \
    --center_rewards_coefficient 1e-2
