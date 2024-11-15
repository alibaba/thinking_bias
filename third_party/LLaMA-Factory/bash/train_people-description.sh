#!/bin/bash

export PATH=/usr/local/cuda/bin:$PATH # using nvcc in cuda-dir, for cpu_adam compile
export PYTHONPATH='.'

ngpus="${NGPUS:-8}"
world_size="${WORLD_SIZE:-1}"

model_name_or_path="${MODEL_NAME_OR_PATH:-/path/to/pretrained/models}"
output_dir="${OUTPUT_DIR:-/path/to/your/output/dir}"
dataset="people-description"

learning_rate="${LEARNING_RATE:-7e-6}"
weight_decay="${WEIGHT_DECAY:-1e-1}"
num_epochs="${NUM_EPOCHS:-3}"
global_batch_size="${GLOBAL_BATCH_SIZE:-16}"
batch_size_per_device="${BATCH_SIZE_PER_DEVICE:-2}"

accum_steps=$((${global_batch_size}/${world_size}/${ngpus}/${batch_size_per_device}))
mkdir -p ${output_dir}

deepspeed --num_gpus ${ngpus} src/train.py \
    --model_name_or_path ${model_name_or_path} \
    --stage pt \
    --do_train \
    --finetuning_type full \
    --deepspeed examples/deepspeed/ds_z3_config.json \
    --dataset ${dataset} \
    --packing false \
    --cutoff_len 1024 \
    --max_samples 100000000 \
    --overwrite_cache \
    --preprocessing_num_workers 16 \
    --output_dir ${output_dir} \
    --logging_steps 10 \
    --save_steps -1 \
    --plot_loss \
    --overwrite_output_dir \
    --per_device_train_batch_size ${batch_size_per_device} \
    --gradient_accumulation_steps ${accum_steps} \
    --learning_rate ${learning_rate} \
    --weight_decay ${weight_decay} \
    --num_train_epochs ${num_epochs} \
    --lr_scheduler_type constant \
    --adam_beta1 0.9 \
    --adam_beta2 0.95 \
    --warmup_ratio 0.02 \
    --bf16 \
    --ddp_timeout 180000000