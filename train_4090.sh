#!/bin/bash

# RTX 4090 (24GB) 训练脚本
# 使用 QLoRA + 优化配置

python train_mem.py \
    --data_path data/VideoInstruct-100K/VideoInstruct100K.json \
    --features_folder data/VideoInstruct-100K/activity_clip-14L_temporal_16f \
    --num_frames 16 \
    --output_dir ./checkpoints/Baseline_T16_4090 \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 2 \
    --learning_rate 5e-5 \
    --num_train_epochs 3 \
    --warmup_ratio 0.03 \
    --lr_scheduler_type cosine \
    --weight_decay 0.0 \
    --model_max_length 2048 \
    --logging_steps 50 \
    --logging_dir ./logs/baseline_4090 \
    --report_to tensorboard \
    --save_strategy steps \
    --save_steps 3000 \
    --save_total_limit 3 \
    --gradient_checkpointing True \
    --bf16 True
