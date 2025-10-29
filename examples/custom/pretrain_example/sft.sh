#!/bin/bash
# 自定义模型微调脚本（在预训练模型基础上进行 SFT）
# 使用方法: bash sft.sh

# 设置使用的 GPU
export CUDA_VISIBLE_DEVICES=0

# 预训练后的模型路径（或初始化的模型路径）
MODEL_PATH="./custom_pretrained_model"

# 输出路径
OUTPUT_DIR="./output/custom_sft"

# 微调数据集
DATASET="AI-ModelScope/alpaca-gpt4-data-zh"

echo "=========================================="
echo "开始自定义模型 SFT 微调"
echo "模型路径: ${MODEL_PATH}"
echo "输出路径: ${OUTPUT_DIR}"
echo "数据集: ${DATASET}"
echo "=========================================="

swift sft \
    --custom_register_path examples/custom/pretrain_example/register_model.py \
    --model_type custom_pretrain \
    --model ${MODEL_PATH} \
    --dataset ${DATASET} \
    --train_type lora \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --learning_rate 1e-4 \
    --weight_decay 0.01 \
    --warmup_ratio 0.05 \
    --gradient_accumulation_steps 8 \
    --eval_steps 200 \
    --save_steps 200 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 2048 \
    --output_dir ${OUTPUT_DIR} \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --gradient_checkpointing true \
    --bf16 true

echo "=========================================="
echo "SFT 微调完成!"
echo "模型保存在: ${OUTPUT_DIR}"
echo "=========================================="
