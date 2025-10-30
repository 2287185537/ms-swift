#!/bin/bash
# 自定义模型全量预训练脚本（从零开始训练）
# 使用 swift pt 命令进行预训练
# 使用方法: bash pretrain.sh

# 设置使用的 GPU（可以设置多个，用逗号分隔，如 "0,1,2,3"）
export CUDA_VISIBLE_DEVICES=0

# 模型路径（你初始化的随机权重模型路径）
MODEL_PATH="./custom_pretrained_model"

# 输出路径
OUTPUT_DIR="./output/custom_pretrain"

# 预训练数据集
# 支持 ModelScope/HuggingFace 数据集 ID，或本地数据路径
# 预训练数据格式：纯文本或包含 "text" 字段的 JSON/JSONL
DATASET="AI-ModelScope/wikipedia-cn-20230720-filtered"

# 开始全量预训练
echo "=========================================="
echo "开始自定义模型全量预训练（从零开始）"
echo "模型路径: ${MODEL_PATH}"
echo "输出路径: ${OUTPUT_DIR}"
echo "数据集: ${DATASET}"
echo "=========================================="

# swift pt 命令：用于预训练（Pre-Training）
swift pt \
    --custom_register_path examples/custom/pretrain_example/register_model.py \
    --model_type custom_pretrain \
    --model ${MODEL_PATH} \
    --dataset ${DATASET} \
    --train_type full \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --learning_rate 1e-4 \
    --weight_decay 0.1 \
    --warmup_ratio 0.01 \
    --gradient_accumulation_steps 4 \
    --eval_steps 1000 \
    --save_steps 1000 \
    --save_total_limit 3 \
    --logging_steps 10 \
    --max_length 2048 \
    --output_dir ${OUTPUT_DIR} \
    --dataloader_num_workers 4 \
    --dataset_num_proc 8 \
    --gradient_checkpointing true \
    --bf16 true \
    --save_only_model true \
    --use_flash_attn true

echo "=========================================="
echo "预训练完成!"
echo "模型保存在: ${OUTPUT_DIR}"
echo "后续可以继续预训练或进行微调（SFT）"
echo "=========================================="
