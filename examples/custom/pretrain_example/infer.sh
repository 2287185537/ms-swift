#!/bin/bash
# 自定义模型推理脚本
# 使用方法: bash infer.sh

# 设置使用的 GPU
export CUDA_VISIBLE_DEVICES=0

# 模型路径（可以是初始化的模型、预训练后的模型，或微调后的模型）
MODEL_PATH="./custom_pretrained_model"

# 如果是微调后的模型，需要指定 adapter 路径
# ADAPTER_PATH="./output/custom_sft/vx-xxx/checkpoint-xxx"

echo "=========================================="
echo "开始自定义模型推理"
echo "模型路径: ${MODEL_PATH}"
echo "=========================================="

# 基础推理（不使用 adapter）
swift infer \
    --custom_register_path examples/custom/pretrain_example/register_model.py \
    --model_type custom_pretrain \
    --model ${MODEL_PATH} \
    --infer_backend pt \
    --max_new_tokens 512 \
    --temperature 0.7 \
    --top_p 0.9

# 如果需要使用微调后的 adapter，使用以下命令：
# swift infer \
#     --custom_register_path examples/custom/pretrain_example/register_model.py \
#     --adapters ${ADAPTER_PATH} \
#     --load_data_args true \
#     --infer_backend pt \
#     --max_new_tokens 512 \
#     --temperature 0.7 \
#     --top_p 0.9

echo "=========================================="
echo "推理完成!"
echo "=========================================="
