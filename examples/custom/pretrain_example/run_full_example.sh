#!/bin/bash
# 完整的端到端示例：从训练分词器到预训练模型
# 这个脚本展示了完整的流程

set -e  # 遇到错误立即退出

echo "=========================================="
echo "自定义模型预训练完整流程示例"
echo "=========================================="
echo ""
echo "流程说明:"
echo "1. 训练自定义分词器"
echo "2. 根据分词器参数初始化模型"
echo "3. 准备预训练数据（示例）"
echo "4. 开始预训练（可选）"
echo ""

# 设置路径
TOKENIZER_DIR="./custom_tokenizer"
MODEL_DIR="./custom_pretrained_model"
DATA_DIR="./data"

# ========================================
# 步骤 1: 训练自定义分词器
# ========================================
echo "=========================================="
echo "步骤 1: 训练自定义分词器"
echo "=========================================="

if [ -d "$TOKENIZER_DIR" ]; then
    echo "分词器目录已存在: $TOKENIZER_DIR"
    read -p "是否重新训练? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$TOKENIZER_DIR"
        python train_tokenizer.py \
            --output_dir "$TOKENIZER_DIR" \
            --vocab_size 32000 \
            --min_frequency 2
    fi
else
    echo "开始训练分词器..."
    python train_tokenizer.py \
        --output_dir "$TOKENIZER_DIR" \
        --vocab_size 32000 \
        --min_frequency 2
fi

echo ""
echo "✓ 分词器准备完成: $TOKENIZER_DIR"
echo ""
read -p "按 Enter 继续..."

# ========================================
# 步骤 2: 根据分词器参数初始化模型
# ========================================
echo ""
echo "=========================================="
echo "步骤 2: 根据分词器参数初始化模型"
echo "=========================================="

if [ -d "$MODEL_DIR" ]; then
    echo "模型目录已存在: $MODEL_DIR"
    read -p "是否重新初始化? (y/n): " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        rm -rf "$MODEL_DIR"
        python init_weights.py \
            --tokenizer_path "$TOKENIZER_DIR" \
            --output_dir "$MODEL_DIR" \
            --hidden_size 768
    fi
else
    echo "开始初始化模型..."
    python init_weights.py \
        --tokenizer_path "$TOKENIZER_DIR" \
        --output_dir "$MODEL_DIR" \
        --hidden_size 768
fi

echo ""
echo "✓ 模型初始化完成: $MODEL_DIR"
echo ""
read -p "按 Enter 继续..."

# ========================================
# 步骤 3: 准备预训练数据（示例）
# ========================================
echo ""
echo "=========================================="
echo "步骤 3: 准备预训练数据"
echo "=========================================="

mkdir -p "$DATA_DIR"

if [ ! -f "$DATA_DIR/sample_pretrain.jsonl" ]; then
    echo "创建示例预训练数据..."
    python prepare_pretrain_data.py sample \
        --output "$DATA_DIR/sample_pretrain.jsonl" \
        --num_samples 1000
else
    echo "预训练数据已存在: $DATA_DIR/sample_pretrain.jsonl"
fi

echo ""
echo "✓ 预训练数据准备完成: $DATA_DIR/sample_pretrain.jsonl"
echo ""

# ========================================
# 步骤 4: 测试模型注册
# ========================================
echo ""
echo "=========================================="
echo "步骤 4: 测试模型注册"
echo "=========================================="
echo ""
echo "运行模型注册测试..."
python register_model.py

echo ""
echo "✓ 模型注册测试完成"
echo ""

# ========================================
# 总结
# ========================================
echo ""
echo "=========================================="
echo "准备工作完成！"
echo "=========================================="
echo ""
echo "已完成的步骤:"
echo "  ✓ 训练自定义分词器: $TOKENIZER_DIR"
echo "  ✓ 初始化模型权重: $MODEL_DIR"
echo "  ✓ 准备预训练数据: $DATA_DIR/sample_pretrain.jsonl"
echo "  ✓ 测试模型注册"
echo ""
echo "下一步操作:"
echo ""
echo "1. 开始预训练（使用示例数据进行快速测试）:"
echo "   bash pretrain.sh"
echo ""
echo "2. 使用自己的数据预训练:"
echo "   修改 pretrain.sh 中的 DATASET 参数，然后运行:"
echo "   bash pretrain.sh"
echo ""
echo "3. 推理测试:"
echo "   bash infer.sh"
echo ""
echo "=========================================="
