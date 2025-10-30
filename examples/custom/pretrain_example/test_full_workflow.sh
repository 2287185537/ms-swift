#!/bin/bash
# 完整工作流程测试脚本
# 验证从分词器训练到模型推理的完整流程

set -e  # 遇到错误立即退出

echo "=========================================="
echo "开始完整工作流程测试"
echo "=========================================="

# 测试目录
TEST_DIR="/tmp/pretrain_test_$(date +%s)"
TOKENIZER_DIR="$TEST_DIR/tokenizer"
MODEL_DIR="$TEST_DIR/model"

mkdir -p "$TEST_DIR"

echo ""
echo "测试环境："
echo "  Python版本: $(python --version)"
echo "  测试目录: $TEST_DIR"
echo ""

# ==========================================
# 步骤 1: 训练分词器
# ==========================================
echo "=========================================="
echo "步骤 1/5: 训练分词器"
echo "=========================================="

python train_tokenizer.py \
    --output_dir "$TOKENIZER_DIR" \
    --vocab_size 1000 \
    --min_frequency 1

if [ ! -f "$TOKENIZER_DIR/tokenizer.json" ]; then
    echo "✗ 分词器训练失败：文件不存在"
    exit 1
fi

echo "✓ 分词器训练成功"

# ==========================================
# 步骤 2: 初始化模型
# ==========================================
echo ""
echo "=========================================="
echo "步骤 2/5: 初始化模型（从分词器导入参数）"
echo "=========================================="

python init_weights.py \
    --tokenizer_path "$TOKENIZER_DIR" \
    --output_dir "$MODEL_DIR" \
    --hidden_size 768

if [ ! -f "$MODEL_DIR/pytorch_model.bin" ] && [ ! -f "$MODEL_DIR/model.safetensors" ]; then
    echo "✗ 模型初始化失败：权重文件不存在"
    exit 1
fi

echo "✓ 模型初始化成功"

# ==========================================
# 步骤 3: 测试模型加载
# ==========================================
echo ""
echo "=========================================="
echo "步骤 3/5: 测试模型加载"
echo "=========================================="

python -c "
from custom_model import CustomModelForCausalLM
from transformers import AutoTokenizer

model = CustomModelForCausalLM.from_pretrained('$MODEL_DIR')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_DIR', trust_remote_code=True)

print(f'✓ 模型加载成功')
print(f'  词汇表大小: {tokenizer.vocab_size}')
print(f'  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M')

# 验证参数一致性
assert model.config.vocab_size == tokenizer.vocab_size, '词汇表大小不一致'
print(f'✓ 验证通过: 模型和分词器参数一致')
"

# ==========================================
# 步骤 4: 测试前向传播
# ==========================================
echo ""
echo "=========================================="
echo "步骤 4/5: 测试前向传播"
echo "=========================================="

python -c "
from custom_model import CustomModelForCausalLM
from transformers import AutoTokenizer
import torch

model = CustomModelForCausalLM.from_pretrained('$MODEL_DIR')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_DIR', trust_remote_code=True)

# 测试1: 简单前向传播
text = '你好，世界！'
input_ids = tokenizer.encode(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(input_ids)
    logits = outputs.logits

print(f'✓ 前向传播测试成功')
print(f'  输入 shape: {input_ids.shape}')
print(f'  输出 logits shape: {logits.shape}')

# 测试2: 带 KV cache 的前向传播
with torch.no_grad():
    outputs = model(input_ids, use_cache=True)
    past_kv = outputs.past_key_values

print(f'✓ KV cache 测试成功')
print(f'  past_key_values 层数: {len(past_kv)}')
"

# ==========================================
# 步骤 5: 测试推理生成
# ==========================================
echo ""
echo "=========================================="
echo "步骤 5/5: 测试推理生成"
echo "=========================================="

python -c "
from custom_model import CustomModelForCausalLM
from transformers import AutoTokenizer
import torch

model = CustomModelForCausalLM.from_pretrained('$MODEL_DIR')
tokenizer = AutoTokenizer.from_pretrained('$MODEL_DIR', trust_remote_code=True)
model.eval()

prompt = '你好'
print(f'提示词: {prompt}')

input_ids = tokenizer.encode(prompt, return_tensors='pt')

# 生成文本（不使用 cache 以避免兼容性问题）
with torch.no_grad():
    output_ids = model.generate(
        input_ids,
        max_new_tokens=10,
        do_sample=False,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
        use_cache=False,
    )

generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(f'生成文本: {generated_text}')
print(f'生成的 token 数: {len(output_ids[0]) - len(input_ids[0])}')
print()
print(f'✓ 推理生成测试成功')
print(f'  注意: 由于模型未经训练，生成的文本是随机的')
"

# ==========================================
# 测试总结
# ==========================================
echo ""
echo "=========================================="
echo "测试完成！"
echo "=========================================="
echo ""
echo "所有测试通过："
echo "  ✓ 分词器训练"
echo "  ✓ 模型初始化（分词器参数导入）"
echo "  ✓ 模型加载"
echo "  ✓ 前向传播"
echo "  ✓ 推理生成"
echo ""
echo "测试文件保存在: $TEST_DIR"
echo ""
echo "下一步："
echo "  1. 准备预训练数据"
echo "  2. 运行预训练: bash pretrain.sh"
echo "  3. 运行推理: bash infer.sh"
echo ""
echo "=========================================="

# 清理测试文件（可选）
read -p "是否删除测试文件？(y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    rm -rf "$TEST_DIR"
    echo "✓ 测试文件已删除"
else
    echo "测试文件保留在: $TEST_DIR"
fi
