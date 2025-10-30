# 快速开始指南

这是一个完整的自定义预训练模型示例，展示了**从训练分词器到预训练模型**的完整流程。

## 🎯 核心特性

本示例的核心创新是：**分词器优先，参数导入**

传统流程（容易出错）：
```
❌ 手动指定 vocab_size → 创建模型 → 训练分词器 → 可能不一致
```

本示例流程（最佳实践）：
```
✅ 训练分词器 → 从分词器导入参数 → 创建模型 → 确保一致性
```

## 📦 安装依赖

```bash
# 1. 安装 ms-swift
pip install ms-swift

# 2. 安装分词器训练工具
pip install tokenizers>=0.13.0

# 3. 或者一次性安装所有依赖
pip install -r requirements.txt
```

## 🚀 三种使用方式

### 方式一：一键运行（最快）

```bash
bash run_full_example.sh
```

这个脚本会自动：
- ✅ 训练一个 32000 词汇的 BPE 分词器
- ✅ 根据分词器参数初始化模型
- ✅ 准备示例预训练数据
- ✅ 测试模型注册和推理

### 方式二：分步执行（学习流程）

#### 步骤 1: 训练分词器

```bash
# 使用示例数据训练（快速测试）
python train_tokenizer.py \
    --output_dir ./custom_tokenizer \
    --vocab_size 32000

# 使用自己的数据训练
python train_tokenizer.py \
    --input_files corpus1.txt corpus2.txt corpus3.txt \
    --output_dir ./custom_tokenizer \
    --vocab_size 32000
```

#### 步骤 2: 初始化模型（从分词器导入参数）

```bash
python init_weights.py \
    --tokenizer_path ./custom_tokenizer \
    --output_dir ./custom_pretrained_model \
    --hidden_size 768
```

**关键：这一步会从分词器导入以下参数到模型配置**
- `vocab_size`: 词汇表大小
- `pad_token_id`: PAD token ID
- `bos_token_id`: BOS token ID  
- `eos_token_id`: EOS token ID

#### 步骤 3: 预训练

```bash
# 编辑 pretrain.sh 设置你的数据集
# 然后运行
bash pretrain.sh
```

### 方式三：代码层面使用

```python
from train_tokenizer import train_custom_tokenizer
from custom_model import CustomModelConfig, CustomModelForCausalLM

# 1. 训练分词器
tokenizer = train_custom_tokenizer(
    input_files=['data1.txt', 'data2.txt'],
    output_dir='./my_tokenizer',
    vocab_size=32000
)

# 2. 从分词器导入参数
vocab_size = tokenizer.vocab_size
pad_token_id = tokenizer.pad_token_id
bos_token_id = tokenizer.bos_token_id
eos_token_id = tokenizer.eos_token_id

# 3. 创建模型配置（使用分词器参数）
config = CustomModelConfig(
    vocab_size=vocab_size,  # 从分词器导入
    pad_token_id=pad_token_id,  # 从分词器导入
    bos_token_id=bos_token_id,  # 从分词器导入
    eos_token_id=eos_token_id,  # 从分词器导入
    hidden_size=768,
    num_hidden_layers=12,
)

# 4. 初始化模型
model = CustomModelForCausalLM(config)

# 5. 保存
model.save_pretrained('./my_model')
tokenizer.save_pretrained('./my_model')
```

## 📝 文件说明

| 文件 | 用途 |
|------|------|
| `train_tokenizer.py` | 训练 BPE 分词器 |
| `custom_model.py` | 自定义模型定义（Transformer 解码器） |
| `init_weights.py` | 初始化模型（从分词器导入参数） |
| `register_model.py` | 注册模型到 ms-swift |
| `prepare_pretrain_data.py` | 预训练数据准备工具 |
| `pretrain.sh` | 预训练脚本（使用 `swift pt`） |
| `sft.sh` | 微调脚本 |
| `infer.sh` | 推理脚本 |
| `run_full_example.sh` | 一键运行完整流程 |
| `test_model_only.py` | 测试模型基础功能 |
| `README.md` | 详细文档 |

## 🔍 验证流程

### 测试 1: 模型定义测试（不需要分词器）

```bash
python test_model_only.py
```

这个测试会验证：
- ✓ 模型创建
- ✓ 前向传播
- ✓ 损失计算
- ✓ 文本生成
- ✓ 模型保存和加载
- ✓ 分词器参数导入流程（模拟）

### 测试 2: 完整流程测试（需要分词器）

```bash
python test_workflow.py
```

这个测试会验证：
- ✓ 分词器训练
- ✓ 模型初始化（真实的参数导入）
- ✓ 模型前向传播
- ✓ 文本生成

## ❓ 常见问题

### Q1: 为什么要先训练分词器？

**A**: 因为模型的词汇表大小必须与分词器完全一致。先训练分词器，再从分词器导入参数，可以确保一致性。

### Q2: 我可以使用已有的分词器吗？

**A**: 可以！使用 `--tokenizer_path` 参数：

```bash
python init_weights.py \
    --tokenizer_path /path/to/existing/tokenizer \
    --output_dir ./my_model
```

### Q3: 我想用不同的词汇表大小怎么办？

**A**: 重新训练分词器：

```bash
python train_tokenizer.py \
    --vocab_size 50000 \
    --output_dir ./new_tokenizer

python init_weights.py \
    --tokenizer_path ./new_tokenizer \
    --output_dir ./my_model
```

### Q4: 这个流程和微调有什么区别？

**A**:
- **预训练**（本示例）：从随机权重开始，使用大规模无标注文本，学习语言基础知识
- **微调**：在已有预训练权重基础上，使用标注数据，学习特定任务

### Q5: 如何使用自己的数据？

**A**: 准备纯文本文件（或 JSONL），然后：

```bash
# 训练分词器
python train_tokenizer.py \
    --input_files your_data.txt \
    --output_dir ./tokenizer

# 初始化模型
python init_weights.py \
    --tokenizer_path ./tokenizer \
    --output_dir ./model

# 修改 pretrain.sh 中的 DATASET 参数
# 然后运行预训练
bash pretrain.sh
```

## 🎓 学习路径

1. **第一步**：运行 `python test_model_only.py` 理解模型结构
2. **第二步**：阅读 `custom_model.py` 理解 Transformer 实现
3. **第三步**：运行 `python train_tokenizer.py` 理解分词器训练
4. **第四步**：阅读 `init_weights.py` 理解参数导入流程
5. **第五步**：运行 `bash run_full_example.sh` 走通完整流程
6. **第六步**：使用自己的数据训练分词器和模型

## 📚 延伸阅读

- [完整文档](README.md)
- [ms-swift 官方文档](https://github.com/modelscope/ms-swift)
- [HuggingFace Transformers](https://huggingface.co/docs/transformers)
- [BPE 分词算法](https://huggingface.co/docs/tokenizers/pipeline)

## 🤝 反馈

如有问题，请在 GitHub 上提 Issue。
