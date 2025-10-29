# 自定义预训练模型完整示例

本示例展示如何在 ms-swift 框架中**从零开始创建和预训练**一个完全自定义的模型。

**核心流程：**
1. 定义符合 HuggingFace 标准的模型结构
2. 初始化随机权重并保存
3. 注册模型到 ms-swift 框架
4. 使用 `swift pt` 命令进行**全量预训练**（类似于预训练 GPT/LLaMA）
5. 可选：继续预训练或进行微调（SFT）

**与微调的区别：**
- 本示例是**预训练（Pre-training）**：从随机初始化的权重开始训练
- 微调（Fine-tuning）：在已有预训练权重的基础上训练
- 使用 `swift pt` 而非 `swift sft` 命令

## 📋 目录结构

```
pretrain_example/
├── README.md               # 本文档
├── custom_model.py         # 自定义模型定义（符合 HuggingFace 标准）
├── init_weights.py         # 模型权重初始化脚本
├── register_model.py       # 模型注册到 ms-swift 框架
├── pretrain.sh            # 预训练脚本
├── sft.sh                 # 微调脚本
└── infer.sh               # 推理脚本
```

## 📦 依赖安装

在开始之前，确保安装必要的依赖：

```bash
# 安装 ms-swift (如果还没安装)
pip install ms-swift

# 安装分词器训练依赖
pip install tokenizers>=0.13.0

# 或者安装示例目录下的所有依赖
pip install -r requirements.txt
```

## 🚀 快速开始

### 方式一：一键运行完整流程（推荐）

使用一键脚本完成所有准备工作：

```bash
cd /path/to/ms-swift/examples/custom/pretrain_example

# 运行完整示例（包括训练分词器、初始化模型、准备数据）
bash run_full_example.sh
```

这个脚本会自动完成：
1. ✅ 训练自定义分词器（BPE，32000 词汇）
2. ✅ 根据分词器参数初始化模型
3. ✅ 准备示例预训练数据
4. ✅ 测试模型注册和推理

### 方式二：分步执行（学习流程）

#### 第一步：训练自定义分词器

**重要：先训练分词器，因为模型的词汇表大小等参数需要从分词器导入**

```bash
# 使用默认配置训练（会自动创建示例数据）
python train_tokenizer.py \
    --output_dir ./custom_tokenizer \
    --vocab_size 32000

# 使用自己的数据训练
python train_tokenizer.py \
    --input_files corpus1.txt corpus2.txt \
    --output_dir ./custom_tokenizer \
    --vocab_size 32000 \
    --min_frequency 2
```

**参数说明：**
- `--input_files`: 训练数据文件（纯文本，每行一个样本）
- `--output_dir`: 分词器保存路径
- `--vocab_size`: 词汇表大小（推荐 32000/50000）
- `--min_frequency`: 最小词频阈值

#### 第二步：根据分词器参数初始化模型

**关键流程：从分词器导入词汇表大小等参数到模型配置**

```bash
# 使用已训练的分词器初始化模型
python init_weights.py \
    --tokenizer_path ./custom_tokenizer \
    --output_dir ./custom_pretrained_model \
    --hidden_size 768

# 或者同时训练新分词器并初始化模型
python init_weights.py \
    --train_tokenizer \
    --output_dir ./custom_pretrained_model \
    --hidden_size 768
```

这个脚本的流程：
1. 加载或训练自定义分词器
2. **从分词器获取词汇表大小、特殊 token ID 等参数**
3. **使用这些参数创建模型配置**
4. 初始化模型随机权重
5. 保存模型和分词器
6. 验证模型和分词器的一致性

**参数说明：**
- `--tokenizer_path`: 已有分词器路径（如果不指定，会训练新分词器）
- `--train_tokenizer`: 强制训练新分词器
- `--output_dir`: 模型保存路径
- `--hidden_size`: 隐藏层维度（默认 768，可选 768/1024/2048）

**输出文件：**
```
custom_pretrained_model/
├── config.json              # 模型配置（包含从分词器导入的参数）
├── pytorch_model.bin        # 模型权重（随机初始化）
├── tokenizer_config.json    # 分词器配置
├── tokenizer.json           # 分词器词汇表
└── special_tokens_map.json  # 特殊 token 映射
```

#### 第三步：测试模型注册

验证模型是否正确注册到 ms-swift 框架：

```bash
python register_model.py
```

如果看到类似以下输出，说明注册成功：

```
✓ 模型加载成功!
  - 词汇表大小: 32000
✓ 推理成功!
```

### 第三步：准备预训练数据（可选）

如果你有自己的文本语料，可以使用工具脚本准备数据：

```bash
# 创建示例数据（用于快速测试）
python prepare_pretrain_data.py sample \
    --output ./data/sample_pretrain.jsonl \
    --num_samples 1000

# 从文本文件转换
python prepare_pretrain_data.py convert \
    --input corpus1.txt corpus2.txt \
    --output ./data/train.jsonl

# 从目录批量转换
python prepare_pretrain_data.py corpus \
    --input_dir ./corpus \
    --output ./data/train.jsonl

# 验证数据格式
python prepare_pretrain_data.py validate \
    --input ./data/train.jsonl
```

### 第四步：全量预训练模型

使用大规模文本数据从零开始训练模型：

```bash
bash pretrain.sh
```

**预训练脚本说明：**
- 使用 `swift pt` 命令进行**全量预训练**（Pre-training）
- 从随机初始化的权重开始训练
- 适合大规模无标注文本数据
- 支持多 GPU 训练（设置 `CUDA_VISIBLE_DEVICES`）
- 支持梯度检查点（节省显存）
- 支持 BF16 混合精度训练

**预训练 vs 微调：**
- **预训练（pt）**：从零开始训练，学习语言的基础知识
- **微调（sft）**：在已有模型基础上训练，学习特定任务

**可以修改的参数：**
- `MODEL_PATH`: 你初始化的随机权重模型路径
- `DATASET`: 预训练数据集（推荐使用大规模文本语料）
  - 示例：`AI-ModelScope/wikipedia-cn-20230720-filtered`
  - 或本地文本文件：`--dataset train.jsonl`
- `OUTPUT_DIR`: 输出路径
- 训练超参数：学习率、batch size、训练轮数等

**预训练数据格式：**
```jsonl
{"text": "这是第一段预训练文本..."}
{"text": "这是第二段预训练文本..."}
```

或纯文本文件（每行一个样本）：
```
这是第一段预训练文本...
这是第二段预训练文本...
```

### 第五步：微调模型（可选）

在预训练模型基础上进行 SFT 微调：

```bash
bash sft.sh
```

**微调脚本说明：**
- 使用 LoRA 进行高效微调
- 可以切换为全参数微调（`--train_type full`）
- 支持自定义数据集

### 第六步：推理测试

使用训练好的模型进行推理：

```bash
bash infer.sh
```

进入交互式对话：
```
<<< 你好，请介绍一下你自己
>>> （模型回复）

<<< 讲个笑话
>>> （模型回复）
```

## 📚 详细说明

### 1. 自定义模型结构 (`custom_model.py`)

这个文件定义了一个完整的 Transformer 解码器模型，包括：

**主要组件：**
- `CustomModelConfig`: 模型配置类（继承自 `PretrainedConfig`）
- `CustomAttention`: 多头自注意力层
- `CustomMLP`: 前馈神经网络层
- `CustomTransformerLayer`: Transformer 解码器层
- `CustomModel`: 模型主体
- `CustomModelForCausalLM`: 因果语言建模模型（带 LM Head）

**关键特性：**
- ✅ 符合 HuggingFace 标准接口
- ✅ 支持梯度检查点（节省显存）
- ✅ 支持 KV Cache（加速推理）
- ✅ 支持 Flash Attention（需要安装）
- ✅ 完整的中文注释

**模型架构：**
```
输入 tokens
    ↓
Token Embedding + Position Embedding
    ↓
Transformer Layer × N
    ├── Layer Norm
    ├── Multi-Head Attention
    ├── Residual Connection
    ├── Layer Norm
    ├── Feed Forward Network
    └── Residual Connection
    ↓
Final Layer Norm
    ↓
LM Head (输出 logits)
```

### 2. 分词器训练 (`train_tokenizer.py`)

这个脚本负责训练 BPE 分词器：

**训练流程：**
1. 创建 BPE (Byte Pair Encoding) 模型
2. 在文本数据上训练，学习最优的 subword 切分
3. 保存为 HuggingFace 格式
4. 自动测试分词效果

**关键特性：**
- 支持任意 Unicode 字符
- 可自定义特殊 token
- 自动生成词汇表
- 提供编码/解码测试

**输出：**
```
custom_tokenizer/
├── tokenizer_config.json    # 分词器配置
├── tokenizer.json           # 词汇表和合并规则
└── special_tokens_map.json  # 特殊 token 映射
```

### 3. 权重初始化 (`init_weights.py`)

**核心流程：分词器参数 → 模型配置**

这个脚本的关键创新是：**先准备分词器，再根据分词器参数初始化模型**

```python
# 伪代码说明参数导入流程
tokenizer = load_or_train_tokenizer()  # 步骤1: 准备分词器

# 步骤2: 从分词器导入参数
vocab_size = tokenizer.vocab_size          # 词汇表大小
pad_token_id = tokenizer.pad_token_id      # PAD token ID
bos_token_id = tokenizer.bos_token_id      # BOS token ID
eos_token_id = tokenizer.eos_token_id      # EOS token ID

# 步骤3: 使用这些参数创建模型配置
config = CustomModelConfig(
    vocab_size=vocab_size,        # 从分词器导入！
    pad_token_id=pad_token_id,    # 从分词器导入！
    bos_token_id=bos_token_id,    # 从分词器导入！
    eos_token_id=eos_token_id,    # 从分词器导入！
    hidden_size=768,              # 用户指定
    num_hidden_layers=12,         # 用户指定
    ...
)

# 步骤4: 初始化模型
model = CustomModelForCausalLM(config)

# 步骤5: 验证一致性
assert model.config.vocab_size == tokenizer.vocab_size
```

**为什么要这样做？**
1. ✅ **保证一致性**：模型的词汇表大小必须与分词器完全一致
2. ✅ **避免错误**：手动指定容易出错，从分词器导入更可靠
3. ✅ **灵活性**：可以方便地使用不同词汇表大小的分词器
4. ✅ **符合最佳实践**：这是工业界标准流程

**初始化策略：**
- 使用正态分布初始化权重（标准差 0.02）
- Linear 层的 bias 初始化为 0
- Embedding 层使用正态分布
- LayerNorm 的 weight 初始化为 1，bias 初始化为 0

**输出文件：**
```
custom_pretrained_model/
├── config.json              # 模型配置（包含从分词器导入的参数）
├── pytorch_model.bin        # 模型权重（随机初始化）
├── tokenizer_config.json    # 分词器配置（从分词器复制）
├── tokenizer.json           # 分词器词汇表（从分词器复制）
└── special_tokens_map.json  # 特殊 token 映射（从分词器复制）
```

### 3. 模型注册 (`register_model.py`)

这个文件将自定义模型注册到 ms-swift 框架：

**注册内容：**
1. **模型加载函数** (`get_custom_model_tokenizer`)
   - 负责加载模型和 tokenizer
   - 支持自定义加载逻辑
   - 可以添加预处理/后处理

2. **对话模板** (`register_template`)
   - 定义用户和助手的对话格式
   - 设置特殊 token（如 `<|endoftext|>`）
   - 可以自定义 system prompt

3. **模型元信息** (`register_model`)
   - 模型类型标识
   - 支持的模型路径
   - 依赖包列表
   - 模型标签

**对话模板示例：**
```
<|system|>
You are a helpful assistant.
<|user|>
你好，请介绍一下你自己
<|assistant|>
（模型回复）
<|endoftext|>
```

### 4. 训练脚本

#### 预训练 (`pretrain.sh`)

**swift pt 命令说明（Pre-training，全量预训练）：**
```bash
swift pt \
    --custom_register_path register_model.py \  # 自定义模型注册文件
    --model_type custom_pretrain \              # 模型类型
    --model ./custom_pretrained_model \         # 随机初始化的模型路径
    --dataset <dataset_name> \                  # 预训练数据集
    --train_type full \                         # 全参数训练（预训练必须用 full）
    --num_train_epochs 1 \                      # 训练轮数
    --per_device_train_batch_size 4 \           # 每个设备的 batch size
    --learning_rate 1e-4 \                      # 学习率（预训练通常用较大学习率）
    --weight_decay 0.1 \                        # 权重衰减（预训练推荐 0.1）
    --warmup_ratio 0.01 \                       # 预热比例（预训练推荐 0.01-0.03）
    --gradient_accumulation_steps 4 \           # 梯度累积步数
    --max_length 2048 \                         # 最大序列长度
    --output_dir ./output/custom_pretrain \     # 输出路径
    --bf16 true \                               # 使用 BF16 混合精度
    --use_flash_attn true                       # 使用 Flash Attention 加速
```

**预训练数据格式（纯文本语料）：**

1. **JSONL 格式**（推荐）：
```jsonl
{"text": "这是一段预训练文本，可以是维基百科、书籍、网页等任意文本..."}
{"text": "另一段预训练文本..."}
```

2. **纯文本格式**：
```
这是第一段预训练文本。
这是第二段预训练文本。
```

3. **预训练数据集推荐**：
   - 中文：`AI-ModelScope/wikipedia-cn-20230720-filtered`
   - 英文：`c4`，`pile`
   - 代码：`bigcode/the-stack`
   - 多语言：`mc4`

**注意事项：**
- 预训练**必须使用 `--train_type full`**（全参数训练）
- 预训练数据应该是**大规模无标注文本**
- 预训练需要较长时间和大量数据（通常需要数十 GB 到数百 GB 文本）
- 学习率通常比微调更大（1e-4 vs 1e-5）
- 权重衰减通常更大（0.1 vs 0.01）

#### 微调 (`sft.sh`)

与预训练类似，但使用 `swift sft` 命令，并且：
- 默认使用 LoRA 训练（更高效）
- 可以在预训练模型基础上继续训练
- 支持指令微调数据集

#### 推理 (`infer.sh`)

**推理模式：**
1. **交互式推理**（默认）
   - 启动后进入对话模式
   - 输入问题，模型生成回复

2. **批量推理**
   ```bash
   swift infer \
       --custom_register_path register_model.py \
       --model ./custom_pretrained_model \
       --val_dataset <dataset> \
       --infer_backend pt
   ```

3. **使用微调后的 adapter**
   ```bash
   swift infer \
       --adapters ./output/custom_sft/checkpoint-xxx \
       --load_data_args true
   ```

## 🔧 高级用法

### 1. 自定义模型结构

如果需要修改模型结构，编辑 `custom_model.py`：

**增加层数：**
```python
config = CustomModelConfig(
    num_hidden_layers=24,  # 从 12 改为 24
    ...
)
```

**修改注意力头数：**
```python
config = CustomModelConfig(
    num_attention_heads=16,  # 从 12 改为 16
    hidden_size=1024,        # 必须能被 num_attention_heads 整除
    ...
)
```

**添加 RoPE (Rotary Position Embedding)：**
可以参考 LLaMA 的实现，在 `CustomAttention` 中添加 RoPE。

### 2. 使用自己的 Tokenizer

如果需要训练自己的 tokenizer：

```python
from tokenizers import Tokenizer, models, trainers, pre_tokenizers

# 训练 BPE tokenizer
tokenizer = Tokenizer(models.BPE())
tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel()

trainer = trainers.BpeTrainer(
    vocab_size=32000,
    special_tokens=["<|endoftext|>", "<|user|>", "<|assistant|>"]
)

# 在你的文本数据上训练
tokenizer.train(files=["your_data.txt"], trainer=trainer)

# 保存
tokenizer.save("custom_tokenizer.json")
```

然后在 `init_weights.py` 中加载你的 tokenizer。

### 3. 使用自定义数据集

创建自定义数据集（参考 `examples/custom/dataset.py`）：

```python
from swift.llm import DatasetMeta, ResponsePreprocessor, register_dataset

class MyPreprocessor(ResponsePreprocessor):
    def preprocess(self, row):
        return {
            'messages': [
                {'role': 'user', 'content': row['input']},
                {'role': 'assistant', 'content': row['output']}
            ]
        }

register_dataset(
    DatasetMeta(
        ms_dataset_id='my_dataset',
        preprocess_func=MyPreprocessor(),
    ))
```

然后在训练时：
```bash
swift pt \
    --custom_register_path register_model.py \
                           my_dataset.py \
    --dataset my_dataset \
    ...
```

### 4. 多 GPU 训练

**数据并行：**
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 \
swift pt \
    --model ./custom_pretrained_model \
    --dataset <dataset> \
    ...
```

**DeepSpeed ZeRO-2（节省显存）：**
```bash
swift pt \
    --model ./custom_pretrained_model \
    --dataset <dataset> \
    --deepspeed default-zero2 \
    ...
```

**DeepSpeed ZeRO-3（极致节省显存）：**
```bash
swift pt \
    --model ./custom_pretrained_model \
    --dataset <dataset> \
    --deepspeed default-zero3 \
    ...
```

### 5. 模型量化

**使用 GPTQ 量化：**
```bash
swift export \
    --model ./output/custom_pretrain/checkpoint-xxx \
    --quant_bits 4 \
    --quant_method gptq \
    --output_dir ./quantized_model
```

**使用 AWQ 量化：**
```bash
swift export \
    --model ./output/custom_pretrain/checkpoint-xxx \
    --quant_bits 4 \
    --quant_method awq \
    --output_dir ./quantized_model
```

## 📊 性能优化建议

### 显存优化

1. **使用梯度检查点**
   ```bash
   --gradient_checkpointing true
   ```
   - 用时间换显存
   - 可以节省 30-50% 显存
   - 训练速度降低 20-30%

2. **使用混合精度训练**
   ```bash
   --bf16 true  # 或 --fp16 true
   ```
   - 节省 50% 显存
   - 加速训练 2-3 倍
   - 数值稳定性好（BF16 优于 FP16）

3. **减小 batch size，增加梯度累积**
   ```bash
   --per_device_train_batch_size 1 \
   --gradient_accumulation_steps 16
   ```
   - 等效 batch size = 1 × 16 = 16
   - 显存占用更少

4. **使用 LoRA 而非全参数训练**
   ```bash
   --train_type lora \
   --lora_rank 8
   ```
   - 只训练 < 1% 的参数
   - 显存占用大幅降低

### 训练速度优化

1. **使用 Flash Attention**
   ```bash
   pip install flash-attn
   ```
   - 自动启用（如果安装）
   - 加速注意力计算 2-4 倍

2. **增加数据加载 workers**
   ```bash
   --dataloader_num_workers 8 \
   --dataset_num_proc 16
   ```

3. **使用更大的 batch size**
   ```bash
   --per_device_train_batch_size 4 \
   --gradient_accumulation_steps 4
   ```

## 🐛 常见问题

### 1. OOM (Out of Memory) 错误

**解决方案：**
- 减小 `per_device_train_batch_size`
- 启用 `gradient_checkpointing`
- 使用 DeepSpeed ZeRO
- 减小 `max_length`

### 2. 训练速度很慢

**解决方案：**
- 安装 Flash Attention
- 使用 BF16/FP16 混合精度
- 增加 `dataloader_num_workers`
- 检查是否在使用 CPU（应该使用 GPU）

### 3. 模型加载失败

**检查：**
- 是否运行了 `init_weights.py`
- `register_model.py` 中的模型路径是否正确
- 是否导入了 `custom_model.py`

### 4. 生成质量不好

**可能原因：**
- 训练数据不足
- 训练轮数太少
- 学习率不合适
- 需要更多的训练数据和更长的训练时间

## 📖 参考资料

- [ms-swift 官方文档](https://github.com/modelscope/ms-swift)
- [HuggingFace Transformers 文档](https://huggingface.co/docs/transformers)
- [深度学习优化技巧](https://github.com/modelscope/ms-swift/blob/main/docs)

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！

## 📄 许可证

本示例遵循 Apache 2.0 许可证。
