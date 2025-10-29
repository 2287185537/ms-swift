# Copyright (c) Alibaba, Inc. and its affiliates.
"""
训练自定义分词器（Tokenizer）
这个脚本展示如何从零开始训练一个 BPE 分词器，并保存为 HuggingFace 格式
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders, processors


def train_custom_tokenizer(
    input_files: List[str],
    output_dir: str,
    vocab_size: int = 32000,
    min_frequency: int = 2,
    special_tokens: Optional[List[str]] = None
):
    """
    训练 BPE 分词器
    
    Args:
        input_files: 训练数据文件列表（纯文本文件）
        output_dir: 分词器保存路径
        vocab_size: 词汇表大小
        min_frequency: 最小词频（低于此频率的词不会被加入词汇表）
        special_tokens: 特殊 token 列表
    """
    print("="*60)
    print("开始训练自定义分词器")
    print("="*60)
    
    # 1. 定义特殊 token
    if special_tokens is None:
        special_tokens = [
            "<|endoftext|>",    # 文本结束标记
            "<|padding|>",      # 填充标记
            "<|unknown|>",      # 未知词标记
            "<|user|>",         # 用户标记
            "<|assistant|>",    # 助手标记
            "<|system|>",       # 系统标记
        ]
    
    print(f"\n特殊 token: {special_tokens}")
    print(f"词汇表大小: {vocab_size}")
    print(f"最小词频: {min_frequency}")
    
    # 2. 创建 BPE 模型
    # BPE (Byte Pair Encoding) 是目前最流行的分词算法
    print(f"\n正在初始化 BPE 分词器...")
    tokenizer = Tokenizer(models.BPE())
    
    # 3. 设置预处理器（将文本转换为字节序列）
    # ByteLevel 预处理可以处理任意 Unicode 字符
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    
    # 4. 设置解码器
    tokenizer.decoder = decoders.ByteLevel()
    
    # 5. 创建训练器
    trainer = trainers.BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=special_tokens,
        show_progress=True,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet()
    )
    
    # 6. 在数据上训练
    print(f"\n正在训练分词器...")
    print(f"训练数据文件: {input_files}")
    
    if not input_files or not all(os.path.exists(f) for f in input_files):
        print(f"警告: 某些训练文件不存在，将使用示例数据训练")
        # 创建示例训练数据
        sample_file = "/tmp/sample_tokenizer_data.txt"
        with open(sample_file, 'w', encoding='utf-8') as f:
            f.write("这是一段示例文本，用于训练分词器。\n")
            f.write("Hello, this is sample text for training tokenizer.\n")
            f.write("分词器会学习如何将文本分解为 token。\n")
            f.write("The tokenizer learns how to split text into tokens.\n")
            # 重复以增加训练数据量
            for i in range(100):
                f.write(f"示例句子 {i}: 人工智能、深度学习、自然语言处理。\n")
                f.write(f"Sample sentence {i}: AI, deep learning, NLP.\n")
        input_files = [sample_file]
        print(f"使用示例数据: {sample_file}")
    
    tokenizer.train(files=input_files, trainer=trainer)
    print(f"✓ 训练完成!")
    
    # 7. 设置后处理器（添加特殊 token）
    # 为对话格式设置模板
    tokenizer.post_processor = processors.ByteLevel(trim_offsets=False)
    
    # 8. 保存原始 tokenizer
    os.makedirs(output_dir, exist_ok=True)
    tokenizer_file = os.path.join(output_dir, "tokenizer.json")
    tokenizer.save(tokenizer_file)
    print(f"\n✓ 原始分词器已保存到: {tokenizer_file}")
    
    # 9. 转换为 HuggingFace PreTrainedTokenizer 格式
    print(f"\n正在转换为 HuggingFace 格式...")
    from transformers import PreTrainedTokenizerFast
    
    # 创建 PreTrainedTokenizerFast 包装器
    hf_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<|unknown|>",
        pad_token="<|padding|>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
        clean_up_tokenization_spaces=False,
    )
    
    # 设置特殊 token ID
    hf_tokenizer.pad_token_id = hf_tokenizer.convert_tokens_to_ids("<|padding|>")
    hf_tokenizer.bos_token_id = hf_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    hf_tokenizer.eos_token_id = hf_tokenizer.convert_tokens_to_ids("<|endoftext|>")
    hf_tokenizer.unk_token_id = hf_tokenizer.convert_tokens_to_ids("<|unknown|>")
    
    # 保存为 HuggingFace 格式
    hf_tokenizer.save_pretrained(output_dir)
    print(f"✓ HuggingFace 格式分词器已保存到: {output_dir}")
    
    # 10. 验证和测试
    print(f"\n" + "="*60)
    print("分词器信息:")
    print("="*60)
    print(f"词汇表大小: {hf_tokenizer.vocab_size}")
    print(f"PAD token: {hf_tokenizer.pad_token} (ID: {hf_tokenizer.pad_token_id})")
    print(f"BOS token: {hf_tokenizer.bos_token} (ID: {hf_tokenizer.bos_token_id})")
    print(f"EOS token: {hf_tokenizer.eos_token} (ID: {hf_tokenizer.eos_token_id})")
    print(f"UNK token: {hf_tokenizer.unk_token} (ID: {hf_tokenizer.unk_token_id})")
    
    # 11. 测试分词器
    print(f"\n" + "="*60)
    print("分词测试:")
    print("="*60)
    
    test_texts = [
        "你好，世界！",
        "Hello, world!",
        "这是一个测试句子。",
        "This is a test sentence."
    ]
    
    for text in test_texts:
        tokens = hf_tokenizer.tokenize(text)
        ids = hf_tokenizer.encode(text, add_special_tokens=False)
        decoded = hf_tokenizer.decode(ids)
        print(f"\n原始文本: {text}")
        print(f"分词结果: {tokens[:20]}{'...' if len(tokens) > 20 else ''}")
        print(f"Token IDs: {ids[:20]}{'...' if len(ids) > 20 else ''}")
        print(f"解码结果: {decoded}")
    
    print(f"\n" + "="*60)
    print("分词器训练完成!")
    print(f"保存路径: {output_dir}")
    print("="*60)
    
    return hf_tokenizer


def main():
    parser = argparse.ArgumentParser(description='训练自定义分词器')
    parser.add_argument(
        '--input_files',
        nargs='+',
        help='训练数据文件（纯文本，一行一个样本）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./custom_tokenizer',
        help='分词器保存路径'
    )
    parser.add_argument(
        '--vocab_size',
        type=int,
        default=32000,
        help='词汇表大小（推荐: 32000/50000/100000）'
    )
    parser.add_argument(
        '--min_frequency',
        type=int,
        default=2,
        help='最小词频（低于此频率的词不会被加入词汇表）'
    )
    parser.add_argument(
        '--special_tokens',
        nargs='+',
        help='自定义特殊 token'
    )
    
    args = parser.parse_args()
    
    # 训练分词器
    train_custom_tokenizer(
        input_files=args.input_files or [],
        output_dir=args.output_dir,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
        special_tokens=args.special_tokens
    )


if __name__ == '__main__':
    # 如果直接运行，显示使用说明
    import sys
    if len(sys.argv) == 1:
        print("="*60)
        print("自定义分词器训练工具")
        print("="*60)
        print("\n使用方法:")
        print("\n1. 使用默认配置训练（会自动创建示例数据）:")
        print("   python train_tokenizer.py")
        print("\n2. 使用自己的数据训练:")
        print("   python train_tokenizer.py \\")
        print("       --input_files corpus1.txt corpus2.txt \\")
        print("       --output_dir ./custom_tokenizer \\")
        print("       --vocab_size 32000")
        print("\n3. 自定义特殊 token:")
        print("   python train_tokenizer.py \\")
        print("       --input_files data.txt \\")
        print("       --special_tokens '<|endoftext|>' '<|user|>' '<|assistant|>'")
        print("\n" + "="*60)
        print("\n提示: 如果不指定 input_files，将使用示例数据进行训练")
        print()
    
    main()
