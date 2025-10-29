# Copyright (c) Alibaba, Inc. and its affiliates.
"""
初始化模型权重并保存
这个脚本展示如何创建一个从零开始的模型，初始化权重，并保存为 HuggingFace 格式
"""

import os
import argparse
from transformers import AutoTokenizer

# 导入自定义模型
from custom_model import CustomModelConfig, CustomModelForCausalLM


def init_and_save_model(
    output_dir: str,
    tokenizer_path: str = None,
    train_tokenizer: bool = False,
    hidden_size: int = 768
):
    """
    初始化模型权重并保存
    
    流程：
    1. 先加载或训练自定义分词器
    2. 从分词器获取词汇表大小等参数
    3. 使用这些参数初始化模型配置
    4. 初始化模型权重
    5. 保存模型和分词器
    
    Args:
        output_dir: 模型保存路径
        tokenizer_path: 分词器路径（如果为 None 且 train_tokenizer=False，则训练新分词器）
        train_tokenizer: 是否训练新的分词器
        hidden_size: 隐藏层维度
    """
    print("="*60)
    print("开始初始化自定义模型")
    print("="*60)
    
    # ========== 步骤1: 准备分词器 ==========
    print(f"\n步骤1: 准备分词器")
    print("-"*60)
    
    if train_tokenizer or tokenizer_path is None:
        # 训练新的分词器
        print(f"正在训练新的分词器...")
        from train_tokenizer import train_custom_tokenizer
        
        tokenizer_dir = os.path.join(os.path.dirname(output_dir), "custom_tokenizer")
        tokenizer = train_custom_tokenizer(
            input_files=[],  # 使用示例数据
            output_dir=tokenizer_dir,
            vocab_size=32000,
            min_frequency=2
        )
        print(f"✓ 分词器训练完成，保存在: {tokenizer_dir}")
    else:
        # 加载已有的分词器
        print(f"正在从 {tokenizer_path} 加载分词器...")
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
        print(f"✓ 分词器加载完成")
    
    # ========== 步骤2: 从分词器获取参数 ==========
    print(f"\n步骤2: 从分词器获取参数")
    print("-"*60)
    
    # 从分词器获取关键参数
    vocab_size = tokenizer.vocab_size
    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    bos_token_id = tokenizer.bos_token_id if tokenizer.bos_token_id is not None else 1
    eos_token_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else 2
    
    print(f"从分词器获取的参数:")
    print(f"  - 词汇表大小: {vocab_size}")
    print(f"  - PAD token ID: {pad_token_id} ({tokenizer.pad_token})")
    print(f"  - BOS token ID: {bos_token_id} ({tokenizer.bos_token})")
    print(f"  - EOS token ID: {eos_token_id} ({tokenizer.eos_token})")
    
    # ========== 步骤3: 创建模型配置（使用分词器参数） ==========
    print(f"\n步骤3: 创建模型配置")
    print("-"*60)
    
    config = CustomModelConfig(
        vocab_size=vocab_size,  # 从分词器导入
        hidden_size=hidden_size,
        num_hidden_layers=12,  # 12层 Transformer
        num_attention_heads=12,  # 12个注意力头
        intermediate_size=hidden_size * 4,  # FFN 中间层是 hidden_size 的 4 倍
        max_position_embeddings=2048,  # 支持最长 2048 tokens
        hidden_dropout_prob=0.1,
        attention_probs_dropout_prob=0.1,
        initializer_range=0.02,
        layer_norm_eps=1e-12,
        pad_token_id=pad_token_id,  # 从分词器导入
        bos_token_id=bos_token_id,  # 从分词器导入
        eos_token_id=eos_token_id,  # 从分词器导入
        tie_word_embeddings=False,  # 不共享输入输出 embedding
    )
    
    print(f"模型配置:")
    print(f"  - 词汇表大小: {config.vocab_size} (来自分词器)")
    print(f"  - 隐藏层维度: {config.hidden_size}")
    print(f"  - Transformer 层数: {config.num_hidden_layers}")
    print(f"  - 注意力头数: {config.num_attention_heads}")
    print(f"  - 最大序列长度: {config.max_position_embeddings}")
    print(f"  - PAD/BOS/EOS token ID: {pad_token_id}/{bos_token_id}/{eos_token_id} (来自分词器)")
    
    # ========== 步骤4: 初始化模型（自动初始化随机权重） ==========
    print(f"\n步骤4: 初始化模型权重")
    print("-"*60)
    
    model = CustomModelForCausalLM(config)
    
    # 统计参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"总参数量: {total_params:,} ({total_params / 1e6:.2f}M)")
    print(f"可训练参数: {trainable_params:,} ({trainable_params / 1e6:.2f}M)")
    
    # 验证模型和分词器的词汇表大小一致
    assert model.config.vocab_size == tokenizer.vocab_size, \
        f"模型词汇表大小 ({model.config.vocab_size}) 与分词器 ({tokenizer.vocab_size}) 不一致!"
    print(f"✓ 验证通过: 模型词汇表大小与分词器一致")
    
    # ========== 步骤5: 保存模型和分词器 ==========
    print(f"\n步骤5: 保存模型和分词器")
    print("-"*60)
    print(f"保存路径: {output_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    # 保存模型
    model.save_pretrained(output_dir)
    print(f"✓ 模型已保存")
    
    # 保存分词器
    tokenizer.save_pretrained(output_dir)
    print(f"✓ 分词器已保存")
    
    # ========== 步骤6: 验证保存的模型可以加载 ==========
    print(f"\n步骤6: 验证模型加载")
    print("-"*60)
    
    loaded_model = CustomModelForCausalLM.from_pretrained(output_dir)
    loaded_tokenizer = AutoTokenizer.from_pretrained(output_dir, trust_remote_code=True)
    print(f"✓ 模型和分词器加载成功!")
    
    # 验证参数一致性
    assert loaded_model.config.vocab_size == loaded_tokenizer.vocab_size, \
        "加载后的模型和分词器词汇表大小不一致!"
    print(f"✓ 验证通过: 词汇表大小一致 ({loaded_tokenizer.vocab_size})")
    
    # ========== 步骤7: 简单的前向传播测试 ==========
    print(f"\n步骤7: 前向传播测试")
    print("-"*60)
    
    import torch
    
    # 测试1: 使用随机 token IDs
    test_input_ids = torch.tensor([[bos_token_id, 100, 200, 300, eos_token_id]])
    with torch.no_grad():
        outputs = loaded_model(test_input_ids)
        print(f"✓ 随机 token 前向传播成功!")
        print(f"  输入 shape: {test_input_ids.shape}")
        print(f"  输出 logits shape: {outputs.logits.shape}")
    
    # 测试2: 使用分词器编码实际文本
    test_text = "你好，世界！Hello, world!"
    encoded = loaded_tokenizer.encode(test_text, return_tensors="pt")
    with torch.no_grad():
        outputs = loaded_model(encoded)
        print(f"✓ 文本编码前向传播成功!")
        print(f"  输入文本: {test_text}")
        print(f"  输入 shape: {encoded.shape}")
        print(f"  输出 logits shape: {outputs.logits.shape}")
        
    # 解码测试
    decoded_text = loaded_tokenizer.decode(encoded[0])
    print(f"  解码文本: {decoded_text}")
    
    print(f"\n" + "="*60)
    print(f"模型初始化完成!")
    print("="*60)
    print(f"\n关键信息:")
    print(f"  模型保存路径: {output_dir}")
    print(f"  词汇表大小: {loaded_tokenizer.vocab_size}")
    print(f"  模型参数量: {total_params / 1e6:.2f}M")
    print(f"  特殊 token:")
    print(f"    - PAD: {loaded_tokenizer.pad_token} (ID: {loaded_tokenizer.pad_token_id})")
    print(f"    - BOS: {loaded_tokenizer.bos_token} (ID: {loaded_tokenizer.bos_token_id})")
    print(f"    - EOS: {loaded_tokenizer.eos_token} (ID: {loaded_tokenizer.eos_token_id})")
    print(f"\n下一步: 使用 swift pt 命令进行预训练")
    print(f"命令示例: bash pretrain.sh")
    print("="*60)
    
    return model, tokenizer, config


def main():
    parser = argparse.ArgumentParser(
        description='初始化自定义模型权重（先创建分词器，再根据分词器参数初始化模型）'
    )
    parser.add_argument(
        '--output_dir',
        type=str,
        default='./custom_pretrained_model',
        help='模型保存路径'
    )
    parser.add_argument(
        '--tokenizer_path',
        type=str,
        default=None,
        help='已有分词器的路径（如果不指定，将训练新的分词器）'
    )
    parser.add_argument(
        '--train_tokenizer',
        action='store_true',
        help='是否训练新的分词器（即使指定了 tokenizer_path）'
    )
    parser.add_argument(
        '--hidden_size',
        type=int,
        default=768,
        help='隐藏层维度 (建议: 768/1024/2048)'
    )
    
    args = parser.parse_args()
    
    print("\n初始化流程:")
    print("1. 准备分词器（训练新分词器 或 加载已有分词器）")
    print("2. 从分词器获取词汇表大小等参数")
    print("3. 使用这些参数创建模型配置")
    print("4. 初始化模型随机权重")
    print("5. 保存模型和分词器")
    print("6. 验证加载和前向传播\n")
    
    # 初始化并保存模型
    init_and_save_model(
        output_dir=args.output_dir,
        tokenizer_path=args.tokenizer_path,
        train_tokenizer=args.train_tokenizer,
        hidden_size=args.hidden_size
    )


if __name__ == '__main__':
    main()
