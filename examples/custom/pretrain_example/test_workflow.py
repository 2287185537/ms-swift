# Copyright (c) Alibaba, Inc. and its affiliates.
"""
测试完整工作流程
验证分词器训练 → 模型初始化 → 注册 → 推理的完整流程
"""

import os
import sys
import shutil
import tempfile
from pathlib import Path


def test_tokenizer_training():
    """测试分词器训练"""
    print("\n" + "="*60)
    print("测试 1: 分词器训练")
    print("="*60)
    
    from train_tokenizer import train_custom_tokenizer
    
    with tempfile.TemporaryDirectory() as tmpdir:
        tokenizer_dir = os.path.join(tmpdir, "test_tokenizer")
        
        try:
            tokenizer = train_custom_tokenizer(
                input_files=[],  # 使用示例数据
                output_dir=tokenizer_dir,
                vocab_size=1000,  # 小词汇表用于快速测试
                min_frequency=1
            )
            
            print(f"\n✓ 分词器训练成功")
            print(f"  词汇表大小: {tokenizer.vocab_size}")
            print(f"  PAD token ID: {tokenizer.pad_token_id}")
            print(f"  BOS token ID: {tokenizer.bos_token_id}")
            print(f"  EOS token ID: {tokenizer.eos_token_id}")
            
            # 测试编码解码
            text = "测试文本"
            encoded = tokenizer.encode(text)
            decoded = tokenizer.decode(encoded)
            print(f"  编码测试: '{text}' -> {encoded[:10]}... -> '{decoded}'")
            
            return True, tokenizer_dir
            
        except Exception as e:
            print(f"\n✗ 分词器训练失败: {e}")
            import traceback
            traceback.print_exc()
            return False, None


def test_model_initialization(tokenizer_dir):
    """测试模型初始化（从分词器导入参数）"""
    print("\n" + "="*60)
    print("测试 2: 模型初始化（从分词器导入参数）")
    print("="*60)
    
    from transformers import AutoTokenizer
    from custom_model import CustomModelConfig, CustomModelForCausalLM
    
    with tempfile.TemporaryDirectory() as tmpdir:
        model_dir = os.path.join(tmpdir, "test_model")
        
        try:
            # 加载分词器
            print("加载分词器...")
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir, trust_remote_code=True)
            
            # 从分词器导入参数
            vocab_size = tokenizer.vocab_size
            pad_token_id = tokenizer.pad_token_id or 0
            bos_token_id = tokenizer.bos_token_id or 1
            eos_token_id = tokenizer.eos_token_id or 2
            
            print(f"\n从分词器导入的参数:")
            print(f"  vocab_size: {vocab_size}")
            print(f"  pad_token_id: {pad_token_id}")
            print(f"  bos_token_id: {bos_token_id}")
            print(f"  eos_token_id: {eos_token_id}")
            
            # 创建模型配置（使用分词器参数）
            config = CustomModelConfig(
                vocab_size=vocab_size,  # 从分词器导入
                hidden_size=256,  # 小模型用于快速测试
                num_hidden_layers=2,
                num_attention_heads=4,
                intermediate_size=1024,
                max_position_embeddings=512,
                pad_token_id=pad_token_id,  # 从分词器导入
                bos_token_id=bos_token_id,  # 从分词器导入
                eos_token_id=eos_token_id,  # 从分词器导入
            )
            
            # 初始化模型
            print("\n初始化模型...")
            model = CustomModelForCausalLM(config)
            
            # 验证一致性
            assert model.config.vocab_size == tokenizer.vocab_size, \
                f"词汇表大小不一致: 模型={model.config.vocab_size}, 分词器={tokenizer.vocab_size}"
            
            print(f"✓ 模型初始化成功")
            print(f"  模型词汇表大小: {model.config.vocab_size}")
            print(f"  分词器词汇表大小: {tokenizer.vocab_size}")
            print(f"  ✓ 验证通过: 词汇表大小一致")
            
            # 保存模型和分词器
            os.makedirs(model_dir, exist_ok=True)
            model.save_pretrained(model_dir)
            tokenizer.save_pretrained(model_dir)
            
            print(f"  模型已保存到: {model_dir}")
            
            return True, model_dir
            
        except Exception as e:
            print(f"\n✗ 模型初始化失败: {e}")
            import traceback
            traceback.print_exc()
            return False, None


def test_model_forward(model_dir):
    """测试模型前向传播"""
    print("\n" + "="*60)
    print("测试 3: 模型前向传播")
    print("="*60)
    
    import torch
    from transformers import AutoTokenizer
    from custom_model import CustomModelForCausalLM
    
    try:
        # 加载模型和分词器
        print("加载模型和分词器...")
        model = CustomModelForCausalLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        # 测试前向传播
        test_text = "你好，世界！"
        print(f"\n测试文本: {test_text}")
        
        encoded = tokenizer.encode(test_text, return_tensors="pt")
        print(f"编码后 shape: {encoded.shape}")
        
        with torch.no_grad():
            outputs = model(encoded)
            logits = outputs.logits
        
        print(f"输出 logits shape: {logits.shape}")
        print(f"预期 shape: (1, {encoded.shape[1]}, {model.config.vocab_size})")
        
        assert logits.shape == (1, encoded.shape[1], model.config.vocab_size), \
            f"输出 shape 不正确: {logits.shape}"
        
        print(f"\n✓ 前向传播测试成功")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 前向传播测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation(model_dir):
    """测试文本生成"""
    print("\n" + "="*60)
    print("测试 4: 文本生成")
    print("="*60)
    
    import torch
    from transformers import AutoTokenizer
    from custom_model import CustomModelForCausalLM
    
    try:
        # 加载模型和分词器
        print("加载模型和分词器...")
        model = CustomModelForCausalLM.from_pretrained(model_dir)
        tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
        
        # 生成文本
        prompt = "你好"
        print(f"\n提示词: {prompt}")
        
        input_ids = tokenizer.encode(prompt, return_tensors="pt")
        
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        
        generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        print(f"生成文本: {generated_text}")
        
        print(f"\n✓ 文本生成测试成功")
        print(f"  注意: 由于模型未训练，生成的文本是随机的")
        
        return True
        
    except Exception as e:
        print(f"\n✗ 文本生成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("自定义预训练模型工作流程测试")
    print("="*60)
    print("\n测试流程:")
    print("1. 训练自定义分词器")
    print("2. 根据分词器参数初始化模型")
    print("3. 测试模型前向传播")
    print("4. 测试文本生成")
    print()
    
    results = []
    
    # 测试 1: 分词器训练
    success, tokenizer_dir = test_tokenizer_training()
    results.append(("分词器训练", success))
    
    if not success:
        print("\n分词器训练失败，后续测试跳过")
        print_summary(results)
        return
    
    # 测试 2: 模型初始化
    success, model_dir = test_model_initialization(tokenizer_dir)
    results.append(("模型初始化（参数导入）", success))
    
    if not success:
        print("\n模型初始化失败，后续测试跳过")
        print_summary(results)
        return
    
    # 测试 3: 前向传播
    success = test_model_forward(model_dir)
    results.append(("模型前向传播", success))
    
    # 测试 4: 文本生成
    success = test_generation(model_dir)
    results.append(("文本生成", success))
    
    # 打印总结
    print_summary(results)


def print_summary(results):
    """打印测试总结"""
    print("\n" + "="*60)
    print("测试总结")
    print("="*60)
    
    for test_name, success in results:
        status = "✓ 通过" if success else "✗ 失败"
        print(f"{status}: {test_name}")
    
    total = len(results)
    passed = sum(1 for _, success in results if success)
    
    print(f"\n总计: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n🎉 所有测试通过！工作流程验证成功！")
        print("\n下一步:")
        print("1. 运行 bash run_full_example.sh 查看完整流程")
        print("2. 使用自己的数据训练分词器和模型")
        print("3. 开始预训练: bash pretrain.sh")
    else:
        print(f"\n⚠ {total - passed} 个测试失败，请检查错误信息")
    
    print("="*60)


if __name__ == '__main__':
    # 确保在正确的目录
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # 运行测试
    main()
