# Copyright (c) Alibaba, Inc. and its affiliates.
"""
简化测试：仅测试模型定义和基本功能（不需要 tokenizers 包）
"""

import torch
from custom_model import CustomModelConfig, CustomModelForCausalLM


def test_model_creation():
    """测试模型创建"""
    print("="*60)
    print("测试 1: 模型创建")
    print("="*60)
    
    try:
        # 创建小型配置用于快速测试
        config = CustomModelConfig(
            vocab_size=1000,
            hidden_size=256,
            num_hidden_layers=2,
            num_attention_heads=4,
            intermediate_size=1024,
            max_position_embeddings=512,
            pad_token_id=0,
            bos_token_id=1,
            eos_token_id=2,
        )
        
        print(f"模型配置:")
        print(f"  vocab_size: {config.vocab_size}")
        print(f"  hidden_size: {config.hidden_size}")
        print(f"  num_hidden_layers: {config.num_hidden_layers}")
        print(f"  num_attention_heads: {config.num_attention_heads}")
        
        # 创建模型
        model = CustomModelForCausalLM(config)
        
        # 统计参数
        total_params = sum(p.numel() for p in model.parameters())
        print(f"\n模型参数量: {total_params:,} ({total_params/1e6:.2f}M)")
        
        print(f"\n✓ 模型创建成功")
        return True, model, config
        
    except Exception as e:
        print(f"\n✗ 模型创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False, None, None


def test_forward_pass(model, config):
    """测试前向传播"""
    print("\n" + "="*60)
    print("测试 2: 前向传播")
    print("="*60)
    
    try:
        # 创建测试输入
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        
        print(f"输入 shape: {input_ids.shape}")
        
        # 前向传播
        with torch.no_grad():
            outputs = model(input_ids)
        
        logits = outputs.logits
        print(f"输出 logits shape: {logits.shape}")
        print(f"预期 shape: ({batch_size}, {seq_length}, {config.vocab_size})")
        
        assert logits.shape == (batch_size, seq_length, config.vocab_size), \
            f"输出 shape 不正确"
        
        print(f"\n✓ 前向传播测试成功")
        return True
        
    except Exception as e:
        print(f"\n✗ 前向传播失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_loss_computation(model, config):
    """测试损失计算"""
    print("\n" + "="*60)
    print("测试 3: 损失计算")
    print("="*60)
    
    try:
        batch_size = 2
        seq_length = 10
        input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        labels = torch.randint(0, config.vocab_size, (batch_size, seq_length))
        
        print(f"输入 shape: {input_ids.shape}")
        print(f"标签 shape: {labels.shape}")
        
        # 前向传播并计算损失
        with torch.no_grad():
            outputs = model(input_ids, labels=labels)
        
        loss = outputs.loss
        print(f"损失值: {loss.item():.4f}")
        
        assert loss is not None and not torch.isnan(loss), "损失计算异常"
        
        print(f"\n✓ 损失计算测试成功")
        return True
        
    except Exception as e:
        print(f"\n✗ 损失计算失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_generation(model, config):
    """测试生成功能"""
    print("\n" + "="*60)
    print("测试 4: 文本生成")
    print("="*60)
    
    try:
        input_ids = torch.tensor([[config.bos_token_id, 100, 200]])
        
        print(f"输入 token IDs: {input_ids.tolist()[0]}")
        
        # 生成
        with torch.no_grad():
            output_ids = model.generate(
                input_ids,
                max_new_tokens=5,
                do_sample=False,
                pad_token_id=config.pad_token_id,
                eos_token_id=config.eos_token_id,
            )
        
        print(f"生成的 token IDs: {output_ids.tolist()[0]}")
        print(f"生成了 {output_ids.shape[1] - input_ids.shape[1]} 个新 token")
        
        print(f"\n✓ 文本生成测试成功")
        print(f"  注意: 由于模型未训练，生成的 token 是随机的")
        return True
        
    except Exception as e:
        print(f"\n✗ 文本生成失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_save_load(model, config):
    """测试保存和加载"""
    print("\n" + "="*60)
    print("测试 5: 模型保存和加载")
    print("="*60)
    
    import tempfile
    import os
    
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            save_dir = os.path.join(tmpdir, "test_model")
            
            # 保存
            print(f"保存模型到: {save_dir}")
            model.save_pretrained(save_dir)
            
            # 加载
            print(f"从 {save_dir} 加载模型")
            loaded_model = CustomModelForCausalLM.from_pretrained(save_dir)
            
            # 验证参数一致
            orig_params = sum(p.numel() for p in model.parameters())
            loaded_params = sum(p.numel() for p in loaded_model.parameters())
            
            assert orig_params == loaded_params, "参数数量不一致"
            
            # 测试前向传播
            input_ids = torch.randint(0, config.vocab_size, (1, 5))
            with torch.no_grad():
                orig_out = model(input_ids).logits
                loaded_out = loaded_model(input_ids).logits
            
            # 验证输出一致
            assert torch.allclose(orig_out, loaded_out, atol=1e-5), "输出不一致"
            
            print(f"\n✓ 模型保存和加载测试成功")
            print(f"  原始参数量: {orig_params:,}")
            print(f"  加载参数量: {loaded_params:,}")
            print(f"  ✓ 验证通过: 参数和输出一致")
            return True
            
    except Exception as e:
        print(f"\n✗ 保存和加载失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_tokenizer_params_import():
    """测试从分词器导入参数的流程（模拟）"""
    print("\n" + "="*60)
    print("测试 6: 分词器参数导入流程（模拟）")
    print("="*60)
    
    try:
        # 模拟分词器参数
        class MockTokenizer:
            vocab_size = 32000
            pad_token_id = 0
            bos_token_id = 1
            eos_token_id = 2
        
        tokenizer = MockTokenizer()
        
        print("模拟分词器参数:")
        print(f"  vocab_size: {tokenizer.vocab_size}")
        print(f"  pad_token_id: {tokenizer.pad_token_id}")
        print(f"  bos_token_id: {tokenizer.bos_token_id}")
        print(f"  eos_token_id: {tokenizer.eos_token_id}")
        
        # 从分词器导入参数创建配置
        config = CustomModelConfig(
            vocab_size=tokenizer.vocab_size,  # 从分词器导入
            hidden_size=768,
            num_hidden_layers=12,
            num_attention_heads=12,
            intermediate_size=3072,
            pad_token_id=tokenizer.pad_token_id,  # 从分词器导入
            bos_token_id=tokenizer.bos_token_id,  # 从分词器导入
            eos_token_id=tokenizer.eos_token_id,  # 从分词器导入
        )
        
        print(f"\n创建的模型配置:")
        print(f"  vocab_size: {config.vocab_size} (从分词器导入)")
        print(f"  pad_token_id: {config.pad_token_id} (从分词器导入)")
        print(f"  bos_token_id: {config.bos_token_id} (从分词器导入)")
        print(f"  eos_token_id: {config.eos_token_id} (从分词器导入)")
        
        # 验证一致性
        assert config.vocab_size == tokenizer.vocab_size, "词汇表大小不一致"
        assert config.pad_token_id == tokenizer.pad_token_id, "PAD token ID 不一致"
        assert config.bos_token_id == tokenizer.bos_token_id, "BOS token ID 不一致"
        assert config.eos_token_id == tokenizer.eos_token_id, "EOS token ID 不一致"
        
        print(f"\n✓ 参数导入流程测试成功")
        print(f"  ✓ 所有参数从分词器正确导入到模型配置")
        return True
        
    except Exception as e:
        print(f"\n✗ 参数导入流程失败: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试"""
    print("="*60)
    print("自定义模型基础功能测试")
    print("="*60)
    print("\n这个测试脚本验证:")
    print("1. 模型创建")
    print("2. 前向传播")
    print("3. 损失计算")
    print("4. 文本生成")
    print("5. 模型保存和加载")
    print("6. 分词器参数导入流程（模拟）")
    print()
    
    results = []
    
    # 测试 1: 模型创建
    success, model, config = test_model_creation()
    results.append(("模型创建", success))
    
    if not success:
        print_summary(results)
        return
    
    # 测试 2: 前向传播
    success = test_forward_pass(model, config)
    results.append(("前向传播", success))
    
    # 测试 3: 损失计算
    success = test_loss_computation(model, config)
    results.append(("损失计算", success))
    
    # 测试 4: 文本生成
    success = test_generation(model, config)
    results.append(("文本生成", success))
    
    # 测试 5: 保存和加载
    success = test_save_load(model, config)
    results.append(("模型保存和加载", success))
    
    # 测试 6: 分词器参数导入
    success = test_tokenizer_params_import()
    results.append(("分词器参数导入流程", success))
    
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
        print("\n🎉 所有测试通过！模型定义正确！")
        print("\n下一步:")
        print("1. 安装 tokenizers: pip install tokenizers")
        print("2. 训练分词器: python train_tokenizer.py")
        print("3. 初始化模型: python init_weights.py --tokenizer_path ./custom_tokenizer")
        print("4. 运行完整流程: bash run_full_example.sh")
    else:
        print(f"\n⚠ {total - passed} 个测试失败，请检查错误信息")
    
    print("="*60)


if __name__ == '__main__':
    main()
