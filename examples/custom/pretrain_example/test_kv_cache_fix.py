#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试 KV cache 修复是否有效
模拟 swift infer 的行为
"""

import sys
import torch
sys.path.insert(0, '/home/runner/work/ms-swift/ms-swift/examples/custom/pretrain_example')

from custom_model import CustomModelForCausalLM
from transformers import AutoTokenizer


def test_generation_with_cache(model_path):
    """测试使用 KV cache 的生成"""
    print("="*60)
    print("测试 KV Cache 修复")
    print("="*60)
    
    # 加载模型和分词器
    print("\n加载模型和分词器...")
    model = CustomModelForCausalLM.from_pretrained(model_path)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    
    print(f"✓ 模型加载成功")
    print(f"  词汇表大小: {tokenizer.vocab_size}")
    print(f"  模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
    
    # 测试用例
    test_prompts = [
        "你好",
        "Hello",
        "测试一下",
    ]
    
    print(f"\n" + "="*60)
    print("测试生成 (use_cache=True)")
    print("="*60)
    
    success_count = 0
    for i, prompt in enumerate(test_prompts, 1):
        print(f"\n测试 {i}/{len(test_prompts)}: {prompt}")
        
        try:
            input_ids = tokenizer.encode(prompt, return_tensors='pt')
            
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=5,
                    do_sample=False,
                    pad_token_id=tokenizer.pad_token_id,
                    eos_token_id=tokenizer.eos_token_id,
                    use_cache=True,  # 关键：使用 KV cache
                )
            
            generated_text = tokenizer.decode(output[0], skip_special_tokens=True)
            print(f"  ✓ 生成成功")
            print(f"    生成文本: {generated_text}")
            print(f"    生成 token 数: {output.shape[1] - input_ids.shape[1]}")
            success_count += 1
            
        except Exception as e:
            print(f"  ✗ 生成失败: {e}")
            import traceback
            traceback.print_exc()
    
    # 总结
    print(f"\n" + "="*60)
    print("测试总结")
    print("="*60)
    print(f"总测试数: {len(test_prompts)}")
    print(f"成功: {success_count}")
    print(f"失败: {len(test_prompts) - success_count}")
    
    if success_count == len(test_prompts):
        print("\n✅ 所有测试通过！KV cache 修复成功！")
        print("   现在可以正常使用 swift infer 命令")
        return True
    else:
        print(f"\n❌ {len(test_prompts) - success_count} 个测试失败")
        return False


if __name__ == '__main__':
    import sys
    
    if len(sys.argv) > 1:
        model_path = sys.argv[1]
    else:
        model_path = '/tmp/test_kv_cache/model'
        print(f"使用默认模型路径: {model_path}")
    
    success = test_generation_with_cache(model_path)
    sys.exit(0 if success else 1)
