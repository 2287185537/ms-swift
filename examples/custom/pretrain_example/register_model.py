# Copyright (c) Alibaba, Inc. and its affiliates.
"""
注册自定义模型到 ms-swift 框架
这个文件展示如何将自定义模型注册到 ms-swift，使其可以像 Qwen/LLaMA 等模型一样使用
"""

from typing import Any, Dict

from swift.llm import (
    Model,
    ModelGroup,
    ModelInfo,
    ModelMeta,
    TemplateMeta,
    register_model,
    register_template
)


def get_custom_model_tokenizer(
    model_dir: str,
    model_info: ModelInfo,
    model_kwargs: Dict[str, Any],
    load_model: bool = True,
    **kwargs
):
    """
    自定义的模型和 tokenizer 加载函数
    
    这个函数负责:
    1. 加载模型配置
    2. 加载 tokenizer
    3. 加载模型权重（如果 load_model=True）
    
    Args:
        model_dir: 模型路径（本地路径或 ModelScope/HuggingFace ID）
        model_info: 模型信息对象，包含 torch_dtype 等
        model_kwargs: 传递给模型的额外参数
        load_model: 是否加载模型权重（推理时为 True，仅需要 tokenizer 时为 False）
        **kwargs: 其他参数
        
    Returns:
        (model, tokenizer) 元组
    """
    from transformers import AutoConfig, AutoTokenizer
    
    # 导入自定义模型类
    # 这里需要确保 custom_model.py 中的模型已经注册到 transformers
    from custom_model import CustomModelForCausalLM, CustomModelConfig
    
    print(f"正在从 {model_dir} 加载自定义模型...")
    
    # 1. 加载配置
    model_config = AutoConfig.from_pretrained(model_dir, trust_remote_code=True)
    print(f"✓ 配置加载成功: {model_config.model_type}")
    
    # 2. 加载 tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    
    # 确保 tokenizer 有必要的特殊 token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    
    print(f"✓ Tokenizer 加载成功 (词汇表大小: {len(tokenizer)})")
    
    # 3. 加载模型（如果需要）
    model = None
    if load_model:
        model = CustomModelForCausalLM.from_pretrained(
            model_dir,
            config=model_config,
            torch_dtype=model_info.torch_dtype,  # 使用指定的数据类型 (如 bfloat16)
            trust_remote_code=True,
            **model_kwargs
        )
        print(f"✓ 模型加载成功")
    
    return model, tokenizer


# 注册对话模板
# 这里使用一个简单的对话格式，你可以根据需要自定义
register_template(
    TemplateMeta(
        template_type='custom_pretrain',  # 模板类型标识
        prefix=[],  # 对话开始前的前缀
        prompt=['<|user|>\n{{QUERY}}<|assistant|>\n'],  # 用户和助手的对话格式
        chat_sep=['\n'],  # 多轮对话之间的分隔符
        suffix=['<|endoftext|>'],  # 对话结束标记
        system_prefix=['<|system|>\n{{SYSTEM}}\n'],  # 系统提示词格式
        default_system='You are a helpful assistant.',  # 默认系统提示词
        stop_words=['<|endoftext|>'],  # 生成停止词
    ))


# 注册模型
register_model(
    ModelMeta(
        model_type='custom_pretrain',  # 模型类型标识，用于命令行 --model_type 参数
        model_groups=[
            ModelGroup([
                # 这里添加你的模型路径
                # 可以是本地路径或 ModelScope/HuggingFace 模型 ID
                Model(
                    'custom_pretrained_model',  # 模型 ID/路径
                    'custom_pretrained_model'   # 别名（可选）
                )
            ])
        ],
        template='custom_pretrain',  # 使用上面注册的模板
        get_function=get_custom_model_tokenizer,  # 使用自定义加载函数
        architectures=['CustomModelForCausalLM'],  # 模型架构名称，用于自动识别
        requires=[],  # 依赖包列表（可选）
        tags=['text-generation', 'pretrain'],  # 标签（可选）
        is_multimodal=False,  # 是否为多模态模型
    ))


if __name__ == '__main__':
    """
    测试模型注册是否成功
    """
    from swift.llm import get_model_tokenizer, InferRequest, PtEngine, RequestConfig
    
    print("="*60)
    print("测试自定义模型注册")
    print("="*60)
    
    # 方法1: 直接使用 get_model_tokenizer
    print("\n方法1: 使用 get_model_tokenizer 加载模型")
    model_dir = './custom_pretrained_model'
    try:
        model, tokenizer = get_model_tokenizer(
            model_dir,
            model_type='custom_pretrain',
            torch_dtype='auto'
        )
        print(f"✓ 模型加载成功!")
        print(f"  - 模型类型: {type(model).__name__}")
        print(f"  - Tokenizer 类型: {type(tokenizer).__name__}")
        print(f"  - 词汇表大小: {len(tokenizer)}")
    except Exception as e:
        print(f"✗ 加载失败: {e}")
        print(f"  提示: 请先运行 init_weights.py 初始化模型")
    
    # 方法2: 使用 PtEngine 进行推理
    print("\n方法2: 使用 PtEngine 进行推理测试")
    try:
        engine = PtEngine(model_dir, model_type='custom_pretrain')
        
        # 创建推理请求
        infer_request = InferRequest(
            messages=[
                {'role': 'user', 'content': 'Hello, who are you?'}
            ]
        )
        
        # 配置推理参数
        request_config = RequestConfig(
            max_tokens=50,
            temperature=0.7,
            top_p=0.9
        )
        
        # 执行推理
        print("正在生成回复...")
        response = engine.infer([infer_request], request_config)
        generated_text = response[0].choices[0].message.content
        
        print(f"✓ 推理成功!")
        print(f"生成的文本: {generated_text}")
        
    except Exception as e:
        print(f"✗ 推理失败: {e}")
        print(f"  提示: 请先运行 init_weights.py 初始化模型")
    
    print("\n" + "="*60)
    print("测试完成!")
    print("="*60)
