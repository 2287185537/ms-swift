# Copyright (c) Alibaba, Inc. and its affiliates.
"""
自定义模型结构定义
这个文件展示如何定义一个符合 HuggingFace 标准的自定义模型
"""

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
from transformers import PretrainedConfig, PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import CausalLMOutputWithPast


class CustomModelConfig(PretrainedConfig):
    """
    自定义模型配置类
    继承自 PretrainedConfig 以确保与 HuggingFace 生态兼容
    """
    model_type = "custom_model"  # 模型类型标识，用于自动识别
    
    def __init__(
        self,
        vocab_size=32000,  # 词汇表大小
        hidden_size=768,  # 隐藏层维度
        num_hidden_layers=12,  # Transformer 层数
        num_attention_heads=12,  # 注意力头数
        intermediate_size=3072,  # FFN 中间层维度
        hidden_dropout_prob=0.1,  # Dropout 概率
        attention_probs_dropout_prob=0.1,  # 注意力 Dropout 概率
        max_position_embeddings=2048,  # 最大序列长度
        initializer_range=0.02,  # 参数初始化范围
        layer_norm_eps=1e-12,  # LayerNorm epsilon
        pad_token_id=0,  # Padding token ID
        bos_token_id=1,  # Begin of sequence token ID
        eos_token_id=2,  # End of sequence token ID
        tie_word_embeddings=False,  # 是否共享输入输出 embedding
        use_cache=True,  # 是否使用 KV cache（用于推理加速）
        **kwargs
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.initializer_range = initializer_range
        self.layer_norm_eps = layer_norm_eps
        self.use_cache = use_cache
        
        super().__init__(
            pad_token_id=pad_token_id,
            bos_token_id=bos_token_id,
            eos_token_id=eos_token_id,
            tie_word_embeddings=tie_word_embeddings,
            **kwargs
        )


class CustomAttention(nn.Module):
    """多头自注意力层"""
    
    def __init__(self, config: CustomModelConfig):
        super().__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                f"hidden_size ({config.hidden_size}) 必须能被 num_attention_heads ({config.num_attention_heads}) 整除"
            )
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = config.hidden_size // config.num_attention_heads
        self.all_head_size = self.num_attention_heads * self.attention_head_size
        
        # QKV 线性投影
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        
        # 输出投影（从 all_head_size 回到 hidden_size）
        self.out_proj = nn.Linear(self.all_head_size, config.hidden_size)
        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
        
    def transpose_for_scores(self, x: torch.Tensor) -> torch.Tensor:
        """将 tensor 重塑为多头形式"""
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(new_x_shape)
        return x.permute(0, 2, 1, 3)  # (batch, heads, seq_len, head_dim)
    
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        # 计算 QKV
        query_layer = self.transpose_for_scores(self.query(hidden_states))
        key_layer = self.transpose_for_scores(self.key(hidden_states))
        value_layer = self.transpose_for_scores(self.value(hidden_states))
        
        # 使用缓存的 KV (用于推理加速)
        if past_key_value is not None:
            key_layer = torch.cat([past_key_value[0], key_layer], dim=2)
            value_layer = torch.cat([past_key_value[1], value_layer], dim=2)
        
        # 计算注意力分数
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        
        # 应用注意力 mask (用于因果语言建模)
        if attention_mask is not None:
            attention_scores = attention_scores + attention_mask
        
        # Softmax + Dropout
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self.dropout(attention_probs)
        
        # 加权求和
        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        
        # 输出投影
        output = self.out_proj(context_layer)
        
        outputs = (output,)
        if use_cache:
            outputs += ((key_layer, value_layer),)
        
        return outputs


class CustomMLP(nn.Module):
    """前馈神经网络层"""
    
    def __init__(self, config: CustomModelConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.hidden_size, config.intermediate_size)
        self.fc2 = nn.Linear(config.intermediate_size, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = nn.functional.gelu(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.fc2(hidden_states)
        return hidden_states


class CustomTransformerLayer(nn.Module):
    """Transformer 解码器层"""
    
    def __init__(self, config: CustomModelConfig):
        super().__init__()
        self.attention = CustomAttention(config)
        self.mlp = CustomMLP(config)
        self.ln_1 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.ln_2 = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_value: Optional[Tuple[torch.Tensor]] = None,
        use_cache: bool = False,
    ) -> Tuple[torch.Tensor, Optional[Tuple[torch.Tensor]]]:
        # Pre-LN: 先 LayerNorm 再 Attention
        residual = hidden_states
        hidden_states = self.ln_1(hidden_states)
        
        attn_outputs = self.attention(
            hidden_states,
            attention_mask=attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        hidden_states = attn_outputs[0]
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        # Pre-LN: 先 LayerNorm 再 MLP
        residual = hidden_states
        hidden_states = self.ln_2(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = residual + hidden_states
        
        outputs = (hidden_states,)
        if use_cache:
            outputs += (attn_outputs[1],)
        
        return outputs


class CustomModel(PreTrainedModel):
    """
    自定义模型主体
    继承自 PreTrainedModel 以确保与 HuggingFace 生态兼容
    """
    config_class = CustomModelConfig
    base_model_prefix = "transformer"
    supports_gradient_checkpointing = True
    
    def __init__(self, config: CustomModelConfig):
        super().__init__(config)
        self.config = config
        
        # Token Embedding + Position Embedding
        self.token_embeddings = nn.Embedding(config.vocab_size, config.hidden_size)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
        # Transformer 层
        self.layers = nn.ModuleList([
            CustomTransformerLayer(config) for _ in range(config.num_hidden_layers)
        ])
        
        # 最终的 LayerNorm
        self.ln_f = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        
        # 初始化权重
        self.post_init()
    
    def get_input_embeddings(self):
        return self.token_embeddings
    
    def set_input_embeddings(self, value):
        self.token_embeddings = value
    
    def _init_weights(self, module):
        """初始化权重 - 使用标准的正态分布初始化"""
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=self.config.initializer_range)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Tuple:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        batch_size, seq_length = input_ids.shape
        
        # Position IDs
        if position_ids is None:
            past_length = 0
            if past_key_values is not None and len(past_key_values) > 0 and past_key_values[0] is not None and len(past_key_values[0]) > 0:
                past_length = past_key_values[0][0].size(2)
            position_ids = torch.arange(past_length, seq_length + past_length, dtype=torch.long, device=input_ids.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
        
        # 获取 embeddings
        token_embeds = self.token_embeddings(input_ids)
        position_embeds = self.position_embeddings(position_ids)
        hidden_states = token_embeds + position_embeds
        hidden_states = self.dropout(hidden_states)
        
        # 准备因果 attention mask
        if attention_mask is not None:
            # attention_mask shape: (batch_size, seq_length)
            # 扩展为 (batch_size, 1, 1, seq_length) 用于广播
            attention_mask = attention_mask[:, None, None, :]
            attention_mask = attention_mask.to(dtype=hidden_states.dtype)
            attention_mask = (1.0 - attention_mask) * torch.finfo(hidden_states.dtype).min
        
        # 通过 Transformer 层
        present_key_values = () if use_cache else None
        all_hidden_states = () if output_hidden_states else None
        
        for i, layer in enumerate(self.layers):
            if output_hidden_states:
                all_hidden_states = all_hidden_states + (hidden_states,)
            
            past_key_value = past_key_values[i] if past_key_values is not None else None
            
            layer_outputs = layer(
                hidden_states,
                attention_mask=attention_mask,
                past_key_value=past_key_value,
                use_cache=use_cache,
            )
            
            hidden_states = layer_outputs[0]
            
            if use_cache:
                present_key_values = present_key_values + (layer_outputs[1],)
        
        # 最终 LayerNorm
        hidden_states = self.ln_f(hidden_states)
        
        if output_hidden_states:
            all_hidden_states = all_hidden_states + (hidden_states,)
        
        if not return_dict:
            return tuple(v for v in [hidden_states, present_key_values, all_hidden_states] if v is not None)
        
        return {
            'last_hidden_state': hidden_states,
            'past_key_values': present_key_values,
            'hidden_states': all_hidden_states,
        }


class CustomModelForCausalLM(PreTrainedModel, GenerationMixin):
    """
    用于因果语言建模的自定义模型
    添加 LM Head 用于预测下一个 token
    继承 GenerationMixin 以支持文本生成功能
    """
    config_class = CustomModelConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _tied_weights_keys = ["lm_head.weight"]
    
    def __init__(self, config: CustomModelConfig):
        super().__init__(config)
        self.model = CustomModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        
        # 初始化权重（包括可能的权重共享）
        self.post_init()
    
    def get_input_embeddings(self):
        return self.model.token_embeddings
    
    def set_input_embeddings(self, value):
        self.model.token_embeddings = value
    
    def get_output_embeddings(self):
        return self.lm_head
    
    def set_output_embeddings(self, new_embeddings):
        self.lm_head = new_embeddings
    
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        
        # 调用 base model
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )
        
        hidden_states = outputs['last_hidden_state']
        logits = self.lm_head(hidden_states)
        
        # 计算损失
        loss = None
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(shift_logits.view(-1, self.config.vocab_size), shift_labels.view(-1))
        
        if not return_dict:
            output = (logits,) + tuple(v for k, v in outputs.items() if k != 'last_hidden_state')
            return ((loss,) + output) if loss is not None else output
        
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.get('past_key_values'),
            hidden_states=outputs.get('hidden_states'),
        )
    
    def prepare_inputs_for_generation(
        self, input_ids, past_key_values=None, attention_mask=None, **kwargs
    ):
        """为生成准备输入（支持 KV cache 加速）"""
        if past_key_values is not None:
            # 只需要最后一个 token
            input_ids = input_ids[:, -1:]
        
        return {
            "input_ids": input_ids,
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        }
    
    @staticmethod
    def _reorder_cache(past_key_values, beam_idx):
        """用于 beam search 时重排序 KV cache"""
        reordered_past = ()
        for layer_past in past_key_values:
            reordered_past += (
                tuple(past_state.index_select(0, beam_idx) for past_state in layer_past),
            )
        return reordered_past


# 注册配置和模型到 AutoConfig 和 AutoModel
from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("custom_model", CustomModelConfig)
AutoModelForCausalLM.register(CustomModelConfig, CustomModelForCausalLM)
