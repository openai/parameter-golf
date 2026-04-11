"""
QUANTUM-FUSION-PLUS: 核心模型实现

包含:
- QuantumFusionGPT: 主模型
- TransformerBlock: Transformer块(含递归)
- MultiHeadAttention: 多头注意力(含QK-Gain)
- FeedForward: 前馈网络
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict
import math


class QuantumFusionGPT(nn.Module):
    """
    QUANTUM-FUSION-PLUS: 融合量化与递归架构的参数优化模型
    
    核心创新:
    1. 3层深度递归 (第4-5层循环)
    2. 并行残差通道
    3. QK-Gain注意力优化
    4. Hadamard旋转量化
    5. AWQ显著性感知
    """
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 嵌入层
        self.embedding = nn.Embedding(config.model.vocab_size, config.model.hidden_size)
        self.pos_embedding = nn.Embedding(config.model.max_seq_length, config.model.hidden_size)
        
        # 输出投影
        if config.model.tie_embeddings:
            self.output_projection = None
        else:
            self.output_projection = nn.Linear(config.model.hidden_size, config.model.vocab_size)
        
        # Transformer层
        self.layers = nn.ModuleList([
            TransformerBlock(config) 
            for _ in range(config.model.num_layers)
        ])
        
        # 递归配置
        self.recurrent_layers = config.recurrence.recurrent_layers
        self.num_recurrence = config.recurrence.num_recurrence
        
        # 最终层归一化
        self.final_ln = nn.LayerNorm(config.model.hidden_size, eps=config.model.layer_norm_eps)
        
        # 初始化权重
        self._init_weights()
    
    def _init_weights(self):
        """初始化权重"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)
    
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        use_cache: bool = False
    ) -> Tuple[torch.Tensor, Optional[Dict]]:
        """
        前向传播
        
        Args:
            input_ids: [batch_size, seq_len]
            attention_mask: [batch_size, seq_len]
            use_cache: 是否使用KV缓存
            
        Returns:
            logits: [batch_size, seq_len, vocab_size]
            cache: 可选的KV缓存
        """
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # 嵌入
        x = self.embedding(input_ids)
        
        # 位置嵌入
        positions = torch.arange(seq_len, device=device).unsqueeze(0)
        x = x + self.pos_embedding(positions)
        
        # 应用Hadamard旋转(如果启用)
        if self.config.quantization.hadamard_rotation:
            x = self._apply_hadamard_rotation(x)
        
        # 通过Transformer层(含递归)
        cache = {} if use_cache else None
        
        for i, layer in enumerate(self.layers):
            # 检查是否是递归层
            if i in self.recurrent_layers:
                # 递归处理
                for recurrence_idx in range(self.num_recurrence):
                    x = layer(x, attention_mask, cache)
            else:
                x = layer(x, attention_mask, cache)
        
        # 最终层归一化
        x = self.final_ln(x)
        
        # 输出投影
        if self.output_projection is not None:
            logits = self.output_projection(x)
        else:
            # 共享嵌入
            logits = x @ self.embedding.weight.t()
        
        return logits, cache
    
    def _apply_hadamard_rotation(self, x: torch.Tensor) -> torch.Tensor:
        """应用Hadamard旋转"""
        # 获取维度
        d = x.shape[-1]
        
        # 创建Hadamard矩阵
        H = self._create_hadamard_matrix(d).to(x.device).to(x.dtype)
        
        # 应用旋转
        x_rotated = torch.matmul(x, H.t())
        
        return x_rotated
    
    @staticmethod
    def _create_hadamard_matrix(n: int) -> torch.Tensor:
        """创建Hadamard矩阵"""
        if n == 1:
            return torch.tensor([[1.0]])
        
        # 找到最接近的2的幂
        k = int(math.ceil(math.log2(n)))
        size = 2 ** k
        
        # 递归构造
        H = torch.tensor([[1.0]])
        for _ in range(k):
            H = torch.kron(H, torch.tensor([[1.0, 1.0], [1.0, -1.0]]))
        
        # 归一化
        H = H / math.sqrt(size)
        
        # 截断到所需大小
        if size > n:
            H = H[:n, :n]
        
        return H


class TransformerBlock(nn.Module):
    """Transformer块,包含注意力和前馈"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # 注意力和前馈
        self.attention = MultiHeadAttention(config)
        self.mlp = FeedForward(config)
        
        # 层归一化
        self.ln1 = nn.LayerNorm(config.model.hidden_size, eps=config.model.layer_norm_eps)
        self.ln2 = nn.LayerNorm(config.model.hidden_size, eps=config.model.layer_norm_eps)
        
        # 并行残差通道
        self.parallel_residual = config.recurrence.parallel_residual
        if self.parallel_residual:
            self.residual_weight = nn.Parameter(
                torch.tensor(config.recurrence.residual_weight_init)
            )
    
    def forward(self, x, attention_mask=None, cache=None):
        """前向传播"""
        # 注意力
        attn_out = self.attention(self.ln1(x), attention_mask, cache)
        
        # MLP
        mlp_out = self.mlp(self.ln2(x))
        
        # 并行残差
        if self.parallel_residual:
            x = x + self.residual_weight * attn_out + (1 - self.residual_weight) * mlp_out
        else:
            x = x + attn_out
            x = x + mlp_out
        
        return x


class MultiHeadAttention(nn.Module):
    """多头注意力,含QK-Gain优化"""
    
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.model.num_attention_heads
        self.num_kv_heads = config.model.num_kv_heads
        self.head_dim = config.model.hidden_size // config.model.num_attention_heads
        self.qk_gain = config.recurrence.qk_gain
        
        # 投影层
        self.q_proj = nn.Linear(config.model.hidden_size, config.model.hidden_size)
        
        kv_size = config.model.hidden_size // (config.model.num_attention_heads // config.model.num_kv_heads)
        self.k_proj = nn.Linear(config.model.hidden_size, kv_size)
        self.v_proj = nn.Linear(config.model.hidden_size, kv_size)
        
        self.out_proj = nn.Linear(config.model.hidden_size, config.model.hidden_size)
        
        self.dropout = nn.Dropout(config.model.dropout)
    
    def forward(self, x, attention_mask=None, cache=None):
        """前向传播"""
        batch_size, seq_len, _ = x.shape
        
        # 投影
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)
        
        # 重塑为多头
        q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_kv_heads, self.head_dim).transpose(1, 2)
        
        # 计算注意力
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        
        # 应用QK-Gain
        scores = scores * self.qk_gain
        
        # 应用掩码
        if attention_mask is not None:
            scores = scores.masked_fill(attention_mask[:, None, None, :] == 0, float('-inf'))
        
        # Softmax
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用到值
        attn_output = torch.matmul(attn_weights, v)
        
        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, -1)
        
        # 输出投影
        output = self.out_proj(attn_output)
        
        return output


class FeedForward(nn.Module):
    """前馈网络"""
    
    def __init__(self, config):
        super().__init__()
        self.fc1 = nn.Linear(config.model.hidden_size, config.model.intermediate_size)
        self.fc2 = nn.Linear(config.model.intermediate_size, config.model.hidden_size)
        self.activation = nn.GELU()
        self.dropout = nn.Dropout(config.model.dropout)
    
    def forward(self, x):
        """前向传播"""
        x = self.fc1(x)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x
