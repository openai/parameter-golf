import torch
import torch.nn as nn
from torch.nn import functional as F

class ExclusiveSelfAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        q = self.q_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        
        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.view(batch_size, seq_len, d_model)
        
        for i in range(seq_len):
            v_i = v[:, :, i:i+1, :]
            v_norm_sq = torch.sum(v_i ** 2, dim=-1, keepdim=True)
            alignment = torch.sum(attn_output[:, i:i+1, :].view(batch_size, 1, self.num_heads, self.head_dim) * v_i, dim=-1, keepdim=True)
            alignment = alignment / (v_norm_sq + 1e-8)
            attn_output[:, i:i+1, :] = attn_output[:, i:i+1, :] - (alignment * v[:, :, i:i+1, :].view(batch_size, 1, d_model))
        
        output = self.out_proj(attn_output)
        return output


class MLP(nn.Module):
    def __init__(self, d_model, expansion=2.0):
        super().__init__()
        hidden_dim = int(d_model * expansion)
        self.fc1 = nn.Linear(d_model, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, d_model)
        
    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TransformerBlock(nn.Module):
    def __init__(self, d_model, num_heads, expansion=2.0):
        super().__init__()
        self.attn = ExclusiveSelfAttention(d_model, num_heads)
        self.mlp = MLP(d_model, expansion)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class BaselineTransformer(nn.Module):
    def __init__(self, d_model=512, num_layers=11, num_heads=8, vocab_size=1024):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([TransformerBlock(d_model, num_heads) for _ in range(num_layers)])
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.embedding.weight
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        logits = self.lm_head(x)
        return logits
