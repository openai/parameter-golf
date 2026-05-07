import torch
import torch.nn as nn
from torch.nn import functional as F

class WeightSynthesisEngine(nn.Module):
    def __init__(self, d_model, num_heads, expansion=3.0, latent_dim=64):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.expansion = expansion
        self.latent_dim = latent_dim
        
        attn_params = num_heads * (d_model // num_heads) * d_model * 4
        mlp_params = int(d_model * expansion) * d_model * 2
        total_params = attn_params + mlp_params
        
        self.entropy_encoder = nn.Sequential(
            nn.Linear(d_model, latent_dim),
            nn.ReLU(),
            nn.Linear(latent_dim, latent_dim)
        )
        
        self.attn_generator = nn.Linear(latent_dim, attn_params)
        self.mlp_generator = nn.Linear(latent_dim, mlp_params)
        
    def forward(self, x):
        batch_size, seq_len, d_model = x.shape
        
        entropy = -torch.sum(F.softmax(x, dim=-1) * F.log_softmax(x, dim=-1), dim=-1)
        entropy = entropy.mean(dim=1)
        
        latent = self.entropy_encoder(entropy)
        
        attn_params = self.attn_generator(latent)
        mlp_params = self.mlp_generator(latent)
        
        return {
            'latent': latent,
            'attn_params': attn_params,
            'mlp_params': mlp_params,
            'entropy': entropy
        }


class ContextAdaptiveLayer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, expansion=3.0):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.expansion = expansion
        
        self.wse = WeightSynthesisEngine(d_model, num_heads, expansion)
        
        self.base_attn = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.base_mlp = nn.Sequential(
            nn.Linear(d_model, int(d_model * expansion)),
            nn.GELU(),
            nn.Linear(int(d_model * expansion), d_model)
        )
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
    def forward(self, x):
        synthesis_info = self.wse(x)
        
        attn_out, _ = self.base_attn(
            self.norm1(x), self.norm1(x), self.norm1(x)
        )
        
        mlp_out = self.base_mlp(self.norm2(x))
        
        scaling_factor = 1.0 + synthesis_info['latent'].mean().tanh()
        attn_out = attn_out * scaling_factor
        mlp_out = mlp_out * scaling_factor
        
        return x + attn_out + mlp_out, synthesis_info


class AdaptiveTransformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, vocab_size=1024, expansion=3.0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embed = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)
        
        self.adaptive_layer = ContextAdaptiveLayer(d_model, num_heads, expansion)
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.embedding.weight
        
    def forward(self, x):
        seq_len = x.shape[1]
        x = self.embedding(x)
        x = x + self.pos_embed[:, :seq_len, :]
        
        x, synthesis_info = self.adaptive_layer(x)
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits, synthesis_info
