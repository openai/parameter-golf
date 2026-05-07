import torch
import torch.nn as nn
from torch.nn import functional as F

class GaloisFieldHashEmbedding(nn.Module):
    def __init__(self, vocab_size, embedding_dim, num_hash_functions=4):
        super().__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_hash_functions = num_hash_functions
        
        table_size = 8192
        self.hash_embedding_tables = nn.ParameterList([
            nn.Parameter(torch.randn(table_size, embedding_dim) * 0.02)
            for _ in range(num_hash_functions)
        ])
        
        self.hash_seeds = [0x9e3779b9, 0xbf58476d, 0x94d049bb, 0xc1b53c1a]
        
    def gf256_hash(self, bigram, seed):
        x = bigram ^ seed
        x = ((x ^ (x >> 16)) * 0x7feb352d) & 0xffffffff
        x = ((x ^ (x >> 15)) * 0x846ca68b) & 0xffffffff
        x = (x ^ (x >> 16)) & 0xffffffff
        return x % 8192
    
    def forward(self, token_ids):
        batch_size, seq_len = token_ids.shape
        
        hash_outputs = []
        for h in range(self.num_hash_functions):
            hashes = []
            for i in range(seq_len - 1):
                bigram = (token_ids[:, i] << 10) | token_ids[:, i + 1]
                hash_idx = self.gf256_hash(bigram, self.hash_seeds[h])
                hashes.append(hash_idx)
            
            if hashes:
                hashes = torch.stack(hashes, dim=1)
                hash_emb = self.hash_embedding_tables[h][hashes]
                hash_outputs.append(hash_emb)
        
        if hash_outputs:
            combined = torch.mean(torch.stack(hash_outputs, dim=0), dim=0)
        else:
            combined = torch.zeros(batch_size, seq_len - 1, self.embedding_dim, device=token_ids.device)
        
        return combined


class GFHEAugmentedModel(nn.Module):
    def __init__(self, d_model=512, num_heads=8, vocab_size=1024, embedding_dim=128):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.gfhe = GaloisFieldHashEmbedding(vocab_size, embedding_dim, num_hash_functions=4)
        
        self.fusion_gate = nn.Sequential(
            nn.Linear(d_model + embedding_dim, d_model),
            nn.Sigmoid()
        )
        
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, num_heads, batch_first=True),
                'mlp': nn.Sequential(
                    nn.Linear(d_model, int(d_model * 3)),
                    nn.GELU(),
                    nn.Linear(int(d_model * 3), d_model)
                ),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            })
            for _ in range(11)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)
        self.lm_head.weight = self.embedding.weight
    
    def forward(self, x):
        seq_len = x.shape[1]
        
        x_emb = self.embedding(x)
        
        gfhe_emb = self.gfhe(x)
        
        padding = torch.zeros(gfhe_emb.shape[0], 1, gfhe_emb.shape[2], device=x.device)
        gfhe_emb = torch.cat([padding, gfhe_emb], dim=1)
        
        fused = torch.cat([x_emb, gfhe_emb], dim=-1)
        gate = self.fusion_gate(fused)
        x = x_emb * gate + gfhe_emb * (1 - gate)
        
        for block in self.transformer_blocks:
            attn_out, _ = block['attn'](block['norm1'](x), block['norm1'](x), block['norm1'](x))
            x = x + attn_out
            mlp_out = block['mlp'](block['norm2'](x))
            x = x + mlp_out
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
