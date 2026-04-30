"""
Hardik SOTA Run: SP8192 + 3-Layer Depth Recurrence + Parallel Residuals + Muon + Legal Score-First TTT
Designed for Parameter Golf 16MB / 10min track.
"""

import os
import sys
import time
import math
import glob
import uuid
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
import numpy as np
import sentencepiece as spm
from pathlib import Path

# -----------------------------
# HYPERPARAMETERS
# -----------------------------

class Hyperparameters:
    data_path = os.environ.get("DATA_PATH", "./data/datasets/fineweb10B_sp8192")
    train_files = os.path.join(data_path, "fineweb_train_*.bin")
    val_files = os.path.join(data_path, "fineweb_val_*.bin")
    tokenizer_path = os.environ.get("TOKENIZER_PATH", "./data/tokenizers/fineweb_8192_bpe.model")
    run_id = os.environ.get("RUN_ID", str(uuid.uuid4()))
    seed = int(os.environ.get("SEED", 1337))

    val_batch_size = 524_288
    val_loss_every = 1000
    train_log_every = 200

    iterations = 20000
    warmdown_iters = 1500
    warmup_steps = 20
    train_batch_tokens = 524_288
    train_seq_len = 1024
    max_wallclock_seconds = 600.0

    vocab_size = 8192
    num_layers = 12
    model_dim = 512
    num_heads = 8
    num_kv_heads = 4
    mlp_mult = 3
    tie_embeddings = True
    qk_gain_init = 5.25
    logit_softcap = 30.0

    # Optimizer
    matrix_lr = 0.045
    muon_momentum = 0.96
    adam_beta1 = 0.9
    adam_beta2 = 0.95
    weight_decay = 0.095

# -----------------------------
# MUON OPTIMIZER
# -----------------------------

def zeropower_via_newtonschulz5(G, steps=10, eps=1e-7):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= X.norm() + eps
    if G.size(0) > G.size(1):
        X = X.T
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
    return X.T if G.size(0) > G.size(1) else X

class Muon(torch.optim.Optimizer):
    def __init__(self, params, lr=0.02, momentum=0.95, steps=5):
        defaults = dict(lr=lr, momentum=momentum, steps=steps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            steps = group['steps']
            for p in group['params']:
                if p.grad is None: continue
                g = p.grad
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(g)
                buf = state['momentum_buffer']
                buf.mul_(momentum).add_(g)
                u = zeropower_via_newtonschulz5(buf, steps=steps)
                p.add_(u, alpha=-lr)

# -----------------------------
# MODEL ARCHITECTURE
# -----------------------------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))
    def forward(self, x):
        return F.rms_norm(x, (x.size(-1),), self.weight, self.eps)

class ParallelBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.ln = RMSNorm(config.model_dim)
        self.attn = nn.Linear(config.model_dim, 3 * config.model_dim, bias=False)
        self.proj = nn.Linear(config.model_dim, config.model_dim, bias=False)
        self.mlp_fc = nn.Linear(config.model_dim, config.mlp_mult * config.model_dim, bias=False)
        self.mlp_proj = nn.Linear(config.mlp_mult * config.model_dim, config.model_dim, bias=False)
        self.head_dim = config.model_dim // config.num_heads
        self.num_heads = config.num_heads
        
        # QK Gain for better stability
        self.q_gain = nn.Parameter(torch.full((config.num_heads,), config.qk_gain_init))

    def forward(self, x):
        h = self.ln(x)
        # Parallel Attention and MLP
        qkv = self.attn(h)
        q, k, v = qkv.split(qkv.size(-1)//3, dim=-1)
        
        # Reshape for multi-head
        q = q.view(q.size(0), q.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        k = k.view(k.size(0), k.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        v = v.view(v.size(0), v.size(1), self.num_heads, self.head_dim).transpose(1, 2)
        
        # Apply QK Gain
        q = q * self.q_gain[None, :, None, None]
        
        attn_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        attn_out = attn_out.transpose(1, 2).reshape(x.shape)
        attn_out = self.proj(attn_out)
        
        mlp_out = self.mlp_proj(F.gelu(self.mlp_fc(h)))
        
        return x + attn_out + mlp_out

class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.tok_emb = nn.Embedding(config.vocab_size, config.model_dim)
        self.blocks = nn.ModuleList([ParallelBlock(config) for _ in range(config.num_layers)])
        self.ln_f = RMSNorm(config.model_dim)
        self.lm_head = nn.Linear(config.model_dim, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.tok_emb.weight

    def forward(self, idx, targets=None):
        x = self.tok_emb(idx)
        x0 = x # For recurrence if needed
        
        # Depth Recurrence: Loop over layers multiple times
        # Here we do a simple loop for L3-5 as in the SOTA run
        for i, block in enumerate(self.blocks):
            if 3 <= i <= 5:
                # Recurrence loop
                for _ in range(2):
                    x = block(x)
            else:
                x = block(x)
        
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        # Softcap logits for stability
        logits = self.config.logit_softcap * torch.tanh(logits / self.config.logit_softcap)
        
        if targets is not None:
            return F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits

# -----------------------------
# TRAINING LOOP (STRIPPED)
# -----------------------------

def main():
    # Setup distributed, data loading, etc.
    # This is a placeholder for the full script which would follow the standard template
    # but with the model above.
    pass

if __name__ == "__main__":
    main()
