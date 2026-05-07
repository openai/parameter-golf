import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTQQuantizer(nn.Module):
    def __init__(self, bit_width=6):
        super().__init__()
        self.bit_width = bit_width
        self.scale = None
        self.zero_point = None
        
    def compute_hessian_diagonal(self, X):
        H = torch.matmul(X.T, X)
        return torch.diag(H)
    
    def quantize_weights(self, W, X, bit_width=None):
        if bit_width is None:
            bit_width = self.bit_width
        
        qmax = 2 ** (bit_width - 1) - 1
        qmin = -(2 ** (bit_width - 1))
        
        H_diag = self.compute_hessian_diagonal(X)
        H_diag = torch.clamp(H_diag, min=1e-6)
        
        scale = (W.abs().max() / qmax).unsqueeze(-1)
        scale = torch.clamp(scale, min=1e-8)
        
        W_q = torch.round(W / scale).clamp(qmin, qmax)
        W_dq = W_q * scale
        
        error = W - W_dq
        
        return W_dq, scale, error
    
    def full_hessian_quantize(self, W, X, bit_width=None):
        if bit_width is None:
            bit_width = self.bit_width
        
        qmax = 2 ** (bit_width - 1) - 1
        
        H = torch.matmul(X.T, X)
        
        try:
            H_inv = torch.linalg.inv(H + 1e-6 * torch.eye(H.shape[0], device=H.device))
        except:
            H_diag = torch.diag(H)
            H_inv = torch.diag(1.0 / (H_diag + 1e-6))
        
        scale = (W.abs().max() / qmax).unsqueeze(-1)
        W_q = torch.round(W / scale).clamp(-qmax - 1, qmax)
        W_dq = W_q * scale
        
        error = (W - W_dq).flatten()
        H_inv_diag = torch.diag(H_inv)
        
        correction = torch.matmul(H_inv, error.unsqueeze(-1)).squeeze(-1)
        
        return W_dq, scale, error, correction
    
    def ar_self_generated_calibration(self, model, num_sequences=64, seq_length=2048):
        calibration_sequences = []
        
        with torch.no_grad():
            for _ in range(num_sequences):
                seq = torch.randint(0, 1024, (1, 1024), device=next(model.parameters()).device)
                
                for _ in range(seq_length - 1024):
                    logits = model(seq)
                    next_token = torch.multinomial(F.softmax(logits[0, -1, :], dim=-1), num_samples=1)
                    seq = torch.cat([seq, next_token.unsqueeze(0)], dim=1)
                    if seq.shape[1] > seq_length:
                        seq = seq[:, :seq_length]
                
                calibration_sequences.append(seq.cpu())
        
        return torch.cat(calibration_sequences, dim=0)


class QuantizedLinear(nn.Module):
    def __init__(self, in_features, out_features, bit_width=6):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bit_width = bit_width
        
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.bias = nn.Parameter(torch.zeros(out_features))
        
        self.weight_scale = nn.Parameter(torch.ones(out_features, 1))
        self.weight_q = None
        
    def quantize(self, X):
        qmax = 2 ** (self.bit_width - 1) - 1
        qmin = -(2 ** (self.bit_width - 1))
        
        W_q = torch.round(self.weight / self.weight_scale).clamp(qmin, qmax)
        self.weight_q = W_q
        
    def forward(self, x):
        if self.weight_q is not None:
            W = self.weight_q * self.weight_scale
        else:
            W = self.weight
        
        return F.linear(x, W, self.bias)


class QuantizedTransformer(nn.Module):
    def __init__(self, d_model=512, num_heads=8, vocab_size=1024, bit_width=6):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        
        self.transformer_blocks = nn.ModuleList([
            nn.ModuleDict({
                'attn': nn.MultiheadAttention(d_model, num_heads, batch_first=True),
                'mlp.fc1': QuantizedLinear(d_model, int(d_model * 3), bit_width),
                'mlp.fc2': QuantizedLinear(int(d_model * 3), d_model, bit_width),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            })
            for _ in range(11)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.lm_head = QuantizedLinear(d_model, vocab_size, bit_width)
    
    def forward(self, x):
        x = self.embedding(x)
        
        for block in self.transformer_blocks:
            attn_out, _ = block['attn'](block['norm1'](x), block['norm1'](x), block['norm1'](x))
            x = x + attn_out
            
            mlp_out = block['mlp.fc2'](F.gelu(block['mlp.fc1'](block['norm2'](x))))
            x = x + mlp_out
        
        x = self.norm(x)
        logits = self.lm_head(x)
        
        return logits
