"""
Simplified State-Space Model (S4/Mamba-inspired) layer.
OpenAI specifically requested SSM submissions.

Why SSM for Parameter Golf:
- Linear complexity O(n) vs O(n²) for attention
- Implicit infinite context window
- Fewer parameters for same effective receptive field
- The TACITUS parallel: SSMs process sequences like a diplomat
  reading a cable — maintaining a compressed state of everything
  seen so far, updating beliefs incrementally.

This implements a simplified S4D (diagonal state-space) layer
that can REPLACE one attention layer to save parameters.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class S4DLayer(nn.Module):
    """
    Simplified diagonal state-space model.
    x'(t) = Ax(t) + Bu(t)
    y(t) = Cx(t) + Du(t)
    
    With diagonal A, this becomes N independent 1D recurrences
    that can be computed as a convolution in parallel.
    """
    def __init__(self, d_model, state_dim=64, dt_min=0.001, dt_max=0.1):
        super().__init__()
        self.d_model = d_model
        self.state_dim = state_dim
        
        # Diagonal A matrix (complex, for oscillatory dynamics)
        # Initialize with HiPPO theory — proven optimal for sequence modeling
        log_A_real = torch.log(0.5 * torch.ones(d_model, state_dim))
        A_imag = math.pi * torch.arange(state_dim).float().unsqueeze(0).expand(d_model, -1)
        self.log_A_real = nn.Parameter(log_A_real)
        self.A_imag = nn.Parameter(A_imag)
        
        # B, C projections
        self.B = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
        self.C = nn.Parameter(torch.randn(d_model, state_dim) * 0.01)
        self.D = nn.Parameter(torch.ones(d_model))  # Skip connection
        
        # Learnable timestep
        log_dt = torch.rand(d_model) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)
        self.log_dt = nn.Parameter(log_dt)
        
        # Input/output projections
        self.input_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj = nn.Linear(d_model, d_model, bias=False)
        self.output_proj.weight.data.zero_()  # Start as identity
        
        # Gate (like Mamba's selection mechanism)
        self.gate = nn.Linear(d_model, d_model, bias=True)
        nn.init.constant_(self.gate.bias, -2.0)
    
    def forward(self, x):
        """
        x: [batch, seq_len, d_model]
        Uses convolution mode for parallel training.
        """
        B, T, D = x.shape
        
        u = self.input_proj(x)  # [B, T, D]
        
        # Discretize A using ZOH
        dt = torch.exp(self.log_dt)  # [D]
        A = -torch.exp(self.log_A_real) + 1j * self.A_imag  # [D, N] complex
        
        # Discretized A_bar = exp(A * dt)
        A_bar = torch.exp(A * dt.unsqueeze(-1))  # [D, N]
        B_bar = self.B * dt.unsqueeze(-1)  # [D, N]
        
        # Compute convolution kernel K
        # K[t] = C * A_bar^t * B_bar for t = 0, ..., T-1
        powers = A_bar.unsqueeze(0) ** torch.arange(T, device=x.device).unsqueeze(-1).unsqueeze(-1)
        K = (self.C.unsqueeze(0) * powers * B_bar.unsqueeze(0)).sum(-1).real  # [T, D]
        
        # Causal convolution via FFT
        K_f = torch.fft.rfft(K.T, n=2*T, dim=-1)  # [D, T+1]
        u_f = torch.fft.rfft(u.transpose(1,2), n=2*T, dim=-1)  # [B, D, T+1]
        y = torch.fft.irfft(K_f.unsqueeze(0) * u_f, n=2*T, dim=-1)[..., :T]  # [B, D, T]
        y = y.transpose(1, 2)  # [B, T, D]
        
        # Add skip connection
        y = y + self.D.unsqueeze(0).unsqueeze(0) * u
        
        # Gated residual
        g = torch.sigmoid(self.gate(x))
        return x + g * self.output_proj(y)

    def param_count(self):
        return sum(p.numel() for p in self.parameters())


if __name__ == "__main__":
    layer = S4DLayer(d_model=512, state_dim=64)
    x = torch.randn(2, 128, 512)
    y = layer(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    print(f"Params: {layer.param_count():,}")
    print(f"Est. size (fp16): {layer.param_count()*2/1024/1024:.2f} MB")
    g = torch.sigmoid(layer.gate(x))
    print(f"Gate mean: {g.mean():.4f} (should be ~0.12)")
    print("S4D layer working.")
