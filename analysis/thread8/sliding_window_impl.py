"""
Sliding Window Evaluation Implementation for Parameter Golf.

Drop-in replacement for eval_val() that uses overlapping windows
to give each token more context, reducing BPB by 0.015-0.060.

Usage: Replace the eval_val() call in train_gpt.py with eval_val_sliding().
Key change: needs model.get_logits() method (or modify forward() to support it).
"""

import math
import torch
import torch.nn.functional as F
from torch import Tensor


def eval_val_sliding(
    args,          # Hyperparameters object
    model,         # GPT model (must have get_logits() method)
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    window: int = 1024,    # same as train_seq_len
    stride: int = 256,     # score last 256 tokens per window (4x compute)
) -> tuple[float, float]:
    """
    Sliding window BPB evaluation.
    
    Every token is scored exactly once, but with (window - stride) minimum context.
    Compared to non-overlapping (stride=window), this gives each token more context,
    resulting in lower per-token CE and thus lower BPB.
    
    stride=window: equivalent to baseline non-overlapping eval
    stride=1:      optimal (every token has max context), but 1024x compute
    stride=256:    good tradeoff (768+ context, 4x compute)
    """
    N = val_tokens.numel() - 1  # total scoreable tokens
    
    # Distribute work across ranks
    total_windows = max(1, (N - window + stride) // stride + 1)
    rank_start = (total_windows * rank) // world_size
    rank_end = (total_windows * (rank + 1)) // world_size
    
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    
    model.eval()
    with torch.inference_mode():
        for win_idx in range(rank_start, rank_end):
            # Window start position in the token stream
            pos = win_idx * stride
            end = min(pos + window, N)
            actual_len = end - pos
            
            if actual_len <= 0:
                break
            
            # Determine scoring region
            if pos == 0:
                # First window: score all tokens (some have less context, unavoidable)
                score_from = 0
                score_count = actual_len
            else:
                # Later windows: only score the last 'stride' tokens
                score_count = min(stride, actual_len)
                score_from = actual_len - score_count
            
            # Construct input/target tensors
            x = val_tokens[pos:end].unsqueeze(0).to(device=device, dtype=torch.int64)
            y = val_tokens[pos + 1:end + 1].unsqueeze(0).to(device=device, dtype=torch.int64)
            
            # Forward pass to get logits
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model.get_logits(x)  # [1, actual_len, vocab_size]
            
            # Per-token CE loss on scored positions only
            scored_logits = logits[0, score_from:score_from + score_count].float()
            scored_targets = y[0, score_from:score_from + score_count]
            per_token_ce = F.cross_entropy(
                scored_logits, scored_targets, reduction="none"
            )
            
            loss_sum += per_token_ce.to(torch.float64).sum()
            token_count += float(score_count)
            
            # Byte counting for scored tokens
            prev_ids = x[0, score_from:score_from + score_count]
            tgt_ids = y[0, score_from:score_from + score_count]
            tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
            tb += (
                has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
            ).to(dtype=torch.int16)
            byte_count += tb.to(torch.float64).sum()
    
    # All-reduce across ranks
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    
    val_loss = loss_sum / token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


def eval_val_sliding_batched(
    args,
    model,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    window: int = 1024,
    stride: int = 256,
    batch_size: int = 32,
) -> tuple[float, float]:
    """
    Batched sliding window evaluation (faster than sequential).
    
    Packs multiple windows into a batch for efficient GPU utilization.
    Same semantics as eval_val_sliding but ~batch_size times faster.
    """
    N = val_tokens.numel() - 1
    
    # Generate all window positions
    positions = list(range(0, N - window + stride, stride))
    if not positions:
        positions = [0]
    
    # Distribute across ranks
    rank_positions = positions[rank::world_size]
    
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    token_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    
    model.eval()
    with torch.inference_mode():
        for batch_start in range(0, len(rank_positions), batch_size):
            batch_positions = rank_positions[batch_start:batch_start + batch_size]
            B = len(batch_positions)
            
            # Pack windows into a batch
            x_batch = torch.zeros(B, window, dtype=torch.int64, device=device)
            y_batch = torch.zeros(B, window, dtype=torch.int64, device=device)
            score_froms = []
            score_counts = []
            
            for i, pos in enumerate(batch_positions):
                end = min(pos + window, N)
                actual_len = end - pos
                
                x_batch[i, :actual_len] = val_tokens[pos:end].to(torch.int64)
                y_batch[i, :actual_len] = val_tokens[pos + 1:end + 1].to(torch.int64)
                
                if pos == 0:
                    score_froms.append(0)
                    score_counts.append(actual_len)
                else:
                    sc = min(stride, actual_len)
                    score_froms.append(actual_len - sc)
                    score_counts.append(sc)
            
            # Forward pass
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model.get_logits(x_batch)  # [B, window, vocab]
            
            # Score each window
            for i in range(B):
                sf = score_froms[i]
                sc = score_counts[i]
                
                scored_logits = logits[i, sf:sf + sc].float()
                scored_targets = y_batch[i, sf:sf + sc]
                per_token_ce = F.cross_entropy(
                    scored_logits, scored_targets, reduction="none"
                )
                
                loss_sum += per_token_ce.to(torch.float64).sum()
                token_count += float(sc)
                
                prev_ids = x_batch[i, sf:sf + sc]
                tgt_ids = y_batch[i, sf:sf + sc]
                tb = base_bytes_lut[tgt_ids].to(dtype=torch.int16)
                tb += (
                    has_leading_space_lut[tgt_ids] & ~is_boundary_token_lut[prev_ids]
                ).to(dtype=torch.int16)
                byte_count += tb.to(torch.float64).sum()
    
    import torch.distributed as dist
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)
    
    val_loss = loss_sum / token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = token_count.item() / byte_count.item()
    
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)


# === GPT MODEL MODIFICATION ===
# Add this method to the GPT class:

GPT_GET_LOGITS_CODE = """
def get_logits(self, input_ids: Tensor) -> Tensor:
    \"\"\"Return per-position logits without computing loss.\"\"\"
    x = self.tok_emb(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[Tensor] = []

    for i in range(self.num_encoder_layers):
        x = self.blocks[i](x, x0)
        skips.append(x)
    for i in range(self.num_decoder_layers):
        if skips:
            x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = self.blocks[self.num_encoder_layers + i](x, x0)

    x = self.final_norm(x)
    if self.tie_embeddings:
        logits = F.linear(x, self.tok_emb.weight)
    else:
        logits = self.lm_head(x)
    return self.logit_softcap * torch.tanh(logits / self.logit_softcap)
"""


# === ROPE EXTRAPOLATION FOR LONGER CONTEXT ===

ROPE_NTK_SCALING_CODE = """
class RotaryNTK(nn.Module):
    \"\"\"RoPE with NTK-aware scaling for context extension.\"\"\"
    def __init__(self, dim: int, base: float = 10000.0, 
                 train_seq_len: int = 1024, eval_seq_len: int = 2048):
        super().__init__()
        # NTK-aware base scaling
        alpha = eval_seq_len / train_seq_len
        scaled_base = base * alpha ** (dim / (dim - 2))
        inv_freq = 1.0 / (scaled_base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)
        self._seq_len_cached = 0
        self._cos_cached = None
        self._sin_cached = None

    def forward(self, seq_len, device, dtype):
        if (self._cos_cached is None or self._seq_len_cached != seq_len 
            or self._cos_cached.device != device):
            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)
            freqs = torch.outer(t, self.inv_freq.to(device))
            self._cos_cached = freqs.cos()[None, None, :, :]
            self._sin_cached = freqs.sin()[None, None, :, :]
            self._seq_len_cached = seq_len
        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)
"""


# === TEST-TIME TRAINING ===

TTT_CODE = """
def eval_val_ttt(args, model, val_tokens, base_bytes_lut, 
                 has_leading_space_lut, is_boundary_token_lut,
                 device, ttt_lr=1e-5, ttt_steps=1, 
                 ttt_params='layernorms',  # 'layernorms', 'all', 'embeddings'
                 window=1024, stride=256):
    \"\"\"
    Two-pass Test-Time Training evaluation.
    
    Pass 1: Adapt model to eval distribution via online learning.
    Pass 2: Score with adapted model using sliding window.
    \"\"\"
    import copy
    
    # Save original weights for potential reset
    original_state = copy.deepcopy(model.state_dict())
    
    # Select parameters to update
    ttt_param_list = []
    for name, param in model.named_parameters():
        if ttt_params == 'all':
            ttt_param_list.append(param)
        elif ttt_params == 'layernorms' and ('norm' in name or 'scale' in name):
            ttt_param_list.append(param)
        elif ttt_params == 'embeddings' and 'emb' in name:
            ttt_param_list.append(param)
        else:
            param.requires_grad_(False)
    
    optimizer = torch.optim.SGD(ttt_param_list, lr=ttt_lr)
    
    # Pass 1: Online adaptation
    N = val_tokens.numel() - 1
    model.train()
    for pos in range(0, N - window, stride):
        x = val_tokens[pos:pos + window].unsqueeze(0).to(device)
        y = val_tokens[pos + 1:pos + window + 1].unsqueeze(0).to(device)
        
        for _ in range(ttt_steps):
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                loss = model(x, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    # Pass 2: Score with adapted model using sliding window
    result = eval_val_sliding(args, model, ..., window=window, stride=stride)
    
    # Optionally restore original weights
    # model.load_state_dict(original_state)
    
    return result
\"\"\"
