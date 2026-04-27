import math
import glob
from pathlib import Path
import numpy as np
import torch
from torch import Tensor, nn
import torch.nn.functional as F
import torch.distributed as dist
import sentencepiece as spm
from data_utils import load_data_shard

def build_sentencepiece_luts(
    sp: spm.SentencePieceProcessor, vocab_size: int, device: torch.device
) -> tuple[Tensor, Tensor, Tensor]:
    sp_vocab_size = int(sp.vocab_size())
    table_size = max(sp_vocab_size, vocab_size)
    base_bytes_np = np.zeros((table_size,), dtype=np.int16)
    has_leading_space_np = np.zeros((table_size,), dtype=np.bool_)
    is_boundary_token_np = np.ones((table_size,), dtype=np.bool_)
    for token_id in range(sp_vocab_size):
        if sp.is_control(token_id) or sp.is_unknown(token_id) or sp.is_unused(token_id):
            continue
        is_boundary_token_np[token_id] = False
        if sp.is_byte(token_id):
            base_bytes_np[token_id] = 1
            continue
        piece = sp.id_to_piece(token_id)
        if piece.startswith("▁"):
            has_leading_space_np[token_id] = True
            piece = piece[1:]
        base_bytes_np[token_id] = len(piece.encode("utf-8"))
    return (
        torch.tensor(base_bytes_np, dtype=torch.int16, device=device),
        torch.tensor(has_leading_space_np, dtype=torch.bool, device=device),
        torch.tensor(is_boundary_token_np, dtype=torch.bool, device=device),
    )


def load_validation_tokens(pattern: str, seq_len: int) -> Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    tokens = torch.cat([load_data_shard(file) for file in files]).contiguous()
    usable = ((tokens.numel() - 1) // seq_len) * seq_len
    if usable <= 0:
        raise ValueError(f"Validation split is too short for TRAIN_SEQ_LEN={seq_len}")
    return tokens[: usable + 1]


def eval_val(
    args,
    model: nn.Module,
    rank: int,
    world_size: int,
    device: torch.device,
    grad_accum_steps: int,
    val_tokens: Tensor,
    base_bytes_lut: Tensor,
    has_leading_space_lut: Tensor,
    is_boundary_token_lut: Tensor,
    max_steps: int | None = None,
    stride: int | None = None,
    ttt_lr: float = 0.0,
) -> tuple[float, float]:
    """
    Sliding Window Evaluation:
    If stride is set, each window overlap by (seq_len - stride).
    BPB is computed only on the non-overlapping 'tail' of each window to ensure warm context.
    """
    seq_len = args.train_seq_len
    eff_stride = stride if stride is not None else seq_len
    
    # Calculate how many possible windows we have
    num_possible_windows = (val_tokens.numel() - seq_len - 1) // eff_stride + 1
    
    # Partition windows among ranks
    win_start = (num_possible_windows * rank) // world_size
    win_end = (num_possible_windows * (rank + 1)) // world_size
    
    val_loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    val_token_count = torch.zeros((), device=device, dtype=torch.float64)
    val_byte_count = torch.zeros((), device=device, dtype=torch.float64)
    
    # 0. SAVE GRAD STATE
    orig_grad_state = {n: p.requires_grad for n, p in model.named_parameters()}

    local_batch_tokens = args.val_batch_size // (world_size * grad_accum_steps)
    local_batch_wins = max(1, local_batch_tokens // seq_len)

    # 1. IDENTIFY LORA PARAMETERS FOR TTT
    ttt_params = []
    if ttt_lr > 0:
        for name, p in model.named_parameters():
            if "lora_" in name:
                p.requires_grad = True
                ttt_params.append(p)
            else:
                p.requires_grad = False
        
        optimizer = torch.optim.SGD(ttt_params, lr=ttt_lr)
    else:
        # Standard Eval: No gradients needed anywhere
        for p in model.parameters():
            p.requires_grad = False

    model.eval()
    for batch_idx, b_win_start in enumerate(range(win_start, win_end, local_batch_wins)):
        if max_steps is not None and batch_idx >= max_steps:
            break
        
        b_win_end = min(b_win_start + local_batch_wins, win_end)
        
        # Pack windows into a batch
        batch_x, batch_y, batch_prev = [], [], []
        keep_masks = []
        
        for win_idx in range(b_win_start, b_win_end):
            raw_start = win_idx * eff_stride
            raw_end = raw_start + seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            
            x = local[:-1]
            y = local[1:]
            prev_ids = local[:-1]
            
            mask = torch.zeros((seq_len,), device=device, dtype=torch.bool)
            if win_idx == 0:
                mask[:] = True
            else:
                mask[seq_len - eff_stride:] = True
            
            batch_x.append(x)
            batch_y.append(y)
            batch_prev.append(prev_ids)
            keep_masks.append(mask)

        if not batch_x: continue
        
        x = torch.stack(batch_x)
        y = torch.stack(batch_y)
        prev_ids = torch.stack(batch_prev)
        mask = torch.stack(keep_masks)

        # 1. SCORE PHASE (Must be no_grad to use compiled inference kernels stably)
        with torch.no_grad():
            logits = model.forward_logits(x, use_compiled=True)
            loss_all = F.cross_entropy(logits.permute(0, 2, 1).float(), y, reduction="none")

            kept_loss = loss_all[mask]
            val_loss_sum += kept_loss.sum().to(torch.float64)
            val_token_count += mask.sum().to(torch.float64)
            
            tgt_ids_kept = y[mask]
            prev_ids_kept = prev_ids[mask]
            token_bytes = base_bytes_lut[tgt_ids_kept].to(dtype=torch.int16)
            token_bytes += (has_leading_space_lut[tgt_ids_kept] & ~is_boundary_token_lut[prev_ids_kept]).to(dtype=torch.int16)
            val_byte_count += token_bytes.to(torch.float64).sum()

        # 2. ADAPT PHASE (Legal TTT)
        if ttt_lr > 0:
            # Eager execution for adaptation to avoid Inductor re-compilation on Windows
            logits_ttt = model.forward_logits(x, use_compiled=False)
            loss_for_backprop = F.cross_entropy(logits_ttt.permute(0, 2, 1).float(), y, reduction="none").mean()
            loss_for_backprop.backward()
            optimizer.step()
            optimizer.zero_grad()

    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(val_loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_token_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(val_byte_count, op=dist.ReduceOp.SUM)

    # RESTORE GRAD STATE
    for n, p in model.named_parameters():
        p.requires_grad = orig_grad_state[n]
        
    val_loss = val_loss_sum / val_token_count
    bits_per_token = val_loss.item() / math.log(2.0)
    tokens_per_byte = val_token_count.item() / val_byte_count.item()
    model.train()
    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)
