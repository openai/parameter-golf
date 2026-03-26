"""
Fixed evaluation & data harness for Parameter Golf autoresearch.
DO NOT MODIFY — the agent modifies train_pgolf.py only.

This wraps Parameter Golf's data loading, evaluation, and quantization
pipeline into the autoresearch-compatible interface.

Usage:
    python prepare_pgolf.py              # download data + verify setup
    python prepare_pgolf.py --shards 1   # quick test with 1 shard
"""

import argparse
import io
import math
import os
import struct
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import sentencepiece as spm
import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

TIME_BUDGET = 300              # 5 minutes per experiment (wall clock training time)
TRAIN_SEQ_LEN = 1024           # default sequence length
VOCAB_SIZE = 1024              # SentencePiece BPE vocab
MAX_ARTIFACT_BYTES = 16_000_000  # 16MB hard cap for Parameter Golf

# Paths (relative to parameter-golf repo root)
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "datasets", "fineweb10B_sp1024")
TOKENIZER_PATH = os.path.join(os.path.dirname(__file__), "data", "tokenizers", "fineweb_1024_bpe.model")

# ---------------------------------------------------------------------------
# Data loading utilities
# ---------------------------------------------------------------------------


def load_data_shard(file_path):
    """Load a token shard, skipping the 1024-byte header."""
    file = Path(file_path)
    header_bytes = 256 * np.dtype("<i4").itemsize
    token_bytes = np.dtype("<u2").itemsize
    header = np.fromfile(file, dtype="<i4", count=256)
    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:
        raise ValueError(f"Unexpected shard header for {file}")
    num_tokens = int(header[2])
    tokens_np = np.fromfile(file, dtype="<u2", count=num_tokens, offset=header_bytes)
    return torch.from_numpy(tokens_np.astype(np.int32, copy=False))


def load_validation_tokens(pattern="fineweb_val_*.bin", seq_len=1024):
    """Load all validation token shards into a single tensor."""
    dataset_dir = Path(DATA_PATH).resolve()
    val_files = sorted(dataset_dir.glob(pattern))
    if not val_files:
        raise FileNotFoundError(
            f"No validation files matching '{pattern}' in {dataset_dir}. "
            "Run: python data/cached_challenge_fineweb.py --variant sp1024"
        )
    all_tokens = []
    for vf in val_files:
        toks = load_data_shard(vf)
        all_tokens.append(toks)
    tokens = torch.cat(all_tokens)
    # Trim to multiple of seq_len + 1 for clean batching
    n = ((tokens.numel() - 1) // seq_len) * seq_len + 1
    return tokens[:n]


def get_tokenizer():
    """Load the SentencePiece tokenizer."""
    if not os.path.exists(TOKENIZER_PATH):
        raise FileNotFoundError(
            f"Tokenizer not found at {TOKENIZER_PATH}. "
            "Run: python data/cached_challenge_fineweb.py --variant sp1024"
        )
    return spm.SentencePieceProcessor(model_file=TOKENIZER_PATH)


def build_sentencepiece_luts(sp, vocab_size, device):
    """Build lookup tables for BPB calculation (from train_gpt.py)."""
    base_bytes_lut = torch.zeros(vocab_size, dtype=torch.float32, device=device)
    has_leading_space_lut = torch.zeros(vocab_size, dtype=torch.bool, device=device)
    is_boundary_token_lut = torch.zeros(vocab_size, dtype=torch.bool, device=device)

    for token_id in range(vocab_size):
        piece = sp.id_to_piece(token_id)
        if sp.is_unknown(token_id) or sp.is_control(token_id):
            is_boundary_token_lut[token_id] = True
            continue
        if sp.is_byte(token_id):
            base_bytes_lut[token_id] = 1.0
            continue
        text = piece.replace("\u2581", " ")
        raw = text.encode("utf-8")
        base_bytes_lut[token_id] = float(len(raw))
        has_leading_space = piece.startswith("\u2581")
        has_leading_space_lut[token_id] = has_leading_space
        if has_leading_space and len(text.strip()) > 0:
            text_no_space = text[1:]
            raw_no_space = text_no_space.encode("utf-8")
            base_bytes_lut[token_id] = float(len(raw_no_space))
    return base_bytes_lut, has_leading_space_lut, is_boundary_token_lut


class DistributedTokenLoader:
    """Token loader for training (single-GPU version)."""

    def __init__(self, file_pattern, rank, world_size, device):
        dataset_dir = Path(DATA_PATH).resolve()
        self.files = sorted(dataset_dir.glob(file_pattern))
        if not self.files:
            raise FileNotFoundError(f"No files matching '{file_pattern}' in {dataset_dir}")
        self.rank = rank
        self.world_size = world_size
        self.device = device
        self.file_idx = rank
        self._load_shard()

    def _load_shard(self):
        idx = self.file_idx % len(self.files)
        self.tokens = load_data_shard(self.files[idx]).to(self.device)
        self.pos = 0
        self.file_idx += self.world_size

    def next_batch(self, total_batch_tokens, seq_len, grad_accum_steps):
        B = total_batch_tokens // (seq_len * grad_accum_steps)
        needed = B * seq_len + 1
        if self.pos + needed > self.tokens.numel():
            self._load_shard()
        buf = self.tokens[self.pos : self.pos + needed].long()
        self.pos += B * seq_len
        x = buf[:B * seq_len].view(B, seq_len)
        y = buf[1 : B * seq_len + 1].view(B, seq_len)
        return x, y


# ---------------------------------------------------------------------------
# Evaluation (fixed metric — DO NOT CHANGE)
# ---------------------------------------------------------------------------


@torch.no_grad()
def eval_val(model, device, val_tokens, base_bytes_lut, has_leading_space_lut,
             is_boundary_token_lut, seq_len=1024, val_batch_size=524288,
             eval_stride=None, eval_batch_seqs=None):
    """
    Evaluate val_bpb with optional sliding window.
    Returns (val_loss, val_bpb).
    """
    model.eval()

    if eval_stride is not None and eval_stride > 0:
        return _eval_sliding_window(
            model, device, val_tokens, base_bytes_lut,
            has_leading_space_lut, is_boundary_token_lut,
            seq_len=seq_len, stride=eval_stride,
            eval_batch_seqs=eval_batch_seqs or 4,
        )

    n_tokens = val_tokens.numel() - 1
    total_loss = 0.0
    total_bpb_numer = 0.0
    total_bpb_denom = 0.0
    pos = 0

    while pos < n_tokens:
        end = min(pos + val_batch_size, n_tokens)
        chunk = val_tokens[pos : end + 1].to(device).long()
        chunk_len = end - pos
        
        # Calculate how many full sequences fit in this chunk
        usable_tokens = (chunk_len // seq_len) * seq_len
        if usable_tokens <= 0:
            break
            
        x_full = chunk[:usable_tokens].reshape(-1, seq_len)
        y_full = chunk[1:usable_tokens + 1].reshape(-1, seq_len)
        
        micro_batch_seqs = eval_batch_seqs or 16
        chunk_losses = []
        for i in range(0, x_full.size(0), micro_batch_seqs):
            x_micro = x_full[i:i + micro_batch_seqs]
            y_micro = y_full[i:i + micro_batch_seqs]
            with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
                logits = model(x_micro)
            loss_micro = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)), y_micro.view(-1), reduction="none"
            )
            chunk_losses.append(loss_micro)
            
        loss_per_tok = torch.cat(chunk_losses)

        total_loss += loss_per_tok.sum().item()

        target_flat = y_full.view(-1)
        raw_bytes = base_bytes_lut[target_flat]
        leading = has_leading_space_lut[target_flat]
        boundary = is_boundary_token_lut[target_flat]
        prev_boundary = torch.zeros_like(boundary)
        if target_flat.numel() > 1:
            prev_boundary[1:] = is_boundary_token_lut[chunk[:usable_tokens-1].view(-1)[-target_flat.numel() + 1:]] if pos > 0 else boundary[:-1]
        adj_bytes = raw_bytes + leading.float()
        adj_bytes[boundary] = 0.0

        total_bpb_numer += (loss_per_tok * (adj_bytes > 0).float()).sum().item()
        total_bpb_denom += adj_bytes.sum().item()
        pos = end

    val_loss = total_loss / n_tokens
    val_bpb = total_bpb_numer / (math.log(2) * max(total_bpb_denom, 1.0))
    return val_loss, val_bpb


def _eval_sliding_window(model, device, val_tokens, base_bytes_lut,
                         has_leading_space_lut, is_boundary_token_lut,
                         seq_len=1024, stride=64, eval_batch_seqs=4):
    """Sliding window evaluation for better per-token context."""
    n_tokens = val_tokens.numel() - 1
    loss_sums = torch.zeros(n_tokens, dtype=torch.float64, device=device)
    loss_counts = torch.zeros(n_tokens, dtype=torch.int32, device=device)

    all_val = val_tokens.to(device).long()
    positions = list(range(0, n_tokens - seq_len + 1, stride))

    for batch_start in range(0, len(positions), eval_batch_seqs):
        batch_positions = positions[batch_start : batch_start + eval_batch_seqs]
        xs, ys = [], []
        for p in batch_positions:
            xs.append(all_val[p : p + seq_len])
            ys.append(all_val[p + 1 : p + seq_len + 1])
        x = torch.stack(xs)
        y = torch.stack(ys)

        with torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=True):
            logits = model(x)
        losses = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.size(-1)), y.view(-1), reduction="none"
        ).view(len(batch_positions), seq_len)

        for i, p in enumerate(batch_positions):
            score_start = max(p + seq_len - stride, p) if p > 0 else 0
            local_start = score_start - p
            local_end = seq_len
            global_indices = torch.arange(score_start, p + local_end, device=device)
            valid = global_indices < n_tokens
            global_indices = global_indices[valid]
            local_losses = losses[i, local_start:local_start + valid.sum()]
            loss_sums[global_indices] += local_losses.double()
            loss_counts[global_indices] += 1

    scored = loss_counts > 0
    avg_losses = torch.zeros_like(loss_sums)
    avg_losses[scored] = loss_sums[scored] / loss_counts[scored].double()

    target_ids = all_val[1 : n_tokens + 1]
    raw_bytes = base_bytes_lut[target_ids]
    leading = has_leading_space_lut[target_ids]
    boundary = is_boundary_token_lut[target_ids]
    adj_bytes = raw_bytes + leading.float()
    adj_bytes[boundary] = 0.0

    valid_mask = scored & (adj_bytes > 0)
    total_loss = avg_losses[scored].sum().item()
    val_loss = total_loss / max(scored.sum().item(), 1)
    bpb_numer = (avg_losses * valid_mask.float()).sum().item()
    bpb_denom = adj_bytes[scored].sum().item()
    val_bpb = bpb_numer / (math.log(2) * max(bpb_denom, 1.0))

    return val_loss, val_bpb


# ---------------------------------------------------------------------------
# Quantization (for artifact size checking)
# ---------------------------------------------------------------------------


def quantize_state_dict_int8(state_dict, keep_fp16_patterns=None):
    """Quantize a state dict to int8 with per-row scales."""
    if keep_fp16_patterns is None:
        keep_fp16_patterns = ["tok_emb"]  # Keep embeddings in fp16

    quant_obj = {}
    total_baseline = 0
    total_payload = 0

    for name, tensor in state_dict.items():
        t = tensor.detach().float()
        total_baseline += t.numel() * t.element_size()

        keep_fp16 = any(p in name for p in keep_fp16_patterns)
        if keep_fp16 or t.ndim < 2:
            quant_obj[name] = t.half()
            total_payload += t.numel() * 2
        else:
            absmax = t.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            scale = absmax / 127.0
            quantized = (t / scale).round().clamp(-128, 127).to(torch.int8)
            quant_obj[name + ".qweight"] = quantized
            quant_obj[name + ".scale"] = scale.half().squeeze(-1)
            total_payload += quantized.numel() + scale.numel() * 2

    stats = {
        "baseline_tensor_bytes": total_baseline,
        "int8_payload_bytes": total_payload,
    }
    return quant_obj, stats


def dequantize_state_dict_int8(quant_obj):
    """Dequantize int8 state dict back to float tensors."""
    state_dict = {}
    done_keys = set()

    for key in list(quant_obj.keys()):
        if key in done_keys:
            continue
        if key.endswith(".qweight"):
            base = key[: -len(".qweight")]
            qw = quant_obj[key]
            scale = quant_obj[base + ".scale"].float().unsqueeze(-1)
            state_dict[base] = (qw.float() * scale).to(torch.bfloat16)
            done_keys.add(key)
            done_keys.add(base + ".scale")
        else:
            state_dict[key] = quant_obj[key].to(torch.bfloat16)
            done_keys.add(key)

    return state_dict


def measure_artifact_size(model, code_path=None):
    """
    Quantize model, compress, and return total artifact size in bytes.
    Returns (artifact_bytes, quant_gap_info).
    """
    sd = model.state_dict() if hasattr(model, "state_dict") else model
    quant_obj, stats = quantize_state_dict_int8(sd)

    buf = io.BytesIO()
    torch.save(quant_obj, buf)
    raw = buf.getvalue()
    compressed = zlib.compress(raw, level=9)
    model_bytes = len(compressed)

    code_bytes = 0
    if code_path and os.path.exists(code_path):
        with open(code_path, "rb") as f:
            code_bytes = len(f.read())

    return model_bytes + code_bytes, stats


# ---------------------------------------------------------------------------
# Data download helper
# ---------------------------------------------------------------------------


def download_data(num_shards=10):
    """Download FineWeb data for Parameter Golf."""
    script = os.path.join(os.path.dirname(__file__), "data", "cached_challenge_fineweb.py")
    if not os.path.exists(script):
        print(f"ERROR: Data download script not found at {script}")
        print("Make sure you're running from the parameter-golf repo root.")
        sys.exit(1)

    cmd = f'python "{script}" --variant sp1024'
    if num_shards > 0:
        cmd += f" --train-shards {num_shards}"
    print(f"Running: {cmd}")
    os.system(cmd)


def verify_setup():
    """Verify all required files exist."""
    errors = []
    if not os.path.exists(TOKENIZER_PATH):
        errors.append(f"Tokenizer not found: {TOKENIZER_PATH}")
    dataset_dir = Path(DATA_PATH)
    if not dataset_dir.exists():
        errors.append(f"Dataset directory not found: {DATA_PATH}")
    else:
        train_files = list(dataset_dir.glob("fineweb_train_*.bin"))
        val_files = list(dataset_dir.glob("fineweb_val_*.bin"))
        if not train_files:
            errors.append("No training data files found")
        if not val_files:
            errors.append("No validation data files found")
        else:
            print(f"Found {len(train_files)} training shards, {len(val_files)} validation shards")
    return errors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare data for Parameter Golf autoresearch")
    parser.add_argument("--shards", type=int, default=10, help="Number of training shards (0=all)")
    parser.add_argument("--verify-only", action="store_true", help="Only verify existing setup")
    args = parser.parse_args()

    if args.verify_only:
        errors = verify_setup()
        if errors:
            print("Setup verification FAILED:")
            for e in errors:
                print(f"  - {e}")
            sys.exit(1)
        else:
            print("Setup verification PASSED!")
            sys.exit(0)

    download_data(args.shards)
    print()
    errors = verify_setup()
    if errors:
        print("Setup incomplete:")
        for e in errors:
            print(f"  - {e}")
    else:
        print("Done! Ready to train.")
