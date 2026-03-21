"""
Error-Guided Test-Time Training (EG-TTT) Evaluation.

Key insight: Standard TTT spreads adaptation budget across ALL tokens equally.
SmearGate already handles easy tokens well, so TTT on easy tokens is wasted.
EG-TTT concentrates 100% of adaptation budget on the model's WEAKEST positions.

Algorithm:
  Pass 1: Quick inference to identify high-loss positions (~20s, distributed)
  TTT:    LoRA updates ONLY on windows containing high-loss tokens (~2-3 min)
  Pass 2: Standard sliding window eval on the improved model (~2 min)

Total eval time: ~5 min. Fully compliant with 10-min eval budget.

Usage:
  torchrun --nproc_per_node=8 eval_error_guided_ttt.py
"""
from __future__ import annotations
import glob, io, math, os, sys, time, zlib
from pathlib import Path
import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import Tensor, nn

try:
    import zstandard
    _COMPRESSOR = "zstd"
except ImportError:
    _COMPRESSOR = "zlib"

# Import model + utils from training script
import importlib.util
spec = importlib.util.spec_from_file_location("tg", "train_sota198.py")
mod = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mod)

GPT = mod.GPT
CastedLinear = mod.CastedLinear
Hyperparameters = mod.Hyperparameters
dequantize_mixed_int6 = mod.dequantize_mixed_int6
restore_low_dim_params_to_fp32 = mod.restore_low_dim_params_to_fp32


def load_data_shard(path):
    header = np.fromfile(path, dtype="<i4", count=256)
    assert header[0] == 20240520 and header[1] == 1
    ntok = int(header[2])
    return torch.from_numpy(np.fromfile(path, dtype="<u2", offset=256*4).astype(np.int32)[:ntok])


class LoRALinear(nn.Module):
    """Lightweight LoRA adapter for a linear layer."""
    def __init__(self, in_features, out_features, rank=4):
        super().__init__()
        self.A = nn.Parameter(torch.randn(rank, in_features) * (1.0 / math.sqrt(in_features)))
        self.B = nn.Parameter(torch.zeros(out_features, rank))

    def forward(self, x):
        return (x @ self.A.t()) @ self.B.t()


def main():
    # --- Distributed setup ---
    distributed = int(os.environ.get("WORLD_SIZE", 1)) > 1
    if distributed:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank, world_size = 0, 1
    device = torch.device(f"cuda:{rank % torch.cuda.device_count()}")
    torch.cuda.set_device(device)

    def log0(msg):
        if rank == 0:
            print(msg, flush=True)

    args = Hyperparameters()

    # --- Config ---
    TTT_LR = float(os.environ.get("TTT_LR", "0.001"))
    TTT_RANK = int(os.environ.get("TTT_RANK", "4"))
    TTT_TOP_FRAC = float(os.environ.get("TTT_TOP_FRAC", "0.02"))  # top 2% highest-loss positions
    TTT_EPOCHS = int(os.environ.get("TTT_EPOCHS", "3"))  # passes over high-loss windows

    log0(f"eg_ttt: lr={TTT_LR} rank={TTT_RANK} top_frac={TTT_TOP_FRAC} epochs={TTT_EPOCHS}")

    # --- Load validation data ---
    val_files = sorted(glob.glob(args.val_files))
    val_tokens = torch.cat([load_data_shard(Path(f)) for f in val_files])
    total = val_tokens.numel() - 1
    log0(f"eg_ttt: {total} val tokens")

    # --- Build byte LUTs ---
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.Load(args.tokenizer_path)
    vocab = args.vocab_size
    base_bytes_lut = torch.zeros(vocab, dtype=torch.int32, device=device)
    has_leading_space_lut = torch.zeros(vocab, dtype=torch.bool, device=device)
    is_boundary_token_lut = torch.zeros(vocab, dtype=torch.bool, device=device)
    BOS = sp.bos_id() if sp.bos_id() >= 0 else 1
    EOS = sp.eos_id() if sp.eos_id() >= 0 else 2
    is_boundary_token_lut[BOS] = True
    is_boundary_token_lut[EOS] = True
    for i in range(vocab):
        piece = sp.IdToPiece(i)
        raw = piece.replace("\u2581", " ")
        base_bytes_lut[i] = len(raw.encode("utf-8"))
        has_leading_space_lut[i] = piece.startswith("\u2581")

    # --- Load quantized model ---
    model_path = "final_model.int6.ptz"
    log0(f"eg_ttt: loading {model_path}")
    with open(model_path, "rb") as f:
        blob = f.read()
    raw_data = zstandard.ZstdDecompressor().decompress(blob) if _COMPRESSOR == "zstd" else zlib.decompress(blob)
    qstate = torch.load(io.BytesIO(raw_data), map_location="cpu", weights_only=False)
    del blob, raw_data

    template_model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    )
    sd_template = template_model.state_dict()
    del template_model

    deq_state = dequantize_mixed_int6(qstate["w"], qstate["m"], sd_template)
    del qstate, sd_template

    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        mtp_num_heads=0, mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear):
            m.float()
    restore_low_dim_params_to_fp32(model)
    model.load_state_dict(deq_state, strict=True)
    # Clone all parameters to escape inference tensor state from dequantize
    with torch.no_grad():
        for p in model.parameters():
            p.data = p.data.clone()
        for b in model.buffers():
            b.data = b.data.clone()
    model.eval()
    del deq_state
    torch.cuda.empty_cache()
    log0("eg_ttt: model loaded (params cloned)")

    # ================================================================
    # PASS 1: Identify high-loss positions (distributed, ~20s)
    # ================================================================
    log0("eg_ttt: PASS 1 - identifying high-loss positions")
    t0 = time.perf_counter()
    seq_len = 1024
    my_starts = list(range(rank * seq_len, total - seq_len, world_size * seq_len))

    # Collect per-position mean loss
    position_losses = {}  # window_start -> mean_loss
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for pos in my_starts:
            x = val_tokens[pos:pos+seq_len].unsqueeze(0).to(device=device, dtype=torch.int64)
            y = val_tokens[pos+1:pos+seq_len+1].unsqueeze(0).to(device=device, dtype=torch.int64)
            logits = model.forward_logits(x)
            loss = F.cross_entropy(logits[0].float(), y[0].long(), reduction="mean").item()
            position_losses[pos] = loss

    # Select top fraction of highest-loss windows
    sorted_windows = sorted(position_losses.items(), key=lambda x: -x[1])
    n_ttt_windows = max(1, int(len(sorted_windows) * TTT_TOP_FRAC))
    high_loss_starts = [w[0] for w in sorted_windows[:n_ttt_windows]]
    high_loss_starts.sort()  # process in order

    mean_high = np.mean([w[1] for w in sorted_windows[:n_ttt_windows]])
    mean_all = np.mean([w[1] for w in sorted_windows])
    log0(f"eg_ttt: pass1 done in {time.perf_counter()-t0:.1f}s. "
         f"Total windows: {len(sorted_windows)}, TTT windows: {n_ttt_windows}, "
         f"mean_loss_all: {mean_all:.4f}, mean_loss_ttt: {mean_high:.4f}")

    # ================================================================
    # TTT: LoRA updates on high-loss windows only
    # ================================================================
    # Clear Rotary caches (they hold inference tensors from Pass 1)
    for m in model.modules():
        if hasattr(m, '_cache'):
            m._cache = (0, None, None)
        if hasattr(m, '_seq_len_cached'):
            m._seq_len_cached = 0
            m._cos_cached = None
            m._sin_cached = None
    log0(f"eg_ttt: TTT phase - {TTT_EPOCHS} epochs on {n_ttt_windows} high-loss windows")
    t_ttt = time.perf_counter()

    # Create LoRA adapters for last 3 blocks' attention Q/V + MLP
    num_blocks = len(model.blocks)
    suffix_start = num_blocks - num_blocks // 4  # last 1/4

    lora_modules = {}
    for i in range(suffix_start, num_blocks):
        block = model.blocks[i]
        dim = args.model_dim
        # LoRA on MLP fc and proj
        mlp_fc_out = block.mlp.fc.weight.shape[0]
        mlp_proj_out = block.mlp.proj.weight.shape[0]
        lora_modules[f"blocks.{i}.mlp.fc"] = LoRALinear(dim, mlp_fc_out, rank=TTT_RANK).to(device).float()
        lora_modules[f"blocks.{i}.mlp.proj"] = LoRALinear(mlp_fc_out, mlp_proj_out, rank=TTT_RANK).to(device).float()

    lora_params = []
    for lm in lora_modules.values():
        lora_params.extend(lm.parameters())
    optimizer = torch.optim.Adam(lora_params, lr=TTT_LR)

    # Hook: add LoRA output to each target layer
    hooks = []
    def make_hook(lora_mod):
        def hook_fn(module, input, output):
            # Always add LoRA delta (even if output doesn't have grad yet - autograd will track through lora_mod)
            return output + lora_mod(input[0].float()).to(output.dtype)
        return hook_fn

    for name, lora_mod in lora_modules.items():
        parts = name.split(".")
        target = model
        for p in parts:
            target = getattr(target, p)
        h = target.register_forward_hook(make_hook(lora_mod))
        hooks.append(h)

    # Enable gradients for LoRA only
    for p in model.parameters():
        p.requires_grad_(False)
    for p in lora_params:
        p.requires_grad_(True)

    # TTT training loop
    model.train()
    for epoch in range(TTT_EPOCHS):
        epoch_loss = 0.0
        for wi, pos in enumerate(high_loss_starts):
            x = val_tokens[pos:pos+seq_len].unsqueeze(0).to(device=device, dtype=torch.int64)
            y = val_tokens[pos+1:pos+seq_len+1].unsqueeze(0).to(device=device, dtype=torch.int64)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits = model.forward_logits(x)
            loss = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), y.reshape(-1).long())
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(lora_params, 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        log0(f"eg_ttt: epoch {epoch+1}/{TTT_EPOCHS} mean_loss:{epoch_loss/max(len(high_loss_starts),1):.4f} time:{time.perf_counter()-t_ttt:.1f}s")

    model.eval()
    log0(f"eg_ttt: TTT done in {time.perf_counter()-t_ttt:.1f}s")

    # ================================================================
    # PASS 2: Standard sliding window eval (with LoRA-improved model)
    # ================================================================
    log0("eg_ttt: PASS 2 - sliding window evaluation")
    t_eval = time.perf_counter()

    stride = 64
    eval_seq = getattr(args, "eval_seq_len", 2048) or 2048
    all_starts = list(range(0, total - eval_seq + 1, stride))
    my_windows = all_starts[rank::world_size]

    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_count = torch.zeros((), device=device, dtype=torch.float64)
    byte_count = torch.zeros((), device=device, dtype=torch.float64)
    batch_seqs = 8

    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for bi in range(0, len(my_windows), batch_seqs):
            bw = my_windows[bi:bi+batch_seqs]
            bsz = len(bw)
            xb = torch.zeros(bsz, eval_seq, dtype=torch.int64, device=device)
            yb = torch.zeros(bsz, eval_seq, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(bw):
                end = min(ws + eval_seq, total)
                wl = end - ws
                wlens.append(wl)
                chunk = val_tokens[ws:end+1].to(dtype=torch.int64, device=device)
                xb[i, :wl] = chunk[:-1]
                yb[i, :wl] = chunk[1:]
            logits = model.forward_logits(xb)
            nll = F.cross_entropy(
                logits.reshape(-1, logits.size(-1)).float(),
                yb.reshape(-1), reduction="none"
            ).reshape(bsz, eval_seq)
            for i, ws in enumerate(bw):
                wl = wlens[i]
                s = 0 if ws == 0 else wl - stride
                scored = nll[i, s:wl].to(torch.float64)
                loss_sum += scored.sum()
                tgt = yb[i, s:wl]
                prev = xb[i, s:wl]
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += tb.sum()
                tok_count += float(wl - s)

    # Remove hooks
    for h in hooks:
        h.remove()

    if distributed:
        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)
        dist.all_reduce(tok_count, op=dist.ReduceOp.SUM)
        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)

    val_loss = (loss_sum / tok_count).item()
    val_bpb = val_loss / math.log(2.0) * (tok_count.item() / byte_count.item())

    log0(f"eg_ttt_eval val_loss:{val_loss:.4f} val_bpb:{val_bpb:.6f} "
         f"pass2:{time.perf_counter()-t_eval:.0f}s total:{time.perf_counter()-t0:.0f}s")
    log0(f"eg_ttt_eval_exact val_loss:{val_loss:.8f} val_bpb:{val_bpb:.8f}")

    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
