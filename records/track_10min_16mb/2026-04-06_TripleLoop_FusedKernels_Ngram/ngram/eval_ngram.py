"""Eval-only: run sliding window + n-gram tilt on an existing quantized model.
Usage: torchrun --standalone --nproc_per_node=8 eval_ngram.py --model final_model.int6.ptz
"""
import argparse, glob, io, math, os, time
from pathlib import Path
import numpy as np
import sentencepiece as spm
import torch
import torch.distributed as dist
import torch.nn.functional as F

def load_data_shard(file):
    header = np.fromfile(file, dtype="<i4", count=256)
    return torch.from_numpy(
        np.fromfile(file, dtype="<u2", count=int(header[2]),
                    offset=256 * np.dtype("<i4").itemsize).astype(np.uint16, copy=False))

def build_luts(sp, vocab_size, device):
    sp_vs = int(sp.vocab_size())
    sz = max(sp_vs, vocab_size)
    bb = np.zeros(sz, dtype=np.int16)
    ls = np.zeros(sz, dtype=np.bool_)
    bd = np.ones(sz, dtype=np.bool_)
    for tid in range(sp_vs):
        if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
            continue
        bd[tid] = False
        if sp.is_byte(tid):
            bb[tid] = 1
            continue
        piece = sp.id_to_piece(tid)
        if piece.startswith("\u2581"):
            ls[tid] = True
            piece = piece[1:]
        bb[tid] = len(piece.encode("utf-8"))
    return (torch.tensor(bb, dtype=torch.int16, device=device),
            torch.tensor(ls, dtype=torch.bool, device=device),
            torch.tensor(bd, dtype=torch.bool, device=device))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--code", default="train_gpt.py")
    parser.add_argument("--model", default="final_model.int6.ptz")
    parser.add_argument("--val-pattern", default="./data/datasets/fineweb10B_sp4096/fineweb_val_*.bin")
    parser.add_argument("--tokenizer", default="./data/tokenizers/fineweb_4096_bpe.model")
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--batch-seqs", type=int, default=32)
    parser.add_argument("--base-beta", type=float, default=1.0)
    parser.add_argument("--agree-bonus", type=float, default=0.5)
    parser.add_argument("--within-threshold", type=float, default=0.25)
    parser.add_argument("--within-beta", type=float, default=0.55)
    parser.add_argument("--word-threshold", type=float, default=0.80)
    parser.add_argument("--word-beta", type=float, default=0.50)
    # Model architecture args (must match training)
    parser.add_argument("--vocab-size", type=int, default=4096)
    parser.add_argument("--num-layers", type=int, default=11)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=float, default=4.0)
    parser.add_argument("--logit-softcap", type=float, default=30.0)
    parser.add_argument("--rope-base", type=float, default=10000.0)
    parser.add_argument("--qk-gain-init", type=float, default=5.0)
    parser.add_argument("--xsa-last-n", type=int, default=11)
    parser.add_argument("--rope-dims", type=int, default=16)
    parser.add_argument("--ve-enabled", type=int, default=1)
    parser.add_argument("--ve-dim", type=int, default=128)
    parser.add_argument("--ve-layers", default="9,10")
    parser.add_argument("--recur-layers", default="4,5")
    parser.add_argument("--parallel-start-layer", type=int, default=7)
    args = parser.parse_args()

    # Distributed init
    distributed = "RANK" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
        dist.barrier()
    master = rank == 0

    # Import training code as module
    import importlib.util
    os.environ.setdefault("MODEL_NAME", "eval")
    os.environ.setdefault("SEED", "42")
    import sys
    spec = importlib.util.spec_from_file_location("tg", args.code)
    tg = importlib.util.module_from_spec(spec)
    sys.modules["tg"] = tg
    spec.loader.exec_module(tg)

    # Load val tokens
    val_files = sorted(glob.glob(args.val_pattern))
    val_tokens = torch.cat([load_data_shard(Path(f)) for f in val_files]).contiguous()
    total_tokens = val_tokens.numel() - 1
    if master:
        print(f"Val tokens: {total_tokens:,}")

    # Build LUTs
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer)
    bb_lut, ls_lut, bd_lut = build_luts(sp, args.vocab_size, device)

    # Load model
    model = tg.GPT(tg.Hyperparameters()).to(device).bfloat16()
    tg.restore_fp32_params(model)
    with open(args.model, "rb") as f:
        blob = f.read()
    import brotli
    dec = brotli.decompress(blob)
    if hasattr(tg, "_byte_unshuffle"):
        dec = tg._byte_unshuffle(dec)
    qs = torch.load(io.BytesIO(dec), map_location="cpu")
    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    dq = tg.dequantize_mixed_int6(qs["w"], qs["m"], sd)
    model.load_state_dict(dq, strict=True)
    if hasattr(model, "set_recurrence_active"):
        model.set_recurrence_active(True)
    model.eval()
    if master:
        print("Model loaded.")

    # Compile
    logits_fn = torch.compile(model.forward_logits, dynamic=False, fullgraph=True)

    # Sliding window setup
    seq_len = args.seq_len
    stride = args.stride
    context_size = seq_len - stride
    window_starts = [ws for ws in range(0, total_tokens, stride)
                     if ws + context_size < total_tokens]
    total_windows = len(window_starts)
    my_s = (total_windows * rank) // world_size
    my_e = (total_windows * (rank + 1)) // world_size
    my_windows = window_starts[my_s:my_e]

    # Precompute n-gram hints
    all_hints = np.zeros(total_tokens + 1, dtype=np.int32)
    all_betas = np.zeros(total_tokens + 1, dtype=np.float64)
    if master:
        from fused_expert_ext import ContextMixer
        sp_vs = int(sp.vocab_size())
        sz = max(sp_vs, args.vocab_size)
        bb_np = np.zeros(sz, dtype=np.int16)
        ls_np = np.zeros(sz, dtype=np.uint8)
        bd_np = np.ones(sz, dtype=np.uint8)
        for tid in range(sp_vs):
            if sp.is_control(tid) or sp.is_unknown(tid) or sp.is_unused(tid):
                continue
            bd_np[tid] = 0
            if sp.is_byte(tid):
                bb_np[tid] = 1
                continue
            piece = sp.id_to_piece(tid)
            if piece.startswith("\u2581"):
                ls_np[tid] = 1
                piece = piece[1:]
            bb_np[tid] = len(piece.encode("utf-8"))
        val_np = val_tokens.numpy().astype(np.int64)
        ngram = ContextMixer(
            base_beta=args.base_beta, agree_bonus=args.agree_bonus,
            within_threshold=args.within_threshold, within_beta=args.within_beta,
            word_threshold=args.word_threshold, word_beta=args.word_beta,
            open_table_bits=26, token_threshold_scale=1.0, order_stride=2)
        ngram.set_tokens(val_np)
        ngram.set_luts(bb_np, ls_np, bd_np)
        positions = np.arange(1, total_tokens + 1, dtype=np.int64)
        ngram.get_hints_batch(positions, all_hints[1:], all_betas[1:])
        print(f"N-gram precomputed for {total_tokens} positions")
    if distributed:
        hints_t = torch.from_numpy(all_hints).to(device)
        betas_t = torch.from_numpy(all_betas).to(device)
        dist.broadcast(hints_t, src=0)
        dist.broadcast(betas_t, src=0)
    else:
        hints_t = torch.from_numpy(all_hints).to(device)
        betas_t = torch.from_numpy(all_betas).to(device)

    if master:
        print(f"Windows: {total_windows:,}, my_windows: {len(my_windows):,}")

    # Run eval: compute both base SW and n-gram tilted in one pass
    val_gpu = val_tokens.to(device=device, dtype=torch.int64)
    base_loss = torch.zeros((), device=device, dtype=torch.float64)
    tilt_loss = torch.zeros((), device=device, dtype=torch.float64)
    tc = torch.zeros((), device=device, dtype=torch.float64)
    bc = torch.zeros((), device=device, dtype=torch.float64)
    t0 = time.perf_counter()

    with torch.inference_mode():
        for bi in range(0, len(my_windows), args.batch_seqs):
            batch_ws = my_windows[bi:bi + args.batch_seqs]
            bsz = len(batch_ws)
            x = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            y = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)
            wlens = []
            for i, ws in enumerate(batch_ws):
                we = min(ws + seq_len, total_tokens)
                wlen = we - ws
                wlens.append(wlen)
                chunk = val_gpu[ws:we + 1]
                x[i, :wlen] = chunk[:-1]
                y[i, :wlen] = chunk[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = logits_fn(x)
            logits_f = logits.float()
            nll_all = F.cross_entropy(
                logits_f.reshape(-1, logits_f.size(-1)),
                y.reshape(-1), reduction="none",
            ).reshape(bsz, seq_len)
            for i, ws in enumerate(batch_ws):
                wlen = wlens[i]
                s = 0 if ws == 0 else context_size
                scored_nll = nll_all[i, s:wlen].to(torch.float64)
                base_loss += scored_nll.sum()
                # N-gram tilt
                gp = torch.arange(ws + s + 1, ws + wlen + 1, device=device, dtype=torch.int64)
                hint = hints_t[gp]
                beta = betas_t[gp]
                has_hint = (hint >= 0).to(torch.float64)
                scored_logits = logits_f[i, s:wlen]
                tgt = y[i, s:wlen]
                safe_h = hint.clamp(min=0)
                logit_tgt = scored_logits.gather(-1, tgt.unsqueeze(-1)).squeeze(-1).to(torch.float64)
                logit_hint = scored_logits.gather(-1, safe_h.unsqueeze(-1)).squeeze(-1).to(torch.float64)
                lse = scored_nll + logit_tgt
                p_hint = (logit_hint - lse).exp().clamp(0.0, 1.0)
                Z = 1.0 + p_hint * (beta.exp() - 1.0)
                is_hit = (tgt == hint).to(torch.float64)
                mixed_nll = scored_nll + has_hint * (Z.log() - beta * is_hit)
                tilt_loss += mixed_nll.sum()
                tc += float(wlen - s)
                prev = x[i, s:wlen]
                tb = bb_lut[tgt].to(torch.float64)
                tb += (ls_lut[tgt] & ~bd_lut[prev]).to(torch.float64)
                bc += tb.sum()

    if distributed:
        for t in (base_loss, tilt_loss, tc, bc):
            dist.all_reduce(t, op=dist.ReduceOp.SUM)

    elapsed = time.perf_counter() - t0
    tpb = tc.item() / bc.item()
    base_bpb = (base_loss.item() / tc.item() / math.log(2)) * tpb
    tilt_bpb = (tilt_loss.item() / tc.item() / math.log(2)) * tpb

    if master:
        print(f"\nbase_sw_bpb:  {base_bpb:.8f}")
        print(f"ngram_tilt_bpb: {tilt_bpb:.8f}")
        print(f"delta:        {tilt_bpb - base_bpb:+.8f}")
        print(f"eval_time:    {elapsed:.1f}s")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
