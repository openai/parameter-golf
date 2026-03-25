"""Sliding window eval — standalone or imported by train_gpt.py."""
from __future__ import annotations
import glob, io, math, os, sys, time, zlib
from pathlib import Path
import numpy as np, sentencepiece as spm
import torch, torch.distributed as dist, torch.nn.functional as F
from torch import Tensor, nn
try: import zstandard as zstd; USE_ZSTD = True
except ImportError: USE_ZSTD = False
sys.path.insert(0, os.path.dirname(__file__))
from train_gpt import (
    Hyperparameters, GPT, build_sentencepiece_luts, load_data_shard,
    restore_low_dim_params_to_fp32, CONTROL_TENSOR_NAME_PATTERNS, CastedLinear,
)

def forward_logits(model, input_ids: Tensor) -> Tensor:
    """Forward pass returning logits (no loss)."""
    x = model.tok_emb(input_ids)
    if model.bigram is not None:
        x = x + model.bigram(input_ids)
    x = F.rms_norm(x, (x.size(-1),))
    x = model.smear(x)
    x0 = x
    skips = []
    for i in range(model.num_encoder_layers):
        sort_idx = model._get_layer_sort_idx(i, input_ids)
        x = model.blocks[i](x, x0, sort_idx)
        skips.append(x)
    for i in range(model.num_decoder_layers):
        bi = model.num_encoder_layers + i
        sort_idx = model._get_layer_sort_idx(bi, input_ids)
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = model.blocks[bi](x, x0, sort_idx)
    x = model.final_norm(x)
    lp = F.linear(x, model.tok_emb.weight) if model.tie_embeddings else model.lm_head(x)
    return model.logit_softcap * torch.tanh(lp / model.logit_softcap)

def eval_val_sliding(args, base_model, rank, world_size, device, val_tokens,
                     base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
                     stride=64, batch_seqs=32):
    sl = args.train_seq_len; tot = val_tokens.numel() - 1
    ws_list = [w for w in range(0, tot, stride) if min(w + sl, tot) - w >= stride or w == 0]
    nw = len(ws_list); my_ws = ws_list[(nw * rank) // world_size:(nw * (rank + 1)) // world_size]
    loss_sum = torch.zeros((), device=device, dtype=torch.float64)
    tok_cnt = torch.zeros((), device=device, dtype=torch.float64)
    byte_cnt = torch.zeros((), device=device, dtype=torch.float64)
    base_model.eval()
    with torch.inference_mode():
        for bi in range(0, len(my_ws), batch_seqs):
            bws = my_ws[bi:bi + batch_seqs]; bsz = len(bws)
            xb = torch.zeros(bsz, sl, dtype=torch.int64, device=device)
            yb = torch.zeros(bsz, sl, dtype=torch.int64, device=device)
            wlens = []
            for i, w in enumerate(bws):
                end = min(w + sl, tot); wl = end - w; wlens.append(wl)
                ch = val_tokens[w:end + 1].to(dtype=torch.int64, device=device)
                xb[i, :wl] = ch[:-1]; yb[i, :wl] = ch[1:]
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = forward_logits(base_model, xb)
            nll = F.cross_entropy(logits.reshape(-1, logits.size(-1)).float(), yb.reshape(-1), reduction="none").reshape(bsz, sl)
            for i, w in enumerate(bws):
                wl = wlens[i]; s = 0 if w == 0 else max(wl - stride, 0)
                loss_sum += nll[i, s:wl].to(torch.float64).sum(); tok_cnt += float(wl - s)
                tgt, prev = yb[i, s:wl], xb[i, s:wl]
                byte_cnt += (base_bytes_lut[tgt].to(torch.float64) + (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)).sum()
    if dist.is_available() and dist.is_initialized():
        for t in (loss_sum, tok_cnt, byte_cnt): dist.all_reduce(t, op=dist.ReduceOp.SUM)
    vl = (loss_sum / tok_cnt).item(); base_model.train()
    return vl, vl / math.log(2.0) * (tok_cnt.item() / byte_cnt.item())

def main():
    args = Hyperparameters()
    stride = int(os.environ.get("EVAL_STRIDE", "64"))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sliding Window Eval: window={args.train_seq_len}, stride={stride}")
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    files = [Path(p) for p in sorted(glob.glob(args.val_files))]
    val_tokens = torch.cat([load_data_shard(f) for f in files]).contiguous()
    print(f"Val tokens: {val_tokens.numel():,}")
    model = GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, mlp_mult=args.mlp_mult, tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std, logit_softcap=args.logit_softcap,
        num_experts=args.num_experts, moe_activation=args.moe_activation,
        classical_layers=args.classical_layers, num_kv_heads=args.num_kv_heads,
        rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        layer_partitions=args.layer_partitions, hash_prime1=args.hash_prime1,
        hash_prime2=args.hash_prime2, bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
    ).to(device).bfloat16()
    for m in model.modules():
        if isinstance(m, CastedLinear): m.float()
    restore_low_dim_params_to_fp32(model)
    model_path = os.environ.get("MODEL_PATH", "final_model.int8.ptz")
    print(f"Loading: {model_path}")
    with open(model_path, "rb") as f: blob = f.read()
    raw = zstd.ZstdDecompressor().decompress(blob) if USE_ZSTD else zlib.decompress(blob)
    qs = torch.load(io.BytesIO(raw), map_location="cpu")
    # Mixed int5/int6 dequantize
    sd_cpu = {k: v for k, v in model.state_dict().items()}
    result, meta = qs["w"], qs["m"]
    out = {}
    for name, orig in sd_cpu.items():
        info = meta[name]
        if info in ("passthrough", "passthrough_ctrl", "passthrough_fp16"):
            t = result[name]
            if t.dtype == torch.float16 and orig.dtype in (torch.float32, torch.bfloat16): t = t.to(orig.dtype)
            out[name] = t; continue
        q, s = result[name + ".q"], result[name + ".scale"]
        out[name] = (q.float() * s.float().unsqueeze(-1)).to(orig.dtype) if s.ndim > 0 else (q.float() * float(s.item())).to(orig.dtype)
    model.load_state_dict(out, strict=True)
    t0 = time.perf_counter()
    val_loss, val_bpb = eval_val_sliding(
        args, model, 0, 1, device, val_tokens,
        base_bytes_lut, has_leading_space_lut, is_boundary_token_lut,
        stride=stride, batch_seqs=args.eval_batch_seqs,
    )
    print(f"val_loss: {val_loss:.4f}  val_bpb: {val_bpb:.4f}  time: {time.perf_counter()-t0:.1f}s")

if __name__ == "__main__":
    main()
