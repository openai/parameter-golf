"""Pack trained Trinity ternary model and compute exact val_bpb.

1. Load final_model.pt (24M params, 47% ternary blend after 24h CPU training)
2. Evaluate at alpha=0.47 (as-trained) and alpha=1.0 (full ternary)
3. Pack ternary weights via base-3 encoding (5 trits per byte)
4. LZMA compress → verify under 16MB
5. Compute exact val_bpb using SentencePiece byte LUT
"""
import os, sys, io, math, lzma, time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# Make train_gpt_v3.py importable
THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(THIS_DIR))
import importlib.util
spec = importlib.util.spec_from_file_location("train_gpt_v3", str(THIS_DIR / "train_gpt_v3.py"))
tg = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tg)


def build_sp_luts(tokenizer_path: str, vocab_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rebuild the SentencePiece byte LUTs used for BPB calculation."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor()
    sp.load(tokenizer_path)
    sp_vocab = int(sp.vocab_size())
    table_size = max(sp_vocab, vocab_size)
    base_bytes = np.zeros((table_size,), dtype=np.int16)
    has_leading = np.zeros((table_size,), dtype=np.bool_)
    is_boundary = np.ones((table_size,), dtype=np.bool_)
    for tok_id in range(sp_vocab):
        if sp.is_control(tok_id) or sp.is_unknown(tok_id) or sp.is_unused(tok_id):
            continue
        is_boundary[tok_id] = False
        if sp.is_byte(tok_id):
            base_bytes[tok_id] = 1
            continue
        piece = sp.id_to_piece(tok_id)
        if piece.startswith("▁"):
            has_leading[tok_id] = True
            piece = piece[1:]
        base_bytes[tok_id] = len(piece.encode("utf-8"))
    return base_bytes, has_leading, is_boundary


def compute_exact_bpb(model, val_tokens, base_bytes, has_leading, is_boundary, cfg, alpha: float, max_batches: int = None) -> dict:
    """Compute exact BPB: sum(log2(p(tgt))) / sum(bytes(tgt))."""
    model.eval()
    model.set_ternary_alpha(alpha)
    device = torch.device('cpu')

    batch_size = cfg.batch_size
    seq_len = cfg.seq_len
    total_tokens = val_tokens.numel()
    usable = (total_tokens - 1) // (batch_size * seq_len) * (batch_size * seq_len)

    loss_sum = 0.0
    byte_count = 0
    token_count = 0

    base_bytes_t = torch.from_numpy(base_bytes.astype(np.int64))
    has_lead_t = torch.from_numpy(has_leading.astype(np.bool_))
    is_bnd_t = torch.from_numpy(is_boundary.astype(np.bool_))

    with torch.no_grad():
        batches_done = 0
        for start in range(0, usable, batch_size * seq_len):
            x_flat = val_tokens[start:start + batch_size * seq_len]
            y_flat = val_tokens[start + 1:start + 1 + batch_size * seq_len]
            if y_flat.numel() < batch_size * seq_len:
                break
            x = x_flat.view(batch_size, seq_len).long()
            y = y_flat.view(batch_size, seq_len).long()

            logits, _ = model(x, None)
            nll = F.cross_entropy(
                logits.reshape(-1, cfg.vocab_size).float(),
                y.reshape(-1),
                reduction='none',
            )
            loss_sum += nll.sum().item()

            # Exact byte counting
            tgt = y.reshape(-1)
            prev = x.reshape(-1)
            bytes_per = base_bytes_t[tgt].to(torch.int64)
            bytes_per += (has_lead_t[tgt] & ~is_bnd_t[prev]).to(torch.int64)
            byte_count += bytes_per.sum().item()
            token_count += tgt.numel()

            batches_done += 1
            if batches_done % 10 == 0:
                print(f"  batch {batches_done}: avg_loss={loss_sum/token_count:.4f}", flush=True)
            if max_batches and batches_done >= max_batches:
                break

    avg_loss = loss_sum / token_count
    bits_per_token = avg_loss / math.log(2.0)
    tokens_per_byte = token_count / byte_count
    val_bpb = bits_per_token * tokens_per_byte

    return {
        "alpha": alpha,
        "avg_loss": avg_loss,
        "bits_per_token": bits_per_token,
        "tokens": token_count,
        "bytes": byte_count,
        "tokens_per_byte": tokens_per_byte,
        "val_bpb": val_bpb,
    }


def pack_ternary_base3(trits: torch.Tensor) -> bytes:
    """Pack ternary values {-1, 0, +1} as 5 trits per byte.
    3^5 = 243 < 256, so each byte encodes 5 trits losslessly.
    Value in byte = (t0+1) + 3*(t1+1) + 9*(t2+1) + 27*(t3+1) + 81*(t4+1), range 0..242.
    """
    assert trits.dtype == torch.int8 or trits.dtype == torch.int16 or trits.dtype == torch.int64
    flat = trits.reshape(-1).to(torch.int64)
    # Pad to multiple of 5
    pad = (-len(flat)) % 5
    if pad > 0:
        flat = torch.cat([flat, torch.zeros(pad, dtype=torch.int64)])
    # Group by 5
    groups = flat.view(-1, 5)
    # Shift to {0, 1, 2} and encode base-3
    g = groups + 1
    packed = g[:, 0] + 3 * g[:, 1] + 9 * g[:, 2] + 27 * g[:, 3] + 81 * g[:, 4]
    return packed.to(torch.uint8).numpy().tobytes()


def unpack_ternary_base3(data: bytes, num_trits: int) -> torch.Tensor:
    """Reverse of pack_ternary_base3."""
    packed = np.frombuffer(data, dtype=np.uint8).astype(np.int64)
    trits = []
    for p in packed:
        for i in range(5):
            trits.append((p % 3) - 1)
            p //= 3
    return torch.tensor(trits[:num_trits], dtype=torch.int8)


def build_ternary_artifact(model, alpha: float = 1.0) -> tuple[bytes, dict]:
    """Ternarize all TernaryLinear weights, pack via base-3, LZMA-compress.
    Non-ternary params (embeddings, norms, gains) stored as fp16.
    """
    model.eval()
    state = {}
    meta = {}
    total_trits = 0
    total_fp16_bytes = 0
    raw_parts = {}

    for name, param in model.named_parameters():
        p = param.detach().cpu()
        # Ternarize TernaryLinear weights
        is_ternary_weight = any(name.endswith(f".{module_attr}.weight")
                                 for module_attr in ["qkv", "proj", "fc"])
        if is_ternary_weight and p.ndim == 2:
            # Ternarize with mean-abs scale (BitNet b1.58)
            abs_mean = p.abs().mean().clamp(min=1e-5).item()
            threshold = 0.7 * abs_mean
            q = torch.where(p > threshold, torch.ones_like(p, dtype=torch.int8),
                torch.where(p < -threshold, -torch.ones_like(p, dtype=torch.int8),
                            torch.zeros_like(p, dtype=torch.int8)))
            packed = pack_ternary_base3(q)
            raw_parts[name] = {'type': 'ternary_base3', 'shape': list(p.shape),
                               'scale': float(abs_mean), 'data': packed}
            total_trits += p.numel()
            meta[name] = f"ternary ({p.numel()} trits -> {len(packed)} B)"
        else:
            # fp16 passthrough for embeddings, norms, small tensors
            p16 = p.to(torch.float16)
            buf = io.BytesIO()
            np.save(buf, p16.numpy(), allow_pickle=False)
            raw_parts[name] = {'type': 'fp16', 'shape': list(p.shape), 'data': buf.getvalue()}
            total_fp16_bytes += p.numel() * 2
            meta[name] = f"fp16 ({p.numel() * 2} B)"

    # Serialize (use manual binary layout instead of torch.save for compactness)
    import pickle
    buf = io.BytesIO()
    pickle.dump(raw_parts, buf)
    raw_bytes = buf.getvalue()

    # LZMA compress
    compressed = lzma.compress(raw_bytes, preset=9)

    summary = {
        'total_trits': total_trits,
        'ternary_raw_bytes': sum(len(v['data']) for v in raw_parts.values() if v['type'] == 'ternary_base3'),
        'fp16_raw_bytes': total_fp16_bytes,
        'pickled_raw_bytes': len(raw_bytes),
        'lzma_compressed_bytes': len(compressed),
        'per_param_meta': meta,
    }
    return compressed, summary


def main():
    print("=" * 60, flush=True)
    print("Trinity Ternary CPU — Pack & Eval", flush=True)
    print("=" * 60, flush=True)

    cfg = tg.Config()
    torch.set_num_threads(10)

    model = tg.TrinityTernaryGPT(cfg)
    ckpt_path = str(THIS_DIR / "final_model_v3.pt")
    state = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    print(f"Loaded v3: {ckpt_path} ({sum(p.numel() for p in model.parameters()):,} params)", flush=True)

    # Build SentencePiece byte LUTs
    tok_path = "/Users/ssdm4/Desktop/PROJECTS/CLAUDE/parameter-golf/data/tokenizers/fineweb_1024_bpe.model"
    base_bytes, has_leading, is_boundary = build_sp_luts(tok_path, cfg.vocab_size)
    print(f"SP LUTs built: mean bytes/token = {base_bytes.mean():.2f}", flush=True)

    # Load val tokens
    val_path = "/Users/ssdm4/Desktop/PROJECTS/CLAUDE/parameter-golf/data/datasets/fineweb10B_sp1024/fineweb_val_000000.bin"
    val_np = tg.load_data_shard(val_path)
    val_tokens = torch.from_numpy(val_np.astype(np.int64).copy())
    print(f"Val tokens: {len(val_tokens):,}", flush=True)

    # v3 trained to full ternary — eval primarily at α=1.0
    print("\n--- Eval at alpha=1.0 (FULL TERNARY, as trained for 72h) ---", flush=True)
    result_ternary = compute_exact_bpb(model, val_tokens, base_bytes, has_leading, is_boundary, cfg, alpha=1.0, max_batches=50)
    print(f"  val_loss: {result_ternary['avg_loss']:.4f}", flush=True)
    print(f"  val_bpb:  {result_ternary['val_bpb']:.4f}", flush=True)
    print(f"  tokens/byte: {result_ternary['tokens_per_byte']:.4f}", flush=True)

    # For comparison: alpha=0 (no ternary, fp32 weights) — what's the underlying capacity?
    print("\n--- Eval at alpha=0.0 (fp32 baseline, no ternary) — 20 batches ---", flush=True)
    result_fp32 = compute_exact_bpb(model, val_tokens, base_bytes, has_leading, is_boundary, cfg, alpha=0.0, max_batches=20)
    print(f"  val_loss: {result_fp32['avg_loss']:.4f}", flush=True)
    print(f"  val_bpb:  {result_fp32['val_bpb']:.4f}", flush=True)
    # Set back to alpha=1.0 for packing
    model.set_ternary_alpha(1.0)
    result_trained = result_ternary  # for back-compat in dict below

    # Pack ternary artifact
    print("\n--- Packing ternary artifact ---", flush=True)
    compressed, summary = build_ternary_artifact(model, alpha=1.0)
    print(f"  total_trits: {summary['total_trits']:,}", flush=True)
    print(f"  ternary_raw_bytes: {summary['ternary_raw_bytes']:,}", flush=True)
    print(f"  fp16_raw_bytes: {summary['fp16_raw_bytes']:,}", flush=True)
    print(f"  pickled_raw: {summary['pickled_raw_bytes']:,}", flush=True)
    print(f"  lzma_compressed: {summary['lzma_compressed_bytes']:,}", flush=True)
    print(f"  Under 16MB? {summary['lzma_compressed_bytes'] < 16_000_000}", flush=True)

    # Save artifact (v3)
    out_path = THIS_DIR / "final_model_v3.trinity.ptz"
    with open(out_path, "wb") as f:
        f.write(compressed)
    print(f"  Saved: {out_path}", flush=True)

    # Save eval summary
    import json
    summary_out = {
        'val_bpb_alpha_1_0_trained': result_ternary['val_bpb'],
        'val_bpb_alpha_0_0_fp32_baseline': result_fp32['val_bpb'],
        'val_loss_ternary': result_ternary['avg_loss'],
        'val_loss_fp32': result_fp32['avg_loss'],
        'tokens_per_byte': result_ternary['tokens_per_byte'],
        'artifact_bytes': summary['lzma_compressed_bytes'],
        'total_params': sum(p.numel() for p in model.parameters()),
        'ternary_params': summary['total_trits'],
        'training_hours': 72.04,
        'final_alpha': 1.0,
    }
    with open(THIS_DIR / "eval_results_v3.json", "w") as f:
        json.dump(summary_out, f, indent=2)
    print(f"\nFinal: {summary_out}", flush=True)


if __name__ == "__main__":
    main()
