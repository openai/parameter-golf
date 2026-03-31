#!/usr/bin/env python3
"""Static + CPU checks for a Parameter Golf train_gpt.py (no CUDA required).

Default target: SOTA+ submission (mixed int6, BigramHash, SmearGate, eval_val_sliding).

Also runs sliding-window *coverage* sanity for the non-overlapping scheme (some older
submissions); if your script only uses strided window_starts, that block still passes
as a pure math check, not a proof your eval matches this reference implementation.
"""
from __future__ import annotations

import argparse
import ast
import importlib.util
import io
import os
import sys
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCRIPT = (
    REPO_ROOT
    / "records/track_10min_16mb/2026-03-20_SOTA_TTT_RoPE50K_EMA_Curriculum/train_gpt.py"
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "script",
        type=Path,
        nargs="?",
        default=DEFAULT_SCRIPT,
        help="Path to train_gpt.py",
    )
    return p.parse_args()


def sliding_window_coverage(seq_len: int, stride: int, total_tokens: int) -> tuple[bool, str]:
    windows: list[tuple[int, int, int]] = []
    scored_up_to = 0
    pos = 0
    while scored_up_to < total_tokens:
        end = min(pos + seq_len, total_tokens)
        score_start = scored_up_to
        if score_start >= end:
            break
        windows.append((pos, score_start, end))
        scored_up_to = end
        if end == total_tokens:
            break
        pos += stride

    counts: Counter[int] = Counter()
    for _w, score_start, win_end in windows:
        for t in range(score_start, win_end):
            counts[t] += 1

    doubles = [t for t, c in counts.items() if c > 1]
    missing = set(range(total_tokens)) - set(counts.keys())
    total_scored = sum(counts.values())
    if doubles or missing or total_scored != total_tokens:
        return False, f"doubles={len(doubles)} missing={len(missing)} scored={total_scored} want={total_tokens}"
    return True, f"windows={len(windows)} tokens={total_tokens}"


def load_submission_module(path: Path):
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("NUM_LAYERS", "4")
    os.environ.setdefault("MODEL_DIM", "256")
    os.environ.setdefault("NUM_HEADS", "8")
    os.environ.setdefault("NUM_KV_HEADS", "4")
    os.environ.setdefault("MLP_MULT", "2")
    os.environ.setdefault("BIGRAM_VOCAB_SIZE", "512")
    os.environ.setdefault("BIGRAM_DIM", "64")
    os.environ.setdefault("VOCAB_SIZE", "1024")

    spec = importlib.util.spec_from_file_location("_pg_validate_submission", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not create module spec")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    args = parse_args()
    path = args.script.resolve()
    if not path.is_file():
        print(f"ERROR: not a file: {path}", file=sys.stderr)
        return 2

    text = path.read_text(encoding="utf-8")
    try:
        ast.parse(text)
    except SyntaxError as e:
        print(f"ERROR: syntax: {e}", file=sys.stderr)
        return 1

    lines = text.count("\n") + 1
    nbytes = len(text.encode("utf-8"))
    print(f"OK parse: {lines} lines, {nbytes} UTF-8 bytes")
    if lines > 1500:
        print(f"WARN: over 1500-line soft cap ({lines})", file=sys.stderr)

    for seq_len, stride, total in [(16, 4, 50), (1024, 64, 62021632), (1024, 64, 1000)]:
        ok, msg = sliding_window_coverage(seq_len, stride, total)
        tag = "OK" if ok else "FAIL"
        print(f"  nonoverlap_sliding_ref {tag}: seq={seq_len} stride={stride} total={total} -> {msg}")
        if not ok:
            return 1

    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        print("SKIP torch tests (pip install torch)")
        return 0

    try:
        mod = load_submission_module(path)
    except Exception as e:
        print(f"ERROR: import failed: {e}", file=sys.stderr)
        return 1

    if not hasattr(mod, "GPT") or not hasattr(mod, "mixed_quantize_int6"):
        print("ERROR: expected GPT + mixed_quantize_int6 (PR #198-style submission)", file=sys.stderr)
        return 1

    hp = mod.Hyperparameters()
    model = mod.GPT(
        vocab_size=hp.vocab_size,
        num_layers=hp.num_layers,
        model_dim=hp.model_dim,
        num_heads=hp.num_heads,
        num_kv_heads=hp.num_kv_heads,
        mlp_mult=int(hp.mlp_mult),
        tie_embeddings=hp.tie_embeddings,
        tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap,
        rope_base=hp.rope_base,
        qk_gain_init=hp.qk_gain_init,
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=hp.bigram_vocab_size,
        bigram_dim=hp.bigram_dim,
    )
    model = model.float()
    for m in model.modules():
        if isinstance(m, mod.CastedLinear):
            m.float()
    mod.restore_low_dim_params_to_fp32(model)

    v = hp.vocab_size
    x = torch.randint(0, v, (2, 32))
    y = torch.randint(0, v, (2, 32))
    with torch.no_grad():
        loss_m = model(x, y)
        logits = model.forward_logits(x)
        loss_manual = F.cross_entropy(
            logits.reshape(-1, v).float(), y.reshape(-1), reduction="mean"
        )
    if abs(float(loss_m.item() - loss_manual.item())) > 1e-4:
        print(
            f"ERROR: forward vs forward_logits loss mismatch {loss_m} vs {loss_manual}",
            file=sys.stderr,
        )
        return 1
    print("OK forward_logits matches forward loss")

    sd = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    qres, qmeta = mod.mixed_quantize_int6(sd, {"mlp", "attn"})
    deq = mod.dequantize_mixed_int6(qres, qmeta, sd)
    model.load_state_dict(deq, strict=True)
    with torch.no_grad():
        loss2 = model(x, y)
    print(f"OK mixed int6 roundtrip loss gap {abs(loss2.item() - loss_m.item()):.6f}")

    buf = io.BytesIO()
    torch.save({"w": qres, "m": qmeta}, buf)
    print(f"OK export payload raw_torch_save={len(buf.getvalue())} bytes")

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
