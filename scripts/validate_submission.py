#!/usr/bin/env python3
"""Static + CPU checks for a Parameter Golf train_gpt.py (no CUDA required)."""
from __future__ import annotations

import argparse
import ast
import io
import sys
from collections import Counter
from pathlib import Path


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "script",
        type=Path,
        nargs="?",
        default=Path("records/track_10min_16mb/2026-03-20_AllInOne_SmearGate_Int6QAT_SlidingWindow/train_gpt.py"),
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

    # Sliding-window logic must match submission (copy of algorithm)
    for seq_len, stride, total in [(16, 4, 50), (1024, 64, 62021632), (1024, 64, 1000)]:
        ok, msg = sliding_window_coverage(seq_len, stride, total)
        tag = "OK" if ok else "FAIL"
        print(f"  sliding_window {tag}: seq={seq_len} stride={stride} total={total} -> {msg}")
        if not ok:
            return 1

    try:
        import torch  # noqa: F401
    except ImportError:
        print("SKIP torch tests (pip install torch)")
        return 0

    import os

    os.environ.setdefault("NUM_LAYERS", "4")
    os.environ.setdefault("MODEL_DIM", "128")
    os.environ.setdefault("NUM_HEADS", "4")
    os.environ.setdefault("NUM_KV_HEADS", "2")
    os.environ.setdefault("MLP_MULT", "3")
    os.environ.setdefault("USE_SMEARGATE", "1")
    os.environ.setdefault("BIGRAM_HASH_BUCKETS", "256")
    os.environ.setdefault("BIGRAM_HASH_DIM", "32")
    os.environ.setdefault("USE_INT6_QAT", "1")

    main_idx = text.index("\ndef main()")
    ns: dict = {}
    exec(compile(text[:main_idx], str(path), "exec"), ns, ns)

    GPT = ns["GPT"]
    fake_quantize_int6 = ns["fake_quantize_int6"]
    quantize_state_dict_int6 = ns["quantize_state_dict_int6"]
    dequantize_state_dict_int6 = ns["dequantize_state_dict_int6"]

    model = GPT(
        vocab_size=64,
        num_layers=4,
        model_dim=128,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=3,
        tie_embeddings=True,
        tied_embed_init_std=0.005,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        use_smeargate=True,
        bigram_hash_buckets=256,
        bigram_hash_dim=32,
    )
    for m in model.modules():
        if hasattr(m, "_use_int6_qat"):
            m._use_int6_qat = False
    x = __import__("torch").randint(0, 64, (2, 16))
    y = __import__("torch").randint(0, 64, (2, 16))
    import torch.nn.functional as F

    with __import__("torch").no_grad():
        loss_m = model(x, y)
        logits = model.forward_logits(x)
        loss_manual = F.cross_entropy(logits.reshape(-1, 64).float(), y.reshape(-1), reduction="mean")
    if abs(float(loss_m.item() - loss_manual.item())) > 1e-5:
        print(f"ERROR: forward vs forward_logits loss mismatch {loss_m} vs {loss_manual}", file=sys.stderr)
        return 1
    print("OK forward_logits matches forward loss")

    w = __import__("torch").randn(16, 32, requires_grad=True)
    wq = fake_quantize_int6(w)
    wq.sum().backward()
    if w.grad is None or w.grad.abs().sum() == 0:
        print("ERROR: STE int6 no gradient", file=sys.stderr)
        return 1
    print("OK STE int6 gradients flow")

    qobj, _ = quantize_state_dict_int6(model.state_dict())
    buf = io.BytesIO()
    __import__("torch").save(qobj, buf)
    restored = dequantize_state_dict_int6(__import__("torch").load(io.BytesIO(buf.getvalue()), map_location="cpu"))
    model.load_state_dict(restored, strict=True)
    with __import__("torch").no_grad():
        loss2 = model(x, y)
    print(f"OK int6 roundtrip loss gap {abs(loss2.item() - loss_m.item()):.6f}")

    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
