"""GPU smoke for NGramMixer + BatchUnigramMixer.

Allocates both mixers at real scale (V=8192, bsz=64) on cuda:0, runs the
mix_nll / update_stream / mix_nll_chunk / update_chunk calls end-to-end,
and checks a few invariants. No eval data needed. Takes ~5 seconds.
"""
import importlib.util
import sys
from pathlib import Path
import time

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
src = (HERE / "train_gpt.py").read_text(encoding="utf-8")
start = src.index("class NGramMixer:")
end = src.index("def build_sentencepiece_luts(")
ns = {"torch": torch, "F": F}
exec(src[start:end], ns)
NGramMixer = ns["NGramMixer"]
BatchUnigramMixer = ns["BatchUnigramMixer"]


def main():
    assert torch.cuda.is_available(), "needs CUDA"
    device = torch.device("cuda:0")
    dev_name = torch.cuda.get_device_name(0)
    print(f"device: {dev_name}")

    V = 8192

    # --- NGramMixer full-bigram test ---
    print("\nNGramMixer (bigram, V=8192) ...")
    m = NGramMixer(
        vocab_size=V, device=device,
        alpha=2.0, beta=-0.25, scale=8.0, use_uni_prior=True,
    )
    t0 = time.perf_counter()
    # Simulate a chunk of 2048 tokens (typical eval_val batch)
    x = torch.randint(0, V, (2048,), dtype=torch.int64, device=device)
    y = torch.randint(0, V, (2048,), dtype=torch.int64, device=device)
    nll_nn = torch.rand(2048, device=device, dtype=torch.float32) * 8
    # First call: state is empty — lambda should be high (trust NN)
    out1 = m.mix_nll(x, y, nll_nn)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"  mix_nll #1: {1000*(t1-t0):.1f}ms  mean_nll={out1.mean().item():.4f}")
    m.update_stream(x, y)
    t2 = time.perf_counter()
    # Second call with warmed state
    out2 = m.mix_nll(x, y, nll_nn)
    torch.cuda.synchronize()
    t3 = time.perf_counter()
    print(f"  update_stream + mix_nll #2: {1000*(t3-t2):.1f}ms  mean_nll={out2.mean().item():.4f}")
    mem_mb = torch.cuda.memory_allocated(device) / 1024**2
    print(f"  GPU memory: {mem_mb:.1f} MB")
    # Verify unchanged positions have bitwise equal mixed NLL when state unchanged.
    out2b = m.mix_nll(x, y, nll_nn)
    assert torch.equal(out2, out2b), "mix_nll produced different output on repeated call (state mutated?)"
    print("  mix_nll is deterministic on repeated call — OK")

    # --- BatchUnigramMixer test (bsz=64) ---
    print("\nBatchUnigramMixer (unigram, bsz=64, V=8192) ...")
    bsz, T = 64, 2048
    bm = BatchUnigramMixer(bsz=bsz, vocab_size=V, device=device,
                           alpha=2.0, beta=-0.25, scale=8.0)
    y_b = torch.randint(0, V, (bsz, T), dtype=torch.int64, device=device)
    nll_b = torch.rand(bsz, T, device=device, dtype=torch.float32) * 8
    offsets = torch.randint(0, T - 64, (bsz,), dtype=torch.int64, device=device)
    lens = torch.randint(32, 64, (bsz,), dtype=torch.int64, device=device)
    t0 = time.perf_counter()
    mixed_b = bm.mix_nll_chunk(y_b, nll_b, offsets, lens)
    bm.update_chunk(y_b, offsets, lens)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    print(f"  mix + update: {1000*(t1-t0):.1f}ms  mean_nll={mixed_b.mean().item():.4f}")
    mem_mb = torch.cuda.memory_allocated(device) / 1024**2
    print(f"  GPU memory total: {mem_mb:.1f} MB")
    # Each slot's update count should match its `lens`.
    for b in range(bsz):
        assert bm.uni_total[b].item() == int(lens[b].item()), f"slot {b} total mismatch"
    print("  per-slot update totals match chunk_lens — OK")

    print("\nall GPU smoke checks passed.")


if __name__ == "__main__":
    main()
