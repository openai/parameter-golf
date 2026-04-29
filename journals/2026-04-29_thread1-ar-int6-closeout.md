# Journal · 2026-04-29 · thread1-ar-int6-closeout

Rotated from journal.md on 2026-04-29 00:38 EDT.

## Entries (newest first)

## 2026-04-29 · 0081 (AR self-gen GPTQ int6) · DISCARD (cap-busts) — int6+brotli incompatible in our family; AR self-gen path also broken

**Question**: Does AR self-gen GPTQ int6 free MPS-visible cap on top of 0076 promoted (val 1.9514, artifact 15.91 MB) without regressing val_bpb? (Last untouched thread-1 lever.)

**Result (0081 first run, with try/except fallback)**:
- val_bpb_post_quant 1.94709857 (1.9471), within noise of 0076 single-seed 1.9483
- Artifact 21.331 MB → SIZE_VIOLATION, 5.4 MB over the 16 MB cap
- Train-loss trajectory bit-identical to 0076 family (step 1: 20.6122)
- pre-quant val_bpb 1.9986 (matches 0076 family pre-quant 1.9996)

**Two failures** (the second is the load-bearing one):

1. **AR self-gen path crashed** at production: `Mamba2Block: seq_len=1 must be divisible by chunk_size=64`. AR generates one token at a time → first iter has ctx_len=1 → Mamba2's chunked SSD scan rejects. The previous subagent wrapped this in try/except so the run silently fell back to taking 512 tokens from the train stream as calibration data. **The 0081 result is therefore train-stream-cal GPTQ int6, not AR self-gen.**

2. **int6 + brotli is anti-synergistic in our family**. `pack_int6` stores 4 int6 values per 3 bytes (~8 bits/byte entropy → near-incompressible). Brotli on int6-packed bytes saves only **5%** (ratio 0.948 vs zlib's 1.0). Compare to int8 weights where brotli saves ~30% (0064 entry: 15.18 MB → 13.44 MB on 0051 base; 0076 fits 15.91 MB AFTER brotli). So:
   - int6 raw payload (19.7 MB packed) + small int8 remainder ≈ same raw bytes as int8-only
   - But brotli saves ~5% on int6 vs ~30% on int8
   - Net: artifact GROWS from 15.91 MB → 21.33 MB. **The 25% raw saving from int6 is more than wiped out by lost brotli compressibility.**

**Mechanism story** [LIKELY]: int6 is information-dense at the byte level. brotli compression depends on byte-level redundancy that int8 retains (uniform-byte patterns, repeating low-magnitude values) and packed-int6 destroys. The H100 record uses int6 successfully but their architecture/scale ranges differ; the pattern doesn't transfer to our SSM hybrid. **Future fix attempts must either (a) keep int6 as uint8 [0, 63] unpacked (75% raw bytes but more brotli-compressible — math suggests still net larger), or (b) find a brotli-friendly int6 layout, OR (c) skip brotli for int6 layers and use a dictionary coder that exploits int6 entropy directly.**

**Disconfirming**: artifact_mb < 15.91 with int6 → would refute the brotli-incompressibility narrative. Did not happen.

**Bug fixes applied (in 0081 folder; carried to 0082b fork)**:
- `modules/gptq_int6.py`: `ar_self_gen_calibration_tokens` front-pads context to multiple of CHUNK_ALIGN=64 (causal model — pads at the front don't influence the last-position logits we read)
- `train_gpt.py` SERIALIZATION: removed silent try/except, AR-self-gen failures now crash the run (fail fast)

**0082b launched** to confirm AR self-gen path works mechanically and that val_bpb shifts trivially with calibration source. Predicted: same cap-bust outcome (artifact ~21 MB), val ≈ 1.945-1.950.

## 2026-04-29 · 0082b (AR self-gen GPTQ int6 — fixed) · DISCARD — confirms cap-bust is calibration-independent

**Question**: with the AR self-gen path actually executing (Mamba2 chunk-align fix + no fallback), does the result differ from 0081's train-stream-cal version? In particular, does the artifact size change?

**Result**: AR self-gen ran cleanly, 512 tokens in 31.3 s (~60 ms/token, no KV cache). GPTQ on 27 layers in 5.77 s. val_bpb_post_quant **1.9455** vs 0081's 1.9471 — Δ -0.0016, well within σ_pair=0.006. Artifact **21.324 MB** vs 0081's 21.331 MB — Δ -0.007 MB (essentially identical). brotli/zlib ratio 0.9478 vs 0081's 0.9480 (also identical).

**Conclusion** [VERIFIED for our family]: calibration source (AR self-gen vs train-stream) is NOT the deciding factor for AR int6 in our regime — int6+brotli incompressibility is. Both runs confirm: int6's 25% raw saving is more than wiped out by the ~25% lost brotli compressibility. Net cap GROWS by ~5.4 MB. AR int6 is closed; it does not transfer to our SSM hybrid at the current quant stack. **Thread 1 free-score levers fully exhausted.**

**Disposition**: 0081 was the spec-violating fallback run; 0082b is the spec-compliant rerun confirming 0081's headline. Both discard, both size_violation. The bug-fix code (front-pad ctx + fail-fast removal of try/except) is preserved in 0082b's folder for any future agent that wants to revisit AR int6 with a different bit-layout (e.g., unpacked uint8 storage, brotli-friendly column-interleaving, dictionary coder for int6 entropy).

**Conclusion** [LIKELY]: AR int6 does not transfer to our SSM family at the int8+brotli stack we already have. Thread 1 has only one lever left genuinely "untested at scale" — REPEAT_UNTIE_MLP / mini-DR — both require code changes and conflict with our K=3 L=3 looped triple-parallel topology. **Thread 1 is effectively closed; next session can pivot fully to thread 2 bold candidates.**

## 2026-04-29 · K=4 top_N sweep · cap-fill would help IF cap could be freed (but 0081 doesn't free it)

While 0081 ran, ran offline blended-BPB sweep over K=4 top_N values (K=3 fixed at 100K, model log2p cached, 6 weight tuples, sweep at `scratch/blend_probe/k4_topn_sweep.py`):

| K=4 top_N | best blend | offline BPB | Δ vs 200K |
|---|---|---|---|
| 200,000 | (0.7, 0.10, 0.20) | 1.9504 | (anchor) |
| 280,000 | (0.7, 0.05, 0.25) | 1.9426 | -0.008 |
| 320,000 | (0.7, 0.05, 0.25) | 1.9407 | -0.010 |
| 360,000 | (0.7, 0.05, 0.25) | 1.9387 | -0.012 |
| 400,000 | (0.7, 0.05, 0.25) | 1.9359 | -0.015 |
| 440,000 | (0.7, 0.05, 0.25) | 1.9346 | -0.016 |

Monotonic, no saturation visible at 440K (which would be 5.85 MB raw / ~2.5 MB brotli — needs ~1 MB more cap than 0076 has).

**Implication**: cap-fill works mechanically; the gain is ~half offline (per the 0074/0075 production-vs-offline ratio for per-context α), so production Δ ≈ -0.004 to -0.008 BPB at top_N=400K. **But cap can only be freed by a different mechanism than int6.** Possibilities:
- Smaller MLP_MULT (currently 8 — kills the 0062 SwiGLU mlp=8 win)
- Drop the per-context α int8 buffer (~0.2-0.3 MB; would lose 0076's gate gain)
- Pure ATTN at a position 0 (0065 freed 0.48 MB neutral)
- Brotli-friendly int8 layout tweak

Of these, 0065 (asym pos0) freed cap and is already known-neutral on val. Stacking 0076 + 0065 architecture changes + cap-fill K=4 top_N=300K could be the "free score" path. Untested in this session.


