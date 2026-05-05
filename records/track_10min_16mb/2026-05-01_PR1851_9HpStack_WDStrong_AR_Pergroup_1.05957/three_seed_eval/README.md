# 3-seed re-evaluation

Re-runs the published `train_gpt.py` from this record directory, with the
exact env block from the parent README's "Reproducing" section, on three
seeds: **42, 0, 1234** — the same seeds PR #1855 used, so the comparison
is apples-to-apples.

Hardware: 8xH100 80GB SXM, PyTorch 2.9.1+cu128, FlashAttention 3 from
`windreamer/flash-attention3-wheels/cu128_torch291`.

## Per-seed results

| Seed | Steps | Pre-quant val_bpb | Post-quant val_bpb | **Post-TTT val_bpb** | Artifact bytes |
|---:|---:|---:|---:|---:|---:|
| 42   | 4862 | 1.06323705 | 1.07238064 | **1.05948583** | 15,899,339 |
| 0    | 4849 | 1.06411281 | 1.07321226 | **1.06029333** | 15,903,214 |
| 1234 | 4878 | 1.06430293 | 1.07362536 | **1.06075989** | 15,909,242 |
| **mean** |   |   |   | **1.06017968** | 15,903,932 |

- 3-seed stdev: 0.00064459
- 3-seed spread (max−min): 0.00127406
- All artifacts under the 16 MB cap; tightest margin is seed 1234 with
  867,974 B of headroom.

## Comparison

| Comparator | val_bpb |
|---|---:|
| This candidate, seed 42 (parent README headline) | 1.05956571 |
| **This candidate, 3-seed mean (this re-eval)** | **1.06017968** |
| PR #1855 published seed 42 | 1.05989454 |
| PR #1855 published 3-seed mean | 1.06107587 |

| Delta | BPB |
|---|---:|
| Parent README's claimed delta (seed-42 vs PR #1855 3-seed mean) | −0.00151016 |
| Honest 3-seed-vs-3-seed delta vs PR #1855 | **−0.00089619** |
| Per-seed mean delta vs PR #1855 (same seeds) | −0.00089619 |
| Same-seed 42 reproduction delta vs parent README | −0.00007988 |

Per-seed comparison vs PR #1855 (same seed):

| Seed | Ours | PR #1855 | Delta |
|---:|---:|---:|---:|
| 42   | 1.05948583 | 1.05989454 | −0.00040871 |
| 0    | 1.06029333 | 1.06124613 | −0.00095280 |
| 1234 | 1.06075989 | 1.06208695 | −0.00132706 |

## What this means

1. **Seed 42 reproduces.** 1.05948583 vs README 1.05956571 (Δ = −0.00008).
   The training graph, GPTQ all-rank Hessian averaging, pergroup
   lrzip+brotli compressor, and 3-phase LoRA TTT all behave as documented.
2. **The parent README's headline framing is misleading.** It compares
   this candidate's *single best seed* (1.05957) to PR #1855's *3-seed
   mean* (1.06108) and reports −0.00151. The honest 3-seed-vs-3-seed
   delta is **−0.00090** — about 60 % of the headline gain disappears
   once the comparison is consistent.
3. **Within-run noise is large.** A 3-seed spread of 0.00127 is much
   larger than the 0.00033 same-seed advantage over PR #1855 seed-42
   that the README cites, so a single seed cannot resolve the effect.
4. **The improvement is real, just smaller.** Per-seed deltas are all
   negative (−0.00041, −0.00095, −0.00133), mean −0.00090. The combined
   wd_strong + GPTQ all-rank Hessian + pergroup compressor stack does
   beat PR #1855 — by roughly half what the parent README claims.
5. **Seed 42 is end-to-end lucky.** It has the best pre-quant
   (1.06324), best post-quant (1.07238), and best post-TTT in this
   re-eval — i.e. it's a favourable seed throughout, not a quirk of the
   compressor or TTT path. Picking it as the headline cherry-picks the
   right tail of the seed distribution.

## Logs

`seed42.stdout.log`, `seed0.stdout.log`, `seed1234.stdout.log` are the
torchrun stdout/stderr from each run, including final
`stopping_early`, `Total submission size quantized+pergroup`,
`diagnostic pre-quantization post-ema`, `diagnostic quantized`, and
`quantized_ttt_phased` lines.
