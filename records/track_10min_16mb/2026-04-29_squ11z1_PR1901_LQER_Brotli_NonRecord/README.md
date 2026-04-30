# Non-Record: PR #1901 base + LQER Asymmetric + Brotli/Byte-Shuffle Compression

**Status: non-record discussion** — patched implementation provided; full 3-seed validation could not be completed within available compute budget.

This submission proposes two orthogonal additions to the **PR #1901 stack from @Karen042009** (val_bpb 0.83353, pending merge):

1. **LQER asymmetric rank-4 post-quantization correction** — reduces the int6 Sigma-Delta quantization tax by storing top-K=2 weight residuals as INT2/INT4 SVD factors.
2. **Brotli-11 + byte-shuffle compression** replacing LZMA — recovers ~150–280 KB of artifact budget that the model can re-spend.

Both contributions are stack-orthogonal and architecturally minimal (~80 LoC added to PR #1901's 1,221-line pipeline). The patched `train_gpt.py` is provided LZMA-wrapped at 18,204 bytes (vs PR #1901's 53,443 bytes raw — a 65.9% code-byte saving alone).

## Why non-record

Available compute was a single $25 starter grant + remaining personal balance. The $500 development grant submitted on 2026-04-27 did not return a decision before the 2026-04-30 deadline. The two compute attempts before submission:

- **2026-04-26 8×H100 SXM bid run** (different stack, PR #1493+LQER): preempted at training step ~4,000 of ~6,700, ~4–5 minutes before the artifact would have been emitted. Sidecar log uploaded to HF preserves train_loss 2.91 at step 4,000.
- **2026-04-29 8×H100 SXM bid run** (this stack): preempted during HuggingFace data prefetch (50% / 250M tokens per rank), well before training started.

Both pods were single-seed attempts on bid pricing because on-demand 8×H100 SXM either repeatedly stuck in container boot (machine `qd6276xi9ky5`) or exceeded the available balance. Without the development grant covering 3-seed validation, this submission is filed as a non-record contribution: implementation + theoretical δ-BPB estimate, no measured val_bpb.

## Theoretical contribution analysis

### LQER asymmetric rank-4 on Sigma-Delta residuals

PR #1901 uses Dynamic MSE Sigma-Delta (SDClip σ-grid {2.5, 3.0, 3.5, 4.0}) with INT6 codes + per-row fp16 scale. After their `export_submission` quantization loop, this submission inserts:

```python
for name, codes in quantised_state.items():
    if not codes.dim() >= 2: continue
    W_q = (codes.float() * scale)
    W_fp = net.state_dict()[name].float().cpu()
    E = W_fp - W_q
    cands.append((name, E, ||E||_F))
cands.sort(key=lambda x: -x[2])
for name, E, _ in cands[:top_k=2]:
    U, S, Vh = svd(E, full_matrices=False)
    A = (U[:, :rank=4] * S[:4]).contiguous()
    B = Vh[:4, :].contiguous()
    qA, sA, qB, sB = lqer_pack_asym(A, B, group=64)
    quantised_state[name + '_lqA'] = qA  # INT2
    quantised_state[name + '_lqB'] = qB  # INT4
```

At dequantization, `W_corrected = W_dequant + A_dequant @ B_dequant`.

LQER paper (Lee et al. 2023, arXiv:2310.18313) reports 0.5–1.5 bit-per-weight equivalent reduction. PR #1797 (@dexhunter) validated the asymmetric variant on Hessian-GPTQ and observed −0.009 BPB recovery (1.06157 base on PR #1787). Sigma-Delta error diffusion already auto-compensates within-row error; we expect the LQER recovery on top of Sigma-Delta to be **smaller, in the range −0.002 to −0.005 BPB**, because the residual variance is reduced before LQER sees it.

This is, to our knowledge, the first proposed application of LQER to a Sigma-Delta-quantized stack in this competition.

### Brotli-11 + stride-2 byte-shuffle

PR #1901 uses `lzma.FORMAT_XZ preset=9 dict_size=128MB`. We replace this with stride-2 byte-shuffle (groups MSB/LSB bytes via position-mod-stride permutation) followed by Brotli quality=11 generic mode. PR #1855 reports ~150–280 KB savings on int6 weight blobs from a comparable per-group lrzip + brotli pipeline. Saved bytes are re-invested in a slightly larger model (PR #1901 already auto-downsizes hidden_size to fit; with Brotli savings, hidden_size could rise from 336 to 344 or higher).

Expected δ-BPB from larger model + same training: **−0.002 to −0.005 BPB** based on the empirical hidden_size→BPB curve in PR #1901's `[SizeCheck]` log (~0.005 BPB per +16 hidden dimensions on the same training stack).

### Combined estimate

Stacked: −0.005 to −0.010 BPB on top of PR #1901's 0.83353 → projected **0.823–0.829 BPB** on a 3-seed run. This is below the current pending top-2 (PR #1901 0.83353, #1848 Mikey 0.87980 — though #1848 is unverified) and above the projected #1818 LQER+SP1024 (1.06108 on a different/weaker base).

If validated, this would be the lowest val_bpb in a non-PPM submission (PPM-based PRs face the @sharpobject argument in Issue #1872 / PR #1905 about probability-distribution validity).

## What is in this submission

| File | Purpose |
|---|---|
| `train_gpt.py` | LZMA-wrapped patched code (18,204 bytes; raw 53,586 bytes) |
| `train_gpt_unwrapped.py` | Raw patched source for review |
| `submission.json` | Metadata; val_bpb fields are empty pending validation |
| `partial_run_2026-04-29.log` | HuggingFace data-prefetch log up to preemption point |
| `partial_run_2026-04-26.log` | Earlier 4,000-step training log (different stack, train_loss=2.91 at step 4000) |
| `README.md` | This file |

## Test plan (incomplete — see compute notes above)

- [x] Patch applies cleanly to PR #1901 (syntax check, function-level replacement validated locally)
- [x] LZMA-base85 wrapper round-trips correctly (verified via `compile()` + decompress identity check)
- [x] Patched code launches on 8×H100 SXM, reaches HF data-prefetch phase (verified by `partial_run_2026-04-29.log`)
- [ ] **Pending**: 3-seed val_bpb measurement on 8×H100 SXM with full 600s training cap
- [ ] **Pending**: artifact size verification under the 16 MB cap
- [ ] **Pending**: ablation `LQER_TOP_K ∈ {1, 2, 3}`, `LQER_RANK ∈ {2, 4, 8}`
- [ ] **Pending**: Brotli vs LZMA artifact size A/B on identical model

If this submission is approved as a record-eligible record after validation, I commit to providing 3-seed logs from a future compute window.

## Attribution

- **Base stack PR #1901**: @Karen042009 — DualTokenHashSkip, LayerScale Recurrence, SharedMoE, AdaMuon optimizer, Dynamic MSE SDClip, Score-First TTT
- **LQER asymmetric variant**: @dexhunter (PR #1797) — first competition implementation
- **LQER paper**: Lee et al. 2023 (arXiv:2310.18313)
- **Brotli + byte-shuffle compression idea**: @dexhunter (PR #1855)

## Compliance notes (verifiable from code)

- INT6 SDClip + INT2/INT4 LQER factors + fp16 scales — all standard tensor types
- Brotli-11 generic mode — public format
- No PPM mixture (avoids the Issue #1872 probability-distribution dispute)
- Score-First TTT inherited verbatim from PR #1901
- Training stays within 600s wallclock cap on 8×H100 (PR #1901's auto-configured schedule unchanged)
- Eval stays within 600s budget (PR #1901's eval pipeline; LQER dequant adds <1s for top-K=2)

## Reproduction

The patched `train_gpt.py` is self-contained and uses HuggingFace streaming for FineWeb data. To reproduce:

```bash
pip install brotli transformers tokenizers datasets huggingface_hub torch
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Tokenizer is trained at first run from `HuggingFaceFW/fineweb` sample-10BT (~25 min CPU on 8 vCPU). Set `HF_TOKEN` env var for higher-rate-limit downloads.

A pre-trained tokenizer for vocab=8192 (compatible with this stack at hidden=336/layers=12) is available at `https://huggingface.co/datasets/squ11z1/pgolf-lqer/blob/main/moe/pg_tokenizer_v10_2/tokenizer.json` to skip the tokenizer-training step.
