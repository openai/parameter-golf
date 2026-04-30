# Notable Non-Record Submission: 1.077345 BPB | Control+Tail Score-First TTT

**Author:** zhenyi-ji ([@Gotnhub](https://github.com/Gotnhub))

**Accepted SP8192 3-layer recurrence base + Muon 0.98 + MIN_LR 0.10 + legal score-first TTT restricted to global control/gating parameters and the last 3 transformer blocks**

**val_bpb: 1.07734522** (seed=42, quantized TTT) | **15,990,737 bytes** artifact | 8xH100 SXM

> **This is a non-record / unlimited-compute-style submission.** It is a
> single-seed result, it does not beat the current official SOTA, and its final
> TTT eval time was 776.046s, above the 10-minute eval cutoff. It is submitted
> to document a clean legal TTT-subset ablation: adapting a targeted
> control+tail parameter subspace instead of the whole model during score-first
> TTT. Full development notes are in [RESULTS.md](RESULTS.md).

## Results

| Metric | Value |
|---|---:|
| Seed | 42 |
| Train steps | 4,787 |
| Train time | 588.060s |
| Pre-quant BPB | 1.08399328 |
| Quantized BPB | 1.09540727 |
| Sliding-window BPB | 1.07870003 |
| Quantized TTT BPB | **1.07734522** |
| TTT eval time | 776.046s |
| Artifact size | 15,990,737 bytes |
| TTT parameters updated | 8,681,560 |

Run id: `top1-control-tailall-muon098-minlr010-seed42`

Modal call id: `fc-01KQCBP780FKR45925KMK2QK2X`

## Why This Is Interesting

Most strong legal-TTT submissions either adapt the full model during TTT or add
explicit adapter machinery such as LoRA-style modules. This submission tries a
middle path: use only parameters that already exist in the accepted model, but
choose them as an implicit adapter subspace.

The selected TTT subspace is:

- all existing control/gating parameters across the full network
- all parameters in the last 3 transformer blocks
- no new learned adapter modules
- no validation-token access before scoring
- no SLOT, ETLB, n-gram cache, or logit bias path

The intuition is that the global control/gating scalars can steer routing,
residual balance, attention sharpness, and skip behavior, while the last blocks
provide enough local capacity to adapt output representations. This is less
blunt than all-parameter TTT and much stronger than control-only TTT.

## Base Stack

The base is the accepted
`2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` record stack:

- SP8192 tokenizer
- 11 layers, 512 hidden size, 8 attention heads, 4 KV heads
- 3-layer recurrence over layers 3-5
- parallel residuals from layer 7
- QK gain 5.25
- GPTQ int6 matrices and int8 embeddings
- Brotli compressed artifact
- legal score-first TTT

This submission keeps the base architecture, tokenizer, dataset, compression
path, and score-first evaluation order.

## Key Changes

### Control+Tail TTT Parameter Selection

The main change is `TTT_PARAM_MODE=control_tail_all`.

During TTT, the script freezes the full quantized model except:

- global control/gating parameters:
  `attn_scale`, `mlp_scale`, `resid_mix`, `skip_weights`, `skip_gates`,
  and `q_gain`
- every parameter in the last `TTT_TRAIN_LAST_N=3` transformer blocks

This updates 8,681,560 parameters during TTT, compared with 35,944,536 model
parameters in the base model.

### Warmdown Floor and Momentum

The run also uses:

- `MUON_MOMENTUM=0.98`
- `MIN_LR=0.10`
- `TTT_LR=0.005`
- `TTT_EPOCHS=3`

These defaults are baked into `train_gpt.py`, so the submitted script can be
run directly without extra environment overrides.

## Comparison and Ablations

These are single-seed workspace results, not statistically significant SOTA
claims:

| Run | Seed | TTT BPB | Notes |
|---|---:|---:|---|
| Local accepted-base reproduction | 42 | 1.07960392 | Official accepted stack reproduced locally |
| Muon 0.98 only | 42 | 1.07898847 | Weak improvement over local base |
| Control-only TTT + Muon 0.98 + MIN_LR 0.10 | 42 | 1.08009438 | Too little TTT capacity |
| **Control+tail-all TTT + Muon 0.98 + MIN_LR 0.10** | 42 | **1.07734522** | Best completed local result |
| Control+tail-all last 4 blocks + anchor | 42 | 1.07944810 | More capacity and anchor regularization regressed |

The important ablation is that control-only TTT was too weak, while adapting
the last 3 blocks plus global control parameters gave a clear gain on seed 42.

## Compliance Notes

- Artifact is under 16,000,000 bytes.
- Training is under 600 seconds.
- Evaluation is score-first: each chunk is scored before any update on that
  chunk.
- No validation tokens are used before they are scored.
- No SLOT, pre-quant TTT, ETLB, hashed n-gram cache, or logit-bias shortcut.
- This is single-seed only.
- Final TTT eval is 776.046 seconds, so this is not a leaderboard-runtime
  record.
- The current official SOTA at submission preparation time is much stronger,
  so this is intentionally submitted as non-record.

## Reproduction

From the repository root:

```bash
pip install brotli sentencepiece numpy tqdm huggingface-hub datasets
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

cd records/track_non_record_16mb/2026-04-30_ControlTailAllTTT_Muon098_MinLR010_Seed42
ln -s ../../../data data 2>/dev/null || true
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

The submitted [train_seed42.log](train_seed42.log) is the log for the headline
score.

## Attribution

- Accepted base stack:
  `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` and its listed
  contributors.
- Legal score-first TTT framework: inherited from the accepted base and its
  listed score-first TTT precedents.
- Adapter/local-TTT principle: PR #1767 was used only as mechanism
  inspiration. No PR #1767 code is copied.
- Warmdown-floor signal: PR #1874 was used only as hyperparameter
  inspiration. No PR #1874 code is copied.

## Limitations

This is not a record submission. The result is only one seed, it is below the
current official SOTA, and the final eval time is above the leaderboard cutoff.
The intended contribution is the auditable TTT-subspace mechanism and the
ablation trail showing that this targeted control+tail subspace can improve the
accepted base family on seed 42.

## Included Files

- `README.md`
- `RESULTS.md`
- `submission.json`
- `requirements.txt`
- `train_gpt.py`
- `train_seed42.log`
