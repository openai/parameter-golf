# Hetul Adapter Memory TTT

Final promoted method: rank1024 eval-time adapter memory TTT on top of the 2026-04-09 SP8192 legal score-first TTT stack.

The adapter is created only during evaluation. It adds a fixed random feature projection `A` and a zero-initialized trainable projection `B`, then lets the existing score-first TTT loop update `B` only after each chunk has already been scored. The adapter tensors are not stored in the artifact, so the model bytes are unchanged.

## 1M-Slice Validation

All comparisons used the same 1,048,576-token eval slice, same control setup, and fresh controls for the promoted seeds.

| Seed | Control BPB | Rank1024 BPB | Delta | Eval Tokens | Runtime |
|---:|---:|---:|---:|---:|---:|
| 42 | 1.32169984 | 1.32003903 | -0.00166081 | 1,048,576 | 77.7s |
| 314 | 1.32170177 | 1.32004790 | -0.00165387 | 1,048,576 | 36.3s |
| 999 | 1.32170062 | 1.32020796 | -0.00149266 | 1,048,576 | 36.4s |
| Mean | 1.32170074 | 1.32009830 | -0.00160245 | 1,048,576 | 50.1s |

Seed 42 rank sweep:

| Rank | BPB | Delta vs Control | Runtime | Decision |
|---:|---:|---:|---:|---|
| 0 | 1.32169984 | 0.00000000 | 79.9s | control |
| 256 | 1.32055362 | -0.00114622 | 76.6s | improved |
| 384 | 1.32042157 | -0.00127827 | 77.5s | improved |
| 512 | 1.32033100 | -0.00136884 | 77.2s | improved |
| 768 | 1.32023947 | -0.00146037 | 77.2s | improved |
| 1024 | 1.32003903 | -0.00166081 | 77.7s | promoted |
| 1536 | 1.32033915 | -0.00136069 | 77.5s | rejected |

Official full-validation estimate: the 2026-04-09 baseline reports 1.08100 mean BPB. Applying the measured 1M-slice mean delta gives an estimated full-validation mean of 1.07940 BPB. This is an estimate only, not a measured full-validation score.

## Artifact Budget

The final `train_gpt.py` is 16,920 bytes. The adapter adds no stored model tensors. Using the 2026-04-09 seed logs for quantized model bytes:

| Seed | Model Bytes | Code Bytes | Total Bytes |
|---:|---:|---:|---:|
| 42 | 15,975,300 | 16,920 | 15,992,220 |
| 314 | 15,976,325 | 16,920 | 15,993,245 |
| 999 | 15,976,638 | 16,920 | 15,993,558 |

Max estimated total artifact size is 15,993,558 bytes, under the 16,000,000-byte cap.

## Exact Run Command

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  TTT_ADAPTER_RANK=1024 TTT_ADAPTER_SEED=42 TTT_ADAPTER_SCALE=1.0 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

For other seeds, replace both `SEED` and `TTT_ADAPTER_SEED` with the same seed. The script defaults to rank1024 and adapter seed equal to `SEED`.

## Compliance Notes

- Score-first legal TTT: each chunk is scored before any adapter or model update.
- No future-token leakage.
- No validation data is used during training.
- No DTS, adaptive routing, lexical cache, suffix cache, RDC, CTR, or TTLA-only method is included.
- Official target is 8xH100, 10 minutes training, 10 minutes eval, and total artifact under 16,000,000 bytes.

## Included Files

- `train_gpt.py`
- `submission.json`
- `README.md`
- `eval_seed42.log`
- `eval_seed314.log`
- `eval_seed999.log`
