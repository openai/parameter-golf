# 11L XSA + Mixed INT6 + Adaptive N-gram Cache (2->7 backoff)

## Results

3-seed runs on **March 27, 2026** under the standard **8xH100 SXM / 600s** budget.

`val_bpb: 0.96308303` (3-seed mean, std `0.00035576`)

| Seed | Steps | Sliding val_bpb | Final val_bpb | Artifact bytes |
| --- | ---: | ---: | ---: | ---: |
| 1337 | 6892 | 1.12124241 | 0.96314788 | 15879364 |
| 42 | 6894 | 1.12125743 | 0.96340191 | 15884280 |
| 2024 | 6897 | 1.12043283 | 0.96269931 | 15884064 |

- Mean final `val_bpb`: `0.96308303`
- Inter-seed std: `0.00035576`
- Best final `val_bpb`: `0.96269931` (`SEED=2024`)
- Mean artifact size: `15882569.33333333` bytes

## Architecture

| Component | Setting |
| --- | --- |
| Layers | 11 (512d, 8Q, 4KV) |
| MLP | 3x with `relu2` |
| XSA | All 11 layers |
| Embeddings | Tied |
| Weight averaging | EMA + late SWA |
| Quantization | Post-training mixed INT6 + LZMA |
| Eval | Sliding window, stride 64 |

## Adaptive N-gram Cache

Score-first adaptive n-gram cache with backoff orders `2->7`.

Backward-looking evaluation order:

1. Score each window under `torch.inference_mode()`.
2. Add only already-scored tokens to the cache.
3. Apply the cache only to later positions and later windows.

No training data is accessed during evaluation.

| Parameter | Value |
| --- | --- |
| Orders | `2->7` |
| Adaptive mode | `sigmoid_raw_entropy` |
| Alpha range | `[0.05, 0.60]` |
| Base alpha | `0.40` |
| Entropy center | `4.0` |
| Entropy scale | `2.0` |
| Hash buckets | `4,194,304` |
| Min count | `2` |

## Ablation (seed 1337)

| Configuration | val_bpb |
| --- | ---: |
| Post-EMA, pre-quant | 1.1369 |
| + INT6 roundtrip | 1.14466175 |
| + Sliding window (stride 64) | 1.12124241 |
| + Adaptive n-gram cache | 0.96314788 |

## Reproducibility

From this records folder:

```bash
python3 -m pip install --upgrade pip
python3 -m pip install -r requirements.txt
python3 ../../../data/cached_challenge_fineweb.py --variant sp1024 --train-shards 80
torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
SEED=2024 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Included evidence:

- `train_seed1337.log`
- `train_seed42.log`
- `train_seed2024.log`

No eval-only checkpoint is required.

## Acknowledgments

- XSA lineage: PR #265 and arXiv:2603.09078
- EMA and quantization lineage: PR #414, PR #549
- Early sliding-window evaluation path: PR #77
- Score-first evaluation framing: Issue #402
