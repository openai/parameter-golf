# Novel knob: LeakyReLU² slope A/B (parent P0)

**Parent:** `2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT` (confirmed).

## What the code does today

The on-disk **`train_gpt.py`** is still the **LZMA-compressed** one-liner; inside it, the MLP used to be **hardcoded** `negative_slope=.5` with no env toggle.

This repo also ships **`train_gpt_plain.py`**: same logic, but the MLP reads **`LEAKY_RELU_SLOPE`** (default **`0.5`**) and uses **ReLU²** when the value is **≤ 0**.

## Planned novel axis (same as `docs/PLAN-sp8192-novel-knob.md`)

| Run | `LEAKY_RELU_SLOPE` | Meaning |
|-----|-------------------|---------|
| Baseline A | `0` | ReLU² (no leak) |
| B (parent default) | `0.5` | LeakyReLU(0.5)² — matches README |

## Shipped in this repo

- **`train_gpt_plain.py`** — decompressed parent trainer with **`LEAKY_RELU_SLOPE`** wired on the MLP:
  - `LEAKY_RELU_SLOPE` unset or **`0.5`** → same as published **LeakyReLU(0.5)²** (run B).
  - **`0`** or any **≤ 0** → **ReLU²** baseline (run A).
- Original **`train_gpt.py`** — unchanged LZMA wrapper (byte-compact, same behavior as plain when using default env).

## Plain `train_gpt.py` vs re-wrapping LZMA

| Approach | When to use |
|----------|-------------|
| **Plain** (`train_gpt_plain.py`) | Local A/B runs, code review, diffs. ~49 KB source; negligible vs ~16 MB weights. **Use this for experiments.** |
| **LZMA wrapper** | Only if you need to match the one-line `exec(b85decode…)` layout or shave every KB of *code* in a constraint; recompress with the same `FORMAT_RAW` + `FILTER_LZMA2` pattern as the parent. |

For leaderboard submissions, either name is usually fine if `submission.json` points at the script you actually ran; confirm current challenge rules.

## How to do runs (8× GPU, from repo root)

**1. One-time environment + data (matches parent README)**

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
```

**2. Train — change only `LEAKY_RELU_SLOPE` between A and B; keep `SEED` and all other vars identical.**

Baseline **A** (ReLU²):

```bash
SEED=42 LEAKY_RELU_SLOPE=0 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt_plain.py
```

Run **B** (parent default, LeakyReLU(0.5)²):

```bash
SEED=42 LEAKY_RELU_SLOPE=0.5 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt_plain.py
```

Omitting `LEAKY_RELU_SLOPE` defaults to **`0.5`** (same as B).

**3. Optional second seed** — swap `SEED=314` or `999` as in the parent README table; still only vary the knob between A/B pairs.

**4. Logs** — capture stdout to `train.log` (or per-seed files like the parent’s `train_seed42.log`) for `final_*` / `val_bpb` lines.

### Regenerate plain from wrapped (if you edit the wrapper flow)

```bash
python3 scripts/decompress_record_train_gpt.py \
  records/track_10min_16mb/2026-04-09_SP8192_3LayerRecur_ParResid_QK525_LegalTTT/train_gpt.py \
  -o train_gpt_plain.py
# then re-apply the MLP `forward` line from this folder’s train_gpt_plain.py or merge by hand
```

## Reference

- `mlp_activation_spec.py` — tiny tensor helper + tests for the math.
- `tests/test_sp8192_p0_mlp_activation_spec.py` — CPU tests (no FlashAttention import).
