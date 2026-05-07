# SP8192 + 4-Layer Depth Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT

**val_bpb = pending** (results to be collected on 8xA100s) | **~16MB** | 8xA100

## Summary

Extends the current SOTA (PR #1509, val_bpb=1.0810) by widening the depth recurrence from 3 looped layers to 4 (`LOOP_END=6`). All other hyperparameters and techniques are carried forward unchanged.

The only code change from SOTA:

```python
# Before (SOTA, PR #1509)
loop_end=int(os.environ.get('LOOP_END', 5))
qk_gain_init=float(os.environ.get('QK_GAIN_INIT', 5.))

# After (this PR)
loop_end=int(os.environ.get('LOOP_END', 6))
qk_gain_init=float(os.environ.get('QK_GAIN_INIT', 5.25))  # absorbs the known-good default
```

## Architecture

### Virtual Layer Sequence

**SOTA (loop_end=5)** — 17 virtual layers, 8 U-Net skips:
```
Encoder [0,1,2,3,4,5,3,4]  →  Decoder [5,3,4,5,6,7,8,9,10]
         ─────────────────         ──────────────────────────
         pre  └──loop──┘  2nd      2nd  └──loop──┘  post (par)
```

**This PR (loop_end=6)** — 19 virtual layers, 9 U-Net skips:
```
Encoder [0,1,2,3,4,5,6,3,4]  →  Decoder [5,6,3,4,5,6,7,8,9,10]
         ─────────────────────         ────────────────────────────
         pre  └────loop────┘ 2nd      2nd  └────loop────┘  post (par)
```

Layer 6 is promoted from the non-recurring post-loop section into the recurrence core. It now executes 3 times (like layers 3, 4, 5) instead of once.

### U-Net Skip Connections

With 9 encoder and 10 decoder steps, there are 9 skip connections (encoder[i] feeds decoder[8-i]):

| Skip | Encoder step | Decoder step |
|------|-------------|-------------|
| 0 | L0 (1st pass) | L10 |
| 1 | L1 (1st pass) | L9 |
| 2 | L2 (1st pass) | L8 |
| 3 | L3 (1st pass) | L7 (parallel) |
| 4 | L4 (1st pass) | L6 (3rd loop pass) |
| 5 | L5 (1st pass) | L5 (3rd loop pass) |
| 6 | **L6 (1st pass)** | **L4 (3rd loop pass)** |
| 7 | L3 (2nd pass) | L3 (3rd loop pass) |
| 8 | L4 (2nd pass) | L6 (2nd loop pass, parallel) |

Skip 6 is new in this PR. It connects the first pass through L6 (shallow context) to the third pass through L4 (deep re-processing), providing a residual shortcut that was absent in the 3-layer config.

### Parallel Residuals

Unchanged: layers 7, 8, 9, 10 use GPT-J-style parallel attention+MLP. These all appear in the decoder's post-loop section and are not part of the recurrence.

## Motivation: Compute Budget Equivalence

The 4-layer loop is slower per step (19 vs 17 virtual forward passes), but the total layer-step budget is identical:

| Config | Virtual layers | Est. steps | Layer-steps |
|--------|---------------|-----------|-------------|
| SOTA   | 17            | ~4,550    | ~77,350     |
| This PR | 19           | ~4,071    | ~77,349     |

The prior depth-recurrence progression shows monotonic improvement with more virtual depth:

| Submission | Looped layers | Virtual layers | val_bpb (no TTT) |
|-----------|--------------|---------------|-----------------|
| PR #1260 (2-layer loop) | [4,5] | ~13 | 1.0979 |
| PR #1394 (3-layer loop) | [3,4,5] | 17 | 1.0856 |
| This PR (4-layer loop) | [3,4,5,6] | 19 | pending |

Each expansion has improved BPB without increasing the compute budget. The hypothesis is that depth-per-step is more sample-efficient than breadth-of-passes at the same depth.

## Local Verification

`test_architecture.py` in this directory validates the 4-layer config on CPU (no CUDA, no flash_attn needed):

```
$ python test_architecture.py
============================================================
Config: SOTA (loop_end=5)
  encoder: [0, 1, 2, 3, 4, 5, 3, 4]
  decoder: [5, 3, 4, 5, 6, 7, 8, 9, 10]
  virtual_layers=17, skips=8
  forward pass: OK  shape=torch.Size([2, 32, 256])
  gradient flow: OK  loss=5.5465
  all looped blocks have clean gradients: OK

============================================================
Config: This PR (loop_end=6)
  encoder: [0, 1, 2, 3, 4, 5, 6, 3, 4]
  decoder: [5, 6, 3, 4, 5, 6, 7, 8, 9, 10]
  virtual_layers=19, skips=9
  forward pass: OK  shape=torch.Size([2, 32, 256])
  gradient flow: OK  loss=5.5435
  all looped blocks have clean gradients: OK

All architecture tests PASSED.
```

## Full Technique Stack (carried from SOTA)

1. **SP8192** tokenizer (kevclark/parameter-golf HuggingFace dataset)
2. **4-Layer Depth Recurrence** — layers [3,4,5,6], 3 total passes, activated at `frac=0.35`
3. **Parallel Residuals** — from layer 7, GPT-J style (attention and MLP read same input)
4. **QK-Gain 5.25** — learnable per-head query scaling (now the script default)
5. **MuonEq-R** — row-normalized Muon with Newton-Schulz 5 steps
6. **Legal Score-First TTT** — SGD lr=0.005, momentum=0.9, 3 epochs/32K chunk, cosine LR decay
7. **Full-Hessian GPTQ SDClip** — int6 matrices (k=12.85), int8 embeddings (k=20.0)
8. **Byte-shuffle + Brotli-11** compression
9. **LZMA code wrapper** — ~16.6KB self-extracting code

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

No additional env vars needed. `QK_GAIN_INIT=5.25` and `LOOP_END=6` are now script defaults.

## Expected Behavior

- Training: ~4071 steps in ~588s (vs 4550 for SOTA), 11.8% more compute per step
- Artifact: ~15.99 MB (essentially unchanged — one extra skip_weight/skip_gate row adds ~4KB uncompressed)
- Eval: identical sliding window + TTT pipeline, same timing budget

## Compliance

Identical to SOTA (PR #1509). Score-first TTT, no SLOT, no n-gram cache, no pre-quant TTT, no ETLB.

## Files

- `train_gpt.py` — LZMA-compressed production script (2-line diff from SOTA)
- `train_gpt_human.py` — human-readable version of the same code
- `test_architecture.py` — CPU smoke test, no dependencies beyond PyTorch

## Credits

- **@clarkkev** — SP8192 + GPTQ + SDClip + MuonEq-R + depth recurrence base (PR #1394)
- **@dexhunter** — 3-layer depth recurrence (PR #1331, #1437), legal TTT on SP8192 (PR #1413)
- **@Robby955** — Parallel residuals on SP8192 (PR #1412)
- **@msisovic** — Parallel residuals concept (PR #1204)
- **@X-Abhishek-X** — Hyperparameter tuning: WD=0.095, MLR=0.022, EMA=0.9965 (PR #1445, #1471)
- **@abaybektursun** — Score-first TTT framework (PR #549)
