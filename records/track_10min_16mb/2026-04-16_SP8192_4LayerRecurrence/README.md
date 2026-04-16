# SP8192 + 4-Layer Depth Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT

**val_bpb = pending** (results to be collected on 8xA100s) | **~16MB** | 8xA100

## Summary

This submission extends the current SOTA (PR #1509, val_bpb=1.0810) by expanding the depth recurrence from 3 looped layers to 4 looped layers (`LOOP_END=6`). All other hyperparameters and techniques are carried forward from the SOTA.

The single architectural change:

| | SOTA | This submission |
|---|---|---|
| `LOOP_END` | 5 | **6** |
| Looped layers | [3,4,5] | **[3,4,5,6]** |
| Virtual layers | 17 | **19** |
| Est. training steps | 4550 | **~4071** |

## Motivation

The prior depth-recurrence submissions show a consistent trend: more virtual layers improve BPB:

- 2-layer loop (layers 4,5), 2 passes, ~13 virtual layers: 1.0979
- 3-layer loop (layers 3,4,5), 3 passes, 17 virtual layers: 1.0835 (no TTT)
- 3-layer loop (layers 3,4,5), 3 passes, 17 virtual layers: 1.0810 (with TTT)

The compute budget (layer-steps) for the two approaches is nearly identical:
- SOTA: 4550 steps x 17 virtual layers = 77,350 layer-steps
- This: ~4071 steps x 19 virtual layers = 77,349 layer-steps

So we're spending the same total compute budget but organized as deeper (not wider) per-step operations. If depth is more sample-efficient than repeated passes at the same depth, this should improve BPB.

The 4-layer loop also gives the U-Net skip connections a richer encoder path, connecting deeper intermediate representations to the corresponding decoder layers.

## Architecture

**Identical to SOTA except:**

Looping layers: `[3, 4, 5, 6]` (3 total passes)

Virtual layer sequence:
```
encoder: [0, 1, 2, 3, 4, 5, 6, 3, 4]
decoder: [5, 6, 3, 4, 5, 6, 7, 8, 9, 10]
```

U-Net skip connections (9 pairs):
- L0 (first pass) -> L10
- L1 (first pass) -> L9
- L2 (first pass) -> L8
- L3 (first pass) -> L7 (parallel)
- L4 (first pass) -> L6 (2nd loop pass)
- L5 (first pass) -> L5 (3rd loop pass)
- L6 (first pass) -> L4 (3rd loop pass)
- L3 (second pass) -> L3 (3rd loop pass)
- L4 (second pass) -> L6 (2nd loop pass, parallel section)

Parallel residuals from layer 7+ (unchanged from SOTA). Layers 3-6 in the loop remain sequential.

## Full Technique Stack (carried from SOTA)

1. **SP8192** tokenizer (kevclark/parameter-golf)
2. **4-Layer Depth Recurrence** (this submission: layers 3,4,5,6, activated at `frac=0.35`)
3. **Parallel Residuals** from layer 7 (GPT-J style)
4. **QK-Gain 5.25** (now the default in Hyperparameters, was env var in prior submission)
5. **MuonEq-R** optimizer (row-normalized Muon, Newton-Schulz 5 steps)
6. **Legal Score-First TTT** (SGD lr=0.005, momentum=0.9, 3 epochs/chunk, cosine LR decay)
7. **Full-Hessian GPTQ SDClip** (int6 matrices k=12.85, int8 embeddings k=20.0)
8. **Byte-shuffle + Brotli-11** compression
9. **LZMA code wrapper**

## Code Changes

The only functional changes from SOTA `train_gpt.py` (PR #1509):

```python
# Before (SOTA)
loop_end=int(os.environ.get('LOOP_END',5))
qk_gain_init=float(os.environ.get('QK_GAIN_INIT',5.))

# After (this submission)
loop_end=int(os.environ.get('LOOP_END',6))
qk_gain_init=float(os.environ.get('QK_GAIN_INIT',5.25))
```

The `qk_gain_init` default change absorbs the known-good 5.25 value (previously passed as env var) into the script defaults. The environment variable override still works.

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

No additional env vars needed (QK_GAIN_INIT=5.25 is now the default).

## Expected Behavior

- Training: ~4071 steps in ~588s (vs 4550 in SOTA), due to 11.8% more compute per step
- Eval: identical sliding window + TTT pipeline
- Artifact size: ~15.99 MB (essentially unchanged)

## Compliance

Same as SOTA (PR #1509). Score-first TTT, no SLOT, no n-gram cache, no pre-quant TTT.

## Credits

- **@clarkkev** - SP8192 + GPTQ + SDClip + MuonEq-R + depth recurrence base (PR #1394)
- **@dexhunter** - 3-layer depth recurrence (PR #1331, #1437), legal TTT on SP8192 (PR #1413)
- **@Robby955** - Parallel residuals on SP8192 (PR #1412)
- **@msisovic** - Parallel residuals concept (PR #1204)
- **@X-Abhishek-X** - Hyperparameter tuning: WD=0.095, MLR=0.022, EMA=0.9965 (PR #1445, #1471)
- **@abaybektursun** - Score-first TTT framework (PR #549)
