# Record: SP8192 + Progressive 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT

**val_bpb = 1.0806** (3-seed mean, std 0.0011) | **~15.99 MB** | **8xH100 SXM**

## 3-Seed Results

| Seed | Sliding BPP | **TTT BPP** | Artifact |
|------|-------------|-------------|----------|
| 42   | 1.0818      | **1.0805**  | 15,992,388 |
| 999  | 1.0831      | **1.0818**  | 15,996,018 |
| 1337 | 1.0810      | **1.0796**  | 15,989,841 |
| **Mean** | **1.0820** | **1.0806** | **15,992,749** |
| **Std** | **0.0009** | **0.0011** | |

Current merged record in repo: **1.0810 BPP**. Delta: **-0.0004 BPP**.

## Key Change

This submission keeps the merged `2026-04-09` SP8192 record stack intact and changes only the recurrence schedule.

Baseline recurrence schedule:

- full 3-layer recurrence activates at `frac=0.35`

This submission uses a progressive schedule:

- phase 1 at `frac=0.35`: one extra recurrence pass
- phase 2 at `frac=0.55`: full 3-layer recurrence

The hypothesis is that the strongest recurrence stack benefits from a smoother transition into deeper virtual depth rather than a single hard switch.

## Key Techniques

1. **SP8192 + GPTQ SDClip** — int6 matrices (`k=12.85`), int8 embeddings (`k=20.0`), Brotli-compressed GPTQ state
2. **Progressive 3-Layer Depth Recurrence** — layers `3..5`, with partial recurrence at `0.35` and full recurrence at `0.55`
3. **Parallel Residuals** (`layers 7+`) — attention and MLP read from the same pre-residual input
4. **QK-Gain 5.25** — learnable per-head query scaling
5. **Legal Score-First TTT** — SGD (`lr=0.005`, momentum `0.9`), 3 epochs per 32K-token chunk, score-before-update ordering
6. **Tuned Hyperparameters** — WD `0.095`, matrix LR `0.022`, EMA `0.9965`, warmdown `0.72`
7. **Packed LZMA wrapper** — keeps code overhead low enough for the full artifact to fit under 16 MB

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16 dims), layerwise LN scale, tied embeddings, logit softcap `30.0`, skip gates enabled.

Progressive recurrence schedules the looped `3..5` segment in two phases:

- phase 1 encoder: `[0,1,2,3,4,5,3]`
- phase 1 decoder: `[4,5,6,7,8,9,10]`
- phase 2 encoder: `[0,1,2,3,4,5,3,4]`
- phase 2 decoder: `[5,3,4,5,6,7,8,9,10]`

Parallel residuals start at layer 7.

## Training

MuonEq-R optimizer (row-normalized Muon, Newton-Schulz 5 steps), AdamW for embeddings/scalars, max wallclock `600s`, GPTQ reserve `12s`. Training stops at the wallclock cap around `4699-4711` steps depending on seed.

## Quantization

Full-Hessian GPTQ with SDClip, int6 matrices and int8 token embeddings. Byte-shuffle plus Brotli-11 compression. All three seeds fit under `16,000,000` bytes with the packed code wrapper.

## TTT

Score-first, chunk-based SGD adaptation at eval time:

- chunk size `32K` tokens
- score each chunk under `torch.no_grad()` before any update
- train for 3 epochs on already-scored chunk tokens
- cosine LR decay across chunks
- gradient clipping at `1.0`

## Compliance

Per Issue #1017 Track B:

- strictly causal sliding-window scoring
- normalized softmax over the full vocabulary
- score-before-update TTT ordering
- single-pass token scoring
- no SLOT
- no ETLB
- no n-gram cache
- all artifacts under 16 MB on all 3 seeds
- training under 600s on all 3 seeds

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 QK_GAIN_INIT=5.25 PARALLEL_RESIDUAL_START=7 \
ENABLE_LOOPING_AT=0.35 LOOP_PHASE2_AT=0.55 \
TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 COMPRESSOR=brotli \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **@clarkkev** — SP8192 + GPTQ embeddings + SDClip + MuonEq-R base stack
- **@dexhunter** — 3-layer depth recurrence and legal TTT framework on SP8192
- **@abaybektursun** — score-first TTT framework
- **@Robby955** — parallel residuals on SP8192
- **@msisovic** — parallel residual concept
- **@X-Abhishek-X** — hyperparameter tuning for the strong SP8192 family

## Included Files

- `README.md`
- `submission.json`
- `train_gpt.py`
