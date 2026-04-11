# Record: SP8192 + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT + Asynchronous Data Loader

**val_bpb = 1.0803** (2-seed mean, std 0.0001) | **~15.99 MB** | 8xH100 SXM

## 2-Seed Results

| Seed | Sliding BPP | **TTT BPP** | Artifact |
|------|-------------|-------------|----------|
| 42   | 1.0824      | **1.0802**  | 15,993,138 |
| 314  | 1.0822      | **1.0804**  | 15,994,881 |
| **Mean** | **1.0823** | **1.0803** | **15,994,010** |
| **Std** | **0.0001** | **0.0001** | 

Merged SOTA (PR #1493): **1.0810 BPP**. Delta: **-0.0007 BPP**.

Note: As this submission focuses on system optimizations and preserves the ML logic of the current SOTA, the 0.005-nat improvement requirement is waived per challenge guidelines.

## Key Techniques

1. **SP8192 + GPTQ SDClip** — int6 matrices (k=12.85), int8 embeddings (k=20.0), zero selective pruning (PR #1394 @clarkkev)
2. **3-Layer Depth Recurrence** (layers 3,4,5, activate at frac=0.35) — 17 virtual layers from 11 physical (PR #1331 @dexhunter, PR #1437 @dexhunter)
3. **Parallel Residuals** (layers 7+) — GPT-J style, attention and MLP read from same input (PR #1412 @Robby955, PR #1204 @msisovic)
4. **QK-Gain 5.25** — learnable per-head query scaling, monotonic improvement from 4.0 to 5.25 (PR #1493 @bigbag)
5. **Legal Score-First TTT** — SGD (lr=0.005, momentum=0.9), 3 epochs per 32K-token chunk, cosine LR decay. Score-before-update ordering. (PR #549 @abaybektursun, PR #1413 @dexhunter)
6. **Tuned Hyperparameters** — WD=0.095, MLR=0.022, EMA=0.9965, warmdown=0.72 (PR #1445 @X-Abhishek-X)
7. **LZMA code wrapper** — compress the training code (PR #1493 @bigbag)
8. **Asynchronous Data Loader** — implemented an asynchronous data loader which keeps a queue of loaded batches, yielding a 1.5% throughput increase (effectively saving ~70-80 steps of execution time).

## Details
This submission introduces two system optimizations, both of `ShuffledSequenceLoader`:
1. Migrated all `next_batch` logic to `numpy` to prevent redundant copies. By calling `torch.from_numpy` only at the final return, we reduced `aten::copy_` overhead by 50% on 1xH100 benchmarks.
2. Implemented a multi-threaded producer-consumer queue for batch loading. Worker threads pre-fetch and pin memory to the GPU device asynchronously. This hides the latency of CPU-to-GPU data movement and eliminates compute-starvation during the `next_batch` call.

## Compliance

- All artifacts under 16,000,000 bytes on all seeds
- Training under 600s on all seeds
- Eval (sliding + TTT) under 600s on all seeds

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **@clarkkev** — SP8192 + GPTQ Embeddings + SDClip + MuonEq-R + depth recurrence (PR #1394)
- **@dexhunter** — 3-layer depth recurrence (PR #1331, #1437), legal TTT on SP8192 (PR #1413)
- **@abaybektursun** — Score-first TTT framework (PR #549, merged precedent)
- **@Robby955** — Parallel residuals on SP8192 (PR #1412)
- **@msisovic** — Parallel residuals concept (PR #1204)
- **@X-Abhishek-X** — Hyperparameter tuning: WD=0.095, MLR=0.022, EMA=0.9965 (PR #1445, #1471)
- **@bigbag** - QK-gain improvemen (PR #1493)

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py`
- `train_seed42.log`
- `train_seed314.log`
