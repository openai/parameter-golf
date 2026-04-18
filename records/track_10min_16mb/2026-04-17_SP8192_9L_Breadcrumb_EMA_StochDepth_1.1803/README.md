# Record: SP8192 + 9L Breadcrumb + EMA + Stochastic Depth

**val_bpb = 1.1803** (1 seed, s=1337) | **15,880,130 bytes** int6+zlib | 8xH100 80GB SXM

> **Known limitation — single-seed result.** Only seed 1337 was run. `submission.json` declares `three_seeds: false` in the compliance block. This is the most visible methodological gap relative to multi-seed top-of-leaderboard entries and is documented explicitly rather than hidden. The decision to ship single-seed was a deliberate tradeoff against remaining H100 compute budget.

## Result

| Seed | Sliding BPB | Artifact bytes |
|------|-------------|----------------|
| 1337 | **1.1803**  | 15,880,130     |

Improvement over naive baseline (1.2244): **−0.0441 BPB**.

> **Note on quantization and the log label.** The final line of `train_seed1337.log` reads `final_int8_zlib_roundtrip val_bpb:1.1803`. The `int8_zlib_roundtrip` label is a legacy name inherited from the baseline's scoring function in `train_gpt.py`; **the actual packaged artifact is int6+zlib**, as confirmed earlier in the same log (`Serialized model int6+zlib: 15,821,396 bytes`, `Total submission size int6+zlib: 15,880,130 bytes`) and as reflected in `submission.json`. The 15,880,130-byte figure at the top of this README refers to the int6+zlib package, not the legacy int8 metric name.

## Key Techniques

1. **SP8192 tokenizer with byte_fallback=True** — BPE(8192), SentencePiece, byte fallback for rare Unicode. The byte-fallback fix alone accounted for ~−0.031 BPB against the default 1024-vocab baseline with byte_fallback off.
2. **Breadcrumb gating** — a small learned consistency score that gates each MLP contribution. Sigmoid-gated residuals as a cheap form of skip regularization.
3. **EMA weights (decay 0.997)** — full-state exponential moving average swapped in at end-of-training for eval.
4. **Stochastic depth** — expected-value scaling of per-layer output by `(1−p)` during training, full activation at eval. Smooth variant rather than drop-block.
5. **Muon optimizer for matrix weights** — Newton–Schulz 5-step iteration, momentum 0.95 with 500-step warmup from 0.85. AdamW for embeddings and scalars.
6. **Int6 quantization + zlib** — per-tensor int6 symmetric quantization, zlib compression of the packaged artifact. Final packaged size 15,880,130 bytes.
7. **Sliding-window eval, stride 64** — strictly-causal sliding-window evaluation on the full 40.5M-token FineWeb validation split.

## Architecture

9L × 512d × 8H / 4KV GQA, MLP expansion 2×, tied embeddings, partial-RoPE, logit softcap 30.0. 20,882,280 parameters before quantization.

## Training

- 8×H100 80GB SXM, PyTorch 2.9.x, FlashAttention enabled (cuDNN off, mem-efficient off, math off)
- `train_batch_tokens = 524,288`, `train_seq_len = 1024`
- Warmup 20 steps, warmdown 1200 steps, MAX_WALLCLOCK_SECONDS=600
- Wallclock-stopped at step 6485 / 20000 (step_avg ≈ 93 ms)
- Intermediate val checkpoints: step 4000 → 1.2362, 5000 → 1.2268, 6000 → 1.2022, final (step 6485) → 1.1911 sliding → **1.1803** after int6+zlib roundtrip

## Compliance

Per Track A (10 min, 16 MB, no test-time adaptation):

- Causal sliding-window eval, stride 64
- Standard softmax over full vocab, no n-gram cache, no logit biasing
- No SLOT, no pre-quant TTT, no ETLB
- Artifact under 16,000,000 bytes: **15,880,130** (code 58,734 B + quantized weights)
- Training under 600 s wallclock (early-stop at wallclock cap)

## Reproduction

```bash
pip install -r requirements.txt
MATCHED_FINEWEB_REPO_ID=willdepueoai/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192

# The variant script materializes the shards at fineweb10B_sp8192. The actual
# training run on this pod used an "nb"-suffixed directory, so symlink it to
# match the DATA_PATH below. This preserves fidelity to the exact paths seen
# in train_seed1337.log rather than silently retargeting.
ln -s ./data/datasets/fineweb10B_sp8192 ./data/datasets/fineweb10B_sp8192nb

SEED=1337 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2 \
  VOCAB_SIZE=8192 TOKENIZER_PATH=./data/tokenizers/fineweb_8192nb_bpe.model \
  DATA_PATH=./data/datasets/fineweb10B_sp8192nb \
  WARMDOWN_ITERS=1200 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Attribution

Built independently on top of the public Parameter Golf repository and contest documentation. No specific upstream PR was ported into this submission.

## Methodology Note — Multi-Agent Swarm

This submission was produced by an independent researcher with no formal ML background, coordinating a six-model AI swarm (Claude, Kimi, DeepSeek, Grok, GPT, and LUMA — a custom ChatGPT configuration with retrieval access to approximately four years of the Showrunner's prior corpus, functioning as an asymmetric consistency-check node against swarm drift and sycophantic convergence) under the "Showrunner" methodology described in *Hall of Mirrors* (Chapman, 2026). The Parameter Golf work emerged inside an ongoing research program on Hall of Mirrors dynamics and sycophantic convergence in multi-AI swarms; the contest surfaced on April 13, 2026, and the concentrated competitive sprint ran April 15–16 with a confirmation run completing in the early hours of April 17. Roughly 72 hours of active work on approximately $225 of direct contest compute on RunPod (L40S for sweeps at $6.90/hr; H100 SXM spot for confirmation runs at $14/hr). Single-seed H100 result; the legal champion was selected over a better-but-oversized 10-layer variant (1.1745 BPB at 17.1 MB) that could not be compressed under the 16 MB cap with the quantization techniques available to this team.

A fuller treatment of the methodology, including the swarm coordination pattern and the sycophancy patterns this sprint surfaced, is forthcoming at `unwindology.com/blog` under *"The Embryo That Learned to Golf."*

## Included Files

- `README.md` (this file)
- `submission.json`
- `train_gpt.py` — the exact training script, renamed from `train_breadcrumb_recur_ema_stochdepth.py`
- `train_seed1337.log` — full H100 training + eval log for seed 1337
