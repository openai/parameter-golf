# SP8192 + Gram-NS + Polar Express + 3-Layer Recurrence + Parallel Residuals + QK-Gain 5.25 + Legal TTT

val_bpb = **1.0800** (3-seed mean, std 0.0003) | ~16.02 MB | 8xH100 SXM

## 3-Seed Results


| Seed     | Sliding BPB | TTT BPB    | Artifact (bytes) |
| -------- | ----------- | ---------- | ---------------- |
| 42       | 1.0811      | 1.0796     | 16,024,793       |
| 314      | 1.0816      | 1.0802     | 16,024,488       |
| 999      | 1.0814      | 1.0801     | 16,024,128       |
| **Mean** | **1.0814**  | **1.0800** | **16,024,470**   |
| Std      | 0.0003      | 0.0003     |                  |


Merged SOTA (PR #1493): 1.0810 BPB. Delta: **-0.0010 BPB**.

## Key Techniques

**Gram-NS** -- For rectangular MLP weight matrices (aspect ratio >= 1.5), iterates on the small *nxn* Gram matrix **R = [X@X.T](mailto:X@X.T)** instead of the full *nxm* matrix, reducing Newton-Schulz FLOP cost by ~22% on MLP weights without changing the theoretical fixed point. Square-ish matrices (aspect < 1.5) use the standard NS path unchanged. Based on Zhang, Amsel, Chen & Dao (Dao AI Lab, 2026), who show up to 55% FLOP reduction at equivalent training quality. Not applied in any other competition PR.

**Polar Express coefficients** -- Replaces fixed (a, b, c) = (3.4445, -4.775, 2.0315) with per-iteration optimal minimax tuples from You Jiacheng (arXiv:2505.16932, ICLR 2026). Each NS iteration uses the coefficients minimising the Chebyshev error for its specific position in the step sequence, giving a tighter polynomial approximation to the sign function at every step.

**4 NS steps + extended training budget** -- Reducing NS steps from 5 to 4 saves ~20% optimizer time per step. Combined with Gram-NS, this recovers ~150 additional gradient steps within the 600s wall-clock budget (4700 steps vs 4550 in PR #1493), with optimizer quality maintained through the tighter Polar Express coefficients.

**Recovered GPTQ budget** -- Setting `gptq_reserve_seconds=0.5` (vs the 12.0 default) recovers 11.5s of training time that would otherwise sit idle before quantization.

**SP8192 + GPTQ SDClip** -- int6 matrices (k=12.85), int8 embeddings (k=20.0), zero selective pruning needed (PR #1394 @clarkkev)

**3-Layer Depth Recurrence** -- layers 3, 4, 5 looped twice, activating at frac=0.35. Encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] -- 17 virtual layers from 11 physical (PR #1331, #1437 @dexhunter)

**Parallel Residuals** -- layers 7+, GPT-J style: attention and MLP read from the same pre-residual input (PR #1412 @Robby955, PR #1204 @msisovic)

**QK-Gain 5.25** -- learnable per-head query scaling, monotonic improvement from 4.0 to 5.25 (PR #1493 @bigbag)

**Legal Score-First TTT** -- SGD (lr=0.005, momentum=0.9), 3 epochs per 32K-token chunk, cosine LR decay across chunks, score-before-update ordering (PR #549 @abaybektursun, PR #1413 @dexhunter)

**Tuned Hyperparameters** -- WD=0.095, MLR=0.022, EMA=0.9965, warmdown=0.72, min_lr=0.1 (PR #1445, #1471 @X-Abhishek-X)

## Architecture

11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2, Partial RoPE (16/64 dims), layerwise LN scale, tied embeddings, logit softcap=30.0. Depth recurrence encoder [0,1,2,3,4,5,3,4] decoder [5,3,4,5,6,7,8,9,10] (loops layers 3-5, activates at step ~2076, frac=0.35). Parallel residuals from layer 7: attention and MLP operate on same pre-residual input. Skip gates (sigmoid-gated U-Net connections).

## Training

Muon with Gram-NS + Polar Express dispatch (4 NS steps), AdamW for embeddings/scalars. ~4700 steps in ~599.5s on 8xH100 SXM. Linear warmdown to min_lr=0.1 over final 72% of training. EMA decay 0.9965.

## Quantization

Full-Hessian GPTQ with SDClip: clip = k*sigma per row for principled rate-distortion. int6 for attention/MLP matrices, int8 for token embeddings. Byte-shuffle + Brotli-11 compression. Zero selective pruning -- model fits natively under 16MB.

## TTT (Test-Time Training)

Score-first, chunk-based SGD adaptation at eval time:

1. Chunk val tokens into 32K-token chunks (1238 chunks total)
2. Per chunk: **(1)** score all sliding windows under `torch.no_grad()`, **(2)** train model on scored chunk with SGD
3. 3 epochs per chunk, cosine LR decay across chunks
4. Gradient clipping at 1.0, distributed all-reduce across 8 GPUs

Total TTT eval time: ~370s (within 600s eval budget).

## Compliance

Per Issue #1017 (Track B -- legal eval-time adaptation):

- **Causality**: Sliding-window eval is strictly causal. Each position scored from prefix tokens only.
- **Normalized distribution**: Standard softmax over full vocab. No logit biasing, no n-gram cache.
- **Score before update**: Each chunk fully scored under `torch.no_grad()` before any SGD update. Training only on already-scored tokens.
- **Single pass**: Each token scored exactly once. No rescoring, no multi-pass selection.

Additional:

- No SLOT, no pre-quant TTT on val data, no ETLB, no n-gram tilt
- All artifacts under 16,777,216 bytes on all 3 seeds (largest: 16,024,793)
- Training under 600s on all 3 seeds (~599.5s actual)
- Eval (sliding + TTT) under 600s on all 3 seeds (~495s actual)

## Reproduction

```bash
pip install brotli sentencepiece
pip install flash_attn_3 --no-deps \
  --find-links https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/

MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf \
  python3 data/cached_challenge_fineweb.py --variant sp8192

SEED=42 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
  MIN_LR=0.1 GPTQ_RESERVE_SECONDS=0.5 DATA_DIR=./data \
  PYTORCH_ALLOC_CONF=expandable_segments:True \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Replace `SEED=42` with `SEED=314` and `SEED=999` for the other two seeds.

## Credits

- **@clarkkev** -- SP8192 + GPTQ Embeddings + SDClip + MuonEq-R + depth recurrence base (PR #1394)
- **@dexhunter** -- 3-layer depth recurrence (PR #1331, PR #1437), legal score-first TTT on SP8192 (PR #1413)
- **@abaybektursun** -- Score-first TTT framework (PR #549)
- **@Robby955** -- Parallel residuals on SP8192 (PR #1412)
- **@msisovic** -- Parallel residuals concept (PR #1204)
- **@X-Abhishek-X** -- Hyperparameter tuning: WD=0.095, MLR=0.022, EMA=0.9965 (PR #1445, PR #1471)
- **@bigbag** -- QK-Gain 5.25, full stack integration (PR #1493)
- **Zhang, Amsel, Chen & Dao (Dao AI Lab, 2026)** -- Gram-Newton-Schulz algorithm ([dao-lab.ai/blog/2026/gram-newton-schulz](https://dao-lab.ai/blog/2026/gram-newton-schulz/))
- **You Jiacheng (arXiv:2505.16932, ICLR 2026)** -- Polar Express per-iteration minimax Newton-Schulz coefficients

## Included Files

- `README.md` (this file)
- `train_gpt.py`
- `train_seed42.log`
- `train_seed314.log`
- `train_seed999.log`

## Acknowledgements

Thanks to OpenAI for running this challenge -- it's a genuinely fun format and I learned a huge amount about optimizer math and quantization I wouldn't have touched otherwise.

Big thanks to the Gram-Newton-Schulz team (Zhang, Amsel, Chen & Dao) and You Jiacheng for publishing their work openly. Building on solid, recent research made a real difference here.

And thanks to everyone in the parameter golf community who shared PRs and kept the leaderboard moving -- @clarkkev, @dexhunter, @abaybektursun, @Robby955, @msisovic, @X-Abhishek-X, @bigbag, and all the others. The collaborative spirit here is something special.

I'm an undergrad at Georgia Tech and have been funding my own compute to participate. I've submitted a compute credit request through OpenAI's official form and would be really grateful if it comes through -- there's a lot more I'd love to explore with this. Thank you for making this kind of research possible.

