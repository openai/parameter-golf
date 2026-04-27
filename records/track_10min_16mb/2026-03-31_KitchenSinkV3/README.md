# Record: Window Attention + Mixed Seq_Len Training

**val_bpb: 1.1108** (5-seed mean, std 0.0013) | **1.8755 nats** | **~15.73 MB** | 8xH100 SXM, 600s | No TTT

I started from [PR #1130](https://github.com/openai/parameter-golf/pull/1130) (KitchenSinkV2 Improved), which added split early/late LR banks, MiLe margin loss, cache+backout residual, residual lambdas, bigger bigram/VE, and FA3 on top of the PR #549 stack. On top of that, I ported the fused Triton MLP from [PR #1072](https://github.com/openai/parameter-golf/pull/1072) and the sigmoid-gated skips + brotli+byte-shuffle compression from [PR #1089](https://github.com/openai/parameter-golf/pull/1089). I also increased to 12 layers and tuned qk_gain to 2.5.

The two main contributions of this submission are window attention and mixed seq_len training, described below.

## Results (8xH100 80GB SXM, 600s, no TTT)

| Seed | Steps | ms/step | Post-EMA BPB | **Sliding BPB** | val_loss (nats) | Artifact |
|------|-------|---------|--------------|-----------------|-----------------|----------|
| 2 | 8,428 | 69.6 | 1.1250 | **1.1094** | 1.8731 | 15,726,762 |
| 1337 | 8,428 | 69.6 | 1.1250 | **1.1101** | 1.8742 | 15,721,698 |
| 42 | 8,428 | 69.6 | 1.1250 | **1.1103** | 1.8746 | 15,725,995 |
| 7 | 8,428 | 69.6 | 1.1250 | **1.1119** | 1.8773 | 15,723,346 |
| 22 | 8,428 | 69.6 | 1.1250 | **1.1126** | 1.8785 | 15,720,902 |
| **Mean** | | | | **1.1108** | **1.8755** | **15,723,741** |

Current merged SOTA ([2026-03-25 AR Self-Gen GPTQ + XSA-all + BigramHash 3072x112](https://github.com/openai/parameter-golf/blob/main/records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/README.md)): **1.11473 BPB**.
Delta vs current merged SOTA: **-0.0039 BPB** (**-0.0066 nats**).

## Window attention

Instead of full causal attention on every layer, layers 2, 4, 6, 8, and 10 use a sliding window of 512 tokens via Flash Attention 3's `window_size` parameter. The remaining layers (0, 1, 3, 5, 7, 9, 11) keep full attention.

The motivation was to enable training at longer sequence lengths without proportionally increasing compute. Full quadratic attention at seq_len=6144 is expensive, but with window attention on 5 of 12 layers, those layers run in O(n * w) instead of O(n^2), cutting the per-step cost significantly. The layers with full attention still give the model access to the full context.

I swept several configurations: window sizes (256, 512, 1024), which layers to window (sparse, dense, even), and how many layers. Window 512 on even-indexed layers was the sweet spot — enough layers windowed to get the speedup, enough full-attention layers to preserve long-range modeling.

At seq_len=2048 (where all tokens fit in a 512-wide window anyway for most positions), windowed attention adds a small overhead (~2-3%). The benefit kicks in at longer sequences: 15% faster at 4096, 21% at 6144, 25% at 8192.

## Mixed seq_len training

Different GPUs train with different sequence lengths within the same step. In the final configuration, 5 GPUs train at seq_len=2048 and 3 GPUs train at seq_len=6144. The number of sequences per GPU is set so that the total ms per step stays roughly constant.

The idea came from noticing that the sliding-window eval (which uses long sequences) gave substantially better scores than the standard 2048-token eval, but training at long sequence lengths was slow. By having most GPUs train cheaply at 2048 and a few GPUs see long context at 6144, the model gets the best of both: high step throughput from the short-sequence GPUs and long-range learning from the long-sequence ones.

I ran an extensive sweep of seq_len combinations. Some findings:

- **3x2048 + 1x6144** (eval at 6144) gave the best int6 roundtrip BPB (**1.1292**) in 4-GPU experiments, beating both pure 4x2048 (1.1417) and pure 4x6144 (1.1360)
- Having at least one GPU on a long sequence (4096+) was critical for good quantized performance
- More short-sequence GPUs = more steps in the same wallclock, which helps training loss
- More long-sequence GPUs = better post-EMA loss, but fewer steps and worse quantization
- 8192 was too slow to be worthwhile — the step-time penalty outweighed the context benefit

For the final 8-GPU submission, I used 5x2048 + 3x6144, which balances throughput and long-context exposure.

## Other changes

- **12 layers** (up from 11) with split early/late LR banks
- **Sigmoid-gated skip connections** — `x += sigmoid(gate) * skip` replaces learned scalar skip weights
- **Fused Triton MLP** (PR #1105) — LeakyReLU(0.5)-squared fused with matmuls
- **Brotli + byte-shuffle compression** (PR #1089) — better compression of quantized weights
- **Bigram hash 5120**, VE dim 128, qk_gain 2.5
- **Eval**: sliding window, seq_len=6144, stride=128

## Artifact size (worst-case, seed 2)

| Component | Bytes |
|-----------|-------|
| Model (int6+brotli) | 15,692,661 |
| Code | 34,101 |
| **Total** | **15,726,762** |

Under the 16,000,000 byte limit.

## Acknowledgments

This submission builds on many contributions from the parameter-golf community:

- **Baseline** (modded-nanogpt, @KellerJordan et al.) — Muon optimizer, relu², U-Net skips, softcap, RoPE, Q-gain, ResidMix
- [modded-nanogpt PR #140](https://github.com/KellerJordan/modded-nanogpt/pull/140) (@ClassicLarry / @snimu) — backout residual
- **PR #50** (@mattqlf) — sliding window eval
- **PR #64** (@yesbhautik) — GPTQ-lite (clip percentile search)
- **PR #89 / #95** (@vmfunc / @MatoTeziTanka) — EMA / SWA
- **PR #162** (@raahilshah) — BigramHash
- **PR #287** (@jfprincz) — XSA (cross-head subtracted attention; [arXiv:2603.09078](https://arxiv.org/abs/2603.09078))
- **PR #315** (@jfprincz) — partial RoPE, layerwise LN scale
- **PR #374** (@unnir) — value embeddings
- **PR #399** (@abaybektursun) — parallel Muon with parameter banking
- **PR #493** (@parinzee) — LeakyReLU(0.5)²
- **PR #535** (@raahilshah) — full Hessian GPTQ
- **PR #549** (@abaybektursun) — banked weight matrices, SmearGate
- **PR #726** (@DeepReinforce) — coprime-stride multi-shard data loader
- **PR #1072** (@vimeto) — fused Triton LeakyReLU-squared MLP kernel
- **PR #1089** (@mikeapedia) — sigmoid-gated skip connections, byte-shuffle + brotli compression
- Flash Attention 3 with `window_size` for efficient window attention

## Reproducibility

The main training runs used the following command:

```bash
SEED=$SEED \
MATRIX_LR=0.024 MATRIX_LR_LATE=0.019 \
SCALAR_LR=0.020 SCALAR_LR_LATE=0.038 \
TIED_EMBED_LR=0.022 \
MUON_MOMENTUM=0.985 WARMDOWN_ITERS=4000 \
TRAIN_BATCH_TOKENS=589824 \
NUM_LAYERS=12 BIGRAM_VOCAB_SIZE=5120 VE_DIM=128 \
WINDOW_SIZE=512 WINDOW_ATTN_LAYERS=2,4,6,8,10 \
LOCAL_SEQS_PER_GPU=36,36,36,36,36,10,10,10 \
SEQS_PER_GPU=2048,2048,2048,2048,2048,6144,6144,6144 \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

`brotli` needs to be installed for the final artifact compression path. Flash Attention 3 (`flash_attn_interface`) is required.
