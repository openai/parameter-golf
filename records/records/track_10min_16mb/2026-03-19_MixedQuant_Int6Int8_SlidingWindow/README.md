Four orthogonal improvements over the naive baseline, each contributing independently to the final score.

### Changes from Baseline

**1. Wider MLP (MLP\_MULT=3)**

The baseline uses a 2x MLP expansion (hidden=1024). We widen to 3x (hidden=1536), increasing total parameters from 17.1M to 21.8M. The wider MLP is enabled by the int6 quantization scheme below which keeps the artifact under 16MB.

**2. Mixed-Precision Post-Training Quantization**

Key insight: during training, all `CastedLinear` weights get **fake int6 quantization via Straight-Through Estimator (STE)** — the forward pass uses quantized weights while gradients flow through the originals. This teaches the model weight distributions that survive int6 (31-level) quantization. However, the token embedding (`tok\_emb.weight`) is a plain `nn.Embedding` that **never sees fake quantization during training**.

Previous approaches applied uniform int6 to all 2D tensors, causing a +0.048 BPB quantization penalty dominated by embedding degradation. Our mixed scheme:

* **int6 per-row** (31 levels) on all 2D block weights (attention projections, MLP layers) — these have STE protection
* **int8 per-row** (127 levels) on the token embedding — no STE protection, needs gentler quantization
* Small/control tensors pass through as fp16/fp32

This reduces the quantization penalty from +0.048 to +0.0015 BPB — a 32x improvement. The int6 values are stored in int8 containers; zlib-9 compresses the zero high bits efficiently.

**3. Optimized Training Configuration**

* `TRAIN\_SEQ\_LEN=1024` (down from 4096): Attention is O(N²) in sequence length. Shorter sequences = faster steps (48.4ms vs 55.5ms) = more total training in the 10-minute window. The 512-dim model cannot meaningfully exploit 4K context.
* `TRAIN\_BATCH\_TOKENS=524,288` (up from 393,216): Better GPU saturation at seq\_len=1024, \~33% more tokens per step.
* Result: 12,395 steps × 524K tokens = \~6.50B total tokens (vs \~4.25B with the old config).

**4. Sliding Window Evaluation (stride=64)**

Instead of non-overlapping evaluation where early tokens in each chunk get minimal context, we use overlapping windows advanced by 64 tokens. Each window runs the full 1024-token forward pass, but only the last 64 tokens are scored. Every scored token gets 960 tokens of preceding context.

Sliding window eval improves val\_bpb by \~0.034 with zero artifact cost. stride=64 gives more context per token than stride=256 (960 vs 768), at the cost of longer eval time (\~73s vs \~18s).

### Configuration

```
MLP\_MULT=3
NUM\_LAYERS=9
MODEL\_DIM=512
NUM\_HEADS=8
NUM\_KV\_HEADS=4
VOCAB\_SIZE=1024
TRAIN\_SEQ\_LEN=1024
TRAIN\_BATCH\_TOKENS=524288
TIE\_EMBEDDINGS=1
EVAL\_STRIDE=64
```

Optimizer settings (tuned via env vars, no code changes from baseline optimizer structure):

```
MATRIX\_LR=0.020
SCALAR\_LR=0.020
TIED\_EMBED\_LR=0.030
MUON\_MOMENTUM=0.99
MUON\_MOMENTUM\_WARMUP\_STEPS=1500
MUON\_MOMENTUM\_WARMUP\_START=0.92
WARMDOWN\_ITERS=3000
```

### Run Command

```bash
RUN\_ID=v2\_int6\_qat\_mlp3 \\
MAX\_WALLCLOCK\_SECONDS=600 \\
VAL\_LOSS\_EVERY=2000 \\
TRAIN\_LOG\_EVERY=200 \\
torchrun --standalone --nproc\_per\_node=8 train\_gpt.py
```

### Key Metrics

* Training stopped at **12,395/20,000** steps due to 10-minute wallclock cap
* Step time: **48.41ms** average on 8xH100 SXM
* Total train tokens: \~6,499,880,000 (12,395 steps × 524,288 tokens/step)
* Peak memory: **11,251 MiB** allocated per GPU

|Metric|Value|
|-|-|
|Pre-quant val\_bpb (last step)|1.1950|
|int6/int8 mixed roundtrip val\_bpb (standard)|1.1965|
|**int6/int8 mixed roundtrip val\_bpb (sliding, stride=64)**|**1.1630**|
|Quantization penalty (standard eval)|+0.0015 BPB|
|Sliding window eval time|72.6s|
|Compressed artifact (int6+zlib-9)|15,296,720 bytes|
|Code size|56,770 bytes|
|**Total submission size**|**15,353,490 bytes**|

### Improvement Breakdown

|Component|val\_bpb|Improvement vs baseline|
|-|-|-|
|Naive baseline (int8, standard eval)|1.2244|—|
|+ Wider MLP 3x + seq1024 + 524K batch|1.1950|-0.0294|
|+ Mixed quant (int6 blocks, int8 embed)|1.1965|+0.0015 penalty|
|+ Sliding window stride=64|**1.1630**|-0.0335 additional|
|**Total improvement**||**-0.0614**|

### Included Files

* `train\_gpt.py` — full training + mixed quantization + evaluation script
* `train.log` — complete training log from the 8xH100 run
* `submission.json` — leaderboard metadata
* `README.md` — this file

