# Submission: SP8192 CaseOps + WiderGate32 + PolarNS Muon + GPTQ-int6

**val_bpb: 1.08037** (3-seed mean, std 0.00139) | **~15.9 MB** | 8×H100 SXM, 600s wallclock | TTT eval

## Results

| Seed | Pre-quant val_bpb | Post-quant val_bpb | **Post-TTT val_bpb** | Artifact |
|------|-------------------|--------------------|----------------------|----------|
| 0    | 1.07175           | 1.09419            | **1.08196**          | 15,890,131 |
| 42   | 1.07039           | 1.09076            | **1.07983**          | 15,887,137 |
| 1234 | 1.06982           | 1.09058            | **1.07932**          | 15,888,516 |
| **Mean** |                |                    | **1.08037**          | 15,888,595 |

## Architecture

| Component | Setting | Source |
|-----------|---------|--------|
| Layers | 11 (512d, 8 GQA heads, 4 KV heads) | Baseline |
| MLP | 4× (2048) with LeakyReLU(0.5)² | [#493](https://github.com/openai/parameter-golf/pull/493) |
| Attention | FA3, GQA 2:1 | Baseline |
| RoPE | Partial (16/64 dims), base 10000 | [#315](https://github.com/openai/parameter-golf/pull/315) |
| U-Net skips | Encoder-decoder skip connections + skip gates | [#289](https://github.com/openai/parameter-golf/pull/289) |
| Parallel decoder | 2-lane parallel from layer 8+ | [#1530](https://github.com/openai/parameter-golf/pull/1530) |
| Depth recurrence | Loop layers 3-5, NUM_LOOPS=2 (17 virtual layers) | [#1344](https://github.com/openai/parameter-golf/pull/1344) |
| Logit softcap | 30 | Baseline |
| **Wider AttnOutGate** | Per-head output gate, **GATE_WIDTH=32** (vs standard 12) | [#1787](https://github.com/openai/parameter-golf/pull/1787) + **this work** |
| **SmearGate** | Position-mixing gate, width=32 | [#1667](https://github.com/openai/parameter-golf/pull/1667) |
| **Polar-Express Muon** | 5 NS steps, per-iter minimax tuples, momentum 0.97 | [#1344](https://github.com/openai/parameter-golf/pull/1344) |
| **MIN_LR floor** | 0.10 (warmdown LR floor) | [#1787](https://github.com/openai/parameter-golf/pull/1787) |
| Quantization | GPTQ int6 all weights (EMBED_BITS=6) + brotli-11 | |
| TTT | LoRA rank-96, 1 phase, 2000 prefix docs | [#1610](https://github.com/openai/parameter-golf/pull/1610) |
| Tokenizer | SP8192 CaseOps (bijective case markers) | [#1729](https://github.com/openai/parameter-golf/pull/1729) |

## Key Innovation: Wider Attention Output Gates

Standard AttnOutGate (PR #1787) uses 12 input dimensions from the residual stream to compute per-head gating:

```python
gate_in = x_orig[:, :, :12]  # standard: 12 dims
gate = 2.0 * sigmoid(linear(gate_in, gate_w))  # -> per-head scalar
y = attn_output * gate
```

We widen the gate input to 32 dimensions (`GATE_WIDTH=32`), giving each head a richer view:

```python
gate_in = x_orig[:, :, :gate_w.shape[-1]]  # wider: 32 dims
```

- Gate params per layer: 32 × 8 heads = 256 (vs 96 with width=12)
- Total extra params: 1,760 across 11 layers (float16 passthrough, negligible)
- **Pre-quant improvement: −0.002 BPB** vs width=12

The same widening is applied to SmearGate for consistency.

## Training Configuration

```bash
VOCAB_SIZE=8192
DATA_PATH=./data/datasets/fineweb10B_sp8192_caseops
TOKENIZER_PATH=./data/tokenizers/fineweb_8192_bpe_lossless_caps_caseops_v1_reserved.model
MAX_WALLCLOCK_SECONDS=600
POLAR_EXPRESS_NS=1
LQER_ENABLED=0
MIN_LR=0.10
EMBED_BITS=6
COMPRESSOR=brotli
ATTN_OUT_GATE=1
SMEAR_GATE=1
GATE_WIDTH=32
```

## Reproduction

```bash
pip install torch>=2.9.0 sentencepiece brotli triton
python prepare_caseops_data.py
torchrun --standalone --nproc_per_node=8 train_gpt.py
```
