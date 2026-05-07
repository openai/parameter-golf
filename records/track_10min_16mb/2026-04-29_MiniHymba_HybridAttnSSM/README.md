# Mini-Hymba: Single Hybrid Attention/SSM Layer

**Author:** Aparna Sivanandam ([@aparna-1407](https://github.com/aparna-1407))  
**Track:** Non-record architecture experiment on the 16MB training stack  
**Best validated config:** `HYMBA_LAYERS=4`, `HYMBA_SCAN_CHUNK=64`, `sp1024`  
**val_bpb:** 1.4753 after int8+zlib roundtrip  
**Artifact size:** 9,234,838 bytes total, under the 16MB limit  
**Hardware:** 1x NVIDIA RTX PRO 6000 Blackwell  

---

## Summary

This submission adds a compact **Mini-Hymba** layer to `train_gpt.py`: one transformer attention block is replaced with a hybrid block that runs standard attention heads and Mamba-lite SSM heads in parallel, then concatenates their outputs before the output projection.

The best result so far uses only **one** hybrid layer, at layer 4. Earlier 3-layer experiments trained stably, but the single-layer version learned much faster under the wallclock budget and produced the best BPB.

The implementation deliberately leaves the challenge scorer, tokenizer-aware BPB calculation, optimizer, quantization/export path, and artifact compression path unchanged. The only architectural change is the attention module swap.

---

## Architecture

### Mini-Hymba Layer

For one selected layer:

```text
Input x (B, T, d_model)
  |-- Attention pathway
  |     4 attention heads
  |     4 learnable meta tokens prepended
  |     host RMS norm + RoPE
  |     QK-Gain
  |     GQA with n_attn=4, n_kv=4, rep=1
  |
  |-- SSM pathway
        4 Mamba-lite SSM heads
        diagonal A
        input-dependent B, C, and dt
        signed chunk-parallel fp32 scan

concat(attn_out, ssm_out) -> output projection
```

The `4 attention + 4 SSM` split avoids the GQA mismatch that occurs with a 6-attention / 2-SSM split when `num_kv_heads=4`.

### Chunk-Parallel SSM Scan

The first stable implementation used a sequential fp32 scan over all 1024 tokens. That proved the architecture, but it was slow: about 6.7 seconds per step for the 3-layer variant.

The current implementation uses a signed chunk-parallel scan:

```text
h_t = a_t * h_{t-1} + u_t
p_t = prod(a_0 ... a_t)
h_t = p_t * cumsum(u_i / p_i)
```

It computes this exactly within chunks and carries the final state between chunks. This keeps signed updates intact and reduces Python loop overhead from 1024 token steps to 16 chunk steps at `HYMBA_SCAN_CHUNK=64`.

### KV Sharing

For multi-layer Hymba experiments, adjacent Hymba layers can share K/V projections. In the best one-layer result this is not active, but the code supports it by omitting K/V projection modules from sharing layers so unused tensors do not enter `state_dict`.

---

## Results

Best run:

```text
HYMBA_ENABLED=1
HYMBA_LAYERS=4
HYMBA_SCAN_CHUNK=64
WARMUP_STEPS=100
ITERATIONS=800
MAX_WALLCLOCK_SECONDS=1800
MATRIX_LR=0.004
SCALAR_LR=0.004
EMBED_LR=0.005
TIED_EMBED_LR=0.005
```

| Metric | Value |
|---|---:|
| Final unquantized val_loss | 2.4501 |
| Final unquantized val_bpb | 1.4511 |
| Final int8+zlib roundtrip val_loss | 2.4910 |
| Final int8+zlib roundtrip val_bpb | **1.4753** |
| Training steps | 800 |
| Training time | 877.4 seconds |
| Step time | ~1.097 seconds |
| Peak allocated memory | 14,145 MiB |
| Total int8+zlib submission size | 9,234,838 bytes |

Validation trajectory:

```text
step   0: val_bpb 4.1077
step 200: val_bpb 1.9420
step 400: val_bpb 1.6016
step 600: val_bpb 1.5044
step 800: val_bpb 1.4511
roundtrip: val_bpb 1.4753
```

The curve is smooth and stable, with a modest quantization penalty of about `+0.024 BPB`.

---

## Comparison To Earlier 3-Layer Probe

The original experiment hybridized layers 3, 4, and 5. It trained stably, but the sequential SSM scan made it too slow for the wallclock budget:

```text
3-layer sequential probe:
  step 134 roundtrip val_bpb: 3.0721
  step time: ~6.7s

1-layer chunk-parallel probe:
  step 800 roundtrip val_bpb: 1.4753
  step time: ~1.1s
```

The one-layer version is the recommended configuration for future full runs.

---

## Run Command

```bash
HYMBA_ENABLED=1 \
HYMBA_LAYERS=4 \
HYMBA_SCAN_CHUNK=64 \
WARMUP_STEPS=100 \
ITERATIONS=800 \
MAX_WALLCLOCK_SECONDS=1800 \
MATRIX_LR=0.004 \
SCALAR_LR=0.004 \
EMBED_LR=0.005 \
TIED_EMBED_LR=0.005 \
TRAIN_LOG_EVERY=50 \
VAL_LOSS_EVERY=200 \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
python -u train_gpt.py 2>&1 | tee train_log.txt
```

`hymba_layer.py` must be in the same directory as `train_gpt.py`.

---

## Implementation Notes

- `hymba_layer.py` registers the host script's `CastedLinear`, `Rotary`, and `apply_rotary_emb`, so the hybrid layer uses the same numerical path as `CausalSelfAttention`.
- `torch.compile` is disabled automatically when `HYMBA_ENABLED=1`, because Inductor currently fails on the SSM scan. The chunk-parallel eager scan is still fast enough for this non-record run.
- The run uses `sp1024`. A future 8xH100 run with `sp8192` is the natural next test.
- The result is non-record because it was run on a single GPU for longer than the 10-minute record setting, but the artifact is under 16MB and the code uses the standard scorer/export path.

---

## References

1. **Hymba: A Hybrid-head Architecture for Small Language Models**  
   Xin Dong, Yonggan Fu, ..., Yingyan Lin, Jan Kautz, Pavlo Molchanov  
   ICLR 2025 — [arXiv:2411.13676](https://arxiv.org/abs/2411.13676)

2. **Mamba: Linear-Time Sequence Modeling with Selective State Spaces**  
   Albert Gu, Tri Dao — [arXiv:2312.00752](https://arxiv.org/abs/2312.00752)

3. **CPT: Efficient Deep Neural Network Training via Cyclic Precision**  
   Yingyan Lin et al., Georgia Tech EIC Lab  
   ICLR 2021 Spotlight — [arXiv:2101.09868](https://arxiv.org/abs/2101.09868)
