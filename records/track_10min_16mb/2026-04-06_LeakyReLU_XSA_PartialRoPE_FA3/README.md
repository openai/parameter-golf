# LeakyReLU² + Partial RoPE + XSA + FA3

**val_bpb: 1.2092** (3-seed mean, std 0.0019, int6+lZMA roundtrip) | **~14.39 MB** | 8×H100 SXM

## Results (8×H100 80GB SXM, PyTorch 2.9.1+cu128)

The logs for this run do not use TTT. The relevant numbers are the final validation BPB at the wallclock stop and the exact post-export `final_int6_lzma_roundtrip` BPB.

| Seed | step_avg | steps | Final train-time bpb | **Final int6+lZMA bpb** | Export delta | Artifact |
|------|----------|-------|----------------------|-------------------------|--------------|----------|
| 42 | 50.44ms | 11,895 | 1.1987 | **1.2066** | +0.0079 | 14,387,504 |
| 1337 | 50.35ms | 11,915 | 1.1998 | **1.2102** | +0.0104 | 14,392,816 |
| 2024 | 50.78ms | 11,802 | 1.1989 | **1.2109** | +0.0120 | 14,388,932 |
| **Mean** | **50.52ms** | **11,871** | **1.1991 (std 0.0005)** | **1.2092 (std 0.0019)** | **+0.0101** | **14,389,751** |

## Core Changes vs `train_gpt.py`

This record file differs from the repo baseline `train_gpt.py` in five material ways:

1. The MLP activation is changed from `relu(x)^2` to `leaky_relu(x, 0.75)^2`.
2. Attention uses partial RoPE with `ROPE_DIMS=16`, so only 16 of 64 head dimensions receive rotary embedding.
3. XSA is enabled only on the deepest 4 layers via `XSA_LAST_N=4`.
4. Standard SDPA is replaced with FlashAttention-3 (`flash_attn_3_func`).
5. Export uses GPTQ-style Hessian-aware mixed int6 quantization with lZMA compression and selective pruning, instead of the baseline int8+zlib path.

## Innovations

```python
# Baseline
x = torch.relu(self.fc(x)).square()

# This experiment
x = F.leaky_relu(self.fc(x), 0.75).square()
```

Key contributions in this run:

- **LeakyReLU(0.75)²** replaces `relu²` in the MLP while keeping the same simple squared-activation structure.
- **Partial RoPE (`16/64`)** reduces rotary work per head and helps training throughput relative to full-head rotary application.
- **FlashAttention-3** improves attention kernel efficiency on Hopper, which helps both training speed and final BPB.
- **Deep-layer XSA** keeps the XSA intervention focused on the last 4 layers while preserving the training speed of the rest of the stack.
- **GPTQ-style mixed int6 + lZMA export** keeps the artifact comfortably under the size limit after roundtrip validation.

Architecture and export settings used in the logs:

| Component | Setting |
|-----------|---------|
| Layers | 11 (512d, 8H, 4KV) |
| MLP | 3× with **LeakyReLU(0.75)²** |
| Attention kernel | **FlashAttention-3** |
| XSA | Last 4 layers (`[7, 8, 9, 10]`) |
| RoPE | Partial (`16/64` dims per head) |
| Quantization | GPTQ-style mixed int6 + lZMA |
| Calibration | 64 autoregressive sequences, block size 128 |

Future work:

- Tune `matrix_lr` for the Muon optimizer.
- Reduce `XSA_LAST_N` and measure the BPB / throughput tradeoff.
- Try `SiLU` as the MLP activation.
- Try `LeakyReLU` again with negative slope `0.5`.
- Speed up Newton-Schulz with Gram Newton-Schulz from <https://github.com/Dao-AILab/gram-newton-schulz>.
- Try `BigramHashEmbedding`.

## Run Command

```bash
RUN_ID=train_seed_1337 \
DATA_PATH=./data/datasets/fineweb10B_sp1024/ \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
VOCAB_SIZE=1024 XSA_LAST_N=4 NUM_LAYERS=11 ROPE_DIMS=16 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **XSA on the last 4 layers**: [PR #549](https://github.com/openai/parameter-golf/pull/549)
- **GPTQ + int6 quantization stack**: [PR #1019](https://github.com/openai/parameter-golf/pull/1019)
- **Base model**: Naive Baseline by @0hq
