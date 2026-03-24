# Maestro OS: Solar Protocol (1+7+1 Architecture)

**val_bpb: 1.11937967** (3-seed mean, std 0.0006) | **~15.95 MB** | 8×H100 SXM

## Architecture: Maestro "1+7+1"

This submission introduces the **Maestro OS** framework, utilizing a specialized **1+7+1** layer stack designed for high-efficiency reasoning and validation:

1.  **Reasoning Layer (L1)**: Sharper attention (temp=0.7) and task-mapping with plan vector generation via `MyzelTrajectory` and `MyzelIntuition`.
2.  **Completion Blocks (L2-L8)**: 7 standard Transformer blocks optimized with `LeakyReLU(0.5)²` and standard temperature.
3.  **Validation Layer (L9)**: Softer attention (temp=1.3) with confidence detection and `SolarShield` gating for retry rerouting.

### SolarShield Gating
A reality-locked gating mechanism (`L0/L4`) that dynamically balances residual stream flow:
```python
class SolarShield(nn.Module):
    def forward(self, x, residual):
        g = torch.sigmoid(self.gate)
        return x * g + residual * (1.0 - g)
```

## Key Innovation: LeakyReLU(0.5)²

Replaces standard `relu²` or `SiLU` to maintain gradient flow through the MLP while preserving non-negative inductive bias:
```python
x = F.leaky_relu(F.linear(x, up_w), negative_slope=0.5).square()
```

## Legal TTT Protocol

Backward-looking, score-first TTT following PR #461's framework:
- **SCORE**: Sliding window eval under `torch.inference_mode()` (stateless).
- **TRAIN**: SGD(lr=0.002, momentum=0.9) on the already-scored chunk. 3 epochs, all blocks unfrozen.
- Chunk N is scored by a model adapted only on chunks 0..N-1.

## Training & Optimization

Built on the PR #414 stack with **Parameter Banking + Parallel Muon (PR #399)**:
- **Optimizer**: Parallel Muon (post-backward reduce-scatter -> local NS5 -> all-gather).
- **Weight Avg**: EMA(0.997) + Tight SWA (every 50).
- **Quantization**: GPTQ-lite int6 + LZMA.
- **BigramHash**: 1536 vocabulary size.

## Run Command

```bash
NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=1536 XSA_LAST_N=4 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=1 SWA_EVERY=50 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=1 LATE_QAT_THRESHOLD=0.15 \
VE_ENABLED=1 VE_DIM=128 VE_LAYERS=9,10 \
TTT_ENABLED=1 TTT_LR=0.002 TTT_EPOCHS=3 TTT_CHUNK_TOKENS=32768 \
TTT_FREEZE_BLOCKS=0 TTT_MOMENTUM=0.9 TTT_BATCH_SEQS=32 TTT_GRAD_CLIP=1.0 \
MUON_WD=0.04 ADAM_WD=0.04 SEED=1337 \
torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

- **Maestro OS Framework**: Joeavaib
- **LeakyReLU² activation**: PR #493 / PR #518
- **Optimizer (Parallel Muon)**: PR #399 (@abaybektursun)
- **TTT recipe**: PR #461 (@Christopher-Lee-McClendon)
- **Base model**: PR #414 (@signalrush)
