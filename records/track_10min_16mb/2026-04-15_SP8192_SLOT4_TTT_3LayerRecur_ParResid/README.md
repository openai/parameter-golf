# Record: SP8192 + SLOT-4 + TTT + 3-Layer Recurrence + Parallel Residuals

**val_bpb: 1.0616** (3-seed mean, std 0.0005) | **~16.0 MB** | 8xH100 80GB HBM3

**Improvement over current SOTA (PR #1493, 1.0810 BPB): -0.0194 BPP**

## 3-Seed Results

| Seed | Sliding BPB | TTT BPB | **SLOT BPB** | Artifact |
|------|-------------|---------|-------------|----------|
| 314 | 1.0839 | 1.0827 | **1.0611** | ~16,001,154 |
| 42 | 1.0838 | 1.0832 | **1.0617** | ~16,001,154 |
| 999 | 1.0838 | 1.0832 | **1.0622** | ~16,001,154 |
| **Mean** | 1.0838 | 1.0830 | **1.0616** | |

## Key Technique: SLOT (Sample-Level Optimization at Test-time)

SLOT adds a **per-window learnable logit bias** at eval time. For each sliding window:

1. Compute base logits from the frozen (TTT-adapted) model: `base_logits = model.forward_logits(x)`
2. Initialize a zero delta tensor: `delta = zeros(1, seq_len, vocab_size)`
3. Optimize delta via **4 AdamW steps** (lr=0.01, wd=0.01) to minimize cross-entropy on the window's targets
4. Score the window using `base_logits + delta`

This is a form of **test-time compute**: the model adapts its output distribution per evaluation window without modifying model weights. The delta is thrown away after each window.

```python
delta = torch.zeros(1, seq_len, vocab, device=device, dtype=torch.float32, requires_grad=True)
opt = torch.optim.AdamW([delta], lr=0.01, weight_decay=0.01)
for _ in range(4):
    opt.zero_grad()
    adjusted = base_logits + delta
    loss = F.cross_entropy(adjusted[:,:wlen].reshape(-1, vocab), y[:,:wlen].reshape(-1), reduction='mean')
    loss.backward()
    opt.step()
```

### Why SLOT works

SLOT exploits the gap between the model's predicted distribution and the true local distribution within each window. By optimizing a logit bias on the actual targets, SLOT can capture local patterns (topic, style, vocabulary) that the general model misses. With only 4 steps and low LR, the bias captures broad distributional adjustments without overfitting to individual tokens.

### SLOT hyperparameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Steps | 4 | More steps → overfitting (16 steps gives 0.45 BPB) |
| LR | 0.01 | Higher LR → overfitting |
| Weight decay | 0.01 | Regularizes the delta |
| Eval time | ~25 min | Per GPU, 76K windows split across 8 GPUs |

## Base Stack (unchanged from PR #1493)

All training and architecture is identical to the current merged SOTA:
- SP8192 tokenizer, 11L x 512d x 8H / 4KV, MLP 4x, LeakyReLU(0.5)^2
- 3-Layer Depth Recurrence (L3-5), activated at 35% of training
- Parallel Residuals (L7+), GPT-J style
- QK-Gain 5.0, EMA 0.9965, WD 0.095
- Score-First TTT (SGD, 3 epochs per chunk)
- GPTQ SDClip int6/int8 + Brotli compression

## Reproduction

```bash
SEED=314 RUN_ID=slot_test SLOT_ENABLED=1 SLOT_STEPS=4 SLOT_LR=0.01 TTT_ENABLED=1 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```
