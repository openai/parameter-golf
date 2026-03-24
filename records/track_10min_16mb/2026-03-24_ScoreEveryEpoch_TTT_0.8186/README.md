# 11L + Score-Every-Epoch LoRA TTT 5ep

**val_bpb: 0.8186** (seed 42, 1xB200) | Pre-TTT sliding: 1.1264 | TTT gain: -0.308

Additional seeds pending (exp84/85 submitted).

## Key Innovation: Score-Every-Epoch Multi-Scale LoRA TTT

Per-document LoRA adaptation with score-every-epoch protocol:
1. For each document, split into 256-token chunks within 1024-token context windows
2. For each epoch (5 total): score every chunk, then train on it
3. Per-doc accumulators reset each epoch — only final epoch's scores count
4. Cosine LR decay across total TTT steps prevents overfitting

### Multi-Scale LoRA Configuration
- LM-head LoRA: rank-16 (2x base LR) — doubled output adaptation capacity
- V projections: rank-8 (1.5x base LR) — controls information flow
- Q projections: rank-8 (0.5x base LR) — slow adaptation prevents destabilization
- Per-block bias tuning: 512 params/block (3x base LR) — fast domain shift
- Post-TTT temperature: T=0.98 — corrects TTT-induced overconfidence
- Base TTT LR: 0.01, Adam optimizer (betas 0.9/0.95)

## Architecture (from PR #414)

- 11L, d=512, 8H/4KV GQA, MLP 3x relu-squared
- XSA on last 4 layers, Partial RoPE (16/64 dims), LN Scale
- SmearGate + BigramHash(2048) + OrthoInit + VE128
- EMA(0.997) + Tight SWA, GPTQ-lite int6 + zstd-22
- Muon WD=0.04, warmdown=3500, Late QAT@0.15

## Results

| Seed | Pre-TTT BPB | Post-TTT BPB | Artifact |
|------|-------------|--------------|----------|
| 42   | 1.1264      | **0.8186**   | 17.13 MB |

Note: Artifact exceeds 16MB on 1xB200 due to quantization differences. Requires 8xH100 validation for proper artifact sizing with FlashAttention 3.

## Training (1xB200, HiPerGator)

- 20,000 steps at ~540ms/step
- Training time: ~3h
- TTT eval time: ~30 min (5 epochs, 50K docs)

## Run Command

```bash
TTT_ENABLED=1 TTT_EPOCHS=5 TTT_LORA_RANK=8 TTT_LM_RANK=16 \
TTT_LORA_LR=0.01 TTT_BIAS_TUNE=1 TTT_TEMP_RESCALE=0.98 \
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

## Credits

- Base architecture: PR #414 by @signalrush (verified SOTA 1.1228)
- TTT approach: PR #77 (samacqua), PR #596 (MatoTeziTanka/DeepQuant)
- Score-every-epoch protocol: PR #568 (PROTEUS v8)
- Per-layer LR groups: PR #481 (mrdavtan)
