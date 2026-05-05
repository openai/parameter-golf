# Record: K_KVShare_Wider FLA (Opensens reproduction)

**val_bpb: 1.0339** (3-seed mean, std 0.0012) | **3.1434 nats** | **8xH100 SXM, 600s** | **No TTT**

Independent 3-seed reproduction of GatedDeltaNet K_KVShare_Wider, building on
PR #1687 (@resouer). Improved results (1.0339 vs 1.0409) likely due to hardware
variance (RunPod secure cloud, IN region).

## Results

| Seed | Steps | EMA BPB | **Quantized BPB** | Artifact |
|------|------:|--------:|------------------:|---------:|
| 42 | 1881 | 1.016763 | **1.03527246** | 15,927,295 |
| 1337 | 1890 | 1.013801 | **1.03326043** | 15,830,641 |
| 2025 | 1884 | 1.014923 | **1.03303636** | 15,893,661 |
| **Mean** | **1885** | **1.015162** | **1.03385760** | **15,883,866** |

## Technique

- GatedDeltaNet / Flash Linear Attention (`K_KVShare_Wider` config)
- 10 GDN layers, model_dim=544, 8 heads, head_dim=64
- KV sharing stride=2 (5 unique K/V sets for 10 layers)
- MLP mult=3.0, ReLU-squared, logit softcap=30
- BigramHash(3072, 112) + trigram embeddings
- SP8192 tokenizer (from kevclark/parameter-golf HF dataset)
- Muon optimizer (momentum 0.95, WD 0.04)
- EMA decay=0.997 + SWA every 50 steps
- Late QAT (Int6 STE when LR < 15% of peak)
- Int6 + zstd-22 artifact compression

Not used: no TTT, no SLOT, no n-gram overlay, no XSA eval.

## Reproducibility

```bash
pip install -r requirements.txt

# Download SP8192 data
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
# Or use snapshot_download from huggingface_hub

SEED=$SEED ARCH_MODE=K MAX_WALLCLOCK_SECONDS=600 VAL_LOSS_EVERY=0 EVAL_COMPILE_ENABLED=0 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Attribution

This submission reproduces and validates the architecture from PR #1687 by @resouer.
The GatedDeltaNet architecture is from Yang, Kautz & Hatamizadeh (NVIDIA, ICLR 2025).
Flash Linear Attention library by @sustcsonglin and @yzhangcs.
