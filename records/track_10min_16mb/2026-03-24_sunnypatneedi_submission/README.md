## Record: PLACEHOLDER_TECHNIQUE_NAME

**val_bpb: PLACEHOLDER** (3-seed mean) | **PLACEHOLDER MB** artifact | 8xH100 SXM, 600s

### Results (3 seeds, 8xH100 SXM)

| Seed | Steps | Sliding BPB (s64) | Artifact |
|------|-------|-------------------|----------|
| 42   | XXXX  | X.XXXX            | XX.XX MB |
| 1337 | XXXX  | X.XXXX            | XX.XX MB |
| 2024 | XXXX  | X.XXXX            | XX.XX MB |

**Mean: X.XXXX | Std: X.XXXX**

### Key Innovations

PLACEHOLDER — describe what's new vs prior SOTA.

### Architecture

- 11 layers, 512 dim, 8 heads / 4 KV heads (GQA)
- PLACEHOLDER — list all components

### Training Configuration

- PLACEHOLDER — optimizer, LR, batch size, warmdown

### Quantization

- PLACEHOLDER — int5/int6, GPTQ-lite, zstd-22

### Run Command

```bash
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

### Provenance

Built on PR #414 (signalrush, merged SOTA 1.1228). Key additions from:
- PLACEHOLDER — list PRs and papers we build on

### Test Plan

- [ ] 3 seeds run on 8xH100 SXM
- [ ] All 3 seeds train in <=600s
- [ ] All 3 seeds artifact <=16,000,000 bytes
- [ ] Sliding window eval (stride=64) consistent
