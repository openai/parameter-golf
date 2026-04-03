## <ModelName>

<One sentence blurb matching submission.json blurb>

## Results

| Seed | val_bpb (sliding window) | Steps | Size |
|------|--------------------------|-------|------|
| 444  | <val_bpb_exact>          | <steps> | <bytes_total> B |
| 300  | <val_bpb_exact>          | <steps> | <bytes_total> B |
| **mean** | **<mean_bpb>**       |       | **<max_bytes> B** |

Hardware: 8×H100 SXM · 600s wallclock · `bytes_code`: <bytes_code>

## Architecture changes

- <What changed vs the prior submission — ONE thing>

## Reproduce

```bash
# From repo root, with flash-attention/hopper on PYTHONPATH
SKIP_GPTQ=1 SEED=444 torchrun --standalone --nproc_per_node=8 \
  records/track_10min_16mb/<records_folder>/train_gpt.py
```
