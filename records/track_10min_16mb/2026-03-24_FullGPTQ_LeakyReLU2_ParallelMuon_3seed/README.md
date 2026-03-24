# Full GPTQ + LeakyReLU² + Parallel Muon + BigramHash(3072,80)

**val_bpb: 1.1180** (3-seed mean, std 0.0010) | **~15.93 MB** mean | **8xH100 SXM, 600s** | No TTT

## Results

| Seed | step_avg | steps | Pre-quant bpb | Post-GPTQ sliding bpb | Total size (bytes) |
|------|----------|-------|---------------|------------------------|--------------------|
| 42   | 84.98ms  | 7062  | 1.1384        | **1.11752350**         | 15,890,732         |
| 1337 | 86.65ms  | 6925  | 1.1402        | 1.11918848             | 15,891,780         |
| 2024 | 85.20ms  | 7043  | 1.1383        | **1.11730894**         | 16,013,080         |
| **Mean** | **85.61ms** | **7010** | **1.1390** | **1.11800697** | **15,931,864** |

Std (sample): **0.00102882**

## Notes

- No TTT used.
- Full logs for all 3 seeds are included under this record folder.
- Independent 3-seed run of the Full GPTQ + LeakyReLU² + Parallel Muon direction.
