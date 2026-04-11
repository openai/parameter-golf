# Record: SP8192 + VarLen Attention + Doc-Independent LoRA TTT + Fused MLP — val_bpb 1.0777 (3-seed mean)

**val_bpb = 1.0777** (3-seed mean, std 0.0003) | **~15.99 MB** | 8xH100 SXM

## 3-Seed Results

| Seed | **LoRA TTT BPB** | val_loss (nats) | Artifact |
|------|------------------|-----------------|----------|
| 42   | **1.0775**       | 2.7834          | 15,991,008 |
| 314  | **1.0776**       | 2.7834          | 15,993,539 |
| 999  | **1.0780**       | 2.7845          | 15,991,008 |
| **Mean** | **1.0777**   | **2.7838**      | |

Merged SOTA (PR #1493): **1.0810 BPB / 2.7920 nats**. Delta: **-0.0082 nats**. Clears 0.005 threshold by 0.0032.

## Novel Contribution: Fused Triton MLP via importlib Wrapper

This submission integrates the Triton TMA fused MLP kernel from PR #1523 into PR #1536's VarLen + LoRA TTT stack using a novel **importlib-based code loader**. Triton's @jit requires source files accessible via `inspect.getsourcelines()`, which fails with standard `exec()` from compressed wrappers. Our solution:

```python
_s = importlib.util.spec_from_file_location('__main__', temp_file)
_m = importlib.util.module_from_spec(_s)
_m.__name__ = '__main__'
_s.loader.exec_module(_m)
```

This writes the decompressed code to a temp file and loads it as a proper Python module, enabling Triton JIT compilation while keeping the submission wrapper at ~30KB.

## Full Stack

1. **VarLen Attention** — Flash Attention 3 `flash_attn_varlen_func` with document boundaries (PR #1536 @dexhunter, PR #1530 @samacqua)
2. **Doc-Independent LoRA TTT** — LoRA adapters (rank 96) trained per-document during score-first eval, no inter-document dependence (PR #1536 @dexhunter)
3. **Fused Triton TMA MLP** — `fc → LeakyReLU(0.5) → square` in one kernel, +5% throughput (PR #1523 @abaybektursun)
4. **Triple Depth Recurrence** (L3-5, 17 virtual layers) + **Parallel Residuals** (L7+)
5. **Parameter Banking** + **Muon 0.97** + **QK-Gain 5.25** + **SDClip** + **Brotli**

## Compliance (Track B — Score-First LoRA TTT)

- LoRA TTT: each document scored BEFORE LoRA weight update. No inter-document dependence.
- No SLOT, no hash embedding, no pre-quant TTT, no n-gram cache, no ETLB
- All four conditions from Issue #1017 satisfied
- All artifacts < 16MB, train < 600s, eval < 600s

## Reproduction

```bash
pip install brotli python-minifier
MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192 --skip-manifest
SEED=42 torchrun --standalone --nproc_per_node=8 train_gpt.py
```

## Credits

PR #1536 @dexhunter (VarLen + LoRA TTT base), PR #1523 @abaybektursun (fused Triton MLP + banking), PR #1530 @samacqua (VarLen concept), PR #1394 @clarkkev (SP8192 + SDClip), PR #1493 @bigbag (merged #1 hyperparameters)
