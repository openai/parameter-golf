# Submission Status — sp8192-rebase

## Result
- **Val BPB:** 1.07069667 (quantized_sliding_window)
- **SOTA (bigbag PR #1493):** 1.0810 → **we beat it**
- **Artifact code size:** 15,645 bytes (wrapper) from 43,517 bytes source

## Artifact Size Budget
| Component | Size |
|---|---|
| Model `.ptz` | ~15,945,252 bytes |
| Code wrapper | 15,645 bytes |
| **Total** | **~15,960,897 bytes** |
| Cap | 16,000,000 bytes |
| **Headroom** | **~39,103 bytes** |

## Files
| File | Purpose |
|---|---|
| `train_gpt_submission.py` | Submit this — 2-line LZMA+b85 self-extractor |
| `train_gpt_human_qkgain.py` | Human-readable source (Kevin's minified SP8192 + QK-Gain lever) |
| `build_submission.py` | Compression utility: `python build_submission.py <src> <out>` |
| `train_gpt_sp8192_opt.py` | Full working script (64 KB, NOT for submission) |

## QK-Gain Lever (in train_gpt_human_qkgain.py)
```bash
QK_GAIN_INIT_SCHEDULE="2.0,2.5,3.0,3.5,4.0,4.5,4.5,4.0,3.5,3.0,2.5" \
  DATA_DIR=/workspace/parameter-golf/data SEED=42 \
  python train_gpt_submission.py
```

Env var `QK_GAIN_INIT_SCHEDULE`: comma-separated floats, one per physical layer (11 values for 11 layers). If empty, falls back to uniform `QK_GAIN_INIT` (default 4.0).

## Submission Checklist
- [x] Training produces 1.07069667 BPB (beats SOTA 1.0810)
- [x] Submission wrapper fits within 16MB artifact cap
- [x] Wrapper syntax OK; decompressed payload syntax OK (43,517 bytes roundtrip)
- [x] Committed on `sp8192-rebase`
- [ ] Push to origin/sp8192-rebase
- [ ] Open PR against upstream/main
