# Experiment 051: SwiGLU + Vocab 2048 + 8 Layers

## Status: COMPLETED

## Config
- Our merged script (train_gpt_merged.py)
- VOCAB_SIZE=2048, NUM_LAYERS=8, USE_SWIGLU=1, SWIGLU_HIDDEN=1024
- TRAIN_SEQ_LEN=4096, TRAIN_BATCH_TOKENS=393216
- NorMuon, int6 QAT, no SWA, no LAWA
- Tokenizer: sp2048 (retokenized on animal machine)
- Train data, 8xH100 NV18 NVLink

## Results
| Metric | Value |
|--------|-------|
| Steps | 10,591 @ 56.7ms/step |
| Model params | 19,941,440 |
| Artifact | **15,230,420 bytes ✅** |
| **Standard eval** | **1.1840 BPB** |
| **Sliding eval stride=64** | **1.1739 BPB** |
| Sliding stride=512 | 1.1739 |
| Sliding stride=128 | 1.1739 |

## Comparison with 049 (vocab 1024, 9 layers)
| Metric | 049 (v1024, 9L) | 051 (v2048, 8L) |
|--------|----------------|-----------------|
| Params | 21.8M | 19.9M |
| Standard | 1.1805 | 1.1840 (+0.0035 worse) |
| Sliding | **1.1685** | 1.1739 (+0.0054 worse) |
| Artifact | 16.29MB ❌ | 15.23MB ✅ |

## Key Findings
- Vocab 2048 + 8 layers WORSE than vocab 1024 + 9 layers
- Losing the 9th layer hurt more than bigger vocab helped
- Fewer total params (19.9M vs 21.8M) because bigger embedding table
- Artifact fits under 16MB though (15.23MB)
- **Conclusion: keep vocab=1024, 9 layers for SwiGLU**

## wandb
- Run ID: ocd9m52m
- Run name: 051_swiglu_vocab2048_8layers
- URL: https://wandb.ai/ishanramrakhiani-bindwell/parameter-golf/runs/ocd9m52m
