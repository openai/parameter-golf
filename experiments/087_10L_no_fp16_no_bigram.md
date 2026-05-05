# Experiment 087: 10L + no FP16_KEEP + no BigramHash + leaky_relu(0.5)²

## Config
- 10 layers, model_dim=512, MLP_HIDDEN=1536
- leaky_relu(0.5)² activation
- No FP16_KEEP_NAME_PATTERNS (all params quantized)
- No BigramHash
- NorMuon + QAT (int6), WD=0.04 both, LR=0.025, SCALAR_LR=0.025, TIED_EMBED_LR=0.035
- seq_len=2048, batch_tokens=786432, warmdown=3000

## Results
- Steps: 6,617 @ 90.7ms/step (wallclock cap 600s single GPU)
- **Standard BPB: 1.1602**
- **Sliding BPB: 1.1391** ← BETTER than best 9L (084's 1.1427)!
- Manual+zstd: 16.52MB ❌
- FLAT+zstd: **16.25MB ❌ (252KB over!)**
- Manual+lzma: 16.50MB ❌

## Artifact Breakdown
- Raw torch: 24,335,799 bytes
- Manual raw: 24,260,749 bytes
- FLAT raw: 24,238,240 bytes
- FLAT zstd: 16,162,347 bytes (payload)
- Code: 90,480 bytes
- Total FLAT submission: 16,252,827 bytes (252KB over 16MB limit)

## Analysis
- 10L gives MUCH better BPP: 1.1391 vs 1.1427 (0.0036 improvement over best 9L)
- Removing FP16_KEEP helped: exp086 (10L+bigram) was 16.72MB, now 16.25MB without bigram and without fp16 keep
- Still 252KB over budget. Need INT5_MLP to compress further.

## Comparison
| Exp | Layers | BigramHash | FP16_KEEP | Sliding BPP | FLAT+zstd | Fits? |
|-----|--------|-----------|-----------|-------------|-----------|-------|
| 084 | 9 | no | yes | 1.1427 | 15.76MB | ✅ |
| 086 | 10 | yes | no | 1.1350 | 16.72MB | ❌ |
| **087** | **10** | **no** | **no** | **1.1391** | **16.25MB** | **❌ (252KB over)** |

## Next Step
INT5 MLP quantization should save 1-2MB → easily under 16MB!
PR219 shows int5 [-16,15] stored in int8 compresses at 1.88x vs 1.51x for int6.
Need to also update QAT to use int5 fake quantization for MLP layers.
