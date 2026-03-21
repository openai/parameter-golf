# PR198 Reproduction on Our Hardware — CRITICAL FINDING

## Results
- PR198 exact script on our machine: **12.71MB** (22.4M params, 9L default)
- Our script same params: **15.78MB**
- **GAP: 3.07MB — from our CODE, not hardware!**

## Root Causes Found
1. **FP16 passthrough** (tok_emb + 2 K projections): ~770KB extra
   - We keep tok_emb, blocks.7.c_k, blocks.8.c_k as fp16
   - PR198 quantizes them to int8/int6
   - Fix: remove FP16_KEEP_NAME_PATTERNS

2. **Extra metadata/structure** from disabled features:
   - Outlier splitting code (disabled but adds check overhead)
   - Blockwise quant code (disabled)
   - Manual serialization header
   - These add bytes to the state dict

3. **Possible: different state dict structure**
   - Our code may store more tensors (bigram extras, smeargate)
   - PR198's default doesn't use BigramHash/SmearGate at all

## Action: Strip fp16 passthrough and simplify quantization to match PR198
Expected savings: ~770KB from fp16 → int8 for tok_emb + K projections
Combined with PR198's simpler serialization: could save 2-3MB total

## THIS EXPLAINS WHY 10/11 LAYERS DON'T FIT!
If we match PR198's artifact size, 10L goes from 16.93MB to ~14MB → EASILY FITS!
