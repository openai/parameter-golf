# Bigram Alternatives Analysis (from Grok)

## Problem
BigramHash adds 0.004 BPB (070 vs 069) but costs ~2MB in artifact (17.38 vs 15.34MB).
Need the BPB benefit without the size cost.

## Option 1: Outer Product of Token Embeddings
- `bigram = tok_emb[prev] concat tok_emb[curr]` → project with CastedLinear(1024, 512)
- PROBLEM: The projection is 524K params = ~400KB compressed. Not zero-cost!
- Also: the concat doubles the input dimension, which means a 1024→512 linear.
- With int6 quant, this adds ~400KB to artifact. Not 1.2MB but still significant.
- BUT: it reuses embedding info, might be more compressible than random hash table.
- **VERDICT: MEDIUM — test if projection compresses better than hash table**

## Option 2: Sinusoidal/Procedural Bigram (persistent=False)
- Generate table from seed at init, don't store in state_dict
- TRUE zero stored params — table recomputed from code every time
- Risk: quality might differ from learned table (but Grok claims <0.001 difference)
- The table being non-persistent means it's NOT quantized, NOT compressed, NOT in artifact
- **VERDICT: HIGH PRIORITY — truly zero cost if it works**

## Option 3: Gate Modulation
- Use bigram hash to modulate existing SmearGate or resid_mix
- Zero extra params, zero artifact cost
- Signal is very weak — just a scalar modulation
- **VERDICT: LOW — too weak, unlikely to match learned bigram**

## Experiment Plan
- **Option 2 first**: Change BigramHash to persistent=False with fixed seed
  - If BPB stays within 0.001 of learned bigram → use it
  - Expected artifact: ~15.3MB (same as no-bigram) with ~1.147 BPB
- **Option 1 second**: If option 2 hurts BPB, try outer product with small projection
