# TrigramHash: BigramHash(10240, dim=96) + TrigramHash(10240, dim=32)

**Base**: `2026-03-20_10L_Int5MLP_MuonWD04_SWA50` — 1.14276 BPB
**Status**: Non-record submission — run on 1×H100 (not 8×H100). Final val_bpb pending full run.

## Hypothesis

The SOTA BigramHash(10240, dim=128) captures P(token | prev_token) co-occurrence patterns. English text also has strong **3-gram patterns** (e.g., "New York City", "in the United", determiner+adjective+noun, verb+preposition+article) that a single bigram lookup cannot encode.

**Iso-parametric design**: Split the bigram table from dim=128 → dim=96, and add a new TrigramHash table at dim=32. Total embed parameters are unchanged:
- Old: 10240 × 128 = 1,310,720 params
- New: 10240 × 96 + 10240 × 32 = 983,040 + 327,680 = 1,310,720 params

The 96-dim bigram still captures most bigram signal; the 32-dim trigram adds orthogonal 3-gram structure. Combined expressiveness should exceed a single 128-dim bigram.

## Implementation

**New class**: `TrigramHashEmbedding`
- Hash function: `(36313*t[i] XOR 27191*t[i-1] XOR 18731*t[i-2]) % (vocab-1)` using three independent prime multipliers
- Positions 0,1 → padding bucket (mod), same as bigram for position 0
- Projection: 32 → 512 (model_dim) via learned linear
- Scale: initialized to 0.05 (same as bigram), learned parameter
- Zero-init for embed and proj weights (starts as no-op, learns gradually)

**Changes from SOTA**:
1. `bigram_dim`: 128 → 96
2. `trigram_vocab_size`: 0 → 10240 (new)
3. `trigram_dim`: 0 → 32 (new)
4. `TrigramHashEmbedding` class added
5. `CONTROL_TENSOR_NAME_PATTERNS` extended with `trigram.scale`
6. `_classify_param` extended with `"trigram" in name → "trigram"` category
7. `mixed_quantize_int6` called with `{"mlp", "attn", "bigram", "trigram"}` (trigram → int6)
8. Optimizer: trigram.embed.weight added to tok_params (AdamW), trigram.proj.weight to matrix_params (Muon), trigram.scale to scalar_params

**All other hyperparameters**: identical to SOTA (10L, int5/int6, SWA, Muon, WD=0.04, seq=2048, etc.)

## Expected Artifact Size

Trigram adds 0 net parameters vs SOTA (iso-parametric). Artifact size expected ≈ SOTA (~15.9MB).

## Run Command

```bash
# 8xH100 (competition standard)
torchrun --standalone --nproc_per_node=8 records/track_10min_16mb/2026-03-23_TrigramHash/train_gpt.py

# 1xH100 (tested configuration — see train_seed42_1xH100.log)
torchrun --standalone --nproc_per_node=1 records/track_10min_16mb/2026-03-23_TrigramHash/train_gpt.py
```

## Non-Record Submission Note

This submission was tested on a single H100 (1×H100 SXM, 80GB HBM3) rather than the competition-standard 8×H100. With 1 GPU, the 600s wallclock budget yields ~870 training steps vs ~6700 steps on 8×H100. The architecture and quantization are valid; the val_bpb reflects the reduced compute budget, not an architecture ceiling.

The iso-parametric constraint is exact: total embedding parameters are unchanged vs SOTA (10240×96 + 10240×32 = 10240×128 = 1,310,720). Artifact size with zstd-22 compression is expected ≤ 16MB.

## Idea Origin

Novel — not from a specific paper. Extension of the BigramHashEmbedding technique in PR#162.
