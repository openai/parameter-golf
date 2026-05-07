Structured Embedding Initialization: Pre-computed token positions from PCA-reduced corpus analysis.

Instead of random Xavier init for the 1024-token embedding table, we pre-compute token positions using PCA on large-scale co-occurrence statistics, then selectively override 665/1024 tokens that have sufficient coverage (5+ associations). Remaining tokens keep default init. All overridden embeddings are per-token Xavier-standardized to preserve gradient flow.

Changes from baseline:
- Added ~10 lines after model init to load `structured_embeddings.npy` and `embedding_mask.npy`
- No architecture changes, no hyperparameter changes, no data changes

Configuration (same as baseline):
- Layout: `VOCAB_SIZE=1024 NUM_LAYERS=9 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=4 MLP_MULT=2`
- Tied embeddings, 10-minute wallclock on 8xH100

Result:
- **BPB: 1.2314** (vs 1.2303 baseline — +0.001 worse)
- The structured init converges faster in early steps (BPB 1.380 vs 1.383 at step 1000) but the advantage does not persist through full training
- Analysis suggests the PCA positions optimize for semantic similarity rather than predictive co-occurrence, which is the actual training objective

This submission demonstrates the approach, not a leaderboard improvement. The embedding init framework is designed to be combined with bigram hash tables and curriculum ordering in future iterations.
