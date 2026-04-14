# PR 1281 — MatrixGPT: Matrix-State Recurrent Language Model

**Author:** Ralph Lauren N. Reyes (rlphlrnrys)
**Claimed BPB:** 3.17277005 (5-seed mean, std 0.00122; seeds 7, 42, 333, 777, 1337)
**Artifact size:** 1.33 MB (1,333,704 bytes; 8.3% of 16 MB budget)
**Model params:** 854,272
**Track:** non_record_16mb

## Files retrieved
- `records__track_non_record_16mb__2026-04-02_MatrixGPT__README.md`
- `records__track_non_record_16mb__2026-04-02_MatrixGPT__submission.json`
- `records__track_non_record_16mb__2026-04-02_MatrixGPT__train_matrix.py`

Note: PR touches multiple record directories (LeakyReLU_LegalTTT_ParallelMuon, 74M_Ternary_UNet, ValCalib_GPTQ_XSA_BigramHash3072, DepthRecurrence_MixedPrecisionQuant, 106M_Binary_Asymmetric_UNet, MatrixGPT, TanHop). Only MatrixGPT files extracted per batch assignment.

## Claimed changes (from README, verbatim)

> This submission replaces the standard Transformer attention mechanism with a **sequential matrix product recurrence** over small 2x2 matrices, yielding O(n) inference in sequence length instead of O(n^2).

> MatrixGPT instead maintains a running state via sequential matrix product: `M_t = f_theta(x_t)`, `S_t = M_t @ S_{t-1}`, `h_t = S_t @ v`

> Each layer maintains k parallel 2x2 dynamical systems. Config: num_channels (k)=64, num_layers=6, model_dim=256, train_seq_len=256, train_batch_tokens=131,072, tied embeddings, logit softcap 30.0.

> No RoPE, no GQA/KV heads, no UNet skips, no per-block MLP. Only change vs baseline: GPT -> MatrixGPT, Block -> MatrixLayer. Muon optimizer, int8+zlib quant, tokenizer-agnostic BPB eval kept identical.

> Hardware: 1x NVIDIA GeForce RTX 3060 8 GB. A 6-hour exploratory run (seed 666) reached val_bpb 2.169 at 4347 steps with no plateau. Parallel prefix scan (log2T depth) replaces naive sequential loop per submission.json.
