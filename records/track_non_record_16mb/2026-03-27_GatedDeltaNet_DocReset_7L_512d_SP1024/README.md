# Non-Record Submission: GatedDeltaNet with Document-Boundary State Reset, 32k Context

## Results (seed=1337, 4xH100, 20min)


| Method name   | Context length | Step time | Steps completed | val_bpb 20min wallclock_cap | val_bpb at step 7800 (unquantized) |
| ------------- | -------------- | --------- | --------------- | --------------------------- | ---------------------------------- |
| Baseline      | 1024           | 85ms      | 14100           | 1.2260                      | 1.2581                             |
| GatedDeltaNet | 1024           | 141ms     | 8525            | 1.2733                      | 1.2827                             |
| GatedDeltaNet | 8192           | 145ms     | 8298            | 1.2553                      | 1.2589                             |
| GatedDeltaNet | 16384          | 147ms     | 8179            | 1.2525                      | 1.2530                             |
| GatedDeltaNet | 32768          | 151ms     | 7951            | 1.2519                      | 1.2478                             |


**Submitted result: val_bpb = 1.2519** (32k context, 20min wallclock cap)

## Method Summary

- Replaces softmax attention with **[GatedDeltaNet](https://arxiv.org/abs/2412.06464)** (linear O(n) recurrent attention)
- **Document-boundary state reset:** packed sequences contain multiple documents separated by BOS tokens. For a recurrent model, hidden state bleeds across document boundaries unless explicitly reset. BOS positions are detected per-sequence, converted to `cu_seqlens`, and passed to FLA's variable-length chunked kernel, which zeros recurrent state at each boundary — applied identically in training and validation
- **Gradient clipping**: necessary to prevent `nan` loss for long recurrent chains
- **Architecture reductions**: necessary to stay within the byte limit (9 → 7 layers, smaller MLP expansion 2× → 1.875×)

## Changes from Baseline


| Component         | Baseline (`train_gpt.py`)                   | This submission                                     |
| ----------------- | ------------------------------------------- | --------------------------------------------------- |
| Attention         | Softmax MHA/GQA (8 heads, 4 KV heads, RoPE) | GatedDeltaNet (4 heads × 128 head_dim, linear O(n)) |
| Sequence length   | 1024                                        | 32768 (also tested 1024 / 8192 / 16384)             |
| Depth             | 9 layers                                    | 7 layers                                            |
| MLP expansion     | 2×                                          | 1.875×                                              |
| Gradient clipping | Disabled (norm=0.0)                         | Enabled (norm=1.0)                                  |


## Conclusion

- GatedDeltaNet allows training with very long context (up to 32k tokens) with minimal compute overhead — each context doubling costs only ~2–4ms per step.
- GatedDeltaNet at 32k context slightly beats the baseline for per-step quality (1.2478 vs. 1.2581 at step 7800).
- Longer context yields better val_bpb, but with diminishing returns — likely because context length already exceeds the length of most documents.
- GatedDeltaNet breaks `torch.compile(fullgraph=True)`, so `fullgraph=False` is used; the resulting graph-break overhead is partly responsible for the slower step time vs. baseline.

