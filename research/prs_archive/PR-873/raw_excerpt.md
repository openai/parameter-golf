# PR 873 — E2E TTT: End-to-End Test-Time Training with Meta-Learning

**Author:** not stated in record README
**Claimed BPB:** 1.0467 post-quant (single seed 42)
**Artifact size:** 13.12 MB
**Seeds:** 42 (single seed)

## Files retrieved
- `records__track_10min_16mb__2026-03-26_E2E_TTT_MetaLearning__README.md`
- `records__track_10min_16mb__2026-03-26_E2E_TTT_MetaLearning__submission.json`
- `records__track_10min_16mb__2026-03-26_E2E_TTT_MetaLearning__train_gpt.py`

## Environment variables (from run command in README)
META_ENABLED=1 SEED=42

## Claimed changes (from README, verbatim)
> First E2E TTT submission in the competition. This implements the meta-learning training procedure from Sun et al., "End-to-End Test-Time Training for Long Context" (arXiv:2512.23675).

> E2E TTT is a training procedure (not a new architecture) where the model is trained to be good at adapting at test time, not just good at static prediction. E2E TTT uses meta-learning (MAML-style) so the outer loop explicitly optimizes W0 for post-TTT performance. For each training sequence: W = W0; for each chunk: loss = cross_entropy(model(chunk, W)); total_loss += loss; W = W - η * ∇loss(W); then outer ∇W0(outer_loss).backward() with create_graph=True.

> Inner params: MLP weights of last L/4 blocks (last 3 blocks). Frozen during inner loop: Attention, embeddings, norms.

> The `w + 0` trick: PyTorch nn.Parameter tensors are leaf nodes. To make meta-gradients flow back through the inner loop, we create non-leaf tensors via `w = param + 0`. Manual forward pass during meta-learning steps bypasses torch.compile and runs F.linear(x, inner_weight). Both p.data swaps and torch.func.functional_call break the meta-gradient path.

> Phased training: Phase 1 (0-80% wallclock): Standard training with torch.compile (~1900 steps). Phase 2 (80-100%): Meta-learning fine-tune (~100 meta-steps). Phase 3: GPTQ quantization (45s reserve). Meta-learning is ~3x slower per step than compiled training.

> Results: 2007 steps (1908 standard + 99 meta-learning). Pre-quant val_bpb 1.2423. Post-quant val_bpb 1.0467. Eval time 258s.
