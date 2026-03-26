# E2E TTT: End-to-End Test-Time Training with Meta-Learning

**First E2E TTT submission in the competition.** This implements the meta-learning training procedure from [Sun et al., "End-to-End Test-Time Training for Long Context" (arXiv:2512.23675)](https://arxiv.org/abs/2512.23675), adapted for the parameter golf setting.

## What is E2E TTT?

E2E TTT is a training procedure (not a new architecture) where the model is trained to be good at *adapting* at test time, not just good at static prediction.

The existing TTT submission (PR #549, LoRA TTT) uses **TTT-naive**: standard training + eval-time SGD. The model's weights are never optimized for test-time adaptation — it just happens to work somewhat.

E2E TTT uses **meta-learning** (MAML-style) so the outer loop explicitly optimizes W0 for post-TTT performance:

```
For each training sequence:
    W = W0 (initial MLP weights)
    total_loss = 0
    
    for each chunk of the sequence:
        loss = cross_entropy(model(chunk, W), targets)   # next-token prediction
        total_loss += loss
        W = W - η * ∇loss(W)                             # inner loop: GD on MLP weights
    
    outer_loss = total_loss / num_chunks
    ∇W0(outer_loss).backward()   # outer loop: backprop THROUGH inner GD steps
    optimizer.step()              # update W0 to be a better starting point for TTT
```

The key: `create_graph=True` in `torch.autograd.grad` makes the inner gradient steps differentiable, so the outer optimizer receives **meta-gradients** — gradients that account for how W0 will be updated at test time.

## E2E Properties (matching the paper)

| Property | Paper (2512.23675) | Our Implementation |
|----------|-------------------|-------------------|
| E2E at test time | Inner loss = next-token prediction | ✅ `F.cross_entropy(logits, targets)` |
| E2E at training time | Outer loop backprops through inner steps | ✅ `create_graph=True` meta-gradients |
| Architecture | Standard Transformer (unchanged) | ✅ Same 11L Transformer as baseline |
| Inner params | MLP weights of last L/4 blocks | ✅ `mlp.fc.weight` + `mlp.proj.weight` of last 3 blocks |
| Frozen during inner loop | Attention, embeddings, norms | ✅ Only MLP weights updated in inner loop |

## Implementation Details

**The `w + 0` trick**: PyTorch `nn.Parameter` tensors are leaf nodes in the computation graph. To make meta-gradients flow back to the original parameters through the inner loop, we create non-leaf tensors via `w = param + 0`. This connects the inner loop's computation graph to the original parameters. Without this, `torch.autograd.grad(..., create_graph=True)` computes inner gradients but the outer backward doesn't propagate through them.

**Manual forward pass**: During meta-learning steps, we bypass `torch.compile` and run a manual forward pass that uses `F.linear(x, inner_weight)` for the inner-param MLP layers. This preserves the autograd graph through weight substitutions. Both `p.data` swaps and `torch.func.functional_call` were tested and found to **break** the meta-gradient path.

**Phased training**: Meta-learning is ~3x slower per step than standard compiled training (inner loop overhead + `create_graph=True` memory). To maximize total training quality within 10 minutes:
- Phase 1 (0–80% wallclock): Standard training with `torch.compile` (~1900 steps)
- Phase 2 (80–100% wallclock): Meta-learning fine-tune (~100 meta-steps)
- Phase 3: GPTQ quantization (45s reserve)

The paper also uses phased training (pre-train then meta-learning fine-tune for context extension).

## What's Novel vs Prior Work in This Competition

| Submission | Training | Eval-time TTT | Meta-learning |
|-----------|----------|---------------|---------------|
| PR #549 (LoRA TTT) | Standard | ✅ Score-first SGD | ❌ None |
| PR #738 (ours, v48) | Standard | ✅ Score-first SGD + n-gram + kNN | ❌ None |
| **This (v61 E2E TTT)** | **Meta-learning** | ✅ Score-first SGD + n-gram + kNN | **✅ MAML-style** |

The meta-learning is the contribution. The eval engine (n-gram + kNN) is carried from our v48/v50 work.

## Results

Single seed (42), 8×H100 (Hyperbolic), 10-minute wallclock:

```
Training: 2007 steps (1908 standard + 99 meta-learning)
Pre-quant:  val_bpb = 1.2423
Post-quant: val_bpb = 1.0467
Artifact:   13.12 MB (under 16 MB limit)
Eval time:  258s (under 600s limit)
```

Note: This instance ran at ~230ms/step (vs typical ~94ms/step on other H100 instances), limiting us to ~2000 steps instead of ~5800. On a faster instance, the standard training phase would produce a stronger base model before meta-learning kicks in.

## Limitations and Future Work

1. **Partial meta-learning**: Only the last 20% of training uses meta-learning. Full meta-learning from the start (as in the paper) would require either faster hardware or a more efficient implementation (e.g., JAX with `jax.grad` which handles higher-order derivatives natively).

2. **torch.compile incompatibility**: `create_graph=True` doesn't work with `torch.compile(fullgraph=True)`. The meta-learning phase runs uncompiled, which is ~3x slower. A custom CUDA kernel for the inner loop could close this gap.

3. **Memory overhead**: The inner loop with `create_graph=True` uses ~44GB (vs ~23GB for standard training) due to storing the computation graph through gradient steps. This fits on H100 80GB but limits batch size.

4. **Inner learning rate tuning**: We used a fixed `η_inner = 0.001`. The paper uses a learned per-token learning rate. Making η learnable (as an outer-loop parameter) could improve adaptation quality.

## Command

```bash
META_ENABLED=1 SEED=42 \
python3 -m torch.distributed.run --nproc_per_node=8 train_gpt.py
```

## Included Files

- `train_gpt.py` — full training script with E2E TTT meta-learning
- `train_seed42.txt` — training log
- `submission.json` — metadata
