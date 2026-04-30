# Ablation 1: Remove x0 Residual Highway

## Hypothesis

The x0 residual highway (`resid_alpha`) is a primary cause of the regression from iter-003.5 (val_bpb=1.600) to iter-005.5 (val_bpb=1.98).

**Why it hurts:**

1. **Representation dilution.** `resid_alpha` is initialized to 0, so `sigmoid(0) = 0.5`. At every depth iteration, 50% of the block input is the raw embedding `x0` rather than the refined hidden state. The block never sees a fully processed representation -- it always gets half-raw input. Over 8 iterations, the effective "depth" of the network is severely reduced because each layer starts from a mixture that is anchored to the initial embedding.

2. **Gradient shortcut undermines depth recurrence.** The whole point of depth recurrence is that each iteration refines the representation. The x0 highway creates a gradient shortcut that lets the model "cheat" by routing gradients directly to the embedding, reducing the incentive for the recurrent block to learn meaningful transformations. The block may learn to be a near-identity because the x0 mixing already provides a strong signal.

3. **Conflicts with SSD state carry.** The SSD mixer maintains hidden state across chunks. Mixing in x0 at each iteration resets the "context" the mixer was building, since x0 is the same static embedding at every iteration. This defeats the purpose of the vertical chunk state carry (another iter-005.5 feature).

4. **iter-003.5 didn't have it.** The best result (val_bpb=1.600) passed `x` directly: `x = self.block(x, self.iter_embeds[i])`. No mixing, no highway. The representation refined progressively through 8 iterations, each building on the last.

## Exact Code Change

In `train_gpt.py`, in `SSDGolfModel._core_forward()`:

**Remove** (lines 1242-1244):
```python
            # x0 residual mixing: gradient highway + raw signal access
            alpha = torch.sigmoid(self.resid_alpha[i]).to(x.dtype)
            x_in = alpha * x + (1.0 - alpha) * x0
```

**Replace with:**
```python
            x_in = x
```

Also remove (but not strictly required -- it becomes a dead parameter):
- Line 1172: `self.resid_alpha = nn.Parameter(torch.zeros(n_iters))`
- Line 1228: `x0 = x  # save for residual highway`
- Line 1432: `scalar_params.append(base_model.resid_alpha)`
- Line 120: `"resid_alpha"` from `CONTROL_TENSOR_NAME_PATTERNS`

## Expected Outcome

- **IF val_bpb improves by >0.1** (i.e., drops below ~1.88): The x0 highway was a significant contributor to the regression. Remove it permanently.
- **IF val_bpb improves by >0.2** (i.e., drops below ~1.78): The x0 highway was THE primary cause. This change alone recovers most of the gap.
- **IF val_bpb stays within 0.05 of 1.98**: The x0 highway is not the problem. Look at other iter-005.5 additions (vertical chunk state carry, multi-block, etc.).

## Run Command

```bash
# On Runpod, 1xH100, 10 min smoke test:
cd /workspace/param-golf && \
RUN_ID=ablation_001_no_x0_highway \
MAX_WALLCLOCK_SECONDS=600 \
VAL_LOSS_EVERY=200 \
TRAIN_LOG_EVERY=50 \
torchrun --standalone --nproc_per_node=1 \
  experiments/iter-005-compiled-ssd/iter-005.5/ablation-1/train_gpt.py
```
