# Ablation 2: Disable Vertical State Carry

## Hypothesis

Vertical state carry (passing `chunk_states` from depth iteration i as `vertical_states` to iteration i+1) is the primary cause of the regression from iter-003.5 (val_bpb=1.600) to iter-005.5 (val_bpb=1.98).

The mechanism works as follows: `_ssd_chunk` returns `chunk_states` which represent the accumulated SSM hidden state at each chunk boundary. In `_core_forward`, these are fed back as `vertical_states=chunk_states` to the next depth iteration, where they get *added* to the horizontal chunk state (line 945: `chunk_states = chunk_states_h + vertical_states`). This means each depth iteration's SSD scan is biased by the raw internal state of the *previous* iteration's scan.

Why this likely hurts:
1. **Representation mismatch**: Each depth iteration applies its own AdaLN conditioning, meaning the representations at each depth level live in different subspaces. Adding raw SSM states across these subspaces injects noise.
2. **Gradient interference**: The vertical carry creates a dense dependency chain across all depth iterations, making optimization harder -- gradients must flow through the additive state path across all N iterations.
3. **Untrained pathway**: iter-003.5 trained without vertical carry and achieved 1.600. iter-005.5 introduced it but the model has not learned to use it -- it is an untrained additive perturbation that corrupts the horizontal scan.
4. **Magnitude concern**: `chunk_states` from iteration i may have different scale than the horizontal prefix states in iteration i+1, causing the effective state to be dominated by the wrong signal.

iter-003.5 did NOT do this -- each depth iteration started with fresh `chunk_states=None`. That version achieved val_bpb=1.600 on 1xH100 in 10 minutes (960 steps).

## Exact Code Change

In `_core_forward` (line ~1248-1251), change:

```python
            x, new_horizontal_state, chunk_states = block(
                x_in, self.iter_embeds[i],
                ssd_state=cross_chunk,
                vertical_states=chunk_states,
            )
```

to:

```python
            x, new_horizontal_state, chunk_states = block(
                x_in, self.iter_embeds[i],
                ssd_state=cross_chunk,
                vertical_states=None,  # ABLATION: disable vertical state carry
            )
```

This is a single-line change. `chunk_states` is still returned by each block (needed for the API), but it is never passed to the next iteration. Each depth iteration starts its SSD scan with only horizontal state (if stateful eval/training is on) or zeros.

## Expected Outcome

- val_bpb should recover to approximately 1.60-1.65 range, matching iter-003.5 performance
- If val_bpb does NOT improve, the regression is caused by other changes between iter-003.5 and iter-005.5 (architecture, hyperparameters, Triton kernel differences, etc.)
- Throughput should be slightly better (no vertical state tensor allocation/addition per iteration)

## Run Command

```bash
cd /root/param-golf && \
DATA_PATH=./data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=./data/tokenizers/fineweb_1024_bpe.model \
RUN_ID=ablation_2_no_vertical \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 \
  experiments/iter-005-compiled-ssd/iter-005.5/ablation-2/train_gpt.py
```

Or from the directory containing the script:

```bash
cd /root/param-golf && \
RUN_ID=ablation_2_no_vertical \
MAX_WALLCLOCK_SECONDS=600 \
torchrun --standalone --nproc_per_node=1 \
  experiments/iter-005-compiled-ssd/iter-005.5/ablation-2/train_gpt.py
```
