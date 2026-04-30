# Claude Handoff

## Current situation

This repo has converged away from Mamba/BitLinear/DeltaNet and toward a lean all-attention model.

Best current local results:

- Best `seq256` run:
  - `14x640`, `h8`, `kv4`, no FFN, casted linears
  - log: `logs/next4_14x640_h8_kv4_111139.txt`
  - post-export `val_bpb: 2.15168042`
  - total mixed artifact size: `11,114,675` bytes

- Best `seq512` run:
  - `12x768`, `h8`, `kv4`, no FFN, casted linears
  - log: `logs/seq512_12x768_h8_kv4_113325.txt`
  - post-export `val_bpb: 2.18719738`
  - total mixed artifact size: `13,073,058` bytes

These both fit under the `16MB` cap.

## Strong conclusions so far

- Lean all-attention is winning.
- `BitLinear` is a net negative here.
- Hybrid Mamba+attn lost to all-attention.
- DeltaNet lost badly.
- Adding a SwiGLU FFN made the attention model worse.
- `h8 kv4` has been the strongest head layout.
- Scaling width/depth helped repeatedly.
- `seq256` still scores a bit better locally than `seq512`, but the gap is now small.

## Important implementation changes already made

### In `train_gpt.py`

- Mixed exporter is in place.
- Per-run artifact names are used under `artifacts/` to avoid collisions.
- BOS reset logic exists.
- `BitLinear` uses float weights at eval when `BITLINEAR_EVAL_MODE=float`.
- Attention-only and DeltaNet architectures are selectable.
- File is currently still under the 1500-line limit.

### New helper module

- `attention_playground.py`

This adds opt-in experimental features without bloating `train_gpt.py`:

- `INIT_CKPT`
  - load from a prior checkpoint with `strict=False`
- `ATTN_TIED_LAYERS`
  - use fewer physical attention layers and unroll a tied pattern
- `MEMORY_TOKENS`
  - learned memory K/V slots in attention blocks
- `EMA_DECAY`
  - EMA shadow weights swapped in before final serialization/eval

The hooks are already wired into `train_gpt.py`.

## Current active batch

The current live sweep is:

- script: `sweep_attention_playground_batch.sh`
- launcher: `run_attention_playground_batch.bat`
- results file: `sweep_attention_playground_batch_results.txt`
- launcher log: `logs/attention_playground_batch_launcher.log`

First active run at handoff time:

- `logs/play_cont512_lr010_120051.txt`

The batch includes:

1. `256 -> 512` continuation from the best `256` checkpoint
2. tied-layer high-width attention
3. memory-token attention
4. one EMA control

## Checkpoint paths worth knowing

- Best `seq256` checkpoint:
  - `artifacts/next4_14x640_h8_kv4_111139.final_model.pt`

- Best `seq512` checkpoint:
  - `artifacts/seq512_12x768_h8_kv4_113325.final_model.pt`

## Smoke tests already done

These all passed:

- memory tokens + EMA
- tied layers
- continuation load from the best `seq256` checkpoint

Relevant logs:

- `logs/smoke_mem_ema.txt`
- `logs/smoke_tied.txt`
- `logs/smoke_init512.txt`

## Practical next steps

If the current playground batch produces a clear win:

1. Promote the best new idea into a focused confirm run.
2. If it still holds, this is a good time to move to `8xH100`.

If the playground batch does **not** beat the baseline:

1. Keep the mainline on lean all-attention.
2. Continue scaling or exporter-tuning the best `seq256` and `seq512` models.
3. Use `8xH100` for a faithful high-compute reproduction of:
   - the best plain `seq256` baseline
   - the best plain `seq512` baseline

## Things not worth re-litigating unless new evidence appears

- DeltaNet
- Mamba rescue
- BitLinear as default
- FFN on the winning attention path

## Useful files

- `train_gpt.py`
- `attention_playground.py`
- `sweep_attention_playground_batch.sh`
- `sweep_attention_next4.sh`
- `sweep_attention_seq512_branch.sh`
- `sweep_attention_next4_results.txt`
- `sweep_attention_seq512_branch_results.txt`

## Monitoring commands

From repo root in WSL:

```bash
tail -n 80 logs/attention_playground_batch_launcher.log
tail -n 80 logs/play_cont512_lr010_120051.txt
cat sweep_attention_playground_batch_results.txt
pgrep -af "sweep_attention_playground_batch.sh|python3 .*train_gpt.py"
```

## My current recommendation

The boring baseline is still the thing to beat:

- all-attention
- casted linears
- no FFN
- `h8 kv4`

The current playground batch is the right place to spend novelty budget.
