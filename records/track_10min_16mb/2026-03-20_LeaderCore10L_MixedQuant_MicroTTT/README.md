## Summary

This is a non-record `track_10min_16mb` submission built from the strongest valid `10L` leader-core line, testing a narrow QLoRA-style lesson:

- use more aggressive mixed quantization by subsystem at export time
- keep eval-time adaptation tiny by only updating a small control-parameter subset

The run is valid on both hard constraints:

- `final_int8_zlib_roundtrip_exact val_bpb: 1.24271554`
- `Total submission size int8+zlib: 10401310 bytes`
- `eval_time: 1548ms`

## What Changed

- `int5` export for MLP matrices
- `int6` export for attention matrices
- `int8` export elsewhere
- tiny eval-time SGD on `q_gain`, `attn_scale`, `mlp_scale`, `resid_mix`, and `skip_weights`
- same validity-safe non-overlapping final eval and temperature-only search as the valid leader-core base

## Result

The main result is a useful negative one.

- The aggressive mixed quantization path reduced artifact size dramatically, from the usual `~15.3MB` range on this line to about `10.4MB`.
- Training throughput stayed very strong: `12348` steps in `600s` at `48.59ms/step`.
- But the post-roundtrip quality regressed substantially.

Stage breakdown from the saved `8xH100` run:

- stop metric before export: `val_bpb 1.2060`
- after mixed quant + temperature search: about `1.2431`
- after `32` micro-TTT steps: `1.24271554`

So the compression idea worked mechanically, but this particular export recipe was too destructive, and the tiny control-only TTT recovered almost none of that damage.

## Why Submit It

This seems worth preserving as a non-record artifact because it establishes a concrete point on the size/quality frontier:

- a valid `10L` run near `10MB` total artifact size
- strong train throughput on the valid leader-core recipe
- a clear failure mode for aggressive `int5/int6` post-training quantization on this model family

That makes it a useful base for later work that tries to spend the saved bytes on more model capacity or uses less destructive mixed quantization.

## Files

- `train_gpt.py`: training and export script
- `train.log`: saved full `8xH100` run log
- `submission.json`: metadata for the run
