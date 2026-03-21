# Aria Parameter Golf Win Checklist

This is a practical execution plan tuned for current local hardware (RTX 5060 Ti) and then scaled to a record-track run on 8xH100.

## 0) Rule Lock (non-negotiable)

1. Optimize `val_bpb` on fixed FineWeb validation.
2. Keep artifact under **16,000,000 bytes** (`compressed model bytes + code bytes`).
3. No network/downloads or training-data access during evaluation unless included in artifact budget.
4. Record-track target: train + eval each reproducible under 10 minutes on 8xH100 SXM.

## 1) Local Baseline (done first)

1. Set up venv and dependencies.
2. Pull challenge data subset:
   - `python data/cached_challenge_fineweb.py --variant sp1024 --train-shards 1`
3. Run smoke train to verify pipeline/logging:
   - `RUN_ID=aria_smoke_sp1024 ... torchrun --standalone --nproc_per_node=1 train_gpt.py`
4. Confirm these outputs exist in logs:
   - `Total submission size int8+zlib: ...`
   - `final_int8_zlib_roundtrip_exact val_bpb:...`

## 2) Aria-Inspired Attack Path (model-side, challenge-safe)

Focus on improvements that help language compression quality directly:

1. **Tokenizer efficiency pass** (still rigorously validated):
   - Keep SP-1024 baseline first.
   - Only change tokenizer when we can prove correct `val_bpb` computation.
2. **Architecture sweep**:
   - Start from baseline family (`9x512`, GQA, tied embeddings).
   - Sweep width/depth/KV-heads/LR schedule within size cap.
3. **Compression-aware training**:
   - Train with post-quant roundtrip in mind (int8+zlib score is what counts).
4. **Stability over novelty**:
   - Avoid risky hacks until baseline reproducibility is tight.

## 3) What *Not* To Do (for challenge score)

1. Do not rely on external wrappers, retrieval, API calls, or runtime tools.
2. Do not optimize for chat feel first; optimize `val_bpb` first.
3. Do not ship anything that cannot rerun from record folder alone.

## 4) Iteration Protocol

For every experiment:

1. Set unique `RUN_ID`.
2. Save full train log.
3. Record:
   - config deltas
   - wallclock
   - `final_int8_zlib_roundtrip_exact val_bpb`
   - `Total submission size int8+zlib`
4. Keep top runs in a ranked table (`run_id`, `val_bpb`, bytes, notes).

## 5) Submission Folder Template

When ready, create a new folder under:

- `records/track_non_record_16mb/<DATE>_<RUN_NAME>/`
  or
- `records/track_10min_16mb/<DATE>_<RUN_NAME>/`

Required files:

1. `README.md` (method + exact commands + results)
2. `submission.json`
3. `train.log`
4. self-contained `train_gpt.py` (+ any dependencies/scripts used)

## 6) Immediate Next Actions

1. Finish current smoke run and capture final `val_bpb` + size.
2. Launch first structured sweep (small local budget):
   - 3 configs around baseline:
     - `9x512 kv4` (control)
     - `10x480 kv4`
     - `8x576 kv4`
3. Compare by **roundtrip exact `val_bpb`**, not train loss.
4. Promote best config to a longer non-record run.

