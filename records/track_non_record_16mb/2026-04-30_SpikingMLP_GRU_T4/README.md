# SpikingMLP + GRU readout (non-record, 16 MB)

This non-record submission swaps each Block's relu² MLP for a **Spiking-LIF MLP with a per-token GRU readout** (`SpikingMLP`), exploring two of the wishlist directions at once:

- *Learning adapters on random linear maps* — the GRU acts as a small learnable temporal-readout adapter on top of a binary spike code produced by a fixed-functional LIF dynamics.
- *State-space models / E2E TTT / super long context* — the LIF state is rolled `T=4` times **per token**, giving each token its own internal recurrent state-space without changing the sequence-axis attention.

Architecture is otherwise identical to the root `train_gpt.py` (9× 512-dim block, GQA 8/4 heads, tied embeddings, RoPE, RMSNorm, SP-1024 tokenizer).

## Per-token SpikingMLP

Each Block computes:

```text
val      = W_up · x                    # [B*T_seq, hidden]   single random-feature linear
v, h     = 0
for t in range(T):
    v   = beta * v + val               # leaky integrate (drive = val)
    spk = (v > thresh)                 # surrogate-grad (fast-sigmoid 25)
    v   = v - spk * thresh             # subtract reset
    gated = spk * val                  # binary gate × continuous value
    h   = GRUCell(gated, h)            # learnable temporal readout
out      = W_out · h                   # zero-init, residual-friendly
```

The recurrence happens **inside one token's forward** — `T=4` LIF micro-steps per sequence position, all with the same `val`. The output projection is zero-initialised so the block starts as identity and the SNN only contributes once gradients flow.

Only **one** linear (`W_up`) is learned upstream of the LIF — both the integrate-drive and the spike-gated value come from the same random-feature map, leaning into the wishlist's *adapter on a random linear map* framing and keeping the artifact comfortably under the 16 MB cap.

## Configuration

| Knob | Value |
|------|-------|
| Track | non-record, 16 MB |
| Hardware | 8× H100 80 GB (`H400`) |
| Tokenizer | SentencePiece BPE, vocab=1024 |
| Base layout | NUM_LAYERS=9, MODEL_DIM=512, NUM_HEADS=8, NUM_KV_HEADS=4, MLP_MULT=2, TIE_EMBEDDINGS=1 |
| SNN | SNN_T=4, SNN_H_GRU=64, SNN_BETA=0.9, SNN_THRESH=0.5 |
| Iterations | 20000 |
| Batch | TRAIN_BATCH_TOKENS=524288, TRAIN_SEQ_LEN=1024 |
| Quant | int8 + zlib (per-row int8 for ≥2-D tensors, fp16 otherwise) |

## Launch command

```bash
RUN_ID=snn_8gpu_real_v2 \
DATA_PATH=/data/parameter-golf/data/datasets/fineweb10B_sp1024 \
TOKENIZER_PATH=/data/parameter-golf/data/tokenizers/fineweb_1024_bpe.model \
SNN_T=4 SNN_H_GRU=64 SNN_BETA=0.9 SNN_THRESH=0.5 \
ITERATIONS=20000 VAL_LOSS_EVERY=1000 MAX_WALLCLOCK_SECONDS=0 \
torchrun --standalone --nproc_per_node=8 \
  records/track_non_record_16mb/2026-04-30_SpikingMLP_GRU_T4/train_gpt.py
```

`MAX_WALLCLOCK_SECONDS=0` disables the 10-minute record-track cap so all 20 000 iterations run.

(`DISABLE_COMPILE=1` was used locally because the H400 box was missing python3.10-dev for triton; the script defaults to compile-on for any environment that has it. Setting `DISABLE_COMPILE=0` or unsetting it restores the standard `torch.compile` path.)

## Files

- `train_gpt.py` — code snapshot (only delta vs root: `_SpikeFn`, `SpikingMLP`, four extra Hyperparameters env knobs, `Block` switch, two `DISABLE_COMPILE` guards)
- `train.log` — full remote training log
- `submission.json` — leaderboard metadata

## Notes

This is an exploratory non-record entry. The SNN/GRU adapter increases per-step cost vs the relu² MLP (about 1.4×) without an offsetting BPB gain at the 20K-iteration / 16 MB scale we ran here, so it does not contest the headline record. It is submitted as a working reference for the *adapter on random linear maps* and *per-token state-space* wishlist items, with all the training plumbing (DDP, Muon, int8+zlib quant, val\_bpb roundtrip) preserved from the root script.
