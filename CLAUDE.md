# Parameter Golf

## Project

OpenAI's Parameter Golf challenge: train the best language model that fits in a 16MB artifact (code + int8+zlib compressed weights), under 10 minutes on 8xH100 SXM. Scored by `val_bpb` (bits per byte) on FineWeb validation set. Challenge runs March 18 — April 30, 2026.

## Our Novel Contribution: Memory Tokens

32 learnable tokens (`NUM_MEMORY_TOKENS=32`) overwrite the first 32 positions of every input sequence. All real tokens can attend to them via causal mask, giving every position access to learned global context — like a shared "cheat sheet." Cost: 16K params (0.1% of model). Memory positions use `ignore_index=-100` so they're excluded from loss.

## Current Config

Our best approach combines memory tokens with proven techniques:
- **Memory tokens (32)** — our novel idea
- **TRAIN_SEQ_LEN=2048** — longer context during training
- **Sliding window eval** (`EVAL_STRIDE=64, EVAL_SEQ_LEN=1024`) — scores tokens with near-full context at eval time (runs after training, doesn't count against 10 min cap)
- **FP16 embedding export** (`FP16_EMBED_EXPORT=1`) — keeps tok_emb in fp16 instead of int8
- **Muon weight decay** (`MUON_WEIGHT_DECAY=0.02`) — decoupled weight decay for generalization

MTP (multi-token prediction) was tested and dropped — showed no improvement over memory tokens.

## Experiment Results (1xH100 SXM, 10 min cap)

| Run | Steps | val_bpb | Notes |
|-----|-------|---------|-------|
| Original baseline | 1078 | 1.3500 | |
| Memory tokens (32) + seq 1024 | 1336 | 1.3333 | |
| Memory tokens (32) + seq 2048 | 1379 | 1.3080 | Best confirmed |
| Full combo (pending) | — | — | + sliding window + fp16 + muon wd |

Note: 1xH100 results are for relative comparison only. 8xH100 gets many more steps and much better final val_bpb.

## Repo Structure

- `train_gpt.py` — main training script (model, optimizer, eval, serialization)
- `run_mtp.sh` — current experiment command (1xH100), includes pod setup
- `run_8xh100.sh` — submission run command (8xH100), includes pod setup
- `run_save.sh` — scp artifacts from pod (edit IP/port per pod)
- `data/` — dataset download scripts and tokenizers
- `records/` — leaderboard submissions

## Git Setup

- Fork: `sp00mm/parameter-golf` on GitHub
- Branch: `mtp-auxiliary-heads`
- Remote `origin` = our fork, remote `upstream` = `openai/parameter-golf`
- Local git config uses personal GitHub identity (sp00mm@users.noreply.github.com)

## RunPod Workflow

1. Deploy H100 SXM pod using Parameter Golf template (has torch 2.8+)
2. SSH in, run setup from `run_mtp.sh` (clone, checkout, download data)
3. Run experiment command
4. Copy results via `run_save.sh` (use direct TCP, not relay)
5. Terminate pod

## Submission Requirements

PR to `openai/parameter-golf` adding `records/track_10min_16mb/YYYY-MM-DD_MemoryTokens/`:
- `README.md` — explains memory tokens approach
- `submission.json` — author (Austin Tarango), github_id (sp00mm), val_bpb, byte sizes
- `train_gpt.py` — standalone script
- `train.log` — from 8xH100 SXM run
- Must beat SOTA by 0.005 nats at p < 0.01 (3+ seed runs needed)

## Next Steps

1. Check results of full combo run (memory + seq2048 + sliding window + fp16 + muon wd)
2. If good, do 8xH100 submission run with `run_8xh100.sh`
3. Save `submission.log` and `final_model.int8.ptz` from pod
4. Prepare submission PR
