# VarLenAttn + Phased Global SGD TTT

Builds directly on [PR #1530](https://github.com/openai/parameter-golf/pull/1530). Training is unchanged. Evaluation changes as follows:

1. Run the stock PR1530 LoRA TTT evaluator on its single global length-sorted queue.
2. After `2000` queue-completed documents have been fully scored, pause once.
3. Gather exactly those already-scored documents in queue order.
4. Run distributed global SGD on that scored prefix.
5. Resume the same queue with the updated base model.

This keeps PR1530's fast batched LoRA TTT while adding one legal global score-first adaptation phase.

## Legality

- LoRA scoring happens before LoRA updates on those chunks.
- Global SGD only trains on documents that have already been fully scored.
- After the pause, evaluation resumes on future queue items only.

So no token is used for adaptation before its score has already been counted.

## Results

| Seed | val_loss | val_bpb | eval_time | artifact_size |
|---:|---:|---:|---:|---:|
| 0 | 2.76951521 | 1.07216564 | 500.104 s | 15,996,697 B |
| 1 | 2.77167493 | 1.07300174 | 515.324 s | 15,995,985 B |
| 2 | 2.77232000 | 1.07325147 | 504.949 s | 15,988,805 B |
| **avg** | **2.77117005** | **1.07280628** | **506.792 s** | **15,993,829 B** |

Compared to the original PR1530 submission mean:

| Metric | PR1530 | This submission | Delta |
|---|---:|---:|---:|
| val_loss | 2.77261037 | 2.77117005 | -0.00144032 |
| val_bpb | 1.07336388 | 1.07280628 | -0.00055760 |

All three seeds are under the 600s eval budget.

## Run

Full submission pipeline for one seed, from training through quantization and phased eval:

```bash
SEED=0 ARTIFACT_DIR="runs/varlen0" \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```

Eval-only on an existing checkpoint:

```bash
SEED=0 EVAL_ONLY_PATH="runs/varlen0/final_model.pt" \
PHASED_TTT_ENABLED=1 PHASED_TTT_PREFIX_DOCS=2000 \
  torchrun --standalone --nproc_per_node=8 train_gpt.py
```
