# Autoresearch: Heads4 + Step-based LR + Sliding Window

**Non-record submission** — tested on 1xH100 only. Requesting compute grant for 8xH100 validation.

**val_bpb: 1.2727** (1xH100, 800 steps, sliding window eval stride=256)

## Approach: Automated Experiment Loop

This submission was developed using an autonomous experimentation methodology inspired by Karpathy's autoresearch. An automated experiment loop:
1. Modifies `train_gpt.py` with an experimental change
2. Runs training for a fixed time budget
3. Compares val_bpb against the current best
4. Keeps improvements, reverts failures
5. Repeats indefinitely

**75+ experiments** were run across three phases:
- **Phase 1 (Mac MLX, 40 experiments)**: Hyperparameter search discovered step-based LR schedule, optimal learning rates, and warmdown tuning
- **Phase 2 (1xH100 CUDA, 10 experiments)**: Validated Mac findings on CUDA, discovered NUM_HEADS=4 with head_dim=128 as a major architectural win
- **Phase 3 (1xH100 CUDA, 25 experiments)**: Built on current SOTA code, combining techniques from multiple leaderboard submissions

## Key Findings

### Confirmed Improvements (relative to SOTA baseline on 1xH100)
| Technique | Relative BPB Change | Source |
|-----------|-------------------|--------|
| NUM_HEADS=4, NUM_KV_HEADS=2 (head_dim=128) | **-0.095** | Our experiment |
| Step-based LR schedule (MWS=0) | **-0.483** | Our experiment |
| BigramHash 10240→16384 | -0.025 | Our experiment |
| MATRIX_LR 0.02→0.03 | -0.003 | Our experiment |

### Confirmed Non-improvements
| Technique | Result |
|-----------|--------|
| LoRA test-time training | Worse (-0.09 BPB) — chunk-based eval hurts |
| Block-wise weight sharing (2x) | Worse + 2x slower |
| NUM_KV_HEADS=1 (MQA) | Worse quality |
| SwiGLU activation | Worse than relu^2 |
| seq_len=4096 | Too slow per step |
| 11 layers | Better BPB but over 16MB budget |
| BigramHash(20480) | Better BPB but over 16MB budget |

## Architecture (built on SOTA)
- **10 layers**, 512 dim, **4 heads** (head_dim=128), **2 KV heads** (GQA)
- MLP 3x expansion (hidden=1536), relu^2 activation
- SmearGate + BigramHash(16384, dim=128) + orthogonal init
- SWA (start_frac=0.4, every=50 steps)
- Tied embeddings (FP16 passthrough, not int8 quantized)
- U-Net skip connections

## Training
- Muon optimizer: matrix_lr=0.03, WD=0.04, momentum=0.99
- **Step-based LR schedule** (not wallclock-based): ITERATIONS=800, WARMDOWN_ITERS=170
- seq_len=2048, batch=786K tokens
- grad_clip=0.3

## Evaluation
- Sliding window eval, stride=256, compiled forward_logits
- Int5 MLP / Int6 attention / FP16 embeddings + zstd compression

## Command
```bash
torchrun --standalone --nproc_per_node=1 train_gpt.py
```

Note: Tested on 1xH100 only (800 steps in 600s). On 8xH100 this would get ~13,780 steps and significantly better BPB.

## Experiment Logs

Full experiment history in results_v3.tsv (25 CUDA experiments) and results.tsv (40 Mac experiments).
