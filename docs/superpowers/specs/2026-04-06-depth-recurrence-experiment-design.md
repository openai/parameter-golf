# Experiment 4: Depth Recurrence — Design Spec

## Goal

Beat merged SOTA (1.1147 BPB) using depth recurrence. Target: ~1.09 BPB.

## Strategy

Adopt PR #1421's proven script (1.0925 BPB, 3-seed mean) as our base. Optionally add BigramHash. This is low-risk because the script is already validated at competition scale.

## Why PR #1421 Over Porting Recurrence Into Our Script

Our SP1024 SOTA (1.1147) is already 0.022 BPB behind PR #1421 (1.0925). The gap comes from 6+ independent improvements (SP4096, MuonEq-R, skip gates, parallel residuals, QK-Gain 5.0, WD 0.09, EMA 0.9965) — not just recurrence. Porting all of these into our parameter-bank architecture would be high effort and high risk. Using the proven script directly is the pragmatic choice.

## Architecture (PR #1421, verbatim)

- 11 physical layers, 512d, 8 heads, 4 KV heads (GQA)
- Depth recurrence: layers 4,5 repeat once (13 virtual layers: `[0,1,2,3,4,5,4,5,6,7,8,9,10]`)
- Recurrence activates at step 3000 (~55% through training)
- Skip gates: learnable sigmoid gating on U-Net skip connections
- Parallel residuals: layers 7+ run attention and MLP in parallel lanes, merged via learnable scalar
- SP4096 tokenizer (SentencePiece 4096 BPE)
- MuonEq-R: row normalization before Newton-Schulz orthogonalization
- Value Embedding: dim=128, layers 9,10
- QK-Gain: learnable per-head Q scaling, init=5.0
- Tied embeddings, logit softcap=30.0, partial RoPE (16/64 dims)
- XSA on all 11 layers

## Training Hyperparameters

| Parameter | Value |
|-----------|-------|
| Muon LR | 0.02 |
| Muon momentum | 0.99 |
| Muon WD | 0.09 |
| Muon backend steps | 5 |
| Embed LR | 0.6 |
| Embed WD | 0.09 |
| Head LR | 0.008 |
| Scalar LR | 0.02 |
| Scalar WD | 0.02 |
| Grad clip | 0.3 |
| Batch size | 786,432 tokens/step |
| Seq len | 2048 |
| Warmdown fraction | 0.667 |
| EMA decay | 0.9965 |
| Warmup steps | 20 |
| Wallclock cap | 600s (590s effective) |

## Quantization

- GPTQ int6, percdamp=0.05, 64 calibration batches
- 10s reserved for GPTQ at end of training
- Selective pruning of ~290K lowest-error values
- Brotli compression
- Expected artifact: ~15.95 MB

## Enhancement: BigramHash (Optional)

One modification on top of the proven base:
- BigramHash with 1536 buckets, dim 112 (sized smaller than our SOTA's 3072 to leave room for SP4096 vocab table)
- Requires porting BigramHash and SmearGate classes from our SOTA script into PR #1421's script, plus wiring them into the GPT.__init__ and forward methods. This is ~60 lines of code.
- Decision rule: Run seed 1337 vanilla FIRST to confirm reproduction. Then run seed 1337 with BigramHash. If artifact > 16MB or BPB regresses, strip it.
- Rationale: PR #363 found BigramHash neutral on heavy looping (3x3), but PR #1421's minimal recurrence (2 layers, 1 extra pass) is much closer to flat, so BigramHash may still help.

## RunPod Execution

### Pod Setup
```bash
runpodctl pod create \
  --template-id y5cejece4j \
  --gpu-id "NVIDIA H100 80GB HBM3" \
  --gpu-count 8 \
  --name "pgolf-exp4-recurrence" \
  --cloud-type SECURE
```

### On-Pod Setup
```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
pip install --break-system-packages zstandard brotli
python3 data/cached_challenge_fineweb.py --variant sp4096
```

### Run Sequence

1. **Run 1** (seed 1337): Vanilla PR #1421 script. Verify ~1.0925 BPB reproduction.
2. **Run 2** (seed 1337): With BigramHash added. Compare to Run 1.
3. **Runs 3-4** (seeds 42, 2024): Best config from above, 2 more seeds for 3-seed submission.
4. Stop pod immediately.

Each run: ~10 min training + ~5 min eval = ~15 min. Total: ~60 min. Cost: ~$22.

### Script Transfer
- Extract `train_gpt.py` from PR #1421 diff locally, save to `experiments/exp4_train_gpt.py`
- SCP to pod: `scp -i ~/.runpod/ssh/RunPod-Key-Go -P <port> experiments/exp4_train_gpt.py root@<ip>:/workspace/parameter-golf/train_gpt.py`

## Submission Structure

```
records/track_10min_16mb/2026-04-06_DepthRecurrence_EMA0.9965/
  README.md
  submission.json
  train_gpt.py
  train_seed1337.log
  train_seed42.log
  train_seed2024.log
```

PR from `AbhayAnandUCSD/parameter-golf` fork to `openai/parameter-golf`.

## Expected Results

| Scenario | Expected BPB | Delta vs SOTA (1.1147) |
|----------|-------------|----------------------|
| Vanilla reproduction | ~1.093 | -0.022 |
| With BigramHash | ~1.088-1.093 | -0.022 to -0.027 |
| Worst case | ~1.10 | -0.015 |

## Risk Mitigation

| Risk | Mitigation |
|------|-----------|
| SP4096 data download fails | Modify script for SP1024 (change vocab_size, paths). Lose ~0.01 BPB. |
| BigramHash breaks 16MB budget | Strip it, run vanilla. Already proven at 1.0925. |
| Recurrence compile stall | Forward loop is explicit Python, not traced by torch.compile. Already handled in PR #1421. |
| Pod unavailable | Try community cloud. If unavailable, wait and retry. |
| Reproduction fails (>1.10 BPB) | Check data shard count (must be all shards, not 1). Verify SP4096 data downloaded correctly. |

## Key Lessons From Failed Approach (PR #363)

These informed our strategy but do NOT apply to PR #1421's minimal recurrence:
- Heavy looping (3x3, 2x5) causes 22% step count loss and quantization compounding
- PR #1421 avoids both: only 2 layers repeat once (minimal overhead), activated late at step 3000
- Noisy QAT was critical for heavy looping but unnecessary for minimal recurrence with GPTQ int6
- BigramHash was neutral on heavy loops but may help with minimal recurrence (untested)
