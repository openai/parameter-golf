#!/bin/bash
# Wait for round 5 to finish, write results, shut down

cd /c/Users/deepc/parameter-golf

echo "Waiting for round 5 experiments to finish..."
while ps aux | grep -v grep | grep -q "run_round5"; do
    sleep 60
done
echo "Round 5 complete. Writing results..."

# Collect ALL experiment results into a markdown file
cat > EXPERIMENT_RESULTS.md << 'HEADER'
# Parameter Golf — Experiment Results

**Config:** 12L x 512dim, 8 heads, 4 KV heads, bigram 20K, MLP 3.5x
**Hardware:** RTX 4090 (single GPU), NO_COMPILE
**Baseline leaderboard #1:** 1.1428 bpb (8xH100, 10 min)

## Summary of Best Results

| Config | Steps | Val BPB (quant) | Notes |
|---|---|---|---|
HEADER

# Extract results from all experiments
for name in exp_control exp_bigger_batch exp_12_layers exp_12_layers_v2 exp_bigram_20k exp_bigram_dim256 exp_mlp_3.5x exp_muon_095 exp_wd_002 exp_combined_v2 exp_combined_v3 exp_16L_448d exp_long_3000 exp_16L_kv7 exp_16L_long_3000 exp_r5_control exp_r5_swa_default exp_r5_swa_early exp_r5_swa_freq25 exp_r5_wd5000 exp_r5_wd1500 exp_r5_slide64 exp_r5_slide32; do
    f="logs/${name}.txt"
    [ ! -f "$f" ] && continue
    final=$(grep "^final_int8_zlib_roundtrip_exact" "$f" 2>/dev/null | head -1)
    last_step=$(grep "^step:" "$f" 2>/dev/null | tail -1)
    stopped=$(grep "^stopping" "$f" 2>/dev/null)
    params=$(grep "^model_params" "$f" 2>/dev/null | head -1 | grep -o "[0-9]*" | head -1)
    swa_info=$(grep "^swa:applying" "$f" 2>/dev/null)
    eval_mode=$(grep "^final_eval_mode" "$f" 2>/dev/null)

    bpb=""
    if [ -n "$final" ]; then
        bpb=$(echo "$final" | grep -o "val_bpb:[0-9.]*" | cut -d: -f2)
    fi

    steps=""
    if [ -n "$last_step" ]; then
        steps=$(echo "$last_step" | grep -o "step:[0-9]*/[0-9]*" | head -1)
    fi

    echo "| $name | $steps | $bpb | $swa_info $eval_mode |" >> EXPERIMENT_RESULTS.md
done

cat >> EXPERIMENT_RESULTS.md << 'BODY'

## Round-by-Round Analysis

### Round 1 — Baseline Experiments (1000 steps)
- **Control:** val_bpb 1.5298 (10L, 512d, bigram 10K, MLP 3x)
- BigramHash is critical (-0.55 val_loss without it)
- Width over depth doesn't work (8L/640d much worse)
- Bigger batch hurts (halves steps, 2x slower per step)

### Round 2 — Architecture Sweep (1000 steps)
- **12 layers:** -0.019 bpb (biggest single gain)
- **Bigram 20K vocab:** -0.011 bpb
- **MLP 3.5x:** -0.005 bpb
- Gains are additive: combined_v3 (12L+bigram20K+MLP3.5x) = -0.027 bpb
- Lower WD and lower Muon momentum both hurt

### Round 3 — Architectural Experiments
- **SwiGLU:** worse than squared ReLU (+0.066 train_loss)
- **Trigram hash:** no benefit (+0.073 train_loss)
- **Depth recurrence 6x2:** worse (-0.111, halved unique params)
- **Depth recurrence 4x3:** worse (-0.141)
- **16L x 448d:** best at 1000 steps (1.4987 bpb) but scales worse
- **Long run (2617 steps):** 1.3647 bpb — massive scaling confirmed

### Round 4 — 16L vs 12L Scaling
- **MHA >> GQA** for this model size
- 18L and 20L too slow per step, don't compensate
- **12L x 512d scales better than 16L x 448d** at longer runs
- 12L long run: 1.3647 bpb vs 16L long run: 1.4011 bpb

### Round 5 — SWA, Warmdown, Sliding Eval
- **SWA hurts quantization** — more checkpoints = worse quant bpb
- Pre-quant val_bpb identical with/without SWA (~1.359)
- SWA smooths weights in ways that quantize poorly
- **Warmdown 1500:** nearly ties control (SWA damage minimal with few ckpts)
- **Warmdown 5000:** much worse (enters warmdown too early)
- **Sliding eval:** results pending (stride 64 and 32)

## Confirmed Best Config (for submission)

```
NUM_LAYERS=12
MODEL_DIM=512
NUM_HEADS=8
NUM_KV_HEADS=4
MLP_MULT=3.5
BIGRAM_VOCAB_SIZE=20480
BIGRAM_DIM=128
SWA_ENABLED=0 (hurts quantization on single GPU)
WARMDOWN_ITERS=3000
EVAL_STRIDE=64 (pending final results)
WEIGHT_DECAY=0.04
MUON_MOMENTUM=0.99
```

## Strategy for 8xH100 Submission

See SUBMISSION_STRATEGY.md
BODY

echo "Results written to EXPERIMENT_RESULTS.md"

# Now write the strategy
cat > SUBMISSION_STRATEGY.md << 'EOF'
# Parameter Golf — Submission Strategy

## Current Standing
- Our best single-4090 result: **1.3620 bpb** (2677 steps, no SWA, standard eval)
- Leaderboard #1: **1.1428 bpb** (8xH100, 10 min)
- Gap: ~0.22 bpb

## What 8xH100 Changes

### Compute budget
- 8xH100 with DDP: ~10-20x more steps in 10 min vs single 4090
- Estimate: 15,000-20,000 steps (vs our 2677)
- torch.compile enabled: additional 1.5-2x speedup

### Expected gains from more steps
- 1000 steps → 2677 steps gave -0.14 bpb improvement
- Curve is still steep — 20K steps could give another -0.15 to -0.20 bpb
- Projected: ~1.16-1.21 bpb range

## Submission Plan

### Phase 1: Validate on 8xH100 (RunPod)
1. Spin up 8xH100 instance on RunPod
2. Run our best config with torch.compile + DDP
3. Let it run for full 10 min wallclock
4. Measure actual steps achieved and val_bpb

### Phase 2: Tune for 8xH100
1. **Batch size:** Scale up to 786K tokens (default) — 8 GPUs can handle it
2. **SWA:** Re-test on 8xH100 — may help with 20K steps (only hurt at 2.7K)
3. **Sliding eval stride:** Use stride=64 (free bpb improvement at eval time)
4. **Warmdown:** Test 3000 vs 5000 iters at 20K step scale
5. **Learning rate:** May need tuning for larger effective batch

### Phase 3: Architecture Optimizations
1. **Int5 MLP quantization** (leaderboard #1 uses this): Mixed int5/int6
2. **Zstd level 22** compression: Better compression = more params in 16MB
3. **Magnitude pruning threshold:** Tune the 3% pruning cutoff
4. **FP16 embedding keep:** Currently keeps tok_emb in fp16 — verify this is optimal

### Phase 4: Advanced Techniques (if time permits)
1. **Test-time training (TTT):** LoRA adaptation at eval time (leaderboard has this at 1.1928)
2. **Larger bigram vocab:** Try 30K or 40K
3. **Sequence length 2048:** Winners use this — test impact on 8xH100
4. **QAT (Quantization-Aware Training):** Train with int6 in the loop

## Key Decisions
- **SWA:** OFF unless 8xH100 testing shows clear benefit at 20K steps
- **Eval:** Sliding window stride=64 (pending final confirmation)
- **Architecture:** 12L x 512d x MLP3.5x x Bigram20K (proven best)
- **Quantization:** Mixed int5(MLP)/int6(attn) + zstd-22

## Risk Assessment
- Main risk: our 4090 experiments don't perfectly predict 8xH100 behavior
- Mitigation: budget 2-3 runs on RunPod to validate before final submission
- Deadline: April 30, 2026

## Estimated Compute Cost
- RunPod 8xH100: ~$25/hour
- 5 validation runs × 15 min each = ~$30
- Budget: $50-100 for full tuning cycle
- OpenAI compute grant: apply if available
EOF

echo "Strategy written to SUBMISSION_STRATEGY.md"

# Append sliding eval results if they're done
for name in exp_r5_slide64 exp_r5_slide32; do
    f="logs/${name}.txt"
    if [ -f "$f" ]; then
        final=$(grep "^final_int8_zlib_roundtrip_exact" "$f" 2>/dev/null)
        if [ -n "$final" ]; then
            echo "" >> EXPERIMENT_RESULTS.md
            echo "### Sliding Eval Result: $name" >> EXPERIMENT_RESULTS.md
            echo "$final" >> EXPERIMENT_RESULTS.md
        fi
    fi
done

echo "All done. Shutting down in 60 seconds..."
sleep 60
shutdown /s /t 0
