# Parameter Golf — Submission Strategy

## Current Standing
- Our best single-4090 result: **1.3620 bpb** (2677 steps, no SWA, standard eval)
- Leaderboard #1: **1.1428 bpb** (8xH100, 10 min)
- Gap: ~0.22 bpb

## What 8xH100 Changes

### Compute budget
- 8xH100 with DDP + torch.compile: ~10-20x more steps in 10 min vs single 4090
- Estimate: 15,000-20,000 steps (vs our 2677)
- torch.compile: additional 1.5-2x speedup

### Expected gains from more steps
- 1000 → 2677 steps gave -0.14 bpb
- Curve still steep — 20K steps could give another -0.15 to -0.20 bpb
- Projected: ~1.16-1.21 bpb range

## Submission Plan

### Phase 1: Validate on 8xH100 (RunPod)
1. Spin up 8xH100 instance on RunPod
2. Run best config with torch.compile + DDP enabled
3. Full 10 min wallclock run
4. Measure actual steps achieved and val_bpb
5. Compare with 4090 results to calibrate

### Phase 2: Tune for 8xH100
1. **Batch size:** Use default 786K tokens — 8 GPUs handle it easily
2. **SWA:** Re-test — may help with 20K steps (only proved harmful at 2.7K steps)
3. **Sliding eval stride=64:** Enable (free bpb improvement at eval time)
4. **Warmdown 3000:** Keep as default, test 5000 at 20K step scale
5. **Learning rate:** May need tuning for larger effective batch with DDP

### Phase 3: Quantization Optimization
1. **Int5 MLP quantization** (leaderboard #1 uses mixed int5/int6)
2. **Zstd level 22** compression: Already using, verify level is optimal
3. **Magnitude pruning threshold:** Tune the 3% cutoff — try 2% and 5%
4. **FP16 embedding keep:** Verify tok_emb fp16 is optimal vs quantized
5. **16MB budget analysis:** Calculate exact param budget after quantization

### Phase 4: Advanced Techniques (if time permits)
1. **Larger bigram vocab:** Try 30K or 40K (more room in 16MB with better quant)
2. **QAT (Quantization-Aware Training):** Train with int6 noise in the loop
3. **Sequence length 2048:** Winners use this — test on 8xH100
4. **Test-time training (TTT):** LoRA adaptation at eval (leaderboard has 1.1928 with this)

## Key Decisions
- **SWA:** OFF by default, re-test on 8xH100 at 20K steps
- **Eval:** Sliding window stride=64
- **Architecture:** 12L x 512d x MLP3.5x x Bigram20K (proven best)
- **Quantization:** Mixed int5(MLP)/int6(attn) + zstd-22

## Risk Assessment
- Main risk: 4090 experiments don't perfectly predict 8xH100 behavior
- Mitigation: 2-3 validation runs on RunPod before final submission
- SWA may behave differently at 20K steps — must re-test
- Deadline: April 30, 2026

## Estimated Compute Cost
- RunPod 8xH100: ~$25/hour
- 5 validation runs × 15 min each = ~$30
- Budget: $50-100 for full tuning cycle
- OpenAI compute grant: apply if still available
