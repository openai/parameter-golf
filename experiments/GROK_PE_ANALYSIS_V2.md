# Grok PE Recommendations V2 — Quick Assessment

## Reality check on Grok's claims
Grok claims "+0.02-0.04 BPB" for partial RoPE and ALiBi. These numbers are wildly optimistic.
Our ROPE_BASE=200K test (exp066) showed changing RoPE params is NEUTRAL. The model doesn't care
much about PE details at 2048 seq_len with only 9 layers.

## Worth testing (in priority order)

### 1. Partial RoPE 10-25% — LOW RISK, test it
- Only rotate 10-25% of head dimensions
- Saves some compute → faster steps → more steps
- Easy 8-line change
- Realistic expectation: +0.000-0.002 BPB, maybe 1-2ms/step faster
- **ADD TO EXPERIMENT QUEUE**

### 2. DroPE (drop RoPE at end) — INTERESTING
- Remove rotary for last 200 steps of training
- Could help sliding window eval extrapolation
- BUT: with only 7300 total steps, dropping at step 7100 means very little recalibration time
- Realistic expectation: +0.000-0.003 BPB
- **ADD TO EXPERIMENT QUEUE after partial RoPE**

### 3. ALiBi — SKIP
- Can't use with F.scaled_dot_product_attention + enable_gqa (needs attention bias)
- Would require custom attention implementation
- Not worth the engineering risk for uncertain gain
- **SKIP**

### 4. PoPE — SKIP
- Brand new, unproven at this scale
- Extra complexity for uncertain gain
- **SKIP**

## Bottom line
PE changes are unlikely to be our biggest lever. Our gap is:
1. Artifact size (1.7MB over budget) — CRITICAL
2. BPB improvements (0.003-0.005 possible from tuning) — SECONDARY

Focus on compression + Runpod testing. PE changes are nice-to-have experiments.
