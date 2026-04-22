We implemented a redundancy-only regularizer (TC - DTC) with eigendecomposition-based DTC scaling. Results
 were bad:
 - BPB regression: 1.9495 vs 1.8866 baseline (+0.063)
 - 12x slowdown: eigendecomposition on CPU per layer per step is catastrophically expensive on MLX
 - Lambda (0.005) was also too aggressive

 Root cause of slowdown: mx.linalg.eigh on a 512x512 matrix, dispatched to CPU via stream=mx.cpu, per layer
 per step. Even with stop_gradient, the computation itself blocks the GPU pipeline.

 Root cause of BPB regression: The regularizer was active from step 0, fighting the model before it learned
 useful representations. Combined with too-high lambda, it prevented convergence.

 Baseline Results

 ┌──────┬───────────────┬────────────────┬───────────┬────────────────┐
 │ Seed │ Pre-quant BPB │ Post-quant BPB │ Quant gap │ Artifact bytes │
 ├──────┼───────────────┼────────────────┼───────────┼────────────────┤
 │ 216  │ 1.8864        │ 1.8875         │ 0.0011    │ 15,704,750     │
 ├──────┼───────────────┼────────────────┼───────────┼────────────────┤
 │ 455  │ 1.8983        │ 1.8995         │ 0.0012    │ 15,678,131     │
 ├──────┼───────────────┼────────────────┼───────────┼────────────────┤
 │ 3308 │ 1.8718        │ 1.8729         │ 0.0011    │ 15,730,838     │
 ├──────┼───────────────┼────────────────┼───────────┼────────────────┤
 │ Mean │ 1.8855        │ 1.8866         │ 0.0011    │ 15,704,573     │
 └──────┴───────────────┴────────────────┴───────────┴────────────────┘

 Config: 1500 iterations, 8192 batch tokens, seq_len 1024, warmdown_iters 450.

 Failed Experiment Results

 ┌──────────────────────────────┬────────────────┬───────────────┬──────────┐
 │            Config            │ Post-quant BPB │ Step overhead │ Verdict  │
 ├──────────────────────────────┼────────────────┼───────────────┼──────────┤
 │ TC-DTC, λ=0.005, from step 0 │ 1.9495         │ ~12x          │ Unusable │
 └──────────────────────────────┴────────────────┴───────────────┴──────────┘

 ---
 New Plan: Drop Eigendecomposition Entirely

 Approach: Pure Correlation Frobenius Proxy + Delayed Start

 Strip the regularizer to only the differentiable part — off-diagonal correlation energy. No
 eigendecomposition, no CPU dispatch, no DTC.

 What we keep: tc_proxy = 0.5 * (||R||²_F - d) — penalizes squared off-diagonal correlations in the
 hidden-state correlation matrix. This is O(d²) via one matmul + element-wise ops, fully on GPU.

 What we drop: The entire non-differentiable DTC scaling block (eigendecomposition, redundancy fraction). We
 lose the "only penalize redundancy, not synergy" selectivity, but we gain:
 - Near-zero overhead (one d×d matmul per collected layer)
 - No CPU sync stalls

 What we add: TC_START_STEP env var — regularizer is zero before this step, then ramps in. This lets the model
  learn good representations first.

 Why this is still information-theoretically motivated

 For near-Gaussian activations, ||R||²_F - d is proportional to the sum of squared pairwise mutual
 informations, which is a lower bound on TC. Penalizing it encourages statistical independence across hidden
 dimensions — the same decorrelation that makes weight matrices more compressible under quantization.

 It's no longer "redundancy-only" (we lose DTC selectivity), but it's still a novel TC-proxy regularizer not
 used in prior parameter-golf submissions.

 ---
 Changes to train_gpt_mlx.py

 1. Add TC_START_STEP hyperparameter (line ~110)

 tc_start_step: int = int(os.environ.get("TC_START_STEP", "0"))

 2. Replace redundancy_regularizer function (lines 350-388) with lightweight version

 def correlation_regularizer(hidden_states: list[mx.array]) -> mx.array:
     """Decorrelation regularizer: penalizes off-diagonal correlations in hidden states.
     ||R||²_F - d ∝ sum of squared pairwise mutual informations (Gaussian approx)."""
     reg_total = mx.array(0.0, dtype=mx.float32)
     for h in hidden_states:
         h_flat = h.reshape(-1, h.shape[-1])
         h_c = h_flat - mx.mean(h_flat, axis=0, keepdims=True)
         n = h_flat.shape[0]
         cov = (h_c.T @ h_c) / n + 1e-6 * mx.eye(h_flat.shape[-1])
         std = mx.sqrt(mx.diag(cov))
         corr = cov / (std[:, None] * std[None, :] + 1e-8)
         d = float(corr.shape[0])
         reg_total = reg_total + 0.5 * (mx.sum(corr * corr) - d)
     return reg_total

 ~13 lines, down from ~38.

 3. Update GPT.loss (line ~582) — call new function + respect start step

 if use_tc:
     return ce + self.tc_lambda * correlation_regularizer(hidden_states)
 Function name change only. The start-step gating goes in the loss method or the __call__ method — pass step
 as an argument, or gate return_hidden at the call site.

 Cleanest approach: Gate in loss() by adding a step parameter:
 def loss(self, input_ids, target_ids, step=0):
     use_tc = self.tc_enabled and self.tc_lambda > 0 and step >= self.tc_start_step
     ...
 Then pass step from the training loop (line ~1185 area).

 4. Pass step through the training loop call chain

 The training loop calls loss_and_grad_chunked (line ~879) which calls compiled_loss_and_grad. Need to thread
 step through. Check if mx.compile can handle the extra arg or if we gate before the compiled function.

 Simplest: Gate at the call site — set model.tc_enabled = (step >= tc_start_step) before the loss call each
 step, avoiding any changes to the compiled function signature.

 5. Hidden state collection guards (already done)

 - Encoder: i > 0 — skip first layer ✅
 - Decoder: i < num_decoder_layers - 1 — skip last layer ✅

 ---
 Run Plan

 Experiment 1: Verify near-zero overhead

 RUN_ID=corr_overhead SEED=216 ITERATIONS=50 TRAIN_BATCH_TOKENS=8192 \
   VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 TRAIN_SEQ_LEN=1024 \
   WARMUP_STEPS=5 MAX_WALLCLOCK_SECONDS=0 WARMDOWN_ITERS=10 \
   TC_ENABLED=1 TC_LAMBDA=0.001 TC_START_STEP=0 \
   python3 train_gpt_mlx.py
 Compare step_avg to baseline (~1018ms). Target: < 5% overhead.

 Experiment 2: Delayed start, low lambda (seed 216)

 RUN_ID=corr_s1 SEED=216 ITERATIONS=1500 TRAIN_BATCH_TOKENS=8192 \
   VAL_LOSS_EVERY=500 VAL_BATCH_SIZE=8192 TRAIN_SEQ_LEN=1024 \
   WARMUP_STEPS=20 MAX_WALLCLOCK_SECONDS=0 WARMDOWN_ITERS=450 \
   TC_ENABLED=1 TC_LAMBDA=0.001 TC_START_STEP=500 \
   python3 train_gpt_mlx.py

 Experiment 3: If Exp 2 promising, run seeds 455 and 3308

 Metrics

 ┌────────────────┬─────────────────┬──────────┬────────────────┐
 │     Metric     │ Baseline (mean) │  Target  │ Fail threshold │
 ├────────────────┼─────────────────┼──────────┼────────────────┤
 │ Post-quant BPB │ 1.8866          │ < 1.885  │ > 1.890        │
 ├────────────────┼─────────────────┼──────────┼────────────────┤
 │ Quant gap      │ 0.0011          │ < 0.0008 │ > 0.0015       │
 ├────────────────┼─────────────────┼──────────┼────────────────┤
 │ Step overhead  │ 0%              │ < 5%     │ > 10%          │
 └────────────────┴─────────────────┴──────────┴────────────────┘

 Verification

 1. Syntax check after edits
 2. 50-step overhead test (Experiment 1)
 3. Full 1500-step run seed 216 (Experiment 2)
 4. Compare all 4 metrics to baseline