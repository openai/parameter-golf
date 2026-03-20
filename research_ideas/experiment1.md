I want to create a compelling submission. Look throught the top ten scorers, list what they did and any    
interesting emergent principles across top scorers: https://github.com/openai/parameter-golf/pulls. Next, I  
want to focus on doing very efficient information decomopisition by nth-order polynomials as seen the the    
pdfs in research_ideas. Previously the adami team were limited in transfering their information theory       
approach to a gpu accerlatatable design. Read through these articles and see which have implications for     
super efficent learning if transfered to a parallaizable framework. There may be more than 1 investagatable  
aproach. https://mgi.natsci.msu.edu/labs/adami-lab/publications.aspx, &                                      
https://scholar.google.com/citations?hl=en&user=VXM7wUMAAAAJ&view_op=list_works&sortby=pubdate. cap your     
search to the last 6 years.                                                                                  
 Parameter Golf: Information-Theoretic Submission Plan                                                       
                                                                                                           
 Context

 We want to create a competitive parameter-golf submission by integrating information-theoretic principles
 from the Adami lab's Functional Information Decomposition (FID) research into a GPU-acceleratable
 transformer training pipeline. The current SOTA is ~1.1400 BPB. The competition constrains artifacts to
 16MB and training to 10 minutes on 8xH100.

 ---
 Part 1: Top 10 Leaderboard & Emergent Principles

 Top Scorers (by BPB, lower is better)

 ┌──────┬──────┬────────┬───────────────────────────────────────────────────────────────────┐
 │ Rank │  PR  │  BPB   │                          Key Techniques                           │
 ├──────┼──────┼────────┼───────────────────────────────────────────────────────────────────┤
 │ 1    │ #180 │ 1.1428 │ Mixed-precision int5/int6, 10 layers, SWA, BigramHash, SmearGate  │
 ├──────┼──────┼────────┼───────────────────────────────────────────────────────────────────┤
 │ 2    │ #236 │ 1.1400 │ 11L int6, SmearGate, batch optimization (524K tokens), BigramHash │
 ├──────┼──────┼────────┼───────────────────────────────────────────────────────────────────┤
 │ 3    │ #114 │ 1.1574 │ Int6 + MLP 3x, 2048-token training, selective FP16, GRAD_CLIP=0.3 │
 ├──────┼──────┼────────┼───────────────────────────────────────────────────────────────────┤
 │ 4    │ #243 │ 1.1704 │ Int6, 3x MLP, cosine warmdown schedule, 10 layers                 │
 ├──────┼──────┼────────┼───────────────────────────────────────────────────────────────────┤
 │ 5    │ #230 │ 1.1875 │ Int6 + NorMuon + weight decay, 3x MLP, 11 layers                  │
 ├──────┼──────┼────────┼───────────────────────────────────────────────────────────────────┤
 │ 6    │ #200 │ 1.2012 │ SP4096 tokenizer, int6 QAT, NorMuon                               │
 ├──────┼──────┼────────┼───────────────────────────────────────────────────────────────────┤
 │ 7    │ #61  │ 1.2154 │ Warmdown-quantization optimization, NTK-RoPE extrapolation        │
 ├──────┼──────┼────────┼───────────────────────────────────────────────────────────────────┤
 │ 8    │ #226 │ 1.3446 │ Low-rank all-attention, persistent memory replacing FFN           │
 ├──────┼──────┼────────┼───────────────────────────────────────────────────────────────────┤
 │ 9    │ #232 │ 1.4370 │ Error correction lookup table (eval-time hack)                    │
 ├──────┼──────┼────────┼───────────────────────────────────────────────────────────────────┤
 │ 10   │ #220 │ 1.848  │ SSM LRU baseline (compute-bottlenecked)                           │
 └──────┴──────┴────────┴───────────────────────────────────────────────────────────────────┘

 Emergent Principles Across Top Scorers

 1. Int6 quantization dominates — best compression/quality tradeoff; mixed int5 (MLP) + int6 (attention)
 frees space for extra layers
 2. 3x MLP expansion is standard — 1024→1536 hidden dim, consistent 0.05-0.10 BPB gain
 3. More gradient steps > more tokens — 524K batch (8,900 steps) beats 786K (7,300 steps) in fixed time
 4. SmearGate + orthogonal init + Muon/NorMuon — shared across nearly all top entries
 5. 10-11 layers, 512 dim, 8 heads / 4 KV heads — the "sweet spot" architecture
 6. Warmdown schedules reduce quantization penalty — cosine warmdown, aggressive LR decay near end
 7. Selective precision preservation — embeddings + late-layer key projections stay FP16
 8. 2048-token training length — matches 4096 perf with more steps possible
 9. Zstd-22 compression beats zlib consistently
 10. SWA (stochastic weight averaging) — reduces quantization-sensitive outliers

 ---
 Part 2: Adami Lab Research — Investigatable Approaches

 Research Papers Reviewed

 Local PDFs:
 - 2509.18522v3.pdf — Functional Information Decomposition (FID)
 - rsta.2021.0250.pdf — Emergence of Functional Information from Multivariate Correlations
 - entropy-24-00735-v2.pdf — Information Fragmentation, Encryption and Information Flow

 Additional publications (2020-2026):
 - "What Is Redundancy?" (2026) — synergy/redundancy in LLMs
 - "Detecting Information Relays in Deep Neural Networks" (2023)
 - "How Brains Perceive the World" (2024)

 Key Limitation the Adami Team Faced

 The Adami lab's information decomposition methods were developed for small evolved neural circuits
 (thousands of neurons). Full PID scales super-exponentially with source count. Their implementations are
 CPU-bound, sequential, and not designed for GPU acceleration. However, tractable approximations derived
 from the same mathematical foundations are transferable.

 ---
 Part 3: PID — Full Math, Hardware Limits, and Workarounds

 3.1 The Full PID Calculation

 Two-source case (Williams & Beer 2010):
 Given sources X₁, X₂ and target Y, joint information decomposes into 4 atoms:

 I(X₁, X₂; Y) = Red + Uniq(X₁) + Uniq(X₂) + Syn
 I(X₁; Y)     = Red + Uniq(X₁)
 I(X₂; Y)     = Red + Uniq(X₂)

 3 equations, 4 unknowns → need a redundancy definition to close the system.

 Ibroja definition (Bertschinger et al. 2014):
 I_∩(X₁, X₂; Y) = max_{Q ∈ Δ_P} I_Q(X₁, X₂; Y)
 where Δ_P = {Q(X₁,X₂,Y) : Q preserves marginals P(X₁,Y) and P(X₂,Y)}

 This is a constrained convex optimization over a×b×c joint distribution entries with a×c + b×c marginal
 constraints.

 n-source lattice — the explosion:
 For n sources, PID atoms = antichains of the powerset of {1,...,n}. Count = Dedekind number D(n):

 ┌──────────────────┬──────────────┬────────────────────────┐
 │   n (sources)    │  PID atoms   │         Notes          │
 ├──────────────────┼──────────────┼────────────────────────┤
 │ 2                │ 4            │ tractable              │
 ├──────────────────┼──────────────┼────────────────────────┤
 │ 3                │ 18           │ doable                 │
 ├──────────────────┼──────────────┼────────────────────────┤
 │ 4                │ 166          │ slow                   │
 ├──────────────────┼──────────────┼────────────────────────┤
 │ 5                │ ~7,500       │ impractical            │
 ├──────────────────┼──────────────┼────────────────────────┤
 │ 10               │ ~10²³        │ absurd                 │
 ├──────────────────┼──────────────┼────────────────────────┤
 │ 512 (hidden dim) │ ~10^(10^150) │ physically meaningless │
 └──────────────────┴──────────────┴────────────────────────┘

 Each atom requires its own constrained optimization over distributions of size k^(n+1).

 3.2 Hardware-Level Computational Limits

 Memory wall: Storing p(X₁,...,X₅₁₂,Y) with 2 bins/variable = 2^513 entries ≈ 10^154. Observable universe
 has ~10^80 atoms.

 Compute wall: Each lattice node needs iterative convex optimization. GPUs are built for dense regular
 tensor ops (GEMM), but PID requires:
 - Graph-irregular lattice traversal → warp divergence (threads in a warp take different paths)
 - Variable convergence per node → load imbalance across SMs
 - No spatial locality in lattice access → poor L2 cache utilization
 - H100's 990 TFLOPS is for matmuls; PID uses ~1% of theoretical throughput

 Parallelism wall: SIMT needs uniform work across 32 threads per warp. PID's data-dependent branching is the
  worst case for GPU architectures.

 3.3 First-Principles Workarounds (Tractable PID Surrogates)

 Key insight: We don't need full PID. We need a GPU-native signal that captures the same phenomenon —
 whether layer representations carry redundant vs. synergistic information.

 Workaround 1: Total Correlation via Covariance Determinant (RECOMMENDED)

 For multivariate Gaussian (reasonable after LayerNorm):
 TC(X₁,...,Xₙ) = ½ log( ∏ᵢ σᵢ² / det(Σ) )

 - TC = 0: dimensions independent (no redundancy)
 - TC large: high correlation (massive redundancy, compressible)
 - Cost: one X.T @ X (covariance) + one Cholesky (log-det). For d=512: O(134M FLOPs) = ~0.0001ms on H100
 - PID connection: TC = sum of ALL pairwise and higher-order redundancies

 Workaround 2: Fano's Second-Order Approximation

 H₂(S) = Σᵢ H(Sᵢ) - Σᵢ<ⱼ I(Sᵢ; Sⱼ)
 Δ = H(S) - H₂(S)   ← captures all higher-order (synergy) effects

 For Gaussian: I(Xᵢ;Xⱼ) = -½ log(1 - ρᵢⱼ²), computed from correlation matrix alone.

 Workaround 3: Interaction Information (Cheapest Synergy Detector)

 II(Xᵢ; Xⱼ; Xₖ) = I(Xᵢ; Xⱼ) - I(Xᵢ; Xⱼ | Xₖ)
 - II > 0 → redundancy in triple; II < 0 → synergy
 - Sample ~1000 random triples, compute as batched 3×3 covariance ops

 Workaround 4: Dual Total Correlation (Synergy/Redundancy Split)

 DTC = H(X₁,...,Xₙ) - Σᵢ H(Xᵢ | X₋ᵢ)
 - TC - DTC = total redundancy; DTC = total synergy
 - Both computable from eigenvalues of Σ (one eigendecomposition)

 3.4 The TC Regularizer Function

 def tc_regularizer(hidden_states_per_layer):
     """Total correlation across layers.
     Cost: ~0.01ms on H100 for 11 layers × 512 dim."""
     tc_total = 0.0
     for h in hidden_states_per_layer:  # h: [batch, seq, dim]
         h_flat = h.reshape(-1, h.shape[-1])  # [B*T, d]
         h_centered = h_flat - h_flat.mean(0)
         cov = (h_centered.T @ h_centered) / h_flat.shape[0]
         L = torch.linalg.cholesky(cov + 1e-6 * torch.eye(cov.shape[0], device=cov.device))
         log_det = 2 * L.diagonal().log().sum()
         tc = 0.5 * (cov.diagonal().log().sum() - log_det)
         tc_total += tc
     return tc_total

 # Usage: L = L_CE + λ_tc * tc_regularizer(layer_activations)

 Training overhead: 11 Cholesky decompositions of 512×512 = ~1.5 GFLOPs = 0.003% of step time.

 3.5 Periodic Synergy Audit (Post-Training Quantization Guide)

 Every 200 steps during training (or once post-training), sample 1000 random dimension triples per layer,
 compute interaction information:
 - Layers with II < 0 (synergy-dominant): preserve at FP16 or int6
 - Layers with II > 0 (redundancy-dominant): safe for aggressive int5 quantization

 This replaces ad-hoc "selective precision" rules with information-theoretic justification.

 ---
 Part 4: How PID Approaches Slot Into High-Performer Pipeline

 Top pipelines: ~8,500 steps in 600s → ~70ms/step (30ms fwd, 35ms loss+bwd, 5ms optim)

 Integration Points

 ┌─────────────────────────┬─────────────────┬─────────┬──────────────────────────────────────────────┐
 │         Signal          │      When       │  Cost   │                   Purpose                    │
 ├─────────────────────────┼─────────────────┼─────────┼──────────────────────────────────────────────┤
 │ TC regularizer          │ Every step      │ 0.002ms │ Reduce layer redundancy → better compression │
 ├─────────────────────────┼─────────────────┼─────────┼──────────────────────────────────────────────┤
 │ Synergy audit           │ Every 200 steps │ ~1ms    │ Track layer information structure            │
 ├─────────────────────────┼─────────────────┼─────────┼──────────────────────────────────────────────┤
 │ Relay-info quantization │ Post-training   │ ~30s    │ Information-guided mixed precision           │
 └─────────────────────────┴─────────────────┴─────────┴──────────────────────────────────────────────┘

 Training Loss

 L = L_CE + λ_tc * TC(layer_activations)
 where λ_tc ∈ [0.001, 0.01] (to be tuned on MLX first)

 ---
 Part 5: Investigatable Approaches (Prioritized)

 Approach A: TC Regularizer During Training (PRIMARY — test first on MLX)

 Add total correlation loss to encourage non-redundant representations across layers. Mathematically
 grounded in PID, costs essentially nothing computationally. Expected effect: representations that are
 already structured for compression before quantization.

 Approach B: Synergy-Guided Mixed-Precision Quantization (SECONDARY)

 Use interaction information to assign int5/int6/FP16 per layer based on synergy content. Replaces heuristic
  decisions with principled ones. Run post-training.

 Approach C: Tapered Architecture (EXPLORATORY)

 Variable MLP width matching natural synergy/redundancy distribution: wider in middle layers (synergy
 zones), narrower at edges (redundancy zones). E.g., MLP expansion 2x→4x→2x.

 Approach D: Information-Weighted Token Loss (EXPLORATORY)

 Weight cross-entropy loss per token by information content derived from multivariate position weight
 matrices. More principled than TF-IDF curriculum.

 ---
 Implementation Plan: MLX-First with Top-Performer Features

 User's existing baseline: val_bpb=2.3380 (default MLX config, 200 iterations, 1 shard)

 Phase 1: Port Top-Performer Features to MLX (establish strong baseline)

 File: train_gpt_mlx.py — all changes in this single file

 1a. SmearGate (new class, ~15 lines)

 Insert after RMSNormNoWeight class (line ~288):
 class SmearGate(nn.Module):
     def __init__(self, dim: int):
         super().__init__()
         self.gate = mx.zeros((dim,), dtype=mx.float32)

     def __call__(self, x: mx.array) -> mx.array:
         g = mx.sigmoid(self.gate.astype(x.dtype))[None, None, :]
         x_prev = mx.concatenate([mx.zeros_like(x[:, :1]), x[:, :-1]], axis=1)
         return (1 - g) * x + g * x_prev
 - Add SmearGate instance to GPT.__init__ (after tok_emb)
 - Call in GPT.__call__ right after the initial rms_norm(tok_emb(...)) embedding

 1b. BigramHashEmbedding (new class, ~30 lines)

 Insert after SmearGate:
 class BigramHashEmbedding(nn.Module):
     def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):
         super().__init__()
         self.bigram_vocab_size = bigram_vocab_size
         self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)
         # Zero-init so bigram signal starts small
         self.embed.weight = mx.zeros_like(self.embed.weight)
         self.proj = CastedLinear(bigram_dim, model_dim) if bigram_dim != model_dim else None
         if self.proj is not None:
             self.proj.weight = mx.zeros_like(self.proj.weight)
         self.scale = mx.array(0.05, dtype=mx.float32)

     def bigram_hash(self, tokens: mx.array) -> mx.array:
         t = tokens.astype(mx.int32)
         mod = self.bigram_vocab_size - 1
         out = mx.concatenate([
             mx.full(t[..., :1].shape, mod, dtype=mx.int32),
             (36313 * t[..., 1:] ^ 27191 * t[..., :-1]) % mod
         ], axis=-1)
         return out

     def __call__(self, token_ids: mx.array) -> mx.array:
         h = self.embed(self.bigram_hash(token_ids))
         if self.proj is not None:
             h = self.proj(h)
         return h * self.scale.astype(h.dtype)
 - Add env vars: BIGRAM_VOCAB_SIZE (default 2048), BIGRAM_DIM (default 128)
 - Add BigramHashEmbedding to GPT.__init__
 - Add bigram embeddings to token embeddings in GPT.__call__

 1c. NorMuon (modify existing zeropower_newtonschulz5, ~3 lines changed)

 At line ~172, add per-row normalization before the global normalization:
 def zeropower_newtonschulz5(g: mx.array, steps: int, eps: float = 1e-7) -> mx.array:
     a, b, c = 3.4445, -4.7750, 2.0315
     x = g.astype(mx.float32)
     # NorMuon: per-row normalization
     x = x / (mx.sqrt(mx.sum(x * x, axis=1, keepdims=True)) + eps)
     x = x / (mx.sqrt(mx.sum(x * x)) + eps)
     # ... rest unchanged

 1d. SWA (add to training loop, ~25 lines)

 Add env vars: SWA_ENABLED (default 1), SWA_START_FRAC (default 0.4), SWA_EVERY (default 50)

 In the training loop (after opt.step() at line ~1036):
 if swa_enabled and lr_mul < swa_start_frac and step % swa_every == 0:
     flat_params = dict(tree_flatten(model.parameters()))
     if swa_state is None:
         swa_state = {k: np.array(v) for k, v in flat_params.items()}
         swa_count = 1
     else:
         for k, v in flat_params.items():
             swa_state[k] += np.array(v)
         swa_count += 1
 After training loop (before serialization):
 if swa_enabled and swa_state is not None and swa_count > 1:
     avg = {k: mx.array(v / swa_count) for k, v in swa_state.items()}
     model.update(tree_unflatten(list(avg.items())))

 1e. Env-var config for top-performer hyperparams

 Add/modify in Hyperparameters:
 - MLP_MULT=3 (3x expansion)
 - NUM_LAYERS=11
 - ROPE_BASE=50000
 - GRAD_CLIP_NORM=0.3
 - MATRIX_LR=0.02 (top performers use lower Muon LR)
 - MUON_MOMENTUM=0.99
 - WARMDOWN_ITERS=5000

 1f. Orthogonal initialization (modify GPT.__init__, ~10 lines)

 Replace zero-init of projection weights with orthogonal:
 for b in self.blocks:
     # Orthogonal init for q, k, v projections
     for linear in [b.attn.c_q, b.attn.c_k, b.attn.c_v]:
         # Use QR decomposition for orthogonal init
         w = mx.random.normal(linear.weight.shape)
         q, r = mx.linalg.qr(w if w.shape[0] >= w.shape[1] else w.T)
         linear.weight = (q if w.shape[0] >= w.shape[1] else q.T).astype(mx.float32)
     # Keep proj zero-init (residual stream)
     b.attn.proj.weight = mx.zeros_like(b.attn.proj.weight)
     b.mlp.proj.weight = mx.zeros_like(b.mlp.proj.weight)

 Phase 2: Run Top-Performer MLX Baseline

 RUN_ID=top_baseline ITERATIONS=200 TRAIN_BATCH_TOKENS=8192 \
   VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 \
   NUM_LAYERS=11 MLP_MULT=3 ROPE_BASE=50000 \
   GRAD_CLIP_NORM=0.3 MATRIX_LR=0.02 MUON_MOMENTUM=0.99 \
   python3 train_gpt_mlx.py
 Record val_bpb. This is our new baseline to beat.

 Phase 3: Add TC Regularizer

 1. Add tc_regularizer() function (MLX tensors, uses mx.linalg.cholesky)
 2. Modify GPT.__call__ to optionally return per-layer hidden states
 3. Add GPT.loss_with_tc() method that computes L_CE + λ_tc * TC
 4. Add env vars: TC_LAMBDA (default 0.0 — off by default), TC_ENABLED (default 0)
 5. Run with TC_ENABLED=1 TC_LAMBDA=0.005 and compare vs Phase 2

 Phase 4: Synergy Audit + Quantization

 1. Add interaction information sampling (1000 random triples per layer)
 2. Log synergy/redundancy ratio per layer every 50 steps
 3. Use synergy scores to guide which layers get int8 vs FP16 preservation in the existing quantization
 pipeline

 Phase 5: Full Validation

 # Compare three configs at 200 iterations on 1 shard:
 # A) Default baseline (val_bpb=2.3380)
 # B) Top-performer features only
 # C) Top-performer + TC regularizer

 Critical Files

 - train_gpt_mlx.py — ALL modifications happen here (single file, <1500 lines limit)
 - Data: ./data/datasets/fineweb10B_sp1024 (1 shard already downloaded)

 Modification Summary (line references from current file)

 ┌─────────────────────────────────┬───────────────────────────────────────────────────┬────────────────┐
 │            Location             │                      Change                       │ Lines affected │
 ├─────────────────────────────────┼───────────────────────────────────────────────────┼────────────────┤
 │ Hyperparameters class (L43-93)  │ Add SmearGate, BigramHash, SWA, TC env vars       │ ~15 new lines  │
 ├─────────────────────────────────┼───────────────────────────────────────────────────┼────────────────┤
 │ After RMSNormNoWeight           │ Add SmearGate class                               │ ~12 new lines  │
 │ (L285-288)                      │                                                   │                │
 ├─────────────────────────────────┼───────────────────────────────────────────────────┼────────────────┤
 │ After SmearGate                 │ Add BigramHashEmbedding class                     │ ~30 new lines  │
 ├─────────────────────────────────┼───────────────────────────────────────────────────┼────────────────┤
 │ zeropower_newtonschulz5         │ Add NorMuon per-row norm                          │ ~2 lines       │
 │ (L172-188)                      │                                                   │ changed        │
 ├─────────────────────────────────┼───────────────────────────────────────────────────┼────────────────┤
 │ GPT.init (L378-408)             │ Add SmearGate, BigramHash, orthogonal init        │ ~20 lines      │
 │                                 │                                                   │ changed        │
 ├─────────────────────────────────┼───────────────────────────────────────────────────┼────────────────┤
 │ GPT.call (L414-429)             │ Hook SmearGate, BigramHash, optional hidden state │ ~10 lines      │
 │                                 │  collection                                       │ changed        │
 ├─────────────────────────────────┼───────────────────────────────────────────────────┼────────────────┤
 │ GPT.loss (L431-448)             │ Add TC regularizer path                           │ ~15 new lines  │
 ├─────────────────────────────────┼───────────────────────────────────────────────────┼────────────────┤
 │ Training loop (L997-1050)       │ Add SWA accumulation                              │ ~15 new lines  │
 ├─────────────────────────────────┼───────────────────────────────────────────────────┼────────────────┤
 │ After training loop (L1051-)    │ Add SWA averaging before serialization            │ ~8 new lines   │
 ├─────────────────────────────────┼───────────────────────────────────────────────────┼────────────────┤
 │ New standalone function         │ tc_regularizer()                                  │ ~15 new lines  │
 └─────────────────────────────────┴───────────────────────────────────────────────────┴────────────────┘

 Total: ~140 new lines, well within 1500-line limit (current: 1097 lines)

 