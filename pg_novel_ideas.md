# Parameter Golf: Novel Approaches to Sub-1.10 BPB

*Analysis date: March 30, 2026*
*Current official SOTA: 1.1194 BPB (#549, @sanjeevmadhav)*
*Best pending pure neural: 1.1086 BPB (#1089, @mikeapedia — Turbo-Muon + EngramLite)*
*Best pending n-gram cache: 0.4027 BPB (#1094 — causal BackoffNgramMixer, compliance TBD)*

---

## Competition Context Summary

**Constraints:** 16MB artifact (code + compressed model), 10 min training on 8×H100, 10 min eval.
**Metric:** val_bpb (bits per byte) on FineWeb validation set. Lower = better.
**Baseline:** 1.2244 BPB (9L, 512d, int8+zlib).
**Current stack:** 11L, 512d, 3×MLP, int6+zstd-22, XSA, EMA, GPTQ-lite, BigramHash, SmearGate, Partial RoPE, sliding-window eval.

**Key quantitative constraints from Issue #140 ablation data:**
- 1ms step overhead ≈ 0.006-0.007 BPB cost (at ~83ms/step baseline)
- Int6 quant gap: ~0.0036 BPB (GPTQ is near-optimal at int6)
- Int5 quant gap: ~0.007 BPB per matrix group
- Int4 quant gap: ~0.065 BPB (catastrophic — dead end)
- 3-seed std: ~0.0005-0.0015 BPB
- EMA > SWA by 0.003 BPB (3-seed verified)
- Sliding window (s64, w2048): ~0.034 BPB improvement
- N-gram cache with correct normalization: 1.51 BPB alone (WORSE than neural — #978)

**What's been tried and failed (selected):**
- MoE (optimal sparsity=0 below 500M params)
- Depth recurrence >2 loops (quant error amplifies 900×)
- Knowledge distillation (11ms/step I/O overhead fatal in 600s)
- MTP (no improvement)
- INT4 quantization (catastrophic +0.065 BPB)
- TrigramHash without gating (+0.0049 BPB, hurts compression)
- MC Dropout ensembling (sub-networks lack diversity at 17M params)
- kNN-LM at eval (XSA already captures inter-position patterns)
- Advanced quant algorithms at int6 (Qronos, CDQuant: GPTQ already near-optimal)
- Procrustes rotation (91% MSE reduction but 380% larger artifact — MSE ≠ artifact size)
- Pruning 3% of weights (+728KB artifact — zeroes hurt zstd-22)

---

## Analysis of All 8 Angles

### ANGLE 1: COMPRESSION IS THE OBJECTIVE (BPB-Aware Loss)

**Core insight:** Cross-entropy loss treats all tokens equally, but BPB weights by bytes-per-token. Tokens decoding to more UTF-8 bytes matter more for BPB.

**Will it work?** Partially — but the gain is smaller than it appears.

**Analysis:**
The gap between CE-optimized BPB and byte-weighted BPB is modest for SP1024 tokenization. With a 1024-token vocabulary, most tokens decode to 1-4 bytes, and the distribution of bytes-per-token is relatively concentrated (mean ~1.18 bytes/token for English-dominant FineWeb). The correction factor (token_count/byte_count) is already baked into the BPB formula, so optimizing CE already approximates BPB optimization.

However, there IS a real effect: tokens that decode to many bytes (rare long tokens, Unicode sequences) receive proportionally less gradient signal under CE. Byte-weighting would reallocate gradient toward these tokens.

**Estimated impact:** 0.001-0.003 BPB improvement. The correction is small because:
1. SP1024 has a tiny vocabulary — the variance of bytes-per-token is low
2. High-byte tokens are often rare/noisy (non-English text, HTML entities) and may not be learnable
3. The model already sees these tokens during training — they just get equal weight

**Implementation difficulty:** Very low — multiply CE loss by a pre-computed bytes-per-token lookup.

**Risk of failure:** Moderate. The effect may be below noise floor (0.0005 BPB 3-seed std).

**Compatibility:** Full — one-line change to loss computation. Zero overhead.

**Verdict: LOW PRIORITY.** The theoretical gap is real but the practical gain at SP1024 is likely below significance threshold. Worth a zero-cost experiment but don't build a strategy around it.

```python
# PROOF-OF-CONCEPT: Byte-weighted cross-entropy loss
# Add to train_gpt_mlx_kl.py

def compute_bytes_per_token_lut(tokenizer_path):
    """Pre-compute UTF-8 byte count for each token in vocabulary."""
    import sentencepiece as spm
    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
    vocab_size = sp.get_piece_size()
    bytes_lut = []
    for i in range(vocab_size):
        piece = sp.id_to_piece(i)
        # Decode to bytes, count UTF-8 length
        try:
            byte_count = len(piece.encode('utf-8'))
        except:
            byte_count = 1
        bytes_lut.append(max(1, byte_count))
    return bytes_lut

def byte_weighted_ce_loss(logits, targets, bytes_lut_tensor):
    """
    Cross-entropy loss weighted by bytes-per-token.
    
    logits: (B*T, V) float
    targets: (B*T,) int
    bytes_lut_tensor: (V,) float — bytes per token ID
    
    Instead of: loss = -log(p[target]) averaged over tokens
    We compute: loss = -log(p[target]) * bytes[target] / mean(bytes[target])
    
    This makes the loss proportional to BPB contribution.
    """
    import mlx.core as mx
    import mlx.nn as nn
    
    # Standard CE per token
    ce_per_token = nn.losses.cross_entropy(logits, targets, reduction='none')  # (B*T,)
    
    # Byte weights for each target token
    byte_weights = bytes_lut_tensor[targets]  # (B*T,)
    
    # Normalize so total weight equals number of tokens (preserves LR scale)
    byte_weights = byte_weights / byte_weights.mean()
    
    # Weighted mean
    loss = (ce_per_token * byte_weights).mean()
    return loss
```

---

### ANGLE 2: NON-UNIFORM QUANTIZATION

**Core insight:** Neural network weights are approximately Gaussian — uniform int6 wastes precision on the tails.

**Will it work?** No, for two devastating reasons specific to this competition.

**Analysis:**

**Reason 1: MSE ≠ Artifact Size.** This is the single most important lesson from the competition (#1048, #316). Non-uniform quantization (k-means, log-scale, NF6) reduces reconstruction MSE. But the artifact is compressed with zstd-22, and non-uniform codebook indices have HIGHER entropy than uniform indices. Uniform int6 produces values that cluster around certain bit patterns, which zstd compresses efficiently. K-means indices are essentially random 6-bit values with near-uniform distribution — maximum entropy, minimum compressibility.

Concrete example from #1048: Procrustes rotation reduced MSE by 91% but increased artifact size by 380% because the rotated weights had higher entropy. The same principle applies to non-uniform quant: better MSE, worse compression, net negative for the 16MB budget.

**Reason 2: GPTQ is already near-optimal at int6.** #756 (@abaybektursun, SOTA holder) tested Qronos iterative Hessian (+0.0007 worse) and CDQuant coordinate descent (+0.0005 worse) — both more sophisticated than uniform GPTQ. At int6 with 64 levels, the quant gap is only 0.0036 BPB. There's simply not much room to improve.

**Reason 3: Codebook storage.** A 64-entry codebook per row (or per tensor) costs bytes that offset any quality gain. For 17M params in ~11K rows, even 64×2 bytes per row = ~1.4MB of codebooks.

**Estimated impact:** Net negative (larger artifact for marginal MSE improvement).

**Implementation difficulty:** Moderate (k-means on weights, custom pack/unpack).

**Risk of failure:** Very high — #1048 and #316 both demonstrate the MSE≠artifact principle.

**Compatibility:** Poor — requires custom serialization, breaks zstd compression efficiency.

**Verdict: DO NOT PURSUE.** The fundamental insight "MSE ≠ compressed artifact size" kills this entire angle. Every non-uniform scheme increases index entropy, which defeats zstd. The competition has empirically confirmed this multiple times.

---

### ANGLE 3: ENTROPY-CODED WEIGHTS

**Core insight:** Zstd treats all bytes equally. What if we designed weight distributions for maximum compressibility?

**Will it work?** One sub-idea works marginally, the rest don't.

**Analysis:**

**Weight entropy regularization:** Promising in principle — penalize high-entropy weight distributions during training. But #609's ablation found lzma (which is closer to a custom entropy coder than zstd) achieves 99.7% of Shannon limit on the weight data. **Zstd-22 is already near-optimal.** The bottleneck isn't the entropy coder — it's the weight entropy itself. And the weight entropy is determined by the model's capacity needs, not by the compression algorithm.

**Sparse + dense hybrid:** #1048 proved that 3% pruning INCREASES artifact by 728KB. Zeroing weights doesn't help zstd-22 because the zero values disrupt the statistical patterns zstd exploits. Structured pruning at 50% would be catastrophic for model quality AND artifact size.

**ANS instead of zstd:** #1089 used Brotli+byte-shuffle instead of zstd on mixed int6/int7 — this IS the closest thing to a custom entropy coder that's been tried. The gain was real but small (enough to squeeze in slightly higher precision). ANS tuned to exact weight distribution could save 0.05-0.2MB over zstd-22, which translates to ~100-400K more parameters — worth ~0.001-0.003 BPB at the margin.

**The one promising sub-idea: NuMuon (arXiv:2603.03597).** Nuclear-norm constraint on Muon updates → lower stable rank → better zstd compression. This pushes compressibility into the *optimizer itself*, which is fundamentally different from post-hoc compression. The weights naturally develop lower entropy during training, rather than being forced into compressible patterns afterward.

**Estimated impact:** 0.001-0.003 BPB (ANS/Brotli tuning) or 0.002-0.006 BPB (NuMuon optimizer).

**Implementation difficulty:** ANS is moderate. NuMuon is low (optimizer change).

**Risk of failure:** Moderate for ANS (marginal gains). Low-moderate for NuMuon (backed by theory).

**Compatibility:** Full — orthogonal to everything else.

**Verdict: NuMuon is worth testing (Tier 2 idea from Issue #140). Custom entropy coding is marginal but free.**

---

### ANGLE 4: HYPERNETWORK WEIGHT GENERATION

**Core insight:** Store a tiny network that generates the weight matrices at load time.

**Will it work?** Almost certainly not at competitive quality.

**Analysis:**

This is implicit neural representation (INR) applied to weight matrices. The problem: weight matrices in a trained LLM are NOT smooth or low-frequency. They contain high-frequency, semantically meaningful structure that resists compact representation. A 200K-param hypernetwork cannot generate 37M coherent weights — the information-theoretic compression ratio of ~185× would require the weights to have ~185× redundancy, which they don't.

**The low-rank basis variant** is more realistic: hypernetwork generates a rank-K basis for each weight matrix, and coefficients are stored directly. But this is just low-rank factorization with extra steps, and it's been explored:
- #609 found Hadamard rotation saves -0.0002 BPB but costs +0.5MB (net negative)
- CPSVD (Column-Preserving SVD) is the principled version of this — untried but estimated at 0.003-0.008 BPB
- The fundamental issue: low-rank approximation of weight matrices loses too much information at the precision levels needed

**The real version of this idea that works:** Weight tying + per-layer LoRA deltas (Relaxed Recursive Transformers, ICLR 2025). Share base weights across layers, add tiny per-layer LoRA adaptations. This gives you ~24 virtual layers from ~11 layers of parameters. #686 demonstrated shallow recurrence (2 layers repeated once, +2 virtual depth) at 1.1182 BPB — it works when limited to 2 loops. But >2 loops causes GPTQ error amplification (#579, #363).

**Estimated impact:** Hypernetwork: net negative. Relaxed Recursive: 0.01-0.03 BPB (from deeper effective model).

**Implementation difficulty:** Hypernetwork: high. Relaxed Recursive: moderate.

**Risk of failure:** Hypernetwork: very high. Relaxed Recursive: moderate (2-loop limit).

**Compatibility:** Relaxed Recursive is compatible with the existing stack.

**Verdict: Hypernetwork is a dead end. Relaxed Recursive Transformers with LoRA deltas is the viable realization of this concept, and it's already on the Tier 2 list.**

---

### ANGLE 5: CONTEXT MIXING (N-gram Ensemble)

**Core insight:** Combine multiple simple predictors (bigrams, trigrams, byte-level) with learned mixing weights.

**Will it work?** This is the MOST promising angle — with critical caveats.

**Analysis:**

The competition has extensively explored this direction, and the results are dramatic but complicated:

**What happened with n-gram caches (Mar 25-27):** A wave of submissions used eval-time n-gram caches to achieve sub-1.0 BPB. Then #978 proved that with correct full-vocabulary normalization, standalone n-gram caches degrade to 1.51 BPB — worse than neural baseline. The previous sub-0.1 scores were normalization artifacts. 33+ PRs were closed.

**But causal n-gram mixing IS viable:** Post-enforcement, the correctly-implemented BackoffNgramMixer (#803, #1094) still achieves 0.40-0.44 BPB. The key difference: these produce full normalized probability distributions over the entire vocabulary at each step, blended with the neural model's distribution using learned or entropy-adaptive alpha mixing.

**TrigramHash as a training-time component:** #609 found TrigramHash (without gating) HURTS by +0.0049 BPB. But EngramLite (#1089) with gating + multi-head hashing + trigrams works — part of the new best 1.1086 BPB. **The gating is essential** — it suppresses noisy hash collisions that raw TrigramHash amplifies.

**Byte-level predictor:** H-Net (#1044) attempted learned byte-level tokenization — 1.90 BPB, far behind. Byte-level processing is too slow for the 600s training budget at current architectures.

**Skip-gram hash:** Untried in the competition. Issue #140 lists it as a Tier 1 idea with 0.005-0.015 BPB estimated gain. Uses non-contiguous positions (e.g., tokens[-1, -3, -5]) as context — captures patterns with intervening content. Zero additional memory per context, just hash different positions. Especially effective on FineWeb's structured web text.

**The realistic path:**
1. Use EngramLite-style gated multi-head hashing (bigram + trigram) during training: ~0.003-0.008 BPB
2. At eval time, add a correctly-normalized BackoffNgramMixer with entropy-adaptive alpha: ~0.05-0.15 BPB
3. The combined system achieves the "complementary training" effect (#803): the neural model specializes on what n-grams can't predict

**Estimated impact:** 
- Training-time context mixing (EngramLite): 0.003-0.008 BPB (proven by #1089)
- Eval-time BackoffNgramMixer: 0.05-0.15 BPB additional (proven by #803, #1094)
- Skip-gram hash: 0.005-0.015 BPB (untried, moderate confidence)

**Implementation difficulty:** EngramLite: moderate. BackoffNgramMixer: moderate-high (must get normalization right). Skip-gram: low.

**Risk of failure:** Low for EngramLite (proven). Moderate for BackoffNgramMixer (normalization is tricky). Low for skip-gram (simple extension of proven concept).

**Compatibility:** Excellent — EngramLite replaces BigramHash. BackoffNgramMixer is eval-only.

**Verdict: HIGH PRIORITY. The combination of EngramLite training + BackoffNgramMixer eval + Complementary Training is the most promising path to sub-1.10 BPB.**

```python
# PROOF-OF-CONCEPT: Gated Multi-Head Hash Embedding (EngramLite-inspired)
# Replaces BigramHash in train_gpt_mlx_kl.py

import mlx.core as mx
import mlx.nn as nn

class EngramLiteEmbedding(nn.Module):
    """
    Multi-head hashed n-gram embeddings with learned gating.
    
    Key improvements over BigramHash:
    1. Multiple hash heads (K=4) per n-gram order — reduces collision rate
    2. Trigram support — captures 3-token patterns
    3. Learned gate — sigmoid suppresses noisy lookups
    
    Architecture:
      - For each n-gram order (2, 3):
        - K hash functions map context to table indices
        - Table lookup produces K embeddings
        - Mean pool across heads
        - Sigmoid gate (from context) scales the output
      - Sum across orders → output
    
    Parameter budget (adjusted for 16MB constraint):
      bigram_table: 2048 × 128 × 2 heads = 524K params
      trigram_table: 2048 × 128 × 2 heads = 524K params
      projection: 128 × 1024 = 131K params
      gate: 128 × 2 + 2 = 258 params
      Total: ~1.2M params ≈ 0.9MB in int6 — fits easily
    """
    def __init__(self, hash_size: int = 2048, embed_dim: int = 128,
                 output_dim: int = 1024, n_heads: int = 2, 
                 orders: tuple = (2, 3)):
        super().__init__()
        self.hash_size = hash_size
        self.embed_dim = embed_dim
        self.output_dim = output_dim
        self.n_heads = n_heads
        self.orders = orders
        
        # Hash primes for multi-head hashing (different prime per head)
        self.primes = [31337, 59999, 73721, 97531][:n_heads]
        
        # Separate embedding table per order (small embed_dim, projected later)
        self.tables = {}
        for order in orders:
            table = nn.Embedding(hash_size, embed_dim)
            # Small init — these are additive corrections
            table.weight = table.weight * 0.01
            self.tables[f'order_{order}'] = table
        
        # Project from embed_dim to output_dim (vocab_size or model_dim)
        self.proj = nn.Linear(embed_dim, output_dim, bias=False)
        
        # Learned gate: context → sigmoid scalar per position
        # Input: concatenated n-gram context embeddings
        self.gate_proj = nn.Linear(embed_dim, len(orders), bias=True)
        # Initialize gate bias to -2.0 → sigmoid(-2) ≈ 0.12 → starts mostly suppressed
        # Model learns to trust hash lookups as training progresses
        self.gate_proj.bias = mx.full((len(orders),), -2.0)
    
    def _hash_ngram(self, tokens, order, head_idx):
        """Hash n-gram context to table index."""
        B, T = tokens.shape
        prime = self.primes[head_idx]
        
        if order == 2:
            # Bigram: hash(t-1, t)
            t_prev = tokens[:, :-1]  # (B, T-1)
            t_curr = tokens[:, 1:]   # (B, T-1)
            idx = mx.remainder(t_prev * prime + t_curr, self.hash_size)
            valid_start = 1
        elif order == 3:
            # Trigram: hash(t-2, t-1, t)
            t_prev2 = tokens[:, :-2]  # (B, T-2)
            t_prev1 = tokens[:, 1:-1] # (B, T-2)
            t_curr = tokens[:, 2:]    # (B, T-2)
            idx = mx.remainder(
                t_prev2 * (prime * prime) + t_prev1 * prime + t_curr,
                self.hash_size
            )
            valid_start = 2
        else:
            raise ValueError(f"Order {order} not supported")
        
        return idx, valid_start
    
    def __call__(self, tokens):
        """
        tokens: (B, T) int32
        Returns: (B, T, output_dim) — additive logit bias
        """
        B, T = tokens.shape
        output = mx.zeros((B, T, self.embed_dim))
        
        for oi, order in enumerate(self.orders):
            table = self.tables[f'order_{order}']
            
            # Multi-head: average K hash lookups
            head_embeds = []
            for hi in range(self.n_heads):
                idx, valid_start = self._hash_ngram(tokens, order, hi)
                emb = table(idx)  # (B, T-order+1, embed_dim)
                head_embeds.append(emb)
            
            # Mean pool across heads — reduces collision noise
            ngram_emb = sum(head_embeds) / self.n_heads  # (B, T-valid_start, embed_dim)
            
            # Pad to full sequence length
            pad = mx.zeros((B, valid_start, self.embed_dim))
            ngram_emb = mx.concatenate([pad, ngram_emb], axis=1)  # (B, T, embed_dim)
            
            output = output + ngram_emb
        
        # Project to output dimension and apply gate
        gated = mx.sigmoid(self.gate_proj(output))  # (B, T, n_orders)
        # Average gate across orders for simplicity
        gate_scalar = gated.mean(axis=-1, keepdims=True)  # (B, T, 1)
        
        return self.proj(output) * gate_scalar  # (B, T, output_dim)


# PROOF-OF-CONCEPT: Skip-Gram Hash Embedding  
class SkipGramHashEmbedding(nn.Module):
    """
    Hash embedding using non-contiguous token positions.
    
    Captures patterns like:
    - token[-1] × token[-3] (skip one)
    - token[-1] × token[-5] (skip three)
    
    Effective for structured text (HTML tags, code indentation,
    sentence templates) where intervening content varies.
    """
    def __init__(self, hash_size: int = 4096, dim: int = 1024,
                 skip_patterns: list = None):
        super().__init__()
        self.hash_size = hash_size
        self.dim = dim
        # Each pattern is a tuple of negative offsets, e.g., (-1, -3)
        self.skip_patterns = skip_patterns or [(-1, -3), (-1, -5), (-2, -4)]
        
        self.tables = {}
        for i, pattern in enumerate(self.skip_patterns):
            table = nn.Embedding(hash_size, dim)
            table.weight = table.weight * 0.01
            self.tables[f'skip_{i}'] = table
    
    def __call__(self, tokens):
        B, T = tokens.shape
        output = mx.zeros((B, T, self.dim))
        
        for i, pattern in enumerate(self.skip_patterns):
            table = self.tables[f'skip_{i}']
            min_offset = min(pattern)  # Most negative
            valid_start = abs(min_offset)
            
            # Gather tokens at skip positions
            # pattern = (-1, -3) means: token at t-1 and token at t-3
            hash_val = mx.zeros((B, T - valid_start), dtype=mx.int32)
            prime = 31337
            for offset in pattern:
                start = valid_start + offset
                end = T + offset
                tok_slice = tokens[:, start:end]
                hash_val = hash_val * prime + tok_slice
            
            idx = mx.remainder(mx.abs(hash_val), self.hash_size)
            emb = table(idx)
            
            pad = mx.zeros((B, valid_start, self.dim))
            emb = mx.concatenate([pad, emb], axis=1)
            output = output + emb
        
        return output
```

---

### ANGLE 6: DEPTH RECURRENCE WITH PROGRESSIVE ADAPTATION

**Core insight:** 4 unique blocks × 3 loops = 12 effective layers from 4 layers of parameters.

**Will it work?** Only with ≤2 loops, and the gain is modest.

**Analysis:**

The competition has extensively tested this:
- #344: 2× slower, hurts BPB
- #363: **Quantization error amplifies ~900× over 3 cycles** — this is the killer
- #579: 6×2 loops gives 1.1478 (1-seed), but GPTQ compounds multiplicatively
- #686: **Shallow recurrence works** — layers 4+5 repeated once (11→13 virtual), 1.1182 BPB (3-seed)

**The critical insight from #363:** When you reuse the same quantized weights K times, the error in each weight gets applied K times. At int6, each weight has ~0.016 expected quantization error (range/64). Over 3 cycles, this compounds: effective error ≈ 3 × 0.016 = 0.048, which is approaching int4-level degradation. At 2 cycles, error ≈ 2 × 0.016 = 0.032 — still viable.

**FiLM conditioning** (scale/shift per loop iteration) helps because it differentiates loop passes, but it can't overcome the quantization amplification problem for >2 loops.

**The viable version:** #686's approach — repeat 2 middle layers once, getting 13 virtual layers from 11 layers of parameters. Combined with per-pass learnable scalars (~2K params). Recovers ~70% of independent 12L quality at minimal step cost.

**Budget math with aggressive recurrence:** 4 unique blocks in int6 ≈ 4MB. But we need >4 blocks for competitive quality — the 4-block config is far too small. And even at 4 blocks × 3 loops, the 900× quant error amplification makes it nonviable.

**Estimated impact:** +2 virtual layers via shallow recurrence: ~0.003-0.008 BPB (proven by #686).

**Implementation difficulty:** Low for shallow recurrence. High for full FiLM + multi-loop.

**Risk of failure:** Low for ≤2 loops. Very high for ≥3 loops.

**Compatibility:** Good — drop-in modification to layer stack.

**Verdict: SHALLOW RECURRENCE (2 loops on 2 layers) IS PROVEN AND VIABLE. Full depth recurrence (3+ loops) is dead due to int6 error amplification.**

---

### ANGLE 7: MULTI-MODEL ENSEMBLE IN 16MB

**Core insight:** Two complementary models (small transformer + massive n-gram hash) might beat one larger transformer.

**Will it work?** YES — this is essentially what the top n-gram cache submissions do.

**Analysis:**

This is not a novel idea — it's the dominant strategy on the (compliance-questioned) pending leaderboard. The top submissions (#803 at 0.4416, #1094 at 0.4027) are exactly this: a neural base model + an n-gram cache ensemble, with learned mixing.

**The specific budget breakdown works:**
- Neural model (11L, 512d, int6+zstd): ~15MB
- BackoffNgramMixer (orders 2-10, ~4M hash buckets): 0 MB artifact (built at eval time from already-scored tokens)
- Total: ~15MB ✓

The n-gram component costs ZERO artifact space because it's built incrementally during evaluation from tokens already scored. This is the key insight — the 16MB budget goes entirely to the neural model, and the n-gram mixer is free.

**Why it works despite #978:** #978 showed standalone normalized n-grams achieve only 1.51 BPB. But that's standalone. When MIXED with a neural model, the n-gram component handles high-confidence local patterns (common bigrams, frequent phrases) while the neural model handles everything else. The mixing is complementary — each handles what the other can't.

**Complementary Training (#803):** The neural model is trained with loss weights that down-weight tokens easily predicted by bigram statistics. This forces the model to specialize on what n-grams can't handle, maximizing complementarity. This is the critical innovation that separates 0.44 from 0.55 BPB.

**Estimated impact:** 0.05-0.20 BPB over pure neural, depending on n-gram order and mixing quality.

**Implementation difficulty:** Moderate-high (correct normalization is crucial — #978's lesson).

**Risk of failure:** Low for concept (proven), moderate for compliance (organizer scrutiny ongoing).

**Compatibility:** Eval-time only — compatible with any training stack.

**Verdict: HIGH PRIORITY. This is proven and the highest-impact single technique available. The risk is compliance, not effectiveness.**

```python
# PROOF-OF-CONCEPT: Complementary Training + BackoffNgramMixer
# 
# Two-part system:
# Part 1: Modified training loss (in train_gpt_mlx_kl.py)
# Part 2: Eval-time BackoffNgramMixer

# ============================================================
# PART 1: Complementary Training Loss
# ============================================================

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from collections import defaultdict

def build_bigram_stats(train_data_path, vocab_size=1024):
    """
    Pre-compute bigram transition probabilities from training data.
    Used to identify tokens that are 'easy' for n-gram models.
    
    Returns: bigram_probs[prev_token, next_token] = P(next|prev)
    """
    # Count bigram frequencies
    counts = np.zeros((vocab_size, vocab_size), dtype=np.float32)
    # Read training shards
    import glob, struct
    for shard_path in sorted(glob.glob(f"{train_data_path}/fineweb_train_*.bin")):
        with open(shard_path, 'rb') as f:
            header = struct.unpack('<256i', f.read(1024))
            n_tokens = header[2]
            tokens = np.frombuffer(f.read(n_tokens * 2), dtype=np.uint16)
            for i in range(len(tokens) - 1):
                counts[tokens[i], tokens[i+1]] += 1
    
    # Normalize to probabilities (with Laplace smoothing)
    row_sums = counts.sum(axis=1, keepdims=True) + vocab_size
    bigram_probs = (counts + 1) / row_sums
    return bigram_probs


def complementary_loss(logits, targets, prev_tokens, bigram_probs_mx, 
                       complement_alpha=0.5):
    """
    Weighted CE loss that down-weights tokens easily predicted by bigrams.
    
    For each token:
      p_bigram = bigram_probs[prev_token, target_token]
      weight = 1 - complement_alpha * p_bigram
    
    Tokens with high bigram probability get lower training weight,
    forcing the neural model to specialize on what n-grams can't predict.
    
    logits: (B*T, V)
    targets: (B*T,)
    prev_tokens: (B*T,) — the token at position t-1
    bigram_probs_mx: (V, V) — pre-computed bigram transition probs
    complement_alpha: float — strength of complementary weighting (0=standard CE)
    """
    # Standard CE per token
    ce_per_token = nn.losses.cross_entropy(logits, targets, reduction='none')
    
    # Look up bigram probability of each target given its predecessor
    # p_bigram[i] = bigram_probs[prev_tokens[i], targets[i]]
    p_bigram = bigram_probs_mx[prev_tokens, targets]  # (B*T,)
    
    # Complementary weights: tokens easily predicted by bigrams get low weight
    weights = 1.0 - complement_alpha * p_bigram
    weights = mx.clip(weights, 0.1, 1.0)  # Floor at 0.1 to avoid zero gradients
    
    # Normalize weights to preserve effective learning rate
    weights = weights / weights.mean()
    
    loss = (ce_per_token * weights).mean()
    return loss


# ============================================================
# PART 2: Eval-Time BackoffNgramMixer (Causal, Full-Vocab Normalized)
# ============================================================

class BackoffNgramMixer:
    """
    Causal n-gram language model with Kneser-Ney-style backoff.
    Built incrementally from already-scored tokens (backward-looking).
    
    Key properties for compliance:
    1. Full-vocabulary normalized: produces valid probability distribution
    2. Causal: only uses tokens at positions < current position
    3. Single-pass: score-first, then update cache
    4. No artifact cost: built from scratch during eval
    
    Algorithm:
    - Maintain count tables for orders 1 through max_order
    - For each position t:
      1. Query all orders using context tokens[t-order+1:t]
      2. Backoff from highest to lowest order with interpolation
      3. Produce P_ngram(token | context) for all vocab tokens
      4. Mix with neural model: P = (1-alpha)*P_neural + alpha*P_ngram
      5. Score position t using P
      6. Update count tables with token[t]
    """
    def __init__(self, vocab_size=1024, max_order=7, 
                 hash_buckets=2_000_000, alpha_mode='entropy_adaptive'):
        self.vocab_size = vocab_size
        self.max_order = max_order
        self.hash_buckets = hash_buckets
        self.alpha_mode = alpha_mode
        
        # Count tables: hash(context) -> array of vocab counts
        # Using defaultdict for prototype; production uses fixed hash tables
        self.counts = [defaultdict(lambda: np.zeros(vocab_size, dtype=np.float32))
                       for _ in range(max_order + 1)]
        self.total_counts = [defaultdict(float) for _ in range(max_order + 1)]
    
    def _hash_context(self, context_tokens):
        """Hash a sequence of tokens to a bucket index."""
        h = 0
        for t in context_tokens:
            h = (h * 31337 + int(t)) % self.hash_buckets
        return h
    
    def _get_ngram_probs(self, context_tokens):
        """
        Compute interpolated n-gram probability distribution.
        Uses simple linear interpolation backoff.
        """
        vocab_size = self.vocab_size
        
        # Start with uniform (order 0)
        probs = np.ones(vocab_size, dtype=np.float64) / vocab_size
        
        # Interpolate from low to high order
        for order in range(1, self.max_order + 1):
            if len(context_tokens) < order:
                break
            
            ctx = context_tokens[-order:]
            ctx_hash = self._hash_context(ctx)
            
            counts = self.counts[order][ctx_hash]
            total = self.total_counts[order][ctx_hash]
            
            if total > 0:
                # Interpolation weight increases with total count (confidence)
                lambda_order = total / (total + 5.0)  # Simple discount
                order_probs = (counts + 1e-10) / (total + 1e-10 * vocab_size)
                
                # Ensure normalization
                order_probs = order_probs / order_probs.sum()
                
                probs = (1 - lambda_order) * probs + lambda_order * order_probs
        
        # Final normalization (paranoia)
        probs = probs / probs.sum()
        return probs
    
    def _compute_alpha(self, neural_logits):
        """
        Entropy-adaptive mixing weight.
        When neural model is uncertain (high entropy), trust n-grams more.
        When neural model is confident (low entropy), trust it more.
        """
        if self.alpha_mode == 'fixed':
            return 0.3
        
        # Compute neural model entropy
        probs = np.exp(neural_logits - neural_logits.max())
        probs = probs / probs.sum()
        entropy = -np.sum(probs * np.log2(probs + 1e-10))
        max_entropy = np.log2(self.vocab_size)  # ~10 bits for 1024 vocab
        
        # Map entropy to alpha: high entropy → high alpha
        normalized_entropy = entropy / max_entropy
        alpha = 0.15 + 0.45 * normalized_entropy  # Range: 0.15-0.60
        
        return alpha
    
    def score_and_update(self, position, context_tokens, token_at_pos, 
                         neural_log_probs):
        """
        Score position and update cache. Must be called sequentially.
        
        Returns: log probability of token_at_pos under mixed distribution
        """
        # 1. Get n-gram distribution (causal: uses only tokens before position)
        ngram_probs = self._get_ngram_probs(context_tokens)
        
        # 2. Get mixing weight
        alpha = self._compute_alpha(neural_log_probs)
        
        # 3. Mix distributions
        neural_probs = np.exp(neural_log_probs)
        neural_probs = neural_probs / neural_probs.sum()  # Re-normalize
        
        mixed_probs = (1 - alpha) * neural_probs + alpha * ngram_probs
        mixed_probs = mixed_probs / mixed_probs.sum()  # Ensure normalization
        
        # 4. Score
        log_prob = np.log(mixed_probs[token_at_pos] + 1e-30)
        
        # 5. Update cache (AFTER scoring — backward-looking)
        for order in range(1, self.max_order + 1):
            if len(context_tokens) >= order:
                ctx = context_tokens[-order:]
                ctx_hash = self._hash_context(ctx)
                self.counts[order][ctx_hash][token_at_pos] += 1
                self.total_counts[order][ctx_hash] += 1
        
        return log_prob
```

---

### ANGLE 8: INFORMATION-THEORETIC LOWER BOUND

**Analysis:**

The theoretical analysis is sound:
- Shannon entropy of clean English: ~1.0-1.3 bits/byte
- Best neural compressors on clean English: ~0.8-0.9 bits/byte (exploiting long-range structure)
- FineWeb is web text: noisier, multilingual, more diverse → practical floor ~1.0-1.1 bits/byte
- Current SOTA: 1.1194 (official), 1.1086 (pending)
- LoRA TTT: reached 1.0865 (#628, GEPA+legal TTT on 4×A100)

**The 16MB constraint is the binding limit, not the theoretical floor.** A 16MB artifact encodes ~128M bits of information. The model has ~37M parameters in int6 ≈ 222M bits. After zstd compression, ~124M bits. The validation set is ~60M tokens × ~1.18 bytes/token ≈ ~70M bytes. To achieve 1.0 BPB, we need 70M bits of prediction accuracy from 124M bits of model. That's a ~1.8:1 ratio — tight but feasible.

**Sub-1.10 BPB is achievable within 16MB** — the GEPA+TTT result (1.0865) proves this, though it used 4×A100 for 20K steps (more compute). The question is whether 600s on 8×H100 (~7K steps) provides enough training to reach the same quality.

**Estimated gap breakdown (1.1194 → 1.10):**
- Better quantization/compression (entropy coding, NuMuon): ~0.003
- Better architecture (shallow recurrence, EngramLite): ~0.005-0.008
- Better training (Complementary Training, Mousse/Turbo-Muon): ~0.003-0.005
- Eval-time BackoffNgramMixer: ~0.05-0.10
- **Total estimated: ~0.06-0.12 BPB improvement → 1.00-1.06 BPB**

Sub-1.10 is clearly achievable. Sub-1.05 is plausible. Sub-1.00 is at the edge of feasibility.

---

## Ranked List: Most Promising Ideas

### Tier 1 — Highest Impact, Proven Feasible

| Rank | Idea | Source Angle | Est. BPB Gain | Risk | Difficulty |
|------|------|-------------|---------------|------|------------|
| 1 | **BackoffNgramMixer at eval time** | 5, 7 | 0.05-0.15 | Low (proven) / Moderate (compliance) | Moderate-High |
| 2 | **Complementary Training** | 5, 7 | 0.01-0.03 (over standard training) | Low (proven by #803) | Low |
| 3 | **EngramLite (gated multi-head hash)** | 5 | 0.003-0.008 (over BigramHash) | Low (proven by #1089) | Moderate |

### Tier 2 — Moderate Impact, Good Feasibility

| Rank | Idea | Source Angle | Est. BPB Gain | Risk | Difficulty |
|------|------|-------------|---------------|------|------------|
| 4 | **Shallow recurrence (+2 virtual layers)** | 6 | 0.003-0.008 | Low (proven by #686) | Low |
| 5 | **Skip-gram hash embedding** | 5 | 0.005-0.015 | Moderate (untried) | Low |
| 6 | **NuMuon optimizer** | 3 | 0.002-0.006 | Moderate | Low |
| 7 | **Mousse optimizer** | (Issue #140) | 0.003-0.008 | Low-Moderate | Low |
| 8 | **PPMII-style escape estimation** | 5, 7 | 0.01-0.03 (over basic backoff) | Moderate | Medium |

### Tier 3 — Small/Speculative Impact

| Rank | Idea | Source Angle | Est. BPB Gain | Risk | Difficulty |
|------|------|-------------|---------------|------|------------|
| 9 | **Byte-weighted CE loss** | 1 | 0.001-0.003 | High (below noise) | Very Low |
| 10 | **Custom entropy coding (ANS/Brotli)** | 3 | 0.001-0.003 | Moderate | Moderate |
| 11 | **Logistic-domain mixing** | 5 | 0.002-0.005 | Low | Very Low |

### Dead Ideas (Don't Pursue)

| Idea | Source Angle | Why Dead |
|------|-------------|----------|
| Non-uniform quantization (K-means, NF6) | 2 | MSE ≠ artifact size; higher index entropy defeats zstd |
| Hypernetwork weight generation | 4 | Information-theoretic impossibility at 185× compression |
| Deep recurrence (3+ loops) | 6 | Int6 error amplifies 900× over 3 cycles |
| Weight sparsification | 3 | Zeroing weights INCREASES artifact size (#1048) |
| Byte-level model | 5 | Far too slow for 600s training budget |
| Standalone n-gram (no neural) | 5 | 1.51 BPB with correct normalization — worse than neural |

---

## Top 3: Proof-of-Concept Code

### POC 1: Complementary Training + BackoffNgramMixer

*(Full code stubs provided in Angle 5 and Angle 7 analysis above)*

**Smoke test plan (M1, 100 steps):**
1. Pre-compute bigram statistics from first training shard
2. Modify `train_gpt_mlx_kl.py` loss to use `complementary_loss`
3. Train 100 steps, compare train_loss vs baseline
4. Expected: slightly higher train_loss (we're down-weighting easy tokens) but model learns harder patterns

**Integration with existing stack:**
- Replace BigramHash with EngramLite in model init
- Add `--complement-alpha 0.5` flag
- Pre-compute bigram stats during data loading (one-time cost)
- At eval time, wrap sliding-window eval with BackoffNgramMixer

### POC 2: EngramLite Gated Multi-Head Hash

*(Full code stub provided in Angle 5 analysis above)*

**Smoke test plan (M1, 100 steps):**
1. Replace `BigramHashEmbedding` with `EngramLiteEmbedding` in model
2. Config: hash_size=2048, embed_dim=128, output_dim=1024, n_heads=2, orders=(2,3)
3. Train 100 steps, compare train_loss vs BigramHash baseline
4. Expected: comparable or slightly better loss (gating suppresses noise)

**Key implementation notes:**
- The gating mechanism is essential — without it, trigrams hurt (#609)
- Multi-head averaging reduces hash collision noise
- Parameter budget: 2 tables × 2048 × 128 × 2 heads + projection (128×1024) ≈ 1.2M params
  - In int6+zstd: ~0.9MB — fits within 16MB budget easily

### POC 3: Skip-Gram Hash + Shallow Recurrence Combo

*(Skip-gram code stub provided in Angle 5. Shallow recurrence below.)*

```python
# PROOF-OF-CONCEPT: Shallow Recurrence with Per-Pass Scalars
# Modification to GPT model in train_gpt_mlx_kl.py
#
# Key idea: Repeat layers 4 and 5 once each (11 → 13 virtual layers)
# with per-pass learnable scalar multipliers.

class ShallowRecurrentGPT:
    """
    Modification to existing GPT class.
    
    Original: layers 0,1,2,3,4,5,6,7,8,9,10 (11 layers)
    Modified: layers 0,1,2,3,4,5,4',5',6,7,8,9,10 (13 virtual, 11 unique)
    
    Layers 4' and 5' reuse weights from layers 4 and 5 but with
    per-pass learnable scalars that differentiate the passes.
    
    Cost: ~2K extra parameters (scale + shift per layer per pass)
    Benefit: ~70% of independent 12L quality gain
    
    CRITICAL: Only 2 loops (1 repeat). 3+ loops cause 900× quant error
    amplification at int6.
    """
    
    def __init__(self, config):
        # ... (standard init) ...
        
        # Per-pass scalars for recurrent layers (pseudocode — actual impl
        # would use nn.Module parameters for gradient tracking)
        self.recur_layers = [4, 5]  # Which layers to repeat
        self.n_passes = 2  # Original + 1 repeat
        
        # Learnable scale per pass (FiLM-lite)
        # 2 recurrent layers × 1 repeat = 2 learnable scalars
        # In real implementation: self.pass_scales = {key: nn.Parameter(mx.array(0.9))}
        self.pass_scales = {}
        for layer_idx in self.recur_layers:
            for pass_idx in range(self.n_passes):
                key = f'layer{layer_idx}_pass{pass_idx}'
                if pass_idx == 0:
                    self.pass_scales[key] = 1.0  # Fixed (original pass)
                else:
                    self.pass_scales[key] = 0.9  # Learnable (repeated pass)
    
    def forward(self, x):
        """
        Forward pass with shallow recurrence.
        
        Instead of: 0 → 1 → 2 → 3 → 4 → 5 → 6 → 7 → 8 → 9 → 10
        We do:      0 → 1 → 2 → 3 → 4 → 5 → 4' → 5' → 6 → 7 → 8 → 9 → 10
        
        Where 4' means layer 4 weights with pass_scales['layer4_pass1']
        """
        # Encoder layers (0 through num_encoder-1)
        for i in range(self.num_encoder):
            x = self.layers[i](x)
        
        x0 = x  # Skip connection source
        
        # Decoder layers with recurrence
        virtual_layer_order = []
        for i in range(self.num_encoder, self.num_layers):
            virtual_layer_order.append((i, 0))  # (layer_idx, pass_idx)
            if i in self.recur_layers:
                virtual_layer_order.append((i, 1))  # Repeat
        
        for layer_idx, pass_idx in virtual_layer_order:
            scale = self.pass_scales.get(
                f'layer{layer_idx}_pass{pass_idx}', 1.0
            )
            
            # Standard block forward with scaling
            block_out = self.layers[layer_idx](x, x0)
            
            if pass_idx > 0:
                # For repeated passes, apply dampened residual
                x = x + scale * (block_out - x)
            else:
                x = block_out
        
        return x
```

**Smoke test plan (M1, 100 steps):**
1. Modify GPT forward to add shallow recurrence on layers 4,5
2. Add SkipGramHashEmbedding alongside existing BigramHash
3. Train 100 steps, compare train_loss
4. Expected: slightly slower per step (~5-10% from extra 2 layer passes) but better loss per step

---

## Implementation Details for Each Idea

| Idea | Difficulty | BPB Est. | Risk | Compatible? | Dependencies |
|------|-----------|----------|------|-------------|-------------|
| BackoffNgramMixer | Moderate-High | 0.05-0.15 | Low/Moderate | Yes (eval-only) | numpy |
| Complementary Training | Low | 0.01-0.03 | Low | Yes | Pre-computed bigram stats |
| EngramLite | Moderate | 0.003-0.008 | Low | Yes (replaces BigramHash) | None |
| Shallow Recurrence | Low | 0.003-0.008 | Low | Yes (model arch change) | None |
| Skip-gram Hash | Low | 0.005-0.015 | Moderate | Yes (additive) | None |
| NuMuon | Low | 0.002-0.006 | Moderate | Yes (optimizer swap) | None |
| Byte-weighted CE | Very Low | 0.001-0.003 | High | Yes (loss change) | tokenizer stats |
| Custom entropy coding | Moderate | 0.001-0.003 | Moderate | Yes (post-training) | ANS library |

---

## THE MOONSHOT

### Complementary Training + EngramLite + BackoffNgramMixer: The Integrated Stack

**Status: IMPLEMENTED** in `train_gpt_mlx_kl.py` (April 2026).

**Env vars for full moonshot run (8×H100):**
```
ENGRAM_LITE_ENABLED=1 COMPLEMENT_ALPHA=0.5 NGRAM_MIXER_ENABLED=1 NGRAM_ALPHA=0.25 NGRAM_MAX_ORDER=4
```

**Smoke test (M1, 100 steps):**
```
RUN_ID=moonshot_test ITERATIONS=100 TRAIN_BATCH_TOKENS=8192 VAL_LOSS_EVERY=0 VAL_BATCH_SIZE=8192 \
WARMUP_STEPS=3 ENGRAM_LITE_ENABLED=1 COMPLEMENT_ALPHA=0.5 NGRAM_MIXER_ENABLED=1 EVAL_MODE=standard \
python3 train_gpt_mlx_kl.py
```

**Why this is the single best bet nobody has fully combined:**

The top competition results reveal three independent discoveries that, when properly integrated, form a system greater than the sum of its parts:

1. **EngramLite (#1089):** Gated multi-head hashing makes n-gram features trainable end-to-end, fixing the TrigramHash failure (#609). This is the TRAINING-TIME component — the neural model learns to use n-gram context efficiently.

2. **Complementary Training (#803):** Down-weighting tokens predictable by n-grams forces the neural model to specialize. This creates maximum COMPLEMENTARITY between neural and n-gram components. Without this, the neural model wastes capacity re-learning patterns the n-gram cache will handle at eval time.

3. **BackoffNgramMixer (#1094):** A correctly-normalized eval-time n-gram cache with entropy-adaptive mixing. This is the EVAL-TIME component that adds 0.05-0.15 BPB improvement at zero artifact cost.

**Why nobody has combined all three:**
- EngramLite is new (#1089, March 29)
- Complementary Training is new (#803, March 25)  
- BackoffNgramMixer compliance was only clarified March 27
- The three ideas emerged from different teams in different weeks

**The integrated system:**

```
TRAINING:
  1. Pre-compute bigram/trigram stats from training data
  2. Train with EngramLite (gated bigram+trigram hash, replaces BigramHash)
  3. Use Complementary Training loss (down-weight easy n-gram tokens)
  4. Standard stack: 11L, 512d, 3×MLP, XSA, EMA, GPTQ-lite, etc.
  5. Result: neural model specialized for what n-grams can't predict

EVAL:
  1. Load quantized model (standard sliding-window)
  2. For each token:
     a. Score with neural model → neural_log_probs
     b. Score with BackoffNgramMixer (orders 2-7) → ngram_probs
     c. Entropy-adaptive alpha: high neural uncertainty → trust n-grams more
     d. Mix: P = (1-alpha) * P_neural + alpha * P_ngram  
     e. Record log(P[true_token])
     f. Update n-gram cache (backward-looking)
  3. Result: complementary predictions from specialized components
```

**Expected BPB:**
- Baseline pure neural SOTA: 1.1086 (#1089)
- EngramLite + Complementary Training: ~1.10-1.11 (saving capacity for hard tokens)
- + BackoffNgramMixer at eval: ~0.95-1.05 
- + Skip-gram hash + shallow recurrence: ~0.92-1.00

**Why this could leapfrog the field:**
1. Nobody has done Complementary Training + EngramLite together (complementarity is maximized)
2. The BackoffNgramMixer is free (zero artifact cost) and additive
3. The neural model is BETTER at the tokens that matter because it doesn't waste capacity on n-gram-predictable tokens
4. This is principled compression theory: multiple experts with complementary specializations, mixed with learned weights

**Risk factors:**
- Compliance: BackoffNgramMixer must produce correctly normalized full-vocabulary distributions
- The eval-time n-gram cache quality depends on validation set repetitiveness
- Complementary Training requires careful alpha tuning (too aggressive → model loses basic competence)
- EngramLite's parameter budget must be managed (hash tables compete with model capacity)

**Estimated floor:** Even without the BackoffNgramMixer (which has compliance risk), EngramLite + Complementary Training + the full existing stack should reach ~1.10 BPB on pure neural — matching the current pending best with a principled path forward.

---

## Honest Assessment: What Won't Work

1. **Non-uniform quantization** — MSE ≠ artifact size. Killed by competition data (#1048, #316).
2. **Hypernetworks** — Information-theoretically impossible at required compression ratios.
3. **Deep recurrence (3+ loops)** — Int6 error amplification is a fundamental constraint.
4. **Byte-level models** — Too slow for 600s training. H-Net proved this at 1.90 BPB.
5. **Standalone n-gram replacement for neural** — 1.51 BPB with correct normalization (#978).
6. **Byte-weighted CE as a major lever** — The effect is real but ~0.001-0.003 BPB, below noise.
7. **Knowledge distillation** — 11ms/step I/O overhead fatal at 600s (#1029).
8. **Weight sparsification** — Increases artifact size, doesn't decrease it (#1048).

The only genuine insights in this analysis are:
1. **The Complementary Training + EngramLite + BackoffNgramMixer integrated stack** (Moonshot)
2. **Skip-gram hashing** as a genuinely untried extension of the proven hash embedding approach
3. **The "MSE ≠ artifact size" principle** that eliminates entire categories of ideas

Everything else is either already known to the competition community or below the significance threshold.
