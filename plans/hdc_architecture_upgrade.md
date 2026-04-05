# HDC Architecture Upgrade Plan

## Overview

This plan covers the full set of architectural upgrades discussed across the design sessions. The changes fall into five logical groups:

1. **Semantic Rolling Hash S[p]** — unlimited-context semantic fallback
2. **WHT + Butterfly Consistency** — noise-aware spectrum analysis replacing argmin
3. **Layered Prediction Pipeline** — 5-layer HDC attention analog
4. **Suffix Grammar Table** — subatomic-grounded morphological grammar learning
5. **Continuous Predictive Coding** — error correction woven into Phases 2/3, Phase 4 as executor

Each group maps to specific new files and specific edits to existing files.

---

## Current Architecture (Baseline)

```
Training phases:
  Phase 1   → rolling hash G[p] precomputation (_full_context_hash.py)
  Phase 1.5 → bigram table (fast pre-build)
  Phase 2   → DNA stack: Boyer-Moore majority vote into table_packed
  Phase 3   → multi-pass reinforcement (same merge_winners loop)
  Phase 3.5 → bigram table (full build) + DSV (sem_fwd + sem_bwd)
  Phase 4   → error-residual repair (find wrong buckets, XOR-swap)

Eval waterfall (in order, first confident hit wins):
  G[p] rolling hash table
    → overflow table (radial shell probe)
    → trigram table
    → bigram table
    → transition codebook
    → DSV sem_fwd vote (1-hop)
    → XOR codebook similarity fallback
    → SmearGate soft-blend (table + bigram gates)
```

---

## Target Architecture (After Upgrades)

```
Training phases:
  Phase 0   → pre-training semantic prior (frozen sem_prior_fwd/bwd, 2M tokens)
  Phase 1   → rolling hash G[p] + semantic rolling hash S[p] precomputation (parallel)
  Phase 1.5 → bigram + skip-bigram lag-2..5 tables
  Phase 2   → DNA stack with semantic conflict resolution in merge_winners
  Phase 3   → reinforcement + repair queue building (semantically annotated)
  Phase 3.5 → bigram + DSV + suffix grammar table + XOR orbit diagonal table
  Phase 4   → execution-only repair from pre-built queue (sorted by confidence)

Eval waterfall (in order):
  G[p] rolling hash table (exact, unlimited context)
    → overflow table
    → trigram table
    → bigram table
    → transition codebook
    → S[p] semantic rolling hash → WHT → butterfly check → layered predict
        Layer 1: direct S[p] query
        Layer 2: HDC attention (sem_bwd as key)
        Layer 3: multi-hop forward composition
        Layer 4: backward validation
        Layer 5: self-consistency check
    → skip-bigram lag-2..5 (diagonal-aware)
    → XOR orbit diagonal prediction
    → DSV sem_fwd vote (1-hop)
    → suffix grammar score (morphological gate)
    → SmearGate soft-blend
```

---

## New Files to Create

### 1. `_semantic_rolling_hash.py`

**Purpose**: S[p] accumulation, WHT query, butterfly consistency check, consensus prediction.

**Key classes/functions**:

```python
class SemanticRollingHash:
    """
    Maintains S[p] = XOR accumulation of position-weighted semantic vectors.
    S[p+1] = S[p] XOR (sem_fwd[tokens[p]] * HADAMARD_KEY[p])
    
    Stores checkpoint states at chunk boundaries (same tier system as G[p]).
    Recomputes forward within each chunk during eval — zero new infrastructure.
    """
    
    def __init__(self, W_UINT64: int, alpha: float = 0.005):
        """
        W_UINT64: number of uint64 blocks (16 for 1024-bit vectors)
        alpha: forgetting factor (0.005 → effective window ~200 tokens)
        """
    
    def accumulate(self, S: np.ndarray, sem_fwd_vec: np.ndarray, key: np.uint64) -> np.ndarray:
        """One-step update: S[p+1] = (S[p] XOR flip_mask) XOR (sem_fwd[t] * key)"""
    
    def build_states(self, tokens, sem_fwd_matrix, keys, chunk_boundaries) -> np.ndarray:
        """Build S_states array at chunk boundaries. O(N * W_UINT64) total."""
    
    def wht_predict(self, S_p: np.ndarray, codebook: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Walsh-Hadamard Transform over S_p.
        Returns (correlations, bipolar_S) where correlations[t] = similarity to token t.
        O(vocab * log(vocab)) = O(10K ops) for vocab=1024.
        """
    
    def butterfly_consistency(self, correlations: np.ndarray, winner: int, n_levels: int = 10) -> float:
        """
        Check that butterfly partners of winner are near noise floor.
        Returns consistency score in [0,1]: 1.0 = genuine signal, 0.0 = noise.
        """
    
    def consensus_predict(self, S_states, p, sem_fwd_matrix, codebook, keys, window=5):
        """
        Query S[p], S[p-1], ..., S[p-window+1].
        Agreement across k states → confidence scales as (1/vocab)^(k-1).
        Returns (winner, confidence, n_agreeing).
        """
```

**Forgetting factor implementation**:
```python
# Deterministic flip_mask seeded by position p
rng = np.random.RandomState(int(p) & 0xFFFFFFFF)
n_flips = max(1, int(alpha * W_UINT64 * 64))
flip_positions = rng.choice(W_UINT64 * 64, n_flips, replace=False)
flip_mask = np.zeros(W_UINT64, dtype=np.uint64)
for pos in flip_positions:
    flip_mask[pos // 64] ^= np.uint64(1) << np.uint64(pos % 64)
```

**Storage**: S_states at chunk boundaries only. For 500M tokens with 2M-token chunks: 250 checkpoints × 16 uint64s = 32 KB. Negligible.

---

### 2. `_suffix_grammar.py`

**Purpose**: Suffix hypervector encoding, grammar role learning from corpus, suffix-grammar scoring at inference.

**Key classes/functions**:

```python
class SuffixGrammarTable:
    """
    Learns: suffix_hv → grammatical context signature
    
    Built during Phase 3.5 in one corpus scan.
    At inference: suffix_grammar_score(candidate, S_p) → float in [0,1]
    """
    
    def __init__(self, vocab_size: int, W_UINT64: int, char_hv: CharacterHypervector,
                 tokenizer, suffix_len: int = 3):
        """
        Precomputes suffix_hvs[t] for all vocab tokens.
        suffix_hvs[t] = char_hv.encode_string(token_str[t][-suffix_len:])
        Storage: vocab_size × W_UINT64 × 8 bytes = 128 KB for vocab=1024
        """
    
    def build_from_corpus(self, tokens: np.ndarray, S_states: np.ndarray,
                           time_budget_s: float = 10.0):
        """
        One corpus scan: for each position p, bundle S_states[p] into
        suffix_gram_bundle[tokens[p]].
        
        suffix_gram_bundle[t] = XOR-bundle of all S[p] states that preceded token t.
        This encodes: "what grammatical contexts does token t's suffix appear in?"
        """
    
    def suffix_grammar_score(self, candidate: int, S_p: np.ndarray) -> float:
        """
        Score: how grammatically consistent is candidate's suffix with S_p?
        
        1. Find tokens with similar suffixes (suffix_sim > 0.7)
        2. Bundle their suffix_gram_bundle entries
        3. Hamming similarity of bundle to S_p
        
        Returns float in [0, 1].
        """
    
    def batch_suffix_grammar_scores(self, candidates: np.ndarray, S_p: np.ndarray) -> np.ndarray:
        """Vectorized version for scoring multiple candidates simultaneously."""
```

**Grammar disambiguation example** (learned automatically):
- `-ed` suffix → high score when S[p] contains past-tense markers
- `-ing` suffix → high score when S[p] contains progressive markers  
- `-s` (verb) → high score when S[p] contains 3rd-person singular markers
- `-s` (noun) → high score when S[p] contains plural noun markers

**Storage**: 128 KB suffix_hvs + 128 KB suffix_gram_bundle + 4 KB counts = ~260 KB total.

---

### 3. `_layered_predict.py`

**Purpose**: Full 5-layer prediction pipeline for ambiguous tokens (table miss or low confidence).

**Key functions**:

```python
def wht(bipolar_vec: np.ndarray) -> np.ndarray:
    """
    Walsh-Hadamard Transform. O(N log N) = O(10K ops) for N=1024.
    Input: (1024,) float32 bipolar vector (+1/-1)
    Output: (1024,) float32 correlations
    """

def bipolar(hv: np.ndarray, W_UINT64: int) -> np.ndarray:
    """Convert uint64 hypervector to (1024,) float32 bipolar (+1/-1)."""

def majority_vote(vecs: List[np.ndarray], weights: np.ndarray = None) -> np.ndarray:
    """
    Weighted majority vote over list of uint64 hypervectors.
    Returns uint64 hypervector where each bit is set if weighted sum > 0.
    """

def hdc_attention(S_p, sem_fwd_matrix, sem_bwd_matrix, codebook, vocab_size, W_UINT64, top_k=4):
    """
    HDC analog of transformer attention.
    Query = S_p, Key = sem_bwd[t], Value = sem_fwd[t]
    
    1. key_sims[t] = hamming_sim(S_p, sem_bwd[t's window])  for all t
    2. top_k_tokens = argsort(key_sims)[-top_k:]
    3. value_bundle = majority_vote([sem_fwd[t] for t in top_k], weights=softmax(key_sims[top_k]))
    4. correction = value_bundle XOR S_p
    5. return S_p XOR (correction & attn_mask)
    
    O(vocab × W) = O(16K ops)
    """

def multihop_layer(query, sem_fwd_matrix, codebook, vocab_size, W_UINT64, n_hops=3, top_k=4):
    """
    Multi-hop forward composition.
    Each hop: find top-k candidates, compose their sem_fwd vectors, blend residually.
    Disambiguates polysemous tokens by reasoning about what would logically follow.
    """

def backward_validation(candidates, S_p, sem_bwd_matrix, W_UINT64):
    """
    For each candidate t: hamming_sim(sem_bwd[t], S_p)
    Returns softmax-normalized scores over candidates.
    O(k × W) = O(64 ops) for k=4
    """

def self_consistency_check(candidate, S_p, sem_fwd_matrix, key, W_UINT64, codebook):
    """
    Hypothetical forward: S_hyp = S_p XOR (sem_fwd[candidate] * key)
    Check: does S_hyp produce a coherent WHT spectrum?
    Returns butterfly_consistency score of S_hyp's winner.
    """

def layered_predict(S_p, sem_fwd_matrix, sem_bwd_matrix, codebook, KEY, p,
                    vocab_size, W_UINT64, srh: SemanticRollingHash,
                    suffix_grammar: SuffixGrammarTable = None) -> Tuple[int, float]:
    """
    Full 5-layer pipeline. Only called on table miss / low confidence.
    
    Layer 0: table lookup (caller handles — returns immediately if conf > 0.9)
    Layer 1: direct S[p] WHT query + butterfly check
    Layer 2: HDC attention refinement
    Layer 3: multi-hop forward (3 hops, top-4)
    Layer 4: backward validation on top-8 candidates
    Layer 5: self-consistency check on top-3 after backward filtering
    
    Optional: suffix grammar score as final morphological gate.
    
    Returns (winner_token, confidence).
    Compute: ~50K ops total → ~0.5 μs at 100 GFLOPS.
    """
```

**Regime detection** (from WHT spectrum shape):
- Regime 1 (clean signal): `correlations[winner] >> 0`, all butterfly partners ≈ 0 → use prediction
- Regime 2 (genuine ambiguity): two elevated peaks, both with clean butterfly partners → blend
- Regime 3 (noise): multiple butterfly levels elevated → fall through

---

## Modifications to Existing Files

### 4. `_semantic_layer.py` — Three Additions

#### 4a. Skip-bigram lag-2 to lag-5 vectors

Add to `DirectionalSemanticVec.build_from_tokens()`:

```python
# After existing lag-1 accumulation, add lags 2-5
for lag in range(2, 6):
    a_toks = tokens[:N - lag].astype(np.int32)
    b_toks = tokens[lag:].astype(np.int32)
    dsv._scatter_xor(dsv.sem_fwd_lag[lag], a_toks, b_toks, codebook)
```

New attribute: `sem_fwd_lag: Dict[int, np.ndarray]` — one (uint64_count,) array per lag.

Storage: 4 × 256 KB = 1 MB for lags 2-5.

At eval: query lag-k vector for the token k positions back. Weight by 1/lag. Cross-lag consensus boosts confidence logarithmically.

#### 4b. XOR orbit diagonal table R[k]

```python
def build_xor_orbit_table(self, bigram_counts: np.ndarray, codebook: np.ndarray,
                            threshold: int = 5) -> np.ndarray:
    """
    R[k] = XOR-bundle of codebook[s] for all (t, s) pairs where:
      - t XOR s == k  (same XOR orbit)
      - bigram_count[t, s] > threshold
    
    R[k] encodes: "what semantic jump does XOR offset k represent?"
    Storage: vocab_size × W_UINT64 × 8 = 128 KB
    """
```

At eval: `diagonal_prediction(S_p, R, codebook)` finds which XOR offset the current state is traveling along, then predicts `recent_token XOR winning_k`.

#### 4c. Pre-training semantic prior (frozen)

```python
@classmethod
def build_pretrain_prior(cls, tokens: np.ndarray, codebook: np.ndarray,
                          vocab_size: int, W: int, uint64_count: int,
                          n_tokens: int = 2_000_000) -> "DirectionalSemanticVec":
    """
    Build sem_prior_fwd and sem_prior_bwd from first n_tokens of corpus.
    These vectors are FROZEN — never modified by Phase 2/3/4.
    
    Used for:
    1. Phase 2 conflict resolution (uncontaminated arbiter)
    2. Phase 3 repair queue annotation (prior-derived candidates)
    3. Phase 4 repair validation (independent confidence source)
    4. Correction map: token → k nearest semantic neighbors (one-step gradient)
    
    Storage: 256 KB (same as main DSV)
    Compute: ~2 seconds for 2M tokens
    """

def build_correction_map(self, vocab_size: int, k_neighbors: int = 8) -> Dict[int, List]:
    """
    For each token t, find k nearest semantic neighbors using one-step gradient
    in token ID space (flip each bit of t, measure sem_prior_fwd similarity).
    
    correction_map[t] = [(neighbor_token, similarity), ...]  sorted descending
    Storage: 1024 × 8 × 4 bytes = 32 KB
    Compute: 1024 × 10 × 16 ops ≈ instant
    """

def build_token_distributions(self, vocab_size: int, codebook: np.ndarray,
                                top_k: int = 8) -> Dict[int, List]:
    """
    For each token t, pre-compute P(next | t) as sparse distribution over top-k.
    Uses WHT over sem_prior_fwd[t] + butterfly consistency filter.
    
    prior_distributions[t] = [(next_token, probability), ...]
    Storage: 1024 × 8 × 4 bytes = 64 KB
    """
```

---

### 5. `train_gpt.py` — Phase Integration

#### 5a. Phase 0: Pre-training semantic prior (new block, before Phase 1)

```python
# ═══════════════════════════════════════════════════════════════════
# Phase 0: Pre-training semantic prior (frozen, 2M tokens, ~2s)
# ═══════════════════════════════════════════════════════════════════
sem_prior = None
correction_map = None
prior_distributions = None

try:
    from _semantic_layer import DirectionalSemanticVec
    _p0_start = time.time()
    sem_prior = DirectionalSemanticVec.build_pretrain_prior(
        tokens, codebook, vocab_size, W_UINT64, vocab_size * W_UINT64,
        n_tokens=2_000_000
    )
    correction_map = sem_prior.build_correction_map(vocab_size)
    prior_distributions = sem_prior.build_token_distributions(vocab_size, codebook)
    print(f"[DNA-HDC Phase 0] Semantic prior built in {time.time()-_p0_start:.2f}s "
          f"| correction_map={len(correction_map)} tokens | ~352 KB total")
except Exception as _p0_err:
    print(f"[DNA-HDC Phase 0] Skipped ({_p0_err})")
```

#### 5b. Phase 1: S[p] accumulation alongside G[p]

Add to the rolling hash precomputation block (after `_rh_chunk_g_states` is built):

```python
# ── Semantic rolling hash S[p] — parallel with G[p] precomputation ──────
_srh = None
_srh_chunk_states = None  # checkpoint states at chunk boundaries (same tier as G)

try:
    from _semantic_rolling_hash import SemanticRollingHash
    if dsv is not None:  # requires sem_fwd to be built first (Phase 3.5-DSV)
        # NOTE: S[p] build is deferred to after Phase 3.5-DSV completes
        # (sem_fwd_matrix needed). Placeholder here; actual build in Phase 3.5-SRH.
        _srh = SemanticRollingHash(W_UINT64, alpha=0.005)
        print(f"[DNA-HDC Phase 1] SemanticRollingHash initialized (alpha=0.005, "
              f"effective window ~200 tokens)")
except Exception as _srh_init_err:
    print(f"[DNA-HDC Phase 1] SRH init skipped ({_srh_init_err})")
```

**Note**: S[p] states are built in a new Phase 3.5-SRH block (after DSV, since sem_fwd is needed). The checkpoint states are stored at the same 2M-token chunk boundaries as G[p].

#### 5c. Phase 2: Semantic conflict resolution in `merge_winners()`

Replace the pure Boyer-Moore decrement with semantic arbitration when `sem_prior` is available:

```python
# In merge_winners(), in the "weaken" branch (mismatch case):
# BEFORE (pure Boyer-Moore):
#   new_counts = stored_confs[weaken_mask] - 1
#   _scatter_table(wb, pack_entry_vec(stored_preds[weaken_mask], new_counts))

# AFTER (semantic arbitration):
if sem_prior is not None and prior_distributions is not None:
    for i, (bucket, stored_tok, new_tok) in enumerate(
            zip(wb, stored_preds[weaken_mask], winner_tokens[weaken_mask])):
        ctx_tok = _get_context_token(bucket)  # approximate from recent tokens
        prior = dict(prior_distributions.get(int(ctx_tok), []))
        p_stored = prior.get(int(stored_tok), 1.0 / vocab_size)
        p_new    = prior.get(int(new_tok),    1.0 / vocab_size)
        
        if p_stored > p_new * 2.0:
            # Prior strongly favors stored — crystallise it, stop Boyer-Moore erosion
            _scatter_table(np.array([bucket]), pack_entry_vec(
                np.array([stored_tok]), np.array([3])))
        elif p_new > p_stored * 2.0:
            # Prior strongly favors new token — overwrite immediately
            _scatter_table(np.array([bucket]), pack_entry_vec(
                np.array([new_tok]), np.array([1])))
        else:
            # Genuinely ambiguous — fall back to Boyer-Moore decrement
            new_count = max(0, int(stored_confs[weaken_mask][i]) - 1)
            _scatter_table(np.array([bucket]), pack_entry_vec(
                np.array([stored_tok]), np.array([new_count])))
else:
    # No prior available — original Boyer-Moore
    new_counts = stored_confs[weaken_mask] - 1
    _scatter_table(wb, pack_entry_vec(stored_preds[weaken_mask], new_counts))
```

#### 5d. Phase 3: Repair queue building as byproduct

Add to the Phase 3 reinforcement loop (after each chunk's `merge_winners` call):

```python
# ── Repair queue building (Phase 3 byproduct) ──────────────────────────
# Only runs if S[p] states and semantic prior are available
if _srh is not None and sem_prior is not None and repair_queue is None:
    repair_queue = {}  # bucket → (wrong_token, candidate, confidence, source)

if repair_queue is not None and _srh_chunk_states is not None:
    _rq_chunk_start = chunk_start
    _rq_S_states = _srh.recompute_chunk(
        _rq_chunk_start, chunk_end, tokens, dsv, keys
    )
    
    for _rq_i, (_rq_p, _rq_tok) in enumerate(
            zip(range(chunk_start, chunk_end), tokens[chunk_start:chunk_end])):
        _rq_bucket = int(buckets[_rq_i])
        _rq_stored, _rq_count = unpack_entry(table_packed[_rq_bucket])
        
        if _rq_count >= 3:
            continue  # crystallised — skip
        
        # Check 1: semantic state disagrees with stored token
        _rq_S_p = _rq_S_states[_rq_i]
        _rq_fwd_stored = hamming_sim(_rq_S_p, sem_fwd_vec(_rq_stored, dsv, W_UINT64))
        _rq_fwd_actual = hamming_sim(_rq_S_p, sem_fwd_vec(_rq_tok, dsv, W_UINT64))
        
        if _rq_fwd_actual > _rq_fwd_stored + 0.05:
            # Semantic state prefers actual token — get confident candidate
            _rq_cand, _rq_conf = layered_predict(
                _rq_S_p, dsv, codebook, keys, _rq_p, vocab_size, W_UINT64, _srh
            )
            if _rq_conf > 0.7:
                existing = repair_queue.get(_rq_bucket)
                if existing is None or _rq_conf > existing[2]:
                    repair_queue[_rq_bucket] = (_rq_stored, _rq_cand, _rq_conf, 'semantic')
        
        # Check 2: consensus across neighboring S[p] states
        if _rq_count < 2 and _rq_p >= 5:
            _rq_cons_pred, _rq_cons_conf, _rq_n_agree = _srh.consensus_predict(
                _rq_S_states, _rq_i, dsv, codebook, keys, window=5
            )
            if _rq_n_agree >= 4 and _rq_cons_pred != _rq_stored:
                repair_queue[_rq_bucket] = (
                    _rq_stored, _rq_cons_pred,
                    _rq_cons_conf * _rq_n_agree / 5.0, 'consensus'
                )
```

#### 5e. Phase 3.5-SRH: Build S[p] checkpoint states (new block after DSV)

```python
# ═══════════════════════════════════════════════════════════════════
# Phase 3.5-SRH: Semantic Rolling Hash state precomputation
# ─────────────────────────────────────────────────────────────────
# Runs after DSV is built (needs sem_fwd_matrix).
# Stores S[p] at chunk boundaries — same tier as G[p] checkpoints.
# ═══════════════════════════════════════════════════════════════════
if _srh is not None and dsv is not None:
    try:
        _srh_t0 = time.time()
        _srh_budget = min(30.0, (config.max_wallclock_seconds - (time.time() - start_time)) * 0.20)
        
        sem_fwd_matrix = dsv.sem_fwd.reshape(vocab_size, W_UINT64)
        _srh_chunk_states = _srh.build_states(
            tokens, sem_fwd_matrix, _rh_keys_cache, _rh_chunk_boundaries,
            time_budget_s=_srh_budget
        )
        print(f"[DNA-HDC Phase 3.5-SRH] S[p] states built in {time.time()-_srh_t0:.2f}s "
              f"| {len(_srh_chunk_states)} checkpoints | alpha={_srh.alpha}")
    except Exception as _srh_err:
        print(f"[DNA-HDC Phase 3.5-SRH] Failed ({_srh_err})")
        _srh_chunk_states = None
```

#### 5f. Phase 3.5-SuffixGrammar: Build suffix grammar table (new block)

```python
# ═══════════════════════════════════════════════════════════════════
# Phase 3.5-SuffixGrammar: Suffix-to-grammar-role table
# ─────────────────────────────────────────────────────────────────
# Requires: CharacterHypervector, S[p] states, tokenizer
# Storage: ~260 KB | Compute: one corpus scan (~5-10s)
# ═══════════════════════════════════════════════════════════════════
suffix_grammar = None
if _srh_chunk_states is not None and _TRANSITION_CODEBOOK_AVAILABLE:
    try:
        from _suffix_grammar import SuffixGrammarTable
        _sg_t0 = time.time()
        _sg_budget = min(15.0, (config.max_wallclock_seconds - (time.time() - start_time)) * 0.10)
        
        char_hv = CharacterHypervector(dim=1024, w_uint64=W_UINT64)
        suffix_grammar = SuffixGrammarTable(
            vocab_size, W_UINT64, char_hv, sp_model, suffix_len=3
        )
        suffix_grammar.build_from_corpus(
            tokens, _srh_chunk_states, _srh, time_budget_s=_sg_budget
        )
        print(f"[DNA-HDC Phase 3.5-SuffixGrammar] Built in {time.time()-_sg_t0:.2f}s "
              f"| ~260 KB | suffix_len=3")
    except Exception as _sg_err:
        print(f"[DNA-HDC Phase 3.5-SuffixGrammar] Skipped ({_sg_err})")
```

#### 5g. Phase 4: Execution-only repair from pre-built queue

Replace the current Phase 4 scan-and-repair loop with a two-part structure:

**Part A** (new): Execute the pre-built repair queue (fast, sorted by confidence):

```python
# ── Phase 4 Part A: Execute pre-built repair queue ──────────────────────
if repair_queue is not None and len(repair_queue) > 0:
    _p4a_start = time.time()
    _p4a_budget = min(60.0, (config.max_wallclock_seconds - (time.time() - start_time)) * 0.40)
    
    sorted_repairs = sorted(repair_queue.items(), key=lambda x: x[1][2], reverse=True)
    _p4a_written = 0
    
    for bucket, (wrong_tok, candidate, conf, source) in sorted_repairs:
        if time.time() - _p4a_start > _p4a_budget:
            break
        
        stored_tok, count = unpack_entry(table_packed[bucket])
        if count >= 3:
            continue  # crystallised — skip
        
        # Final gate: butterfly consistency of candidate's sem_fwd vector
        if dsv is not None and _srh is not None:
            cand_fwd = dsv.sem_fwd[candidate * W_UINT64: (candidate + 1) * W_UINT64]
            cand_corr = _srh.wht_predict(cand_fwd, codebook)[0]
            cand_consistency = _srh.butterfly_consistency(cand_corr, candidate)
            if cand_consistency < 0.6:
                continue
        
        table_packed[bucket] = pack_entry(candidate, 3)
        fingerprint_packed[bucket] = compute_fingerprint(bucket)
        _p4a_written += 1
    
    print(f"[DNA-HDC Phase 4A] Queue repairs: {_p4a_written:,}/{len(sorted_repairs):,} "
          f"written in {time.time()-_p4a_start:.2f}s")
```

**Part B** (existing, unchanged): The current error-residual scan loop continues as before, consuming remaining time budget. It now runs on a table that has already been partially repaired by Part A, so it converges faster.

#### 5h. Eval waterfall: S[p] fallback + layered predict + suffix grammar

Insert after the transition codebook block and before the DSV sem_fwd vote:

```python
# ── S[p] Semantic Rolling Hash fallback ─────────────────────────────────
# Fires when table + overflow + trigram + bigram + transition codebook all miss.
# Uses the accumulated semantic history S[p] to predict via WHT + butterfly check.
if _srh is not None and _srh_chunk_states is not None and np.any(low_conf_mask):
    try:
        from _layered_predict import layered_predict as _lp_fn
        lc_lp_idx = np.where(low_conf_mask)[0]
        
        for _lp_i in lc_lp_idx:
            _lp_p = chunk_start + _lp_i
            # Recompute S[p] for this position from nearest checkpoint
            _lp_S_p = _srh.recompute_single(
                _lp_p, tokens, dsv.sem_fwd.reshape(vocab_size, W_UINT64),
                _rh_keys_cache, _srh_chunk_states
            )
            
            # Full layered prediction pipeline
            _lp_winner, _lp_conf = _lp_fn(
                _lp_S_p,
                dsv.sem_fwd.reshape(vocab_size, W_UINT64),
                dsv.sem_bwd.reshape(vocab_size, W_UINT64),
                codebook, _rh_keys_cache, _lp_p,
                vocab_size, W_UINT64, _srh,
                suffix_grammar=suffix_grammar
            )
            
            if _lp_conf > 0.3:  # regime 1 or 2 — genuine signal
                table_preds[_lp_i] = _lp_winner
                table_conf[_lp_i]  = max(1, int(_lp_conf * 10))
        
        low_conf_mask = (table_conf == 0)
    except Exception as _lp_err:
        pass  # fall through to DSV sem_fwd vote

# ── Skip-bigram diagonal fallback (lag-2 to lag-5) ──────────────────────
if dsv is not None and hasattr(dsv, 'sem_fwd_lag') and np.any(low_conf_mask):
    lc_sb_idx = np.where(low_conf_mask)[0]
    for lag in range(2, 6):
        if lag not in dsv.sem_fwd_lag:
            continue
        if not np.any(low_conf_mask):
            break
        lag_matrix = dsv.sem_fwd_lag[lag].reshape(vocab_size, W_UINT64)
        for _sb_i in lc_sb_idx:
            if table_conf[_sb_i] > 0:
                continue
            _sb_p = chunk_start + _sb_i
            if _sb_p < lag:
                continue
            ctx_tok = int(val_tokens[_sb_p - lag])
            lag_vec = lag_matrix[ctx_tok]
            # WHT over lag vector
            _sb_corr = _srh.wht_predict(lag_vec, codebook)[0] if _srh else None
            if _sb_corr is not None:
                _sb_winner = int(np.argmax(_sb_corr))
                _sb_conf = float(_sb_corr[_sb_winner])
                if _sb_conf > 0.15 / lag:  # weight by 1/lag
                    table_preds[_sb_i] = _sb_winner
                    table_conf[_sb_i]  = 1
        low_conf_mask = (table_conf == 0)
```

---

## Data Flow Diagram

```mermaid
graph TD
    A[Raw corpus tokens] --> B[Phase 0: Pre-training prior\n2M tokens, frozen\nsem_prior_fwd/bwd\ncorrection_map\nprior_distributions]
    A --> C[Phase 1: G[p] rolling hash\nprecomputation\nchunk boundary states]
    B --> D[Phase 2: DNA Stack\nmerge_winners with\nsemantic conflict resolution\nusing prior_distributions]
    C --> D
    A --> E[Phase 1.5: bigram +\nskip-bigram lag 2-5]
    D --> F[Phase 3: Reinforcement\n+ repair queue building\nsemanticaly annotated]
    F --> G[Phase 3.5-DSV:\nsem_fwd + sem_bwd\nskip-bigram lags\nXOR orbit diagonals]
    G --> H[Phase 3.5-SRH:\nS[p] checkpoint states\nalpha=0.005]
    H --> I[Phase 3.5-SuffixGrammar:\nsuffix_hvs\nsuffix_gram_bundle]
    F --> J[Phase 4A: Execute\npre-built repair queue\nsorted by confidence]
    I --> J
    J --> K[Phase 4B: Error-residual\nscan loop\nremaining budget]
    K --> L[Eval waterfall:\nG[p] table\noverflow\ntrigram\nbigram\ntransition codebook\nS[p] layered predict\nskip-bigram lags\nDSV sem_fwd\nSmearGate blend]
```

---

## Storage Budget

| Component | Size | Notes |
|---|---|---|
| `table_packed` | 8 MB | Existing (TABLE_BITS=22) |
| `fingerprint_packed` | 4 MB | Existing |
| `bigram_packed` | 2 KB | Existing |
| `trigram_packed` | ~2 MB | Existing |
| `sem_fwd + sem_bwd` (DSV) | 256 KB | Existing |
| `sem_fwd_lag[2..5]` | 1 MB | New — skip-bigram lags |
| `R[k]` XOR orbit table | 128 KB | New — diagonal prediction |
| `sem_prior_fwd/bwd` (frozen) | 256 KB | New — pre-training prior |
| `correction_map` | 32 KB | New — token neighbors |
| `prior_distributions` | 64 KB | New — P(next\|t) sparse |
| `suffix_hvs` | 128 KB | New — suffix encodings |
| `suffix_gram_bundle` | 128 KB | New — grammar signatures |
| `S[p] chunk checkpoints` | ~32 KB | New — 250 checkpoints × 16 uint64 |
| `repair_queue` | ~1 MB peak | New — transient, Phase 3 only |
| **Total new storage** | **~1.8 MB** | Within 5 MB model budget |

---

## Compute Budget (10-minute wall clock)

| Phase | Time allocation | Notes |
|---|---|---|
| Phase 0 (prior) | ~3s | 2M token scan, one-time |
| Phase 1 (G[p] precompute) | ~50s | Existing |
| Phase 1.5 (bigrams) | ~1s | Existing |
| Phase 2 (DNA stack) | ~120s | +~5% overhead for semantic arbitration |
| Phase 3 (reinforcement + queue) | ~120s | +~10% overhead for queue building |
| Phase 3.5-DSV | ~55s | Existing |
| Phase 3.5-SRH | ~20s | New — S[p] checkpoint build |
| Phase 3.5-SuffixGrammar | ~10s | New — one corpus scan |
| Phase 4A (queue execution) | ~30s | New — replaces part of Phase 4 scan |
| Phase 4B (error-residual) | ~120s | Existing loop, faster convergence |
| **Total** | **~530s** | Within 600s budget |

---

## Implementation Order (Recommended)

Implement in this order to allow incremental testing at each step:

1. **`_semantic_rolling_hash.py`** — standalone, no dependencies on other new code
2. **`_layered_predict.py`** — depends on SRH for WHT/butterfly; can stub suffix_grammar
3. **`_suffix_grammar.py`** — depends on CharacterHypervector (already exists)
4. **`_semantic_layer.py` additions** — skip-bigram lags, XOR orbit, pre-training prior
5. **`train_gpt.py` Phase 0** — pre-training prior build (isolated new block)
6. **`train_gpt.py` Phase 3.5-SRH + SuffixGrammar** — new blocks after DSV
7. **`train_gpt.py` Phase 4A** — queue execution (add before existing Phase 4 loop)
8. **`train_gpt.py` Phase 3 queue building** — add to reinforcement loop
9. **`train_gpt.py` Phase 2 semantic arbitration** — modify merge_winners
10. **`train_gpt.py` eval waterfall** — add S[p] + skip-bigram fallbacks

This order ensures each step is testable independently. Steps 1-4 are pure additions (new files/methods). Steps 5-10 are integrations into train_gpt.py, each guarded by `try/except` so failures degrade gracefully to the existing behavior.

---

## Key Design Invariants

1. **All new code is guarded by `try/except`** — any failure falls through to existing behavior. BPB cannot regress due to a new module import error.

2. **S[p] is never stored for all N tokens** — only checkpoint states at chunk boundaries. Recomputed forward within each chunk during eval. Memory cost is O(N/chunk_size) not O(N).

3. **sem_prior is frozen after Phase 0** — never modified by Phase 2/3/4. Independence from training noise is the source of its value as an arbiter.

4. **Layered predict only fires on table miss** — Layer 0 (table lookup) returns immediately if confidence > 0.9. The 5-layer pipeline adds zero overhead to high-confidence predictions.

5. **Suffix grammar gates subatomic expansion** — morphological neighbors are only added to the candidate set after passing the grammar score threshold. This prevents tense/number/POS confusion.

6. **Phase 4A is additive, not replacing Phase 4B** — the queue execution runs first (fast, high-confidence repairs), then the existing error-residual scan runs on the remaining budget. Total repair quality improves; no existing repair logic is removed.

7. **Forgetting factor alpha is tunable** — default 0.005 gives ~200-token effective window. Can be tuned on validation data. The deterministic flip_mask (seeded by position) makes it reproducible across runs.