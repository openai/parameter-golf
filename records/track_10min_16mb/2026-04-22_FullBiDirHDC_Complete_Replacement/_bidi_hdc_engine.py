"""FullBiDirHDC — Bilateral HDC Engine adapted for Language Modeling.

Extracted and adapted from Arc_AGI_3_HDC_Hadamard_DSV_Model-main/
ARC-AGI-3-Agents/agents/templates/bidi_hdc_full.py

Changes from the original:
  - Removed all ARC-AGI-3 game engine dependencies (OxytocinSystem,
    ThalamicSafetySystem, UpgradedSafetyGate, LTGAController, watermark)
  - trust=1.0, safety=1.0 defaults (no game engine safety systems needed)
  - Added train_on_tokens() — O(N) batch training pass over token sequence
  - Added vote_scores_vectorised() — vectorised bilateral scoring for BPB eval
  - SpiralPointerMemory imported from _spiral_dsv_lm (local copy)

All original HDC logic preserved exactly:
  Codebook, ManifoldAxes, ZSignal/ZState, ResonanceSignal,
  RelationshipMemory, ChainManifold, FullBiDirHDC, InferResult
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional, Tuple

try:
    from _spiral_dsv_lm import SpiralPointerMemory, GOLDEN_AXES
except ImportError:
    SpiralPointerMemory = None  # type: ignore[assignment,misc]
    GOLDEN_AXES = None          # type: ignore[assignment]

# ─────────────────────────────────────────────────────────────────────────────
PHI_FRAC = 0.6180339887498949
EPS      = 1e-8

# Anticipation Enhancement v2 thresholds
ACCEL_THRESHOLD   = 0.02
RETRO_THRESHOLD   = 0.55
FORWARD_THRESHOLD = 0.60
CHAIN_MAX_RULES   = 256
CHAIN_JACCARD_THRESHOLD = 0.70

# Fast byte-level popcount lookup
POPCOUNT_TABLE = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)


def _popcount_2d(arr: np.ndarray) -> np.ndarray:
    """(N, W) uint64 → (N,) int32 total set bits per row."""
    return POPCOUNT_TABLE[arr.view(np.uint8).reshape(arr.shape[0], -1)].sum(axis=1).astype(np.int32)


def _cosine_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,W) × (N,W) → (N,) cosine in [0,1]. 0.5=random, 1.0=identical."""
    hamming = _popcount_2d(np.bitwise_xor(a, b)) / (a.shape[1] * 64)
    return np.float32(0.5 + 0.5 * (1.0 - 2.0 * hamming))


def _jaccard_batch(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    inter = _popcount_2d(np.bitwise_and(a, b)).astype(np.float32)
    union = _popcount_2d(np.bitwise_or(a, b)).astype(np.float32)
    return np.where(union > 0, inter / union, np.float32(0.0))


def _entropy_batch(arr: np.ndarray) -> np.ndarray:
    p = _popcount_2d(arr) / (arr.shape[1] * 64)
    p = np.clip(p, EPS, 1.0 - EPS)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return h.astype(np.float32)


def _cosine_single(a: np.ndarray, b: np.ndarray) -> float:
    return float(_cosine_batch(a[np.newaxis], b[np.newaxis])[0])


# ─────────────────────────────────────────────────────────────────────────────

class Codebook:
    """Universal token ↔ HV mapping via XOR bundle.

    encode([a, b, c]) = CB[a] ⊕ CB[b] ⊕ CB[c]
    decode via cosine nearest-neighbour.
    """

    def __init__(self, vocab_size: int, n_words: int, seed: int = 42):
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.W = n_words
        self.vecs: np.ndarray = rng.integers(
            0, np.iinfo(np.uint64).max,
            size=(vocab_size, n_words), dtype=np.uint64
        )

    def encode(self, tokens: list) -> np.ndarray:
        hv = np.zeros(self.W, dtype=np.uint64)
        for t in tokens:
            hv = np.bitwise_xor(hv, self.vecs[t % self.vocab_size])
        return hv

    def decode(self, hv: np.ndarray, top_k: int = 1) -> list:
        sims = _cosine_batch(
            np.tile(hv, (self.vocab_size, 1)),
            self.vecs
        )
        return list(np.argsort(sims)[::-1][:top_k])

    def relationship_strength(self, tok_a: int, tok_b: int) -> float:
        a = self.vecs[tok_a % self.vocab_size][np.newaxis]
        b = self.vecs[tok_b % self.vocab_size][np.newaxis]
        return float(_jaccard_batch(a, b)[0])


class ManifoldAxes:
    Y_SPLIT = 8

    def __init__(self, n_axes: int, n_words: int, axis_offset: int = 0):
        self.n_axes = n_axes
        self.W = n_words
        self.axis_offset = axis_offset
        total_bits = n_words * 64
        self.compass_mask, self.proto_mask = self._build_masks(n_axes, total_bits, axis_offset)

    def _build_masks(self, n: int, total_bits: int, axis_offset: int = 0):
        compass = np.zeros(self.W, dtype=np.uint64)
        proto   = np.zeros(self.W, dtype=np.uint64)
        for local_k in range(n):
            global_k = axis_offset + local_k
            offset   = int(global_k * PHI_FRAC * total_bits) % total_bits
            word_idx = offset // 64
            bit_idx  = offset % 64
            if local_k < self.Y_SPLIT:
                compass[word_idx] ^= np.uint64(1 << bit_idx)
            else:
                proto[word_idx] ^= np.uint64(1 << bit_idx)
        return compass, proto

    def step_mask(self, S: float) -> np.ndarray:
        if S == 0.0:
            return np.bitwise_xor(self.compass_mask, self.proto_mask)
        rng  = np.random.default_rng()
        pick = rng.random(self.W) < abs(S)
        if S < 0:
            return np.where(pick, self.compass_mask, self.proto_mask).astype(np.uint64)
        else:
            return np.where(pick, self.proto_mask, self.compass_mask).astype(np.uint64)

    def extend(self, new_n: int):
        self.compass_mask, self.proto_mask = self._build_masks(
            new_n, self.W * 64, self.axis_offset
        )
        self.n_axes = new_n


@dataclass
class ZState:
    Z_current  : float = 0.0
    Z_prev     : float = 0.0
    Z_prev2    : float = 0.0
    alpha      : float = 0.5
    t_last     : int   = 0
    history    : list  = field(default_factory=list)
    tau        : float = 0.0
    lam        : float = 0.05
    alpha_lr   : float = 0.05


class ZSignal:
    def __init__(self, tau: float = 0.0, lam: float = 0.05):
        self.state = ZState(tau=tau, lam=lam)

    def update(self, Y: float, H: float, t: int) -> float:
        s = self.state
        X   = s.Z_prev - s.Z_prev2
        Z_t = s.alpha * X + (1.0 - s.alpha) * Y
        dt  = max(1, t - s.t_last)
        decay = np.exp(-s.lam * dt)
        S = (1.0 / (1.0 + np.exp(-(Z_t - s.tau)))) - 0.5
        S *= (1.0 / (H + EPS)) * decay * 2.0
        err = Y - Z_t
        s.alpha = float(np.clip(s.alpha + s.alpha_lr * err, 0.1, 0.9))
        s.Z_prev2   = s.Z_prev
        s.Z_prev    = s.Z_current
        s.Z_current = float(Z_t)
        s.t_last    = t
        s.history.append(float(Z_t))
        return float(np.clip(S, -1.0, 1.0))

    def reset(self):
        s = self.state
        s.Z_current = s.Z_prev = s.Z_prev2 = 0.0
        s.history.clear()


class ResonanceSignal:
    def __init__(self, lam: float = 0.05):
        self.lam = lam

    def compute(
        self,
        mag      : float,
        saliency : float,
        trust    : float,
        entropy  : float,
        dt       : int,
        safety   : float = 1.0,
    ) -> float:
        decay = np.exp(-self.lam * dt)
        return float(
            (mag * saliency)
            * (trust / (entropy + EPS))
            * decay
            * safety
        )


class RelationshipMemory:
    def __init__(self, n_words: int, max_rules: int = 512):
        self.W         = n_words
        self.max_rules = max_rules
        self._rules    : list = []
        self._weights  : list = []
        self._action_ids: list = []

    def store(self, action_id: int, rule_hv: np.ndarray, weight: float):
        self._rules.append(rule_hv.copy())
        self._weights.append(weight)
        self._action_ids.append(action_id)
        if len(self._rules) > self.max_rules:
            worst = int(np.argmin(self._weights))
            self._rules.pop(worst)
            self._weights.pop(worst)
            self._action_ids.pop(worst)

    def query_effect(self, action_id: int, context_hv: np.ndarray) -> Optional[np.ndarray]:
        idxs = [i for i, a in enumerate(self._action_ids) if a == action_id]
        if not idxs:
            return None
        rules   = np.stack([self._rules[i] for i in idxs])
        context = np.tile(context_hv, (len(idxs), 1))
        sims    = _cosine_batch(rules, context)
        weights = np.array([self._weights[i] for i in idxs]) * sims
        if weights.sum() < EPS:
            return rules[0]
        idx_sorted = np.argsort(weights)[::-1][:8]
        bundle = np.zeros(self.W, dtype=np.uint64)
        for i in idx_sorted:
            p = weights[i] / weights[idx_sorted].sum()
            flip = np.random.random(self.W) < p
            bundle = np.where(flip, np.bitwise_xor(bundle, rules[i]), bundle).astype(np.uint64)
        return bundle

    def jaccard_similarity(self, hv_a: np.ndarray, hv_b: np.ndarray) -> float:
        a, b = hv_a[np.newaxis], hv_b[np.newaxis]
        return float(_jaccard_batch(a, b)[0])


class ChainManifold:
    """Mini joint manifold for (A→B) bigram storage and retrieval."""

    def __init__(
        self,
        codebook: "Codebook",
        n_hyp: int = 20,
        max_iters: int = 10,
        n_axes: int = 8,
        noise_rate: float = 0.05,
    ):
        self.cb         = codebook
        self.W          = codebook.W
        self.H          = n_hyp
        self.max_iters  = max_iters
        self.noise_rate = noise_rate
        self.manifold      = ManifoldAxes(n_axes, self.W)
        self.rel_mem       = RelationshipMemory(self.W, max_rules=CHAIN_MAX_RULES)
        self._rule_bundle  = np.zeros(self.W, dtype=np.uint64)
        self._rule_weight  = 0.0
        self._n_stored     = 0

    def observe(
        self,
        action_a_hv: np.ndarray,
        action_b_id: int,
        world_fwd_hv: np.ndarray,
        weight: float,
    ) -> None:
        rule_hv = np.bitwise_xor(action_a_hv, world_fwd_hv)
        alpha = weight / (self._rule_weight + weight + EPS)
        flip  = np.random.random(self.W) < alpha
        self._rule_bundle = np.where(
            flip, np.bitwise_xor(self._rule_bundle, rule_hv), self._rule_bundle
        ).astype(np.uint64)
        self._rule_weight += weight
        self.rel_mem.store(action_b_id, rule_hv, weight)
        self._n_stored += 1

    def query(
        self,
        action_a_hv: np.ndarray,
        candidate_action_ids: list,
        action_token_map: dict,
        S: float = 0.0,
    ) -> tuple:
        if not candidate_action_ids or self._rule_weight < EPS:
            return None, 0.0, -1

        n   = len(candidate_action_ids)
        rng = np.random.default_rng()
        full_mask = np.bitwise_xor(self.manifold.compass_mask, self.manifold.proto_mask)
        fwd = np.empty((n * self.H, self.W), dtype=np.uint64)
        bwd = np.empty((n * self.H, self.W), dtype=np.uint64)

        for i, aid in enumerate(candidate_action_ids):
            act_hv = self.cb.encode(action_token_map.get(aid, [aid]))
            fwd_s  = np.bitwise_xor(action_a_hv, act_hv)
            bwd_s  = np.bitwise_xor(action_a_hv, np.bitwise_xor(act_hv, full_mask))
            fb     = np.tile(fwd_s, (self.H, 1))
            bb     = np.tile(bwd_s, (self.H, 1))
            noise_m = rng.random((self.H, self.W)) < self.noise_rate
            noise_v = rng.integers(0, np.iinfo(np.uint64).max, (self.H, self.W), dtype=np.uint64)
            fwd[i*self.H:(i+1)*self.H] = np.where(noise_m, np.bitwise_xor(fb, noise_v), fb).astype(np.uint64)
            bwd[i*self.H:(i+1)*self.H] = np.where(noise_m, np.bitwise_xor(bb, noise_v), bb).astype(np.uint64)

        if self._rule_weight > 0:
            blend  = min(0.5, self._rule_weight / (self._rule_weight + 20.0))
            rule_t = np.tile(self._rule_bundle, (n * self.H, 1))
            bm     = rng.random((n * self.H, self.W)) < blend
            fwd    = np.where(bm, np.bitwise_xor(fwd, rule_t), fwd).astype(np.uint64)

        active  = np.ones(n * self.H, dtype=bool)
        prev_pc = np.zeros(n * self.H, dtype=np.float32)
        for _ in range(self.max_iters):
            if not active.any():
                break
            mask = self.manifold.step_mask(S)
            fwd[active] = np.bitwise_xor(fwd[active], mask)
            cur_pc = _popcount_2d(fwd[active]).astype(np.float32)
            delta  = np.abs(cur_pc - prev_pc[active])
            done   = delta < 1.0
            idxs   = np.where(active)[0]
            active[idxs[done]] = False
            prev_pc[active]    = cur_pc[~done]

        bwd_active  = np.ones(n * self.H, dtype=bool)
        bwd_prev_pc = np.zeros(n * self.H, dtype=np.float32)
        for _ in range(self.max_iters):
            if not bwd_active.any():
                break
            mask = self.manifold.step_mask(-S)
            bwd[bwd_active] = np.bitwise_xor(bwd[bwd_active], mask)
            cur_pc = _popcount_2d(bwd[bwd_active]).astype(np.float32)
            delta  = np.abs(cur_pc - bwd_prev_pc[bwd_active])
            done   = delta < 1.0
            idxs   = np.where(bwd_active)[0]
            bwd_active[idxs[done]] = False
            bwd_prev_pc[bwd_active] = cur_pc[~done]

        fwd_r = fwd.reshape(n, self.H, self.W)
        bwd_r = bwd.reshape(n, self.H, self.W)
        fwd_f = fwd_r.reshape(n * self.H, self.W)
        bwd_f = bwd_r.reshape(n * self.H, self.W)
        consistency = _cosine_batch(fwd_f, bwd_f).reshape(n, self.H)

        if self._rule_weight > 0:
            rb_tile    = np.tile(self._rule_bundle, (n * self.H, 1))
            rule_align = _cosine_batch(fwd_f, rb_tile).reshape(n, self.H)
        else:
            rule_align = np.full((n, self.H), 0.5, dtype=np.float32)

        joint    = 0.6 * consistency + 0.4 * rule_align
        best_h   = joint.argmax(axis=1)
        a_scores = joint[np.arange(n), best_h]
        best_a   = int(a_scores.argmax())
        best_hv  = int(best_h[best_a])

        prior_hv    = fwd_r[best_a, best_hv].copy()
        chain_score = float(a_scores[best_a])
        best_aid    = candidate_action_ids[best_a]
        return prior_hv, chain_score, best_aid

    @property
    def n_stored(self) -> int:
        return self._n_stored


@dataclass
class InferResult:
    best_action           : int
    joint_scores          : dict
    consistency           : float
    goal_sim              : float
    traj_slope            : float
    surprise              : float
    entropy               : float
    resonance             : float
    Z                     : float
    S                     : float
    best_fwd_hv           : np.ndarray
    best_bwd_hv           : np.ndarray
    retrodiction_accuracy : float = 0.5
    traj_accel            : float = 0.0
    chain_hits            : int   = 0


class FullBiDirHDC:
    """Joint Bidirectional HDC Manifold Engine — Language Model adaptation.

    Identical to the ARC-AGI-3 version except:
    - trust=1.0, safety=1.0 defaults (no game engine safety systems)
    - Added train_on_tokens() for O(N) batch training
    - Added vote_scores_vectorised() for fast BPB evaluation
    """

    W_CONSISTENCY = 0.35
    W_GOAL_SIM    = 0.30
    W_TRAJ_SLOPE  = 0.20
    W_ENTROPY     = 0.10
    W_RESONANCE   = 0.05

    def __init__(
        self,
        codebook    : Codebook,
        n_axes      : int   = 19,
        n_hyp       : int   = 200,
        max_iters   : int   = 40,
        noise_rate  : float = 0.05,
        axis_offset : int   = 0,
        pointer_mask: Optional[np.ndarray] = None,
    ):
        self.cb          = codebook
        self.W           = codebook.W
        self.H           = n_hyp
        self.max_iters   = max_iters
        self.noise_rate  = noise_rate
        self.axis_offset = axis_offset
        self.pointer_mask: Optional[np.ndarray] = (
            pointer_mask.astype(np.uint64) if pointer_mask is not None else None
        )

        self.manifold   = ManifoldAxes(n_axes, self.W, axis_offset=axis_offset)
        self.z_signal   = ZSignal()
        self.resonance  = ResonanceSignal()
        self.rel_mem    = RelationshipMemory(self.W)

        self._rule_bundle  = np.zeros(self.W, dtype=np.uint64)
        self._rule_weight  = 0.0

        self.goal_hv       : Optional[np.ndarray] = None
        self._goal_weight  : float = 0.0
        self._base_goal_weight: float = 0.0

        self._corr_buf     : list = []
        self._buf_max      = 8

        self._step         = 0
        self._last_S       = 0.0
        self._last_Z       : float = 0.0
        self._prev_fwd_hv  : Optional[np.ndarray] = None

        self._prev_bwd_hv     : Optional[np.ndarray] = None
        self._prev_present_hv : Optional[np.ndarray] = None
        self._prev_traj_slope : float = 0.0
        self._last_consistency: float = 0.5
        self._last_action_id  : Optional[int] = None
        self._last_action_tokens: list = []

        self.chain_memory = ChainManifold(
            codebook   = codebook,
            n_hyp      = max(10, n_hyp // 10),
            max_iters  = max(5, max_iters // 4),
            n_axes     = min(8, n_axes),
            noise_rate = noise_rate,
        )

        if SpiralPointerMemory is not None:
            self.spiral_mem: Optional[SpiralPointerMemory] = SpiralPointerMemory(
                n_words=self.W, n_levels=4
            )
        else:
            self.spiral_mem = None

        self._spiral_addr_counter: int = 0

    def _encode(self, tokens: list) -> np.ndarray:
        hv = self.cb.encode(tokens)
        if self.pointer_mask is not None:
            hv = np.bitwise_xor(hv, self.pointer_mask).astype(np.uint64)
        return hv

    def observe(
        self,
        before_tokens : list,
        action_id     : int,
        after_tokens  : list,
        reward        : float = 0.0,
    ) -> None:
        before_hv  = self._encode(before_tokens)
        action_hv  = self._encode([action_id])
        after_hv   = self._encode(after_tokens)
        rule_hv    = np.bitwise_xor(np.bitwise_xor(before_hv, action_hv), after_hv)

        res = self.resonance.compute(
            mag=1.0, saliency=1.0, trust=1.0,
            entropy=0.5, dt=1, safety=1.0,
        )
        self._bundle_rule(rule_hv, weight=res + abs(reward))
        self.rel_mem.store(action_id, rule_hv, weight=res + abs(reward))

        if reward > 0.0:
            self._update_goal(after_hv, weight=1.0 + reward * 5.0)

        if self.spiral_mem is not None and reward >= 10.0:
            try:
                self.store_spiral(after_hv)
            except Exception:
                pass

    def train_on_tokens(
        self,
        tokens: np.ndarray,
        bigram_freq: np.ndarray,
        chunk_size: int = 500_000,
        verbose: bool = True,
    ) -> None:
        """O(N) training pass over token sequence.

        For each position p:
            before_tokens = [tokens[p]]
            action_id     = tokens[p]
            after_tokens  = [tokens[p+1]]
            reward        = bigram_freq[tokens[p], tokens[p+1]] * 100.0

        High-reward transitions (reward >= 10) are archived in spiral_mem.

        Args:
            tokens      : (N,) int array of token IDs
            bigram_freq : (vocab_size, vocab_size) float32 — P(b|a)
            chunk_size  : Progress reporting interval
            verbose     : Print progress
        """
        N = len(tokens)
        if N < 2:
            return

        # Vectorised rule bundle update: process all bigrams at once
        # For each (t_prev, t_next) pair:
        #   before_hv = codebook[t_prev]
        #   action_hv = codebook[t_prev]  (action_id = t_prev)
        #   after_hv  = codebook[t_next]
        #   rule_hv   = before_hv XOR action_hv XOR after_hv
        #             = codebook[t_prev] XOR codebook[t_prev] XOR codebook[t_next]
        #             = codebook[t_next]  (since A XOR A = 0)
        # So rule_hv = codebook[t_next] for all positions.
        # The rule bundle accumulates XOR of all next-token HVs weighted by reward.

        t_prev_arr = tokens[:-1].astype(np.int32)
        t_next_arr = tokens[1:].astype(np.int32)
        rewards    = bigram_freq[t_prev_arr, t_next_arr].astype(np.float32) * 100.0

        if verbose:
            print(f"[BiDirHDC] Training on {N-1:,} bigrams...")

        # Process in chunks for memory efficiency
        for chunk_start in range(0, N - 1, chunk_size):
            chunk_end = min(chunk_start + chunk_size, N - 1)
            tp = t_prev_arr[chunk_start:chunk_end]
            tn = t_next_arr[chunk_start:chunk_end]
            rw = rewards[chunk_start:chunk_end]

            for i in range(len(tp)):
                t_p = int(tp[i])
                t_n = int(tn[i])
                r   = float(rw[i])
                self.observe(
                    before_tokens=[t_p],
                    action_id=t_p,
                    after_tokens=[t_n],
                    reward=r,
                )

            if verbose:
                pct = 100.0 * chunk_end / (N - 1)
                print(f"[BiDirHDC] {chunk_end:,}/{N-1:,} ({pct:.1f}%)")

    def vote_scores_vectorised(
        self,
        prev_tokens: np.ndarray,
    ) -> np.ndarray:
        """Vectorised bilateral scoring for all vocab tokens.

        For each prev_token t:
            query_hv = codebook[t] XOR rule_bundle
            fwd_scores[v] = cosine(query_hv, codebook[v])  for all v
            bwd_scores[v] = cosine(codebook[v] XOR rule_bundle, codebook[t])  for all v
            consistency[v] = (fwd_scores[v] + bwd_scores[v]) / 2
            probs = softmax(consistency)

        Args:
            prev_tokens : (batch,) int32 — previous token IDs

        Returns:
            (batch, vocab_size) float32 — probability distribution over next tokens
        """
        cb   = self.cb.vecs                          # (vocab_size, n_words) uint64
        rb   = self._rule_bundle                     # (n_words,) uint64
        batch = len(prev_tokens)
        vocab_size, n_words = cb.shape
        half = float(n_words * 32)                   # n_words × 64 / 2

        # Forward: query_hv = codebook[prev_t] XOR rule_bundle
        query_hvs = cb[prev_tokens] ^ rb[None, :]    # (batch, n_words)

        # fwd_scores[b, v] = cosine(query_hvs[b], cb[v])
        xor_fwd = query_hvs[:, None, :] ^ cb[None, :, :]  # (batch, vocab, n_words)
        pc_fwd  = np.unpackbits(xor_fwd.view(np.uint8), axis=2).sum(axis=2).astype(np.float32)
        fwd_scores = (half - pc_fwd) / half              # (batch, vocab)

        # Backward: bwd_hv[v] = codebook[v] XOR rule_bundle
        bwd_hvs = cb ^ rb[None, :]                       # (vocab, n_words)
        # bwd_scores[b, v] = cosine(bwd_hvs[v], codebook[prev_t])
        xor_bwd = bwd_hvs[None, :, :] ^ cb[prev_tokens][:, None, :]  # (batch, vocab, n_words)
        pc_bwd  = np.unpackbits(xor_bwd.view(np.uint8), axis=2).sum(axis=2).astype(np.float32)
        bwd_scores = (half - pc_bwd) / half              # (batch, vocab)

        # Bilateral consistency → softmax
        consistency = (fwd_scores + bwd_scores) / 2.0   # (batch, vocab_size)
        consistency -= consistency.max(axis=1, keepdims=True)
        probs = np.exp(consistency)
        probs /= probs.sum(axis=1, keepdims=True)
        return probs.astype(np.float32)

    def step(
        self,
        present_tokens  : list,
        action_token_map: dict,
        actual_next_tokens: Optional[list] = None,
        trust           : float = 1.0,
        safety          : float = 1.0,
    ) -> InferResult:
        self._step += 1

        present_hv = self._encode(present_tokens)
        action_hvs = {aid: self._encode(toks) for aid, toks in action_token_map.items()}
        action_ids = list(action_hvs.keys())
        n_actions  = len(action_ids)

        retrodiction_accuracy = 0.5
        if self._prev_bwd_hv is not None and self._prev_present_hv is not None:
            retrodiction_accuracy = _cosine_single(self._prev_bwd_hv, self._prev_present_hv)

        surprise = 0.0
        actual_hv: Optional[np.ndarray] = None
        if actual_next_tokens is not None and self._prev_fwd_hv is not None:
            actual_hv = self._encode(actual_next_tokens)
            a_tile    = actual_hv[np.newaxis]
            p_tile    = self._prev_fwd_hv[np.newaxis]
            surprise  = float(1.0 - _cosine_batch(a_tile, p_tile)[0])

        self._prev_present_hv = present_hv.copy()

        fwd_seeds, bwd_seeds = self._build_seeds(present_hv, action_hvs, action_ids)

        S = self._last_S
        fwd_f, fwd_i, fwd_sim_hist, _chain_hits = self._propagate(fwd_seeds, direction=+1, S=S)
        bwd_f, bwd_i, _, _                       = self._propagate(bwd_seeds, direction=-1, S=S)

        fwd_f = fwd_f.reshape(n_actions, self.H, self.W)
        bwd_f = bwd_f.reshape(n_actions, self.H, self.W)
        fwd_i = fwd_i.reshape(n_actions, self.H)
        bwd_i = bwd_i.reshape(n_actions, self.H)
        T_sim = fwd_sim_hist.shape[1] if fwd_sim_hist.ndim == 2 else 1
        fwd_sim_hist = fwd_sim_hist.reshape(n_actions, self.H, T_sim)

        joint, details = self._score_joint(fwd_f, bwd_f, fwd_i, bwd_i, fwd_sim_hist)

        best_h   = joint.argmax(axis=1)
        a_scores = joint[np.arange(n_actions), best_h]
        best_a   = int(a_scores.argmax())
        best_act = action_ids[best_a]
        best_hv  = int(best_h[best_a])

        best_fwd  = fwd_f[best_a, best_hv]
        best_bwd  = bwd_f[best_a, best_hv]
        b_consist = float(details['consistency'][best_a, best_hv])
        b_gsim    = float(details['goal_sim']   [best_a, best_hv])
        b_slope   = float(details['traj_slope'] [best_a, best_hv])
        b_ent     = float(details['entropy']    [best_a, best_hv])

        Y      = float(np.mean(details['goal_sim']))
        H_mean = float(np.mean(details['entropy']))
        S_new  = self.z_signal.update(Y, H_mean, self._step)
        self._last_Z = self.z_signal.state.Z_current

        traj_accel = b_slope - self._prev_traj_slope
        self._prev_traj_slope = b_slope

        if (traj_accel < -ACCEL_THRESHOLD) or \
           (traj_accel < 0.0 and retrodiction_accuracy < RETRO_THRESHOLD):
            S_new = -abs(S_new)

        self._last_S = S_new

        mag = float(np.abs(b_slope)) + surprise
        res = self.resonance.compute(
            mag=mag, saliency=b_consist, trust=trust,
            entropy=b_ent, dt=1, safety=safety,
        )

        rule_hv = np.bitwise_xor(best_fwd, best_bwd)
        self._bundle_rule(rule_hv, weight=res)
        self.rel_mem.store(best_act, rule_hv, weight=res)

        if b_consist > 0.65:
            self._update_goal(best_fwd, weight=res)

        current_action_tokens = action_token_map.get(best_act, [best_act])
        if (self._last_action_id is not None and
                self._prev_fwd_hv is not None and
                actual_hv is not None):
            forward_acc = 1.0 - surprise
            if (forward_acc > FORWARD_THRESHOLD and
                    retrodiction_accuracy > RETRO_THRESHOLD and
                    self._last_consistency > 0.65):
                action_a_hv = self.cb.encode(self._last_action_tokens)
                self.chain_memory.observe(
                    action_a_hv  = action_a_hv,
                    action_b_id  = best_act,
                    world_fwd_hv = best_fwd,
                    weight       = res,
                )

        self._prev_bwd_hv      = best_bwd.copy()
        self._last_consistency = b_consist
        self._last_action_id   = best_act
        self._last_action_tokens = current_action_tokens

        self._corr_buf.append(best_fwd)
        if len(self._corr_buf) > self._buf_max:
            self._corr_buf.pop(0)
        self._prev_fwd_hv = best_fwd

        return InferResult(
            best_action   = best_act,
            joint_scores  = {aid: float(a_scores[i]) for i, aid in enumerate(action_ids)},
            consistency   = b_consist,
            goal_sim      = b_gsim,
            traj_slope    = b_slope,
            surprise      = surprise,
            entropy       = b_ent,
            resonance     = res,
            Z             = self.z_signal.state.Z_current,
            S             = S_new,
            best_fwd_hv   = best_fwd,
            best_bwd_hv   = best_bwd,
            retrodiction_accuracy = retrodiction_accuracy,
            traj_accel            = traj_accel,
            chain_hits            = _chain_hits,
        )

    def decode_output(self, hv: np.ndarray, top_k: int = 1) -> list:
        return self.cb.decode(hv, top_k=top_k)

    def store_spiral(
        self,
        item_hv: np.ndarray,
        address: Optional[Tuple[int, ...]] = None,
    ) -> Tuple[int, ...]:
        if self.spiral_mem is None:
            raise RuntimeError("SpiralPointerMemory not available")
        if address is None:
            D = self.spiral_mem.D
            n = self._spiral_addr_counter
            self._spiral_addr_counter += 1
            if self.axis_offset > 0:
                k0 = self.axis_offset
                k1 = n % D
                k2 = (n // D) % D
                k3 = (n // (D * D)) % D
            else:
                k0 = n % D
                k1 = (n // D) % D
                k2 = (n // (D * D)) % D
                k3 = (n // (D * D * D)) % D
            address = (k0, k1, k2, k3)
        self.spiral_mem.store(item_hv, address)
        return address

    def retrieve_spiral(self, address: Tuple[int, ...]) -> np.ndarray:
        if self.spiral_mem is None:
            raise RuntimeError("SpiralPointerMemory not available")
        return self.spiral_mem.retrieve(address)

    def reset_level(self):
        self.z_signal.reset()
        self._prev_fwd_hv     = None
        self._prev_bwd_hv     = None
        self._prev_present_hv = None
        self._prev_traj_slope = 0.0
        self._last_S          = 0.0
        self._last_Z          = 0.0
        self._last_consistency = 0.5
        self._last_action_id   = None
        self._last_action_tokens = []

    def _build_seeds(self, present_hv, action_hvs, action_ids):
        n   = len(action_ids)
        rng = np.random.default_rng()
        fwd = np.empty((n * self.H, self.W), dtype=np.uint64)
        bwd = np.empty((n * self.H, self.W), dtype=np.uint64)
        full_mask = np.bitwise_xor(self.manifold.compass_mask, self.manifold.proto_mask)
        for i, aid in enumerate(action_ids):
            eff   = action_hvs[aid]
            fwd_s = np.bitwise_xor(present_hv, eff)
            bwd_s = np.bitwise_xor(present_hv, np.bitwise_xor(eff, full_mask))
            fb    = np.tile(fwd_s, (self.H, 1))
            bb    = np.tile(bwd_s, (self.H, 1))
            noise_m = rng.random((self.H, self.W)) < self.noise_rate
            noise_v = rng.integers(0, np.iinfo(np.uint64).max, (self.H, self.W), dtype=np.uint64)
            fwd[i*self.H:(i+1)*self.H] = np.where(noise_m, np.bitwise_xor(fb, noise_v), fb).astype(np.uint64)
            bwd[i*self.H:(i+1)*self.H] = np.where(noise_m, np.bitwise_xor(bb, noise_v), bb).astype(np.uint64)
        return fwd, bwd

    def _propagate(self, tensor, direction, S):
        N       = tensor.shape[0]
        active  = np.ones(N, dtype=bool)
        iters   = np.full(N, self.max_iters, dtype=np.float32)
        prev_pc = np.zeros(N, dtype=np.float32)
        sim_hist= []

        rule_tile = np.tile(self._rule_bundle, (N, 1)) if self._rule_weight > 0 else None
        goal_tile = np.tile(self.goal_hv, (N, 1)) if self.goal_hv is not None else None

        chain_hits = 0
        if direction == 1 and self._last_action_id is not None and self._last_action_tokens:
            action_a_hv = self.cb.encode(self._last_action_tokens)
            candidate_ids = list(set(self.chain_memory.rel_mem._action_ids))
            if candidate_ids:
                cand_token_map = {aid: [aid] for aid in candidate_ids}
                prior_hv, chain_score, _ = self.chain_memory.query(
                    action_a_hv          = action_a_hv,
                    candidate_action_ids = candidate_ids,
                    action_token_map     = cand_token_map,
                    S                    = S,
                )
                if prior_hv is not None and chain_score > 0.55:
                    chain_blend = min(0.3, self.chain_memory._rule_weight /
                                      (self.chain_memory._rule_weight + 10.0))
                    if chain_blend > 0.0:
                        cp_tile = np.tile(prior_hv, (N, 1))
                        bm = np.random.random((N, self.W)) < chain_blend
                        tensor = np.where(bm, np.bitwise_xor(tensor, cp_tile), tensor).astype(np.uint64)
                        chain_hits = 1

        for t in range(self.max_iters):
            if not active.any():
                break
            mask = self.manifold.step_mask(S if direction == 1 else -S)
            tensor[active] = np.bitwise_xor(tensor[active], mask)

            if rule_tile is not None and self._rule_weight > 0:
                blend = min(0.5, self._rule_weight / (self._rule_weight + 20.0))
                bm    = np.random.random((active.sum(), self.W)) < blend
                idx   = np.where(active)[0]
                tensor[idx] = np.where(
                    bm,
                    np.bitwise_xor(tensor[idx], rule_tile[idx]),
                    tensor[idx]
                ).astype(np.uint64)

            if goal_tile is not None:
                sim_t = _cosine_batch(tensor, goal_tile)
                sim_hist.append(sim_t)

            if t % 4 == 3:
                tensor = self._parity_correct(tensor, active)

            cur_pc = _popcount_2d(tensor[active]).astype(np.float32)
            delta  = np.abs(cur_pc - prev_pc[active])
            done   = delta < 1.0
            idxs   = np.where(active)[0]
            iters[idxs[done]] = t
            active[idxs[done]] = False
            prev_pc[active]    = cur_pc[~done]

        sim_array = np.stack(sim_hist, axis=1) if sim_hist else np.zeros((N, 1))
        return tensor, iters, sim_array, chain_hits

    def _score_joint(self, fwd, bwd, fi, bi, sim_hist):
        n, H, W = fwd.shape
        fwd_flat = fwd.reshape(n * H, W)
        bwd_flat = bwd.reshape(n * H, W)

        consistency = _cosine_batch(fwd_flat, bwd_flat).reshape(n, H)

        if self.goal_hv is not None:
            g_tile   = np.tile(self.goal_hv, (n * H, 1))
            goal_sim = _cosine_batch(fwd_flat, g_tile).reshape(n, H)
        else:
            goal_sim = np.full((n, H), 0.5, dtype=np.float32)

        if sim_hist.ndim == 3 and sim_hist.shape[2] >= 2:
            slope = (sim_hist[:, :, -1] - sim_hist[:, :, 0])
            slope = np.clip(0.5 + slope, 0.0, 1.0).astype(np.float32)
        else:
            slope = np.full((n, H), 0.5, dtype=np.float32)

        entropy    = _entropy_batch(fwd_flat).reshape(n, H)
        confidence = np.float32(1.0) - entropy
        conv       = np.float32(1.0) - (fi + bi) / (np.float32(2.0) * self.max_iters)
        rule_conf  = np.float32(self._rule_weight / (self._rule_weight + 10.0))

        joint = (
              np.float32(self.W_CONSISTENCY) * consistency
            + np.float32(self.W_GOAL_SIM)    * goal_sim
            + np.float32(self.W_TRAJ_SLOPE)  * slope
            + np.float32(self.W_ENTROPY)      * confidence
            + np.float32(self.W_RESONANCE)    * rule_conf
        )

        return joint, dict(
            consistency = consistency,
            goal_sim    = goal_sim,
            traj_slope  = slope - 0.5,
            entropy     = entropy,
            conv_speed  = conv,
        )

    def _bundle_rule(self, rule_hv, weight):
        alpha = weight / (self._rule_weight + weight + EPS)
        flip  = np.random.random(self.W) < alpha
        self._rule_bundle = np.where(
            flip, np.bitwise_xor(self._rule_bundle, rule_hv), self._rule_bundle
        ).astype(np.uint64)
        self._rule_weight += weight

    def _update_goal(self, fwd_hv, weight):
        if self.goal_hv is None:
            self.goal_hv           = fwd_hv.copy()
            self._goal_weight      = weight
            self._base_goal_weight = weight
        else:
            dynamic_weight   = self._base_goal_weight * (1.0 + self._last_Z)
            effective_weight = max(self._goal_weight, dynamic_weight)
            alpha = weight / (effective_weight + weight + EPS)
            flip  = np.random.random(self.W) < alpha
            self.goal_hv = np.where(
                flip, np.bitwise_xor(self.goal_hv, fwd_hv), self.goal_hv
            ).astype(np.uint64)
            self._goal_weight = effective_weight + weight

    def _parity_correct(self, tensor, active):
        if not active.any() or not self._corr_buf:
            return tensor
        sub     = tensor[active]
        bits    = np.unpackbits(sub.view(np.uint8).reshape(sub.shape[0], -1), axis=-1)
        balance = bits.mean(axis=-1)
        bad     = (balance < 0.3) | (balance > 0.7)
        if bad.any():
            median = np.median(
                np.stack(self._corr_buf).view(np.int64), axis=0
            ).astype(np.uint64)
            sub[bad] = median
            tensor[active] = sub
        return tensor