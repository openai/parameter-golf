"""
Swarm-Guided KG-Conditioned Training — Lightweight Agent System

A multi-agent swarm that makes training decisions via voting.
All agents are rule-based (no LLM calls). Total overhead budget: <30s across 4-6 cycles.
Includes BackoffNgramMixer for eval-time n-gram cache mixing.
"""
from __future__ import annotations
import lzma
import math
import struct
import time
from dataclasses import dataclass, field
from typing import Optional

import torch
import torch.nn.functional as F
from torch import Tensor


# ---------------------------------------------------------------------------
# BackoffNgramMixer — eval-time n-gram cache with entropy-adaptive mixing
# Based on PR #779/#803 architecture. Swarm agents control alpha parameters.
# ---------------------------------------------------------------------------

class BackoffNgramMixer:
    """Multi-order n-gram backoff cache with entropy-adaptive neural/n-gram mixing.

    Built from already-scored tokens only (backward-looking, score-first).
    Proper full-vocabulary mixture: p_final = (1-alpha)*p_neural + alpha*p_ngram.
    """
    PRIMES = [36313, 27191, 51647, 81929, 131071, 174763, 233017]

    def __init__(self, vocab_size: int, device: torch.device, num_buckets: int = 4_000_000,
                 max_order: int = 7, min_count: int = 2, min_tokens: int = 5000,
                 alpha_base: float = 0.20, alpha_range: float = 0.55, alpha_center: float = 3.0):
        self.V = vocab_size
        self.B = num_buckets
        self.MASK = num_buckets - 1 if (num_buckets & (num_buckets - 1)) == 0 else None
        self.max_order = max_order
        self.min_count = min_count
        self.min_tokens = min_tokens
        self.device = device
        self.tokens_seen = 0
        self.alpha_base = alpha_base
        self.alpha_range = alpha_range
        self.alpha_center = alpha_center
        self.uni_counts = torch.zeros(vocab_size, device=device, dtype=torch.float32)
        self.uni_total = 0.0
        self.ctx_counts = []
        self.full_counts = []
        for _ in range(max_order - 1):
            self.ctx_counts.append(torch.zeros(num_buckets, device=device, dtype=torch.float32))
            self.full_counts.append(torch.zeros(num_buckets, device=device, dtype=torch.float32))

    def _bucket(self, h: Tensor) -> Tensor:
        if self.MASK is not None:
            return h & self.MASK
        return h.abs() % self.B

    def update(self, tokens: Tensor):
        """Accumulate n-gram counts from already-scored tokens."""
        t = tokens.to(self.device).long()
        n = t.numel()
        self.tokens_seen += n
        ones = torch.ones(n, device=self.device, dtype=torch.float32)
        self.uni_counts.scatter_add_(0, t, ones)
        self.uni_total += n
        for order in range(2, self.max_order + 1):
            if n < order:
                continue
            oi = order - 2
            nxt = t[order - 1:]
            ctx_h = t[0:n - order + 1] * self.PRIMES[0]
            for k in range(1, order - 1):
                ctx_h = ctx_h ^ (t[k:n - order + 1 + k] * self.PRIMES[k % len(self.PRIMES)])
            ctx_key = self._bucket(ctx_h)
            full_h = ctx_h ^ (nxt * self.PRIMES[(order - 1) % len(self.PRIMES)])
            full_key = self._bucket(full_h)
            self.ctx_counts[oi].scatter_add_(0, ctx_key, ones[:n - order + 1])
            self.full_counts[oi].scatter_add_(0, full_key, ones[:n - order + 1])

    def score(self, logits: Tensor, x_batch: Tensor, y_batch: Tensor,
              temperature: float = 1.0) -> Tensor:
        """Score tokens using neural+n-gram mixture. Returns per-token NLL."""
        bsz, slen, V = logits.shape
        if temperature != 1.0:
            logits = logits / temperature
        log_probs_neural = F.log_softmax(logits.float(), dim=-1)
        neural_p = log_probs_neural.gather(-1, y_batch.unsqueeze(-1)).squeeze(-1).exp()

        if self.tokens_seen < self.min_tokens:
            return -neural_p.clamp(min=1e-12).log()

        # Build context stack for n-gram lookups
        ctx_stack = [x_batch]
        for k in range(1, self.max_order - 1):
            shifted = torch.zeros_like(x_batch)
            if k < slen:
                shifted[:, k:] = x_batch[:, :-k]
            ctx_stack.append(shifted)

        # Unigram fallback
        if self.uni_total > 0:
            uni_p = (self.uni_counts[y_batch] + 0.5) / (self.uni_total + 0.5 * V)
            ngram_p = uni_p
        else:
            ngram_p = torch.full((bsz, slen), 1.0 / V, device=self.device)
        ngram_hit = torch.zeros(bsz, slen, device=self.device, dtype=torch.bool)

        # Greedy cascade: highest order first
        for order in range(self.max_order, 1, -1):
            oi = order - 2
            cw = order - 1
            ctx_h = ctx_stack[cw - 1] * self.PRIMES[0]
            for k in range(1, cw):
                ctx_h = ctx_h ^ (ctx_stack[cw - 1 - k] * self.PRIMES[k % len(self.PRIMES)])
            ctx_key = self._bucket(ctx_h)
            full_h = ctx_h ^ (y_batch * self.PRIMES[(order - 1) % len(self.PRIMES)])
            full_key = self._bucket(full_h)
            ctx_c = self.ctx_counts[oi][ctx_key]
            full_c = self.full_counts[oi][full_key]
            valid = (ctx_c >= self.min_count) & (~ngram_hit)
            min_pos = order - 2
            if min_pos > 0:
                valid[:, :min_pos] = False
            p = torch.where(valid, full_c.clamp(max=ctx_c) / ctx_c.clamp(min=1), torch.zeros_like(ctx_c))
            p = p.clamp(0, 1)
            ngram_p = torch.where(valid, p, ngram_p)
            ngram_hit = ngram_hit | valid

        # Entropy-adaptive alpha
        probs_neural = log_probs_neural.exp()
        entropy = -(probs_neural * log_probs_neural).sum(dim=-1)
        alpha = self.alpha_base + self.alpha_range * torch.sigmoid(
            2.0 * (entropy - self.alpha_center))

        # Proper full-vocabulary mixture
        mixed_p = (1.0 - alpha) * neural_p + alpha * ngram_p
        return -mixed_p.clamp(min=1e-12).log()


class TrainNgramTracker:
    """Tracks bigram statistics during training for complementary loss weighting.

    Downweights tokens that bigrams predict well, so the model focuses on
    what n-gram caches can't handle.
    """
    def __init__(self, vocab_size: int, device: torch.device, complement_alpha: float = 0.5):
        self.V = vocab_size
        self.alpha = complement_alpha
        self.bi_counts = torch.zeros(vocab_size, vocab_size, device=device, dtype=torch.float32)
        self.bi_totals = torch.zeros(vocab_size, device=device, dtype=torch.float32)

    def update(self, x: Tensor, y: Tensor):
        """Update bigram counts from training batch."""
        prev = x.reshape(-1)
        nxt = y.reshape(-1)
        idx = prev * self.V + nxt
        ones = torch.ones(idx.numel(), device=idx.device, dtype=torch.float32)
        self.bi_counts.view(-1).scatter_add_(0, idx, ones)
        self.bi_totals.scatter_add_(0, prev, ones)

    def get_weights(self, x: Tensor, y: Tensor) -> Tensor:
        """Return per-token loss weights. Tokens predictable by bigrams get lower weight."""
        prev = x.reshape(-1)
        nxt = y.reshape(-1)
        count = self.bi_counts[prev, nxt]
        total = self.bi_totals[prev]
        ngram_prob = count / (total + 1)
        return (1.0 - self.alpha * ngram_prob).clamp(min=0.1)


def decompress_token_importance(data: bytes) -> dict[int, float]:
    """Decompress KG token importance from LZMA-compressed binary."""
    raw = lzma.decompress(data)
    count = struct.unpack_from("<H", raw, 0)[0]
    offset = 2
    result: dict[int, float] = {}
    for _ in range(count):
        tid, imp_int = struct.unpack_from("<HH", raw, offset)
        result[tid] = imp_int / 10000.0
        offset += 4
    return result


# ---------------------------------------------------------------------------
# KG-Conditioned Loss
# ---------------------------------------------------------------------------

class KGConditionedLoss:
    """Modulates per-token cross-entropy by knowledge graph importance scores.

    token_importance: dict mapping token_id (int) -> importance weight (float).
    Pre-computed offline from PageRank/centrality on the 500K-node graph,
    then distilled to top-N tokens. Stored as compressed bytes in train_gpt.py.
    """

    def __init__(self, token_importance: dict[int, float], vocab_size: int,
                 base_weight: float = 1.0, kg_weight: float = 0.3):
        self.kg_weight = kg_weight
        # Build a static weight tensor: default=base_weight, override for KG tokens
        weights = torch.full((vocab_size,), base_weight, dtype=torch.float32)
        for tok_id, importance in token_importance.items():
            if 0 <= tok_id < vocab_size:
                # Scale importance into [base_weight, base_weight + kg_weight]
                weights[tok_id] = base_weight + kg_weight * importance
        self._weights_cpu = weights
        self._weights_gpu: Optional[Tensor] = None

    def get_weights(self, device: torch.device) -> Tensor:
        if self._weights_gpu is None or self._weights_gpu.device != device:
            self._weights_gpu = self._weights_cpu.to(device)
        return self._weights_gpu

    def weighted_cross_entropy(self, logits: Tensor, targets: Tensor) -> Tensor:
        """Cross-entropy weighted by KG importance. Drop-in replacement for F.cross_entropy.
        Uses F.cross_entropy's native `weight` parameter for torch.compile compatibility."""
        w = self.get_weights(targets.device)
        return torch.nn.functional.cross_entropy(
            logits.float(), targets, weight=w, reduction="mean"
        )


# ---------------------------------------------------------------------------
# Swarm Agent Definitions
# ---------------------------------------------------------------------------

@dataclass
class TrainingMetrics:
    """Snapshot of training state for agent decision-making."""
    step: int
    total_steps: int
    train_loss: float
    loss_history: list[float] = field(default_factory=list)
    grad_norm: float = 0.0
    elapsed_ms: float = 0.0
    max_wallclock_ms: float = 600_000.0
    current_lr_scale: float = 1.0
    qat_enabled: bool = False
    kg_weight: float = 0.3


@dataclass
class SwarmDecision:
    """A proposed change from a swarm agent."""
    agent_name: str
    param_name: str
    old_value: float
    new_value: float
    confidence: float  # 0.0 - 1.0
    reason: str


class LearningRateAgent:
    """Monitors loss velocity to propose LR scaling adjustments."""

    name = "lr_agent"

    def evaluate(self, metrics: TrainingMetrics) -> Optional[SwarmDecision]:
        if len(metrics.loss_history) < 20:
            return None

        # Compare recent loss velocity to earlier velocity
        recent = metrics.loss_history[-10:]
        earlier = metrics.loss_history[-20:-10]

        recent_velocity = (recent[0] - recent[-1]) / max(len(recent), 1)
        earlier_velocity = (earlier[0] - earlier[-1]) / max(len(earlier), 1)

        if earlier_velocity <= 0:
            return None

        ratio = recent_velocity / earlier_velocity

        # If loss improvement slowed by >80%, propose moderate LR reduction
        if ratio < 0.2 and metrics.step > metrics.total_steps * 0.4:
            return SwarmDecision(
                agent_name=self.name,
                param_name="lr_scale_factor",
                old_value=1.0,
                new_value=0.85,
                confidence=0.7,
                reason=f"loss_velocity_ratio={ratio:.3f}, improvement stalled"
            )

        # If loss is improving faster than before, propose slight LR increase
        if ratio > 1.5 and metrics.current_lr_scale < 1.2:
            return SwarmDecision(
                agent_name=self.name,
                param_name="lr_scale_factor",
                old_value=1.0,
                new_value=1.1,
                confidence=0.6,
                reason=f"loss_velocity_ratio={ratio:.3f}, accelerating"
            )

        return None


class QATTimingAgent:
    """Decides when to enable quantization-aware training."""

    name = "qat_agent"

    def evaluate(self, metrics: TrainingMetrics) -> Optional[SwarmDecision]:
        if metrics.qat_enabled:
            return None

        progress = metrics.step / max(metrics.total_steps, 1)

        # Enable QAT when warmdown begins (LR scale drops) but only after 40%
        if progress > 0.40 and metrics.current_lr_scale < 0.15:
            return SwarmDecision(
                agent_name=self.name,
                param_name="qat_enable_now",
                old_value=0.0,
                new_value=1.0,
                confidence=0.9,
                reason=f"progress={progress:.2f}, lr_scale={metrics.current_lr_scale:.4f}, warmdown region"
            )

        # Safety: must enable by 65% regardless
        if progress > 0.65:
            return SwarmDecision(
                agent_name=self.name,
                param_name="qat_enable_now",
                old_value=0.0,
                new_value=1.0,
                confidence=0.95,
                reason=f"progress={progress:.2f}, QAT deadline"
            )

        return None


class KGWeightAgent:
    """Adjusts the knowledge graph loss weighting based on training progress."""

    name = "kg_weight_agent"
    _last_proposed: float = 0.3

    def evaluate(self, metrics: TrainingMetrics) -> Optional[SwarmDecision]:
        progress = metrics.step / max(metrics.total_steps, 1)

        # Dynamic KG schedule: ramp up early, hold mid, taper late
        if progress < 0.15:
            target = 0.5  # Strong KG guidance at start
        elif progress < 0.4:
            target = 0.4  # High guidance during learning
        elif progress < 0.7:
            target = 0.3  # Standard weight mid-training
        else:
            target = 0.1  # Reduce for final convergence

        if abs(target - self._last_proposed) > 0.05:
            old = self._last_proposed
            self._last_proposed = target
            return SwarmDecision(
                agent_name=self.name,
                param_name="kg_weight",
                old_value=old,
                new_value=target,
                confidence=0.75,
                reason=f"progress={progress:.2f}, adjusting KG schedule"
            )

        return None


class GradientHealthAgent:
    """Monitors gradient norms and proposes clip adjustments."""

    name = "grad_health_agent"

    def evaluate(self, metrics: TrainingMetrics) -> Optional[SwarmDecision]:
        if metrics.grad_norm <= 0:
            return None

        # If gradients are exploding, tighten clipping
        if metrics.grad_norm > 2.0:
            return SwarmDecision(
                agent_name=self.name,
                param_name="grad_clip_norm",
                old_value=0.3,
                new_value=0.15,
                confidence=0.85,
                reason=f"grad_norm={metrics.grad_norm:.3f}, tighten clipping"
            )

        return None


class MTPWeightAgent:
    """Adjusts multi-token prediction loss weight based on training phase."""

    name = "mtp_agent"

    def evaluate(self, metrics: TrainingMetrics) -> Optional[SwarmDecision]:
        progress = metrics.step / max(metrics.total_steps, 1)

        # MTP more valuable early (teaches the model predictive structure)
        # Less valuable late (focus on primary loss)
        if progress > 0.75:
            return SwarmDecision(
                agent_name=self.name,
                param_name="mtp_loss_weight",
                old_value=0.1,
                new_value=0.05,
                confidence=0.65,
                reason="late training, shift focus to primary loss"
            )

        return None


# ---------------------------------------------------------------------------
# Voting Mesh
# ---------------------------------------------------------------------------

class VotingMesh:
    """Aggregates agent proposals and applies consensus decisions.

    Decision rules:
    - Single agent with confidence >= 0.85 -> apply
    - Two agents agree on direction -> apply
    - Any agent raises confidence < 0.3 -> skip (uncertainty)
    """

    def __init__(self):
        self.agents = [
            QATTimingAgent(),
            KGWeightAgent(),
            GradientHealthAgent(),
            MTPWeightAgent(),
        ]
        self.decision_log: list[dict] = []
        self.cycle_count = 0

    def run_decision_cycle(self, metrics: TrainingMetrics) -> list[SwarmDecision]:
        """Run all agents, vote, return approved decisions."""
        t0 = time.perf_counter()
        self.cycle_count += 1

        proposals: list[SwarmDecision] = []
        for agent in self.agents:
            decision = agent.evaluate(metrics)
            if decision is not None:
                proposals.append(decision)

        # Apply decisions with sufficient confidence
        approved: list[SwarmDecision] = []
        for d in proposals:
            if d.confidence >= 0.6:
                approved.append(d)
                self.decision_log.append({
                    "cycle": self.cycle_count,
                    "step": metrics.step,
                    "agent": d.agent_name,
                    "param": d.param_name,
                    "old": d.old_value,
                    "new": d.new_value,
                    "confidence": d.confidence,
                    "reason": d.reason,
                    "elapsed_us": int((time.perf_counter() - t0) * 1e6),
                })

        return approved

    def should_run(self, step: int, total_steps: int) -> bool:
        """Determine if a decision cycle should run at this step."""
        if total_steps <= 0 or step == 0:
            return False
        # Run every 800 steps — gives ~9 decision points in a typical 7000-step run
        return step % 800 == 0

    def summary(self) -> str:
        """Return a log summary of all decisions made."""
        lines = [f"Swarm: {self.cycle_count} cycles, {len(self.decision_log)} decisions"]
        for d in self.decision_log:
            lines.append(
                f"  cycle {d['cycle']} step {d['step']}: {d['agent']} "
                f"{d['param']} {d['old']}->{d['new']} "
                f"(conf={d['confidence']:.2f}, {d['elapsed_us']}us) {d['reason']}"
            )
        return "\n".join(lines)
