"""Embedding-space Forest-of-Thought training head.

This module adapts Forest-of-Thought (FoT) from textual test-time prompting to
ToricGT's BPB-facing hidden states.  It is intentionally training-only: the head
adds auxiliary gradients and diagnostics, but the OAI Parameter-Golf artifact can
remain the compact baseline model unless an export path explicitly keeps it.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import Tensor, nn


@dataclass(frozen=True)
class EmbeddingFoTConfig:
    dim: int
    num_trees: int = 4
    max_depth: int = 5
    branching: int = 4
    topk_trees: int = 2
    hidden_dim: int = 192
    max_positions: int = 192
    consensus_buckets: int = 64
    correction_scale: float = 0.08
    ucb_exploration: float = 1.25
    temperature: float = 0.7
    sparse_weight: float = 1.0
    ucb_weight: float = 0.5
    correction_weight: float = 0.5
    consensus_weight: float = 0.75
    tb_weight: float = 1.0
    subtb_weight: float = 0.0
    complexity_weight: float = 0.05
    reward_advanced_bonus: float = 0.0
    reward_mode: str = "bpb_delta"
    bpb_delta_weight: float = 1.0
    reward_graph_weight: float = 0.10
    reward_consensus_weight: float = 0.20
    reward_complexity_weight: float = 0.02
    reward_floor: float = 1.0e-4


class EmbeddingFoTOutput(dict):
    """Dictionary output with a typed name for readability."""


class EmbeddingForestOfThoughtHead(nn.Module):
    """Differentiable FoT head over OAI baseline hidden states.

    The head views selected hidden positions as nodes in several interleaved
    reasoning trees.  It trains sparse activation, UCB-like expansion,
    self-correction, consensus, and trajectory-balance objectives using the
    local negative log likelihood as a dense BPB-native reward.
    """

    def __init__(self, config: EmbeddingFoTConfig) -> None:
        super().__init__()
        if config.dim <= 0:
            raise ValueError("EmbeddingFoTConfig.dim must be positive")
        if config.num_trees < 1:
            raise ValueError("OAI_FOT_NUM_TREES must be at least 1")
        if config.branching < 2:
            raise ValueError("OAI_FOT_BRANCHING must be at least 2")
        if config.max_depth < 1:
            raise ValueError("OAI_FOT_MAX_DEPTH must be at least 1")
        if config.consensus_buckets < 2:
            raise ValueError("OAI_FOT_CONSENSUS_BUCKETS must be at least 2")
        self.config = config
        d = int(config.dim)
        h = int(config.hidden_dim)
        self.node_norm = nn.LayerNorm(d)
        self.activation_head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))
        self.value_head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))
        self.forward_policy = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, config.branching))
        self.backward_policy = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, config.branching))
        self.correction_head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, d))
        self.consensus_head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, config.consensus_buckets))
        self.flow_head = nn.Sequential(nn.Linear(d, h), nn.GELU(), nn.Linear(h, 1))
        self.log_z = nn.Parameter(torch.zeros((), dtype=torch.float32))

    def _selected_indices(self, length: int, device: torch.device) -> Tensor:
        cfg = self.config
        tree_budget = int(cfg.num_trees) * int(cfg.max_depth) * max(1, int(cfg.branching))
        max_positions = max(2, min(int(cfg.max_positions), tree_budget, int(length)))
        if max_positions >= length:
            return torch.arange(length, device=device, dtype=torch.long)
        return torch.linspace(0, length - 1, max_positions, device=device).round().long().unique(sorted=True)

    def _tree_reduce_mean(self, values: Tensor, tree_ids: Tensor, num_trees: int) -> Tensor:
        # values: [B, S] or [B, S, C]
        if values.ndim == 2:
            out = values.new_zeros(values.shape[0], num_trees)
            counts = values.new_zeros(num_trees).scatter_add_(0, tree_ids, torch.ones_like(tree_ids, dtype=values.dtype))
            out.scatter_add_(1, tree_ids.unsqueeze(0).expand(values.shape[0], -1), values)
            return out / counts.clamp_min(1.0).unsqueeze(0)
        out = values.new_zeros(values.shape[0], num_trees, values.shape[-1])
        counts = values.new_zeros(num_trees).scatter_add_(0, tree_ids, torch.ones_like(tree_ids, dtype=values.dtype))
        expand_ids = tree_ids.view(1, -1, 1).expand(values.shape[0], -1, values.shape[-1])
        out.scatter_add_(1, expand_ids, values)
        return out / counts.clamp_min(1.0).view(1, -1, 1)

    def _tree_last_indices(self, tree_ids: Tensor, num_trees: int) -> Tensor:
        last: list[int] = []
        for tree_id in range(num_trees):
            where = torch.nonzero(tree_ids == tree_id, as_tuple=False).flatten()
            last.append(int(where[-1].item()) if where.numel() else 0)
        return torch.tensor(last, device=tree_ids.device, dtype=torch.long)

    def forward(
        self,
        hidden: Tensor,
        target_ids: Tensor,
        per_token_nll: Tensor,
        *,
        advanced_signal: Tensor | None = None,
        target_byte_lengths: Tensor | None = None,
        lm_head_weight: Tensor | None = None,
        logit_softcap: float = 30.0,
        temperature_multiplier: float = 1.0,
        ucb_multiplier: float = 1.0,
        sparse_multiplier: float = 1.0,
    ) -> EmbeddingFoTOutput:
        cfg = self.config
        zero = hidden.new_zeros(())
        if hidden.ndim != 3 or hidden.shape[1] < 2:
            return EmbeddingFoTOutput({"oai_fot_loss": zero, "oai_fot_enabled": zero})

        idx = self._selected_indices(int(hidden.shape[1]), hidden.device)
        if idx.numel() < 2:
            return EmbeddingFoTOutput({"oai_fot_loss": zero, "oai_fot_enabled": zero})

        nodes = self.node_norm(hidden[:, idx, :].float())
        targets = target_ids[:, idx].long()
        nll = per_token_nll[:, idx].detach().float()
        if target_byte_lengths is not None:
            byte_lengths = target_byte_lengths[:, idx].detach().float().clamp_min(1.0)
        else:
            byte_lengths = torch.ones_like(nll)
        batch, n_nodes, _ = nodes.shape
        num_trees = max(1, min(int(cfg.num_trees), int(n_nodes)))
        tree_ids = torch.remainder(torch.arange(n_nodes, device=hidden.device), num_trees).long()
        depth_ids = torch.div(torch.arange(n_nodes, device=hidden.device), num_trees, rounding_mode="floor")
        temperature = max(float(cfg.temperature) * max(float(temperature_multiplier), 1.0e-4), 1.0e-4)
        ucb_exploration = max(float(cfg.ucb_exploration) * max(float(ucb_multiplier), 0.0), 0.0)

        activation_logits = self.activation_head(nodes).squeeze(-1).clamp(-30.0, 30.0)
        values = self.value_head(nodes).squeeze(-1).float()
        flow_values = self.flow_head(nodes).squeeze(-1).float()

        with torch.no_grad():
            norm_nodes = F.normalize(nodes.detach(), dim=-1)
            first_per_tree = torch.tensor(
                [int(torch.nonzero(tree_ids == t, as_tuple=False).flatten()[0].item()) for t in range(num_trees)],
                device=hidden.device,
                dtype=torch.long,
            )
            roots = norm_nodes[:, first_per_tree, :]
            root_for_node = roots[:, tree_ids, :]
            novelty = (1.0 - (norm_nodes * root_for_node).sum(dim=-1)).clamp(0.0, 2.0)
            depth_bonus = depth_ids.to(nodes.dtype).view(1, -1) / max(float(depth_ids.max().item() + 1), 1.0)
            byte_nll = nll / byte_lengths
            target_score = -byte_nll + float(cfg.reward_graph_weight) * novelty + 0.05 * depth_bonus
            tree_target_score = self._tree_reduce_mean(target_score, tree_ids, num_trees)
            activation_target = torch.softmax(tree_target_score / temperature, dim=-1)

        tree_activation_logits = self._tree_reduce_mean(activation_logits, tree_ids, num_trees)
        tree_log_probs = F.log_softmax(tree_activation_logits, dim=-1)
        sparse_loss = F.kl_div(tree_log_probs, activation_target, reduction="batchmean")
        activation_probs = torch.softmax(tree_activation_logits, dim=-1)
        activation_entropy = -(activation_probs * activation_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        activation_entropy_norm = activation_entropy / math.log(max(2, num_trees))
        topk = max(1, min(int(cfg.topk_trees), num_trees))
        active_mass = torch.topk(activation_probs, k=topk, dim=-1).values.sum(dim=-1).mean()
        active_tree_count = torch.exp(activation_entropy).detach()

        if n_nodes > 1:
            f_logits = self.forward_policy(nodes[:, :-1, :]).float().clamp(-30.0, 30.0)
            b_logits = self.backward_policy(nodes[:, 1:, :]).float().clamp(-30.0, 30.0)
            actions = torch.remainder(targets[:, 1:], int(cfg.branching))
            log_pf = F.log_softmax(f_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            log_pb = F.log_softmax(b_logits, dim=-1).gather(-1, actions.unsqueeze(-1)).squeeze(-1)
            visits = torch.arange(1, n_nodes, device=hidden.device, dtype=torch.float32).view(1, -1)
            parent_visits = (visits + num_trees).clamp_min(1.0)
            child_visits = (1.0 + torch.remainder(torch.arange(1, n_nodes, device=hidden.device), num_trees).float()).view(1, -1)
            ucb_target_value = values[:, 1:].detach() + ucb_exploration * torch.sqrt(
                torch.log(parent_visits + 1.0) / child_visits.clamp_min(1.0)
            )
            ucb_weights = torch.softmax(ucb_target_value / temperature, dim=-1)
            transition_nll = F.cross_entropy(
                f_logits.reshape(-1, int(cfg.branching)),
                actions.reshape(-1),
                reduction="none",
            ).reshape(batch, -1)
            ucb_loss = (transition_nll * ucb_weights).sum(dim=-1).mean()

            correction = self.correction_head(nodes[:, :-1, :]).float()
            desired_delta = (nodes[:, 1:, :] - nodes[:, :-1, :]).detach()
            correction_cosine = F.cosine_similarity(correction, desired_delta, dim=-1)
            correction_direction_loss = (1.0 - correction_cosine).mean()
            corrected_nodes = nodes[:, :-1, :] + float(cfg.correction_scale) * correction
            corrected_value = self.value_head(corrected_nodes).squeeze(-1).float()
            value_lift = corrected_value - values[:, :-1].detach()
            correction_lift_loss = F.relu(0.01 - value_lift).mean()
            correction_loss = correction_direction_loss + 0.25 * correction_lift_loss

            raw_byte_nll = (nll / byte_lengths).mean(dim=1)
            corrected_byte_nll = raw_byte_nll
            bpb_delta_reward = hidden.new_zeros(batch, dtype=torch.float32)
            corrected_ce = nll[:, :-1].detach()
            if lm_head_weight is not None and str(cfg.reward_mode).strip().lower() in {"bpb_delta", "ce_delta", "byte_delta"}:
                logits_proj = F.linear(corrected_nodes.reshape(-1, corrected_nodes.shape[-1]), lm_head_weight.float())
                softcap = max(float(logit_softcap), 1.0e-4)
                logits_proj = softcap * torch.tanh(logits_proj / softcap)
                corrected_ce = F.cross_entropy(
                    logits_proj.float(),
                    targets[:, :-1].reshape(-1),
                    reduction="none",
                ).view(batch, -1)
                selected_bytes = byte_lengths[:, :-1].clamp_min(1.0)
                raw_local_byte_nll = (nll[:, :-1].detach() / selected_bytes).mean(dim=1)
                corrected_byte_nll = (corrected_ce / selected_bytes).mean(dim=1)
                bpb_delta_reward = (raw_local_byte_nll - corrected_byte_nll).clamp(-5.0, 5.0)
                reward = torch.exp(
                    -corrected_byte_nll.detach()
                    + float(cfg.bpb_delta_weight) * bpb_delta_reward.detach().clamp(-2.0, 2.0)
                )
            else:
                reward = torch.exp(-raw_byte_nll).clamp_min(1e-8)
            if advanced_signal is not None:
                bonus = torch.as_tensor(advanced_signal, device=hidden.device, dtype=torch.float32)
                while bonus.ndim > 1:
                    bonus = bonus.mean(dim=-1)
                if bonus.ndim == 0:
                    bonus = bonus.expand_as(reward)
                reward = reward * torch.exp(float(cfg.reward_advanced_bonus) * bonus[: reward.shape[0]].detach().clamp(-5.0, 5.0))
            complexity = (n_nodes / max(float(cfg.max_positions), 1.0)) + active_tree_count.float() / max(float(num_trees), 1.0)
            reward = (
                reward
                * torch.exp(float(cfg.reward_consensus_weight) * active_mass.detach().clamp(0.0, 1.0))
                * torch.exp(-float(cfg.reward_complexity_weight) * complexity.detach())
            ).clamp_min(max(float(cfg.reward_floor), 1.0e-8))
            tb_residual = self.log_z.float() + (log_pf - log_pb).mean(dim=1) - reward.log()
            tb_loss = tb_residual.square().mean()
            if n_nodes > 3:
                prefix = torch.cumsum(log_pf - log_pb, dim=1)
                denom = torch.arange(1, nll.shape[1], device=hidden.device, dtype=torch.float32).view(1, -1)
                prefix_mean_byte_nll = torch.cumsum(nll[:, 1:] / byte_lengths[:, 1:].clamp_min(1.0), dim=1) / denom.clamp_min(1.0)
                prefix_reward = torch.exp(-prefix_mean_byte_nll).clamp_min(max(float(cfg.reward_floor), 1.0e-8))
                prefix_flow = flow_values[:, 1:]
                subtb_residual = prefix_flow + prefix - prefix_reward.log()
                subtb_loss = subtb_residual.square().mean()
            else:
                subtb_loss = zero.float()
        else:
            log_pf = log_pb = torch.empty(batch, 0, device=hidden.device)
            ucb_loss = zero.float()
            correction_loss = zero.float()
            correction_cosine = zero.float()
            value_lift = zero.float()
            tb_loss = zero.float()
            subtb_loss = zero.float()
            tb_residual = zero.float().expand(batch)
            reward = torch.exp(-(nll / byte_lengths).mean(dim=1)).clamp_min(max(float(cfg.reward_floor), 1.0e-8))
            complexity = zero.float()
            bpb_delta_reward = zero.float().expand(batch)
            corrected_ce = nll.detach()
            corrected_byte_nll = (nll / byte_lengths).mean(dim=1)

        leaf_idx = self._tree_last_indices(tree_ids, num_trees)
        leaf_nodes = nodes[:, leaf_idx, :]
        leaf_targets = torch.remainder(targets[:, leaf_idx], int(cfg.consensus_buckets))
        leaf_logits = self.consensus_head(leaf_nodes).float().clamp(-30.0, 30.0)
        tree_weights = torch.softmax(tree_activation_logits, dim=-1).unsqueeze(-1)
        consensus_logits = (tree_weights * leaf_logits).sum(dim=1)
        final_target = torch.remainder(target_ids[:, idx[-1]], int(cfg.consensus_buckets))
        consensus_loss = F.cross_entropy(consensus_logits, final_target)
        consensus_probs = torch.softmax(consensus_logits, dim=-1)
        consensus_top2 = torch.topk(consensus_logits, k=min(2, int(cfg.consensus_buckets)), dim=-1).values
        consensus_margin = (
            consensus_top2[:, 0] - consensus_top2[:, -1] if consensus_top2.shape[-1] > 1 else consensus_top2[:, 0]
        ).mean()
        consensus_entropy = -(consensus_probs * consensus_probs.clamp_min(1e-8).log()).sum(dim=-1).mean()
        leaf_pred = leaf_logits.argmax(dim=-1)
        tree_agreement = (leaf_pred == final_target.view(-1, 1)).float().mean()
        leaf_supervision_loss = F.cross_entropy(
            leaf_logits.reshape(-1, int(cfg.consensus_buckets)),
            leaf_targets.reshape(-1),
        )
        consensus_loss = 0.75 * consensus_loss + 0.25 * leaf_supervision_loss

        tree_repr = self._tree_reduce_mean(F.normalize(nodes, dim=-1), tree_ids, num_trees)
        if num_trees > 1:
            sim = torch.matmul(tree_repr, tree_repr.transpose(1, 2))
            mask = ~torch.eye(num_trees, device=hidden.device, dtype=torch.bool).unsqueeze(0)
            tree_diversity = (1.0 - sim.masked_select(mask).view(batch, -1).mean(dim=-1)).mean().clamp(0.0, 2.0)
        else:
            tree_diversity = zero.float()
        complexity_loss = (
            F.relu(active_tree_count.float() / max(float(num_trees), 1.0) - 0.85).square()
            + F.relu(0.10 - tree_diversity).square()
        )

        total = (
            float(cfg.sparse_weight) * max(float(sparse_multiplier), 0.0) * sparse_loss
            + float(cfg.ucb_weight) * ucb_loss
            + float(cfg.correction_weight) * correction_loss
            + float(cfg.consensus_weight) * consensus_loss
            + float(cfg.tb_weight) * tb_loss
            + float(cfg.subtb_weight) * subtb_loss
            + float(cfg.complexity_weight) * complexity_loss
        )
        total = torch.nan_to_num(total, nan=0.0, posinf=1.0e4, neginf=0.0)

        return EmbeddingFoTOutput(
            {
                "oai_fot_loss": total,
                "oai_fot_sparse_activation_loss": sparse_loss.detach(),
                "oai_fot_ucb_loss": ucb_loss.detach(),
                "oai_fot_self_correction_loss": correction_loss.detach(),
                "oai_fot_consensus_loss": consensus_loss.detach(),
                "oai_fot_tb_loss": tb_loss.detach(),
                "oai_fot_subtb_loss": subtb_loss.detach(),
                "oai_fot_complexity_loss": complexity_loss.detach(),
                "oai_fot_activation_entropy": activation_entropy_norm.detach(),
                "oai_fot_active_tree_count": active_tree_count.detach(),
                "oai_fot_active_mass_topk": active_mass.detach(),
                "oai_fot_tree_diversity": tree_diversity.detach(),
                "oai_fot_value_mean": values.detach().mean(),
                "oai_fot_reward_mean": reward.detach().mean(),
                "oai_fot_reward_bpb_delta": bpb_delta_reward.detach().mean(),
                "oai_fot_reward_raw_byte_nll": raw_byte_nll.detach().mean() if "raw_byte_nll" in locals() else zero.detach(),
                "oai_fot_reward_corrected_byte_nll": (
                    corrected_byte_nll.detach().mean() if torch.is_tensor(corrected_byte_nll) else zero.detach()
                ),
                "oai_fot_corrected_ce": corrected_ce.detach().mean() if torch.is_tensor(corrected_ce) else zero.detach(),
                "oai_fot_tb_residual": tb_residual.detach().abs().mean(),
                "oai_fot_correction_cosine": correction_cosine.detach().mean(),
                "oai_fot_correction_bpb_proxy_lift": value_lift.detach().mean() if torch.is_tensor(value_lift) else zero.detach(),
                "oai_fot_consensus_margin": consensus_margin.detach(),
                "oai_fot_consensus_entropy": (
                    consensus_entropy / math.log(max(2, int(cfg.consensus_buckets)))
                ).detach(),
                "oai_fot_consensus_tree_agreement": tree_agreement.detach(),
                "oai_fot_log_z": self.log_z.detach(),
                "oai_fot_num_trees": hidden.new_tensor(float(num_trees)).detach(),
                "oai_fot_node_count": hidden.new_tensor(float(n_nodes)).detach(),
                "oai_fot_topk_trees": hidden.new_tensor(float(topk)).detach(),
                "oai_fot_enabled": hidden.new_tensor(1.0).detach(),
            }
        )

    @torch.no_grad()
    def trace_payload(self, hidden: Tensor, target_ids: Tensor, per_token_nll: Tensor) -> dict[str, object]:
        """Return a compact JSON-serializable forest trace for analysis reports."""
        idx = self._selected_indices(int(hidden.shape[1]), hidden.device)
        nodes = self.node_norm(hidden[:, idx, :].float())
        n_nodes = int(nodes.shape[1])
        num_trees = max(1, min(int(self.config.num_trees), n_nodes))
        tree_ids = torch.remainder(torch.arange(n_nodes, device=hidden.device), num_trees).long()
        activation = self.activation_head(nodes).squeeze(-1).float()[0]
        value = self.value_head(nodes).squeeze(-1).float()[0]
        edges = []
        latest: dict[int, int] = {}
        for local_id, tree_id in enumerate(tree_ids.tolist()):
            parent = latest.get(tree_id)
            if parent is not None:
                edges.append({"source": parent, "target": local_id, "tree_id": tree_id, "kind": "tree"})
            latest[tree_id] = local_id
        return {
            "schema": "toricgt.embedding_forest_of_thought.trace.v1",
            "num_trees": num_trees,
            "node_count": n_nodes,
            "nodes": [
                {
                    "id": int(i),
                    "tree_id": int(tree_ids[i].item()),
                    "position": int(idx[i].item()),
                    "target_id": int(target_ids[0, idx[i]].item()),
                    "nll": float(per_token_nll[0, idx[i]].detach().float().item()),
                    "activation": float(activation[i].detach().cpu().item()),
                    "value": float(value[i].detach().cpu().item()),
                }
                for i in range(n_nodes)
            ],
            "edges": edges,
        }
