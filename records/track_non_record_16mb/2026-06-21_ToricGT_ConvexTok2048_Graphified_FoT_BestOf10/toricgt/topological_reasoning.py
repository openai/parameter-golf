"""Directed persistent topology utilities for reasoning trajectories.

The functions in this module are deliberately lightweight.  They implement the
topological analogue used by the Parameter-Golf adapter without depending on a
full persistent-homology package at training time:

* every local reasoning window is treated as a point cloud of hidden states;
* a radius sweep builds nested Vietoris--Rips/flag complexes up to dimension 2;
* a time/skew biased directed adjacency tracks noncommutative trajectory flow;
* a mutual-reachability sweep supplies an HDBSCAN-style stability gate.
* DEC-inspired residuals audit conservative reasoning flow: divergence,
  vorticity drift, kinetic-energy drift, Hodge balance, and wedge/interior
  product consistency over the same directed simplicial windows.

The PyTorch path returns differentiable losses and finite diagnostics.  The
NumPy path mirrors the same definitions for periodic plots and audits.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch.nn import functional as F

from .gudhi_persistence import torch_persistence_image, torch_persistence_landscape


@dataclass(frozen=True)
class ReasoningTopologyConfig:
    max_points: int = 24
    max_windows: int = 6
    window_size: int = 32
    step_stride: int = 8
    levels: int = 4
    radius_min: float = 0.55
    radius_max: float = 1.65
    temperature: float = 0.12
    skew_scale: float = 0.35
    time_bias: float = 0.18
    hdbscan_min_cluster_size: int = 4
    hdbscan_min_samples: int = 4
    hdbscan_stability_threshold: float = 0.18


def _zero_like(hidden: torch.Tensor) -> dict[str, torch.Tensor]:
    zero = hidden.float().sum() * 0.0
    keys = (
        "reasoning_step_topology_loss",
        "reasoning_step_barcode_loss",
        "reasoning_step_simplex_closure_loss",
        "reasoning_step_filtration_inclusion_loss",
        "reasoning_step_boundary_residual",
        "reasoning_step_dirichlet_energy",
        "reasoning_step_dec_conservation_loss",
        "reasoning_step_dec_mass_residual",
        "reasoning_step_dec_vorticity_drift",
        "reasoning_step_dec_kinetic_energy",
        "reasoning_step_dec_kinetic_energy_drift",
        "reasoning_step_dec_hodge_balance",
        "reasoning_step_dec_wedge_interior_residual",
        "reasoning_step_directed_topology_loss",
        "reasoning_step_directed_transitive_loss",
        "reasoning_step_directed_cycle_flux",
        "reasoning_step_directed_chain_commutator",
        "reasoning_step_directed_asymmetry",
        "reasoning_step_analogical_map_loss",
        "reasoning_step_directed_map_loss",
        "reasoning_step_transport_entropy",
        "reasoning_step_hdbscan_loss",
        "reasoning_step_hdbscan_stability",
        "reasoning_step_hdbscan_noise_fraction",
        "reasoning_step_hdbscan_persistent_edge_density",
        "reasoning_step_hdbscan_core_radius",
        "reasoning_step_ph_landscape_loss",
        "reasoning_step_ph_landscape_norm",
        "reasoning_step_ph_image_energy",
        "reasoning_step_edge_density",
        "reasoning_step_triangle_density",
        "reasoning_step_cycle_rank",
        "reasoning_step_betti0",
        "reasoning_step_windows",
    )
    return {key: zero.detach() if key != "reasoning_step_topology_loss" else zero for key in keys}


def _sample_window_points(window: torch.Tensor, max_points: int) -> torch.Tensor:
    if window.shape[0] <= max_points:
        return window
    index = torch.linspace(0, window.shape[0] - 1, steps=max_points, device=window.device).round().long()
    return window.index_select(0, index)


def _window_starts(length: int, cfg: ReasoningTopologyConfig) -> list[int]:
    if length < 4:
        return []
    window = max(4, min(int(cfg.window_size), length))
    stride = max(1, int(cfg.step_stride))
    starts = list(range(0, max(1, length - window + 1), stride))
    final_start = max(0, length - window)
    if final_start not in starts:
        starts.append(final_start)
    if len(starts) > max(1, int(cfg.max_windows)):
        idx = np.linspace(0, len(starts) - 1, num=max(1, int(cfg.max_windows))).round().astype(int)
        starts = [starts[int(i)] for i in idx]
    return starts


def _antisymmetric_skew(points: torch.Tensor, skew_scale: float) -> torch.Tensor:
    n, width = points.shape
    if width < 2:
        return points.new_zeros((n, n))
    half = max(1, width // 2)
    left = points[:, :half]
    right = points[:, half : half + half]
    if right.shape[1] < left.shape[1]:
        right = F.pad(right, (0, left.shape[1] - right.shape[1]))
    skew = (left @ right.transpose(0, 1) - right @ left.transpose(0, 1)) / math.sqrt(max(1, half))
    normalizer = skew.detach().abs().median().clamp_min(1e-4)
    return (skew / normalizer).clamp(-4.0, 4.0) * float(skew_scale)


def reasoning_step_topology_loss(
    hidden: torch.Tensor,
    *,
    config: ReasoningTopologyConfig | None = None,
) -> dict[str, torch.Tensor]:
    """Differentiable directed topology loss over local reasoning windows.

    ``hidden`` has shape ``[B, T, D]``.  Each sampled window in each batch row is
    a local reasoning-step complex: vertices are hidden states in that window,
    undirected soft edges are radius-thresholded distances, directed soft edges
    additionally include time orientation and an antisymmetric toric form.
    """

    cfg = config or ReasoningTopologyConfig()
    if hidden.ndim != 3 or hidden.shape[1] < 4:
        return _zero_like(hidden)

    x_all = hidden.float()
    zero = x_all.sum() * 0.0
    radii = torch.linspace(
        float(cfg.radius_min),
        max(float(cfg.radius_min) + 1e-4, float(cfg.radius_max)),
        steps=max(2, int(cfg.levels)),
        device=x_all.device,
        dtype=x_all.dtype,
    )
    temperature = max(float(cfg.temperature), 1e-4)
    window_terms: dict[str, list[torch.Tensor]] = {
        "barcode": [],
        "closure": [],
        "inclusion": [],
        "boundary": [],
        "dirichlet": [],
        "dec_conservation": [],
        "dec_mass": [],
        "dec_vorticity_drift": [],
        "dec_kinetic": [],
        "dec_kinetic_drift": [],
        "dec_hodge": [],
        "dec_wedge": [],
        "directed": [],
        "directed_transitive": [],
        "directed_cycle": [],
        "directed_chain": [],
        "directed_asymmetry": [],
        "analogical_map": [],
        "directed_map": [],
        "transport_entropy": [],
        "hdbscan": [],
        "hdbscan_stability": [],
        "hdbscan_noise": [],
        "hdbscan_persistent_edge": [],
        "hdbscan_core": [],
        "ph_landscape_loss": [],
        "ph_landscape_norm": [],
        "ph_image_energy": [],
        "edge_density": [],
        "triangle_density": [],
        "cycle_rank": [],
        "betti0": [],
    }
    starts = _window_starts(int(x_all.shape[1]), cfg)
    if not starts:
        return _zero_like(hidden)

    eye_cache: dict[int, torch.Tensor] = {}
    complex_snapshots: list[dict[str, Any]] = []
    for batch_index in range(x_all.shape[0]):
        sequence = x_all[batch_index]
        for start in starts:
            points = _sample_window_points(sequence[start : start + max(4, min(int(cfg.window_size), sequence.shape[0]))], int(cfg.max_points))
            n = int(points.shape[0])
            if n < 4:
                continue
            points = F.normalize(points, dim=-1)
            dist = torch.cdist(points, points, p=2)
            positive = dist[dist > 1e-6]
            if positive.numel() == 0:
                continue
            scale = positive.detach().median().clamp_min(1e-4)
            dist = dist / scale
            eye = eye_cache.get(n)
            if eye is None or eye.device != points.device or eye.dtype != points.dtype:
                eye = torch.eye(n, device=points.device, dtype=points.dtype)
                eye_cache[n] = eye
            masked_dist = dist + eye * 1.0e6
            nn_deaths = masked_dist.topk(min(2, n - 1), dim=-1, largest=False).values[:, 0].clamp_min(0.0)
            ph_diagram0 = torch.stack([torch.zeros_like(nn_deaths), nn_deaths], dim=-1)
            ph_grid = torch.linspace(
                0.0,
                max(float(cfg.radius_max), float(cfg.radius_min) + 1.0e-4),
                steps=24,
                device=points.device,
                dtype=points.dtype,
            )
            ph_landscape = torch_persistence_landscape(ph_diagram0, ph_grid, layers=3)
            ph_image = torch_persistence_image(ph_diagram0, ph_grid, ph_grid, sigma=max(float(cfg.temperature), 1e-3))
            ph_landscape_loss = ph_landscape.diff(n=2, dim=-1).abs().mean() if ph_landscape.shape[-1] > 2 else zero
            ph_landscape_norm = ph_landscape.norm(p=2) / max(1, ph_landscape.numel())
            ph_image_energy = ph_image.pow(2).mean()
            window_terms["ph_landscape_loss"].append(ph_landscape_loss)
            window_terms["ph_landscape_norm"].append(ph_landscape_norm.detach())
            window_terms["ph_image_energy"].append(ph_image_energy.detach())
            core_k = min(max(1, int(cfg.hdbscan_min_samples)), n - 1)
            core_radius = masked_dist.topk(core_k, dim=-1, largest=False).values[:, -1].detach()
            mutual_reachability = torch.maximum(dist, torch.maximum(core_radius[:, None], core_radius[None, :]))
            mutual_reachability = mutual_reachability + eye * 1.0e6
            skew = _antisymmetric_skew(points, cfg.skew_scale)
            time_index = torch.arange(n, device=points.device, dtype=points.dtype)
            time_orientation = (time_index[None, :] - time_index[:, None]).sign() * float(cfg.time_bias)
            hdbscan_affinities = []
            prev_sym = None
            prev_dir = None
            prev_sym_chain = None
            prev_dir_chain = None
            level_barcode = []
            level_closure = []
            level_inclusion = []
            level_boundary = []
            level_dirichlet = []
            level_dec_conservation = []
            level_dec_mass = []
            level_dec_vorticity_drift = []
            level_dec_kinetic = []
            level_dec_kinetic_drift = []
            level_dec_hodge = []
            level_dec_wedge = []
            level_directed = []
            level_directed_transitive = []
            level_directed_cycle = []
            level_directed_chain = []
            level_directed_asymmetry = []
            level_edge_density = []
            level_triangle_density = []
            level_cycle_rank = []
            level_betti0 = []
            sym_levels: list[torch.Tensor] = []
            directed_levels: list[torch.Tensor] = []
            prev_vorticity = None
            prev_kinetic = None
            nn_scale = masked_dist.topk(min(2, n - 1), dim=-1, largest=False).values[:, 0].mean()
            level_barcode.append(nn_scale)
            for radius in radii:
                sym = torch.sigmoid((radius - dist) / temperature) * (1.0 - eye)
                directed = torch.sigmoid((radius - dist + skew + time_orientation) / temperature) * (1.0 - eye)
                density = torch.sigmoid((radius - mutual_reachability) / temperature) * (1.0 - eye)
                hdbscan_affinities.append(density)

                two_step = sym @ sym / max(1, n - 2)
                closure = (two_step * (1.0 - sym)).mean()
                triangle_mass = torch.einsum("ij,jk,ik->", sym, sym, sym)
                triangle_density = triangle_mass / max(1, n * (n - 1) * (n - 2))
                edge_density = sym.sum() / max(1, n * (n - 1))
                hard_edges = (sym.detach() > 0.5).to(dtype=points.dtype)
                degree = hard_edges.sum(dim=-1)
                # Euler/flag-complex proxies: beta0 is approximated by the
                # trace of a lazy random-walk heat kernel; beta1 by cycle rank
                # minus filled triangle pressure.
                lap = torch.diag(degree) - hard_edges
                beta0_proxy = torch.trace(torch.matrix_exp(-0.25 * lap.float())).to(points.dtype) / max(1, n)
                undirected_edges = hard_edges.sum() / 2.0
                cycle_rank = torch.relu(undirected_edges - float(n) + beta0_proxy * float(n))
                filled_cycle_proxy = torch.minimum(cycle_rank, triangle_density * float(n))

                weights = (sym + sym.transpose(0, 1)) * 0.5
                weights = weights.detach()
                dirichlet = (weights * dist.pow(2)).sum() / weights.sum().clamp_min(1e-8)
                flow = directed - directed.transpose(0, 1)
                divergence = flow.sum(dim=-1) / max(1, n - 1)
                dec_mass = divergence.pow(2).mean()
                degree_weight = sym.detach().sum(dim=-1).clamp_min(1e-6)
                vorticity_node = (flow * sym.detach()).sum(dim=-1) / degree_weight
                edge_mass = sym.detach().sum().clamp_min(1e-8)
                dec_kinetic = (sym.detach() * flow.pow(2)).sum() / edge_mass
                if prev_vorticity is None:
                    dec_vorticity_drift = zero
                    dec_kinetic_drift = zero
                else:
                    dec_vorticity_drift = (vorticity_node - prev_vorticity).pow(2).mean()
                    dec_kinetic_drift = (dec_kinetic - prev_kinetic).pow(2)
                hodge_edge = 1.0 / (core_radius[:, None] + core_radius[None, :] + 1e-6)
                hodge_edge = hodge_edge * (1.0 - eye)
                positive_hodge = hodge_edge[hodge_edge > 0]
                hodge_mean = (
                    positive_hodge.mean().clamp_min(1e-6)
                    if positive_hodge.numel() > 0
                    else hodge_edge.new_tensor(1.0)
                )
                hodge_edge = hodge_edge / hodge_mean
                hodge_energy = (hodge_edge.detach() * sym.detach() * flow.pow(2)).sum() / (
                    hodge_edge.detach() * sym.detach()
                ).sum().clamp_min(1e-8)
                dec_hodge = (hodge_energy / dec_kinetic.detach().abs().clamp_min(1e-8) - 1.0).pow(2)
                avg_vorticity = 0.5 * (vorticity_node[:, None] + vorticity_node[None, :])
                wedge_form = flow * avg_vorticity
                interior_form = directed * vorticity_node[None, :] - directed.transpose(0, 1) * vorticity_node[:, None]
                dec_wedge = (sym.detach() * (wedge_form - interior_form).pow(2)).sum() / edge_mass
                dec_conservation = (
                    dec_mass
                    + 0.25 * dec_vorticity_drift
                    + 0.10 * dec_kinetic_drift
                    + 0.05 * dec_hodge
                    + 0.05 * dec_wedge
                )
                prev_vorticity = vorticity_node.detach()
                prev_kinetic = dec_kinetic.detach()
                directed_two_step = directed @ directed / max(1, n - 2)
                directed_transitive = (directed_two_step * (1.0 - directed)).mean()
                cycle_forward = torch.einsum("ij,jk,ki->", directed, directed, directed)
                cycle_backward = torch.einsum("ji,kj,ik->", directed, directed, directed)
                directed_cycle = (cycle_forward - cycle_backward).abs() / max(1, n * (n - 1) * (n - 2))
                directed_asymmetry = (directed - directed.transpose(0, 1)).abs().mean()
                sym_chain = sym + eye
                sym_chain = sym_chain / sym_chain.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                dir_chain = directed + eye
                dir_chain = dir_chain / dir_chain.sum(dim=-1, keepdim=True).clamp_min(1e-6)
                if prev_sym is None:
                    inclusion = zero
                    chain = zero
                    directed_chain = zero
                else:
                    inclusion = torch.relu(prev_sym - sym).pow(2).mean() + torch.relu(prev_dir - directed).pow(2).mean()
                    chain = (sym_chain @ prev_sym_chain - prev_sym_chain @ sym_chain).pow(2).mean()
                    directed_chain = (dir_chain @ prev_dir_chain - prev_dir_chain @ dir_chain).pow(2).mean()
                # Soft boundary proxy for oriented 2-simplices: a strong
                # triangle should carry its three oriented boundary edges.
                boundary = torch.relu(
                    torch.einsum("ij,jk,ik->ijk", directed, directed, directed)
                    - directed[:, :, None]
                    - directed[:, None, :]
                    - directed[None, :, :]
                ).pow(2).mean()

                level_closure.append(closure)
                level_inclusion.append(inclusion + chain)
                level_boundary.append(boundary)
                level_dirichlet.append(dirichlet)
                level_dec_conservation.append(dec_conservation)
                level_dec_mass.append(dec_mass)
                level_dec_vorticity_drift.append(dec_vorticity_drift)
                level_dec_kinetic.append(dec_kinetic)
                level_dec_kinetic_drift.append(dec_kinetic_drift)
                level_dec_hodge.append(dec_hodge)
                level_dec_wedge.append(dec_wedge)
                level_directed.append(directed_transitive + float(cfg.skew_scale) * directed_cycle + directed_chain)
                level_directed_transitive.append(directed_transitive)
                level_directed_cycle.append(directed_cycle)
                level_directed_chain.append(directed_chain)
                level_directed_asymmetry.append(directed_asymmetry)
                level_edge_density.append(edge_density)
                level_triangle_density.append(triangle_density)
                level_cycle_rank.append(torch.relu(cycle_rank - filled_cycle_proxy))
                level_betti0.append(beta0_proxy)
                sym_levels.append(sym)
                directed_levels.append(directed)
                prev_sym = sym
                prev_dir = directed
                prev_sym_chain = sym_chain
                prev_dir_chain = dir_chain

            persistent = torch.stack(hdbscan_affinities, dim=0).mean(dim=0)
            stable_degree = persistent.sum(dim=-1) / max(1, n - 1)
            stable_member = (stable_degree.detach() * max(1, n - 1)) >= float(max(2, int(cfg.hdbscan_min_cluster_size)) - 1)
            pair_weight = torch.relu(persistent.detach() - float(cfg.hdbscan_stability_threshold)).pow(2)
            pair_weight = pair_weight * stable_member.to(pair_weight.dtype)[:, None] * stable_member.to(pair_weight.dtype)[None, :]
            pair_weight = pair_weight * (1.0 - eye)
            pair_mass = pair_weight.sum()
            if bool((pair_mass > 1e-8).detach().cpu().item()):
                hdbscan = (pair_weight * dist.pow(2)).sum() / pair_mass.clamp_min(1e-8)
            else:
                hdbscan = zero

            window_terms["barcode"].append(torch.stack(level_barcode).mean())
            window_terms["closure"].append(torch.stack(level_closure).mean())
            window_terms["inclusion"].append(torch.stack(level_inclusion).mean())
            window_terms["boundary"].append(torch.stack(level_boundary).mean())
            window_terms["dirichlet"].append(torch.stack(level_dirichlet).mean())
            window_terms["dec_conservation"].append(torch.stack(level_dec_conservation).mean())
            window_terms["dec_mass"].append(torch.stack(level_dec_mass).mean())
            window_terms["dec_vorticity_drift"].append(torch.stack(level_dec_vorticity_drift).mean())
            window_terms["dec_kinetic"].append(torch.stack(level_dec_kinetic).mean())
            window_terms["dec_kinetic_drift"].append(torch.stack(level_dec_kinetic_drift).mean())
            window_terms["dec_hodge"].append(torch.stack(level_dec_hodge).mean())
            window_terms["dec_wedge"].append(torch.stack(level_dec_wedge).mean())
            window_terms["directed"].append(torch.stack(level_directed).mean())
            window_terms["directed_transitive"].append(torch.stack(level_directed_transitive).mean())
            window_terms["directed_cycle"].append(torch.stack(level_directed_cycle).mean())
            window_terms["directed_chain"].append(torch.stack(level_directed_chain).mean())
            window_terms["directed_asymmetry"].append(torch.stack(level_directed_asymmetry).mean())
            window_terms["hdbscan"].append(hdbscan)
            window_terms["hdbscan_stability"].append(stable_degree.mean())
            window_terms["hdbscan_noise"].append(1.0 - stable_member.to(points.dtype).mean())
            window_terms["hdbscan_persistent_edge"].append((pair_weight > 0).to(points.dtype).sum() / max(1, n * (n - 1)))
            window_terms["hdbscan_core"].append(core_radius.mean())
            window_terms["edge_density"].append(torch.stack(level_edge_density).mean())
            window_terms["triangle_density"].append(torch.stack(level_triangle_density).mean())
            window_terms["cycle_rank"].append(torch.stack(level_cycle_rank).mean())
            window_terms["betti0"].append(torch.stack(level_betti0).mean())
            complex_snapshots.append(
                {
                    "points": points,
                    "sym_levels": sym_levels,
                    "directed_levels": directed_levels,
                }
            )

    if not window_terms["closure"]:
        return _zero_like(hidden)

    # Analogical maps between consecutive local reasoning complexes.  This is
    # the differentiable analogue of mapping one collection of interrelated
    # thought vectors to the next: a soft transport sends source vertices to
    # target vertices, and the pushed-forward adjacency/2-skeleton should match
    # the target filtration.  The directed residual keeps the noncommutative
    # order of the graph-of-thought trajectory visible.
    for source, target in zip(complex_snapshots[:-1], complex_snapshots[1:]):
        source_points = source["points"]
        target_points = target["points"]
        if source_points.numel() == 0 or target_points.numel() == 0:
            continue
        transport_logits = -torch.cdist(source_points, target_points, p=2) / temperature
        transport = F.softmax(transport_logits, dim=-1)
        transport_detached = transport.detach()
        entropy = -(transport * (transport + 1e-8).log()).sum(dim=-1).mean()
        entropy = entropy / math.log(max(2, target_points.shape[0]))
        window_terms["transport_entropy"].append(entropy)
        for sym_source, sym_target in zip(source["sym_levels"], target["sym_levels"]):
            pushed = transport_detached.transpose(0, 1) @ sym_source @ transport_detached
            window_terms["analogical_map"].append((pushed - sym_target).pow(2).mean())
        for dir_source, dir_target in zip(source["directed_levels"], target["directed_levels"]):
            pushed_dir = transport_detached.transpose(0, 1) @ dir_source @ transport_detached
            window_terms["directed_map"].append((pushed_dir - dir_target).pow(2).mean())

    def mean_term(name: str) -> torch.Tensor:
        return torch.stack(window_terms[name]).mean()

    barcode_loss = mean_term("barcode")
    closure_loss = mean_term("closure")
    inclusion_loss = mean_term("inclusion")
    boundary_residual = mean_term("boundary")
    dirichlet_energy = mean_term("dirichlet")
    dec_conservation = mean_term("dec_conservation")
    directed_loss = mean_term("directed")
    analogical_map_loss = mean_term("analogical_map") if window_terms["analogical_map"] else zero
    directed_map_loss = mean_term("directed_map") if window_terms["directed_map"] else zero
    transport_entropy = mean_term("transport_entropy") if window_terms["transport_entropy"] else zero.detach()
    hdbscan_loss = mean_term("hdbscan")
    topology_loss = (
        barcode_loss
        + closure_loss
        + 0.15 * inclusion_loss
        + 0.15 * boundary_residual
        + 0.25 * dirichlet_energy
        + 0.05 * mean_term("ph_landscape_loss")
        + 0.15 * dec_conservation
        + 0.35 * directed_loss
        + 0.15 * analogical_map_loss
        + 0.15 * directed_map_loss
        + 0.20 * hdbscan_loss
    )
    return {
        "reasoning_step_topology_loss": topology_loss,
        "reasoning_step_barcode_loss": barcode_loss.detach(),
        "reasoning_step_simplex_closure_loss": closure_loss.detach(),
        "reasoning_step_filtration_inclusion_loss": inclusion_loss.detach(),
        "reasoning_step_boundary_residual": boundary_residual.detach(),
        "reasoning_step_dirichlet_energy": dirichlet_energy.detach(),
        "reasoning_step_dec_conservation_loss": dec_conservation.detach(),
        "reasoning_step_dec_mass_residual": mean_term("dec_mass").detach(),
        "reasoning_step_dec_vorticity_drift": mean_term("dec_vorticity_drift").detach(),
        "reasoning_step_dec_kinetic_energy": mean_term("dec_kinetic").detach(),
        "reasoning_step_dec_kinetic_energy_drift": mean_term("dec_kinetic_drift").detach(),
        "reasoning_step_dec_hodge_balance": mean_term("dec_hodge").detach(),
        "reasoning_step_dec_wedge_interior_residual": mean_term("dec_wedge").detach(),
        "reasoning_step_directed_topology_loss": directed_loss.detach(),
        "reasoning_step_directed_transitive_loss": mean_term("directed_transitive").detach(),
        "reasoning_step_directed_cycle_flux": mean_term("directed_cycle").detach(),
        "reasoning_step_directed_chain_commutator": mean_term("directed_chain").detach(),
        "reasoning_step_directed_asymmetry": mean_term("directed_asymmetry").detach(),
        "reasoning_step_analogical_map_loss": analogical_map_loss.detach(),
        "reasoning_step_directed_map_loss": directed_map_loss.detach(),
        "reasoning_step_transport_entropy": transport_entropy.detach(),
        "reasoning_step_hdbscan_loss": hdbscan_loss.detach(),
        "reasoning_step_hdbscan_stability": mean_term("hdbscan_stability").detach(),
        "reasoning_step_hdbscan_noise_fraction": mean_term("hdbscan_noise").detach(),
        "reasoning_step_hdbscan_persistent_edge_density": mean_term("hdbscan_persistent_edge").detach(),
        "reasoning_step_hdbscan_core_radius": mean_term("hdbscan_core").detach(),
        "reasoning_step_ph_landscape_loss": mean_term("ph_landscape_loss").detach(),
        "reasoning_step_ph_landscape_norm": mean_term("ph_landscape_norm").detach(),
        "reasoning_step_ph_image_energy": mean_term("ph_image_energy").detach(),
        "reasoning_step_edge_density": mean_term("edge_density").detach(),
        "reasoning_step_triangle_density": mean_term("triangle_density").detach(),
        "reasoning_step_cycle_rank": mean_term("cycle_rank").detach(),
        "reasoning_step_betti0": mean_term("betti0").detach(),
        "reasoning_step_windows": hidden.new_tensor(float(len(window_terms["closure"]))).detach(),
    }


def sigmoid_np(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def _component_count(adjacency: np.ndarray) -> tuple[int, list[int]]:
    n = adjacency.shape[0]
    seen = np.zeros(n, dtype=bool)
    sizes: list[int] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for nxt in np.flatnonzero(adjacency[node]):
                if not seen[nxt]:
                    seen[nxt] = True
                    stack.append(int(nxt))
        sizes.append(size)
    return len(sizes), sizes


def _f2_rank(matrix: np.ndarray) -> int:
    """Rank over F2 for small audit matrices."""

    if matrix.size == 0:
        return 0
    mat = (matrix.astype(np.uint8, copy=True) & 1)
    rows, cols = mat.shape
    rank = 0
    for col in range(cols):
        pivot = np.flatnonzero(mat[rank:, col])
        if pivot.size == 0:
            continue
        pivot_row = rank + int(pivot[0])
        if pivot_row != rank:
            mat[[rank, pivot_row]] = mat[[pivot_row, rank]]
        for row in range(rows):
            if row != rank and mat[row, col]:
                mat[row] ^= mat[rank]
        rank += 1
        if rank == rows:
            break
    return int(rank)


def _f2_rref(matrix: np.ndarray) -> tuple[np.ndarray, list[int]]:
    mat = (matrix.astype(np.uint8, copy=True) & 1)
    rows, cols = mat.shape
    pivots: list[int] = []
    rank = 0
    for col in range(cols):
        pivot = np.flatnonzero(mat[rank:, col])
        if pivot.size == 0:
            continue
        pivot_row = rank + int(pivot[0])
        if pivot_row != rank:
            mat[[rank, pivot_row]] = mat[[pivot_row, rank]]
        for row in range(rows):
            if row != rank and mat[row, col]:
                mat[row] ^= mat[rank]
        pivots.append(col)
        rank += 1
        if rank == rows:
            break
    return mat, pivots


def _f2_nullspace(matrix: np.ndarray) -> np.ndarray:
    """Return a column-basis for the nullspace over F2."""

    rows, cols = matrix.shape
    if cols == 0:
        return np.zeros((0, 0), dtype=np.uint8)
    rref, pivots = _f2_rref(matrix)
    pivot_set = set(pivots)
    free_cols = [col for col in range(cols) if col not in pivot_set]
    if not free_cols:
        return np.zeros((cols, 0), dtype=np.uint8)
    basis = np.zeros((cols, len(free_cols)), dtype=np.uint8)
    for basis_col, free_col in enumerate(free_cols):
        basis[free_col, basis_col] = 1
        for pivot_row, pivot_col in enumerate(pivots):
            if rref[pivot_row, free_col]:
                basis[pivot_col, basis_col] = 1
    return basis


def _rank_fraction(matrix: np.ndarray) -> float:
    if matrix.size == 0:
        return 0.0
    return float(_f2_rank(matrix) / max(1, min(matrix.shape)))


def _be_complementary_minor_audit(d1: np.ndarray, d2: np.ndarray, *, max_samples: int = 32) -> dict[str, float]:
    """Buchsbaum-Eisenbud-style complementary-minor audit over F2.

    For a two-step free complex C2 -> C1 -> C0, exactness at C1 implies the
    rank condition rank(d1)+rank(d2)=dim(C1).  Buchsbaum-Eisenbud multipliers
    express compatibility among complementary maximal minors.  We use a
    tractable sampled version: choose disjoint edge-index sets S and C of sizes
    rank(d1) and rank(d2), test whether the corresponding maximal minors can be
    nonzero, and count complementary mismatches.
    """

    edge_count = int(d1.shape[1])
    rank_d1 = _f2_rank(d1)
    rank_d2 = _f2_rank(d2)
    rank_residual = abs(rank_d1 + rank_d2 - edge_count) / max(1, edge_count)
    if edge_count == 0 or rank_d1 == 0 or rank_d2 == 0 or rank_d1 + rank_d2 > edge_count:
        return {
            "buchsbaum_eisenbud_rank_residual": float(rank_residual),
            "buchsbaum_eisenbud_multiplier_residual": float(rank_residual),
            "buchsbaum_eisenbud_multiplier_samples": 0.0,
        }
    samples = min(max_samples, max(1, edge_count))
    rng = np.random.default_rng(1729 + 17 * edge_count + 31 * rank_d1 + 43 * rank_d2)
    mismatches = 0
    used = 0
    for _ in range(samples):
        perm = rng.permutation(edge_count)
        s_cols = np.sort(perm[:rank_d1])
        c_rows = np.sort(perm[rank_d1 : rank_d1 + rank_d2])
        d1_minor_possible = _f2_rank(d1[:, s_cols]) == rank_d1
        d2_minor_possible = _f2_rank(d2[c_rows, :]) == rank_d2
        mismatches += int(bool(d1_minor_possible) != bool(d2_minor_possible))
        used += 1
    multiplier_residual = mismatches / max(1, used)
    # Exactness/rank failure should not be hidden by accidental minor agreement.
    multiplier_residual = max(float(multiplier_residual), float(rank_residual))
    return {
        "buchsbaum_eisenbud_rank_residual": float(rank_residual),
        "buchsbaum_eisenbud_multiplier_residual": float(multiplier_residual),
        "buchsbaum_eisenbud_multiplier_samples": float(used),
    }


def _build_flag_complex_f2(adjacency: np.ndarray, *, max_edges: int = 512, max_triangles: int = 1500) -> dict[str, Any]:
    """Build a small flag complex and F2 boundary matrices up to dimension 2."""

    hard = np.asarray(adjacency, dtype=bool)
    hard = np.logical_or(hard, hard.T)
    np.fill_diagonal(hard, False)
    n = int(hard.shape[0])
    edges = [(i, j) for i in range(n) for j in range(i + 1, n) if hard[i, j]]
    truncated = False
    if len(edges) > max_edges:
        truncated = True
        edges = edges[:max_edges]
    edge_index = {edge: idx for idx, edge in enumerate(edges)}
    triangles: list[tuple[int, int, int]] = []
    if not truncated:
        for i in range(n):
            neigh = [j for j in range(i + 1, n) if hard[i, j]]
            for a, j in enumerate(neigh):
                for k in neigh[a + 1 :]:
                    if hard[j, k]:
                        triangles.append((i, j, k))
                        if len(triangles) > max_triangles:
                            truncated = True
                            triangles = triangles[:max_triangles]
                            break
                if truncated:
                    break
            if truncated:
                break
    d1 = np.zeros((n, len(edges)), dtype=np.uint8)
    for col, (i, j) in enumerate(edges):
        d1[i, col] = 1
        d1[j, col] = 1
    d2 = np.zeros((len(edges), len(triangles)), dtype=np.uint8)
    for col, (i, j, k) in enumerate(triangles):
        for edge in ((i, j), (i, k), (j, k)):
            row = edge_index.get(tuple(sorted(edge)))
            if row is not None:
                d2[row, col] = 1
    rank_d1 = _f2_rank(d1)
    rank_d2 = _f2_rank(d2)
    complex_product = (d1 @ d2) & 1 if d1.size and d2.size else np.zeros((d1.shape[0], d2.shape[1]), dtype=np.uint8)
    variety_complex_residual = float(complex_product.sum() / max(1, complex_product.size))
    h0 = n - rank_d1
    h1 = max(0, len(edges) - rank_d1 - rank_d2)
    be_audit = _be_complementary_minor_audit(d1, d2)
    fitting_rank_residual = (
        (min(d1.shape) - rank_d1) / max(1, min(d1.shape))
        if min(d1.shape) > 0
        else 0.0
    )
    if min(d2.shape) > 0:
        fitting_rank_residual += (min(d2.shape) - rank_d2) / max(1, min(d2.shape))
    fitting_rank_residual *= 0.5
    return {
        "n": n,
        "edges": edges,
        "edge_index": edge_index,
        "triangles": triangles,
        "triangle_set": set(triangles),
        "d1": d1,
        "d2": d2,
        "rank_d1": int(rank_d1),
        "rank_d2": int(rank_d2),
        "h0": int(h0),
        "h1": int(h1),
        "variety_complex_residual": float(variety_complex_residual),
        "fitting_d1_rank_fraction": _rank_fraction(d1),
        "fitting_d2_rank_fraction": _rank_fraction(d2),
        "fitting_minor_rank_residual": float(fitting_rank_residual),
        "buchsbaum_eisenbud_rank_residual": float(be_audit["buchsbaum_eisenbud_rank_residual"]),
        "buchsbaum_eisenbud_multiplier_residual": float(be_audit["buchsbaum_eisenbud_multiplier_residual"]),
        "buchsbaum_eisenbud_multiplier_samples": float(be_audit["buchsbaum_eisenbud_multiplier_samples"]),
        "multigraded_betti_mass": float(h0 + h1),
        "truncated": bool(truncated),
    }


def _chain_map_matrices(
    source: dict[str, Any],
    target: dict[str, Any],
    vertex_map: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Return F0,F1 and edge/triangle validity for a vertex map."""

    f0 = np.zeros((target["n"], source["n"]), dtype=np.uint8)
    for src, dst in enumerate(vertex_map.tolist()):
        if 0 <= int(dst) < target["n"]:
            f0[int(dst), src] = 1
    f1 = np.zeros((len(target["edges"]), len(source["edges"])), dtype=np.uint8)
    valid_edges = 0
    noncollapsed_edges = 0
    for col, (i, j) in enumerate(source["edges"]):
        a, b = int(vertex_map[i]), int(vertex_map[j])
        if a == b:
            valid_edges += 1
            continue
        noncollapsed_edges += 1
        edge = tuple(sorted((a, b)))
        row = target["edge_index"].get(edge)
        if row is not None:
            f1[row, col] = 1
            valid_edges += 1
    valid_triangles = 0
    for tri in source["triangles"]:
        mapped = sorted({int(vertex_map[i]) for i in tri})
        if len(mapped) < 3:
            valid_triangles += 1
            continue
        if tuple(mapped) in target["triangle_set"]:
            valid_triangles += 1
    edge_validity = valid_edges / max(1, len(source["edges"]))
    triangle_validity = valid_triangles / max(1, len(source["triangles"]))
    return f0, f1, float(edge_validity), float(triangle_validity)


def _induced_homology_rank(
    source: dict[str, Any],
    target: dict[str, Any],
    chain_map: np.ndarray,
    degree: int,
) -> int:
    """Rank of the induced homology map over F2."""

    if degree == 0:
        cycles = np.eye(source["n"], dtype=np.uint8)
        target_boundaries = target["d1"]
    elif degree == 1:
        cycles = _f2_nullspace(source["d1"])
        target_boundaries = target["d2"]
    else:
        raise ValueError("only H0 and H1 are supported")
    if cycles.size == 0:
        return 0
    mapped = (chain_map @ cycles) & 1
    if target_boundaries.size == 0:
        base_rank = 0
        combined = mapped
    else:
        base_rank = _f2_rank(target_boundaries)
        combined = np.concatenate([target_boundaries, mapped], axis=1)
    return int(max(0, _f2_rank(combined) - base_rank))


def _exact_persistence_morphism_audit(
    snapshots: list[dict[str, Any]],
    levels: int,
) -> dict[str, Any]:
    """Compute exact F2 persistence-module morphism summaries when tractable."""

    h0_dims: list[list[float]] = []
    h1_dims: list[list[float]] = []
    variety_residuals: list[list[float]] = []
    fitting_residuals: list[list[float]] = []
    be_rank_residuals: list[list[float]] = []
    be_multiplier_residuals: list[list[float]] = []
    betti_masses: list[list[float]] = []
    truncated_count = 0
    for snapshot in snapshots:
        level_h0 = []
        level_h1 = []
        level_variety = []
        level_fitting = []
        level_be_rank = []
        level_be_multiplier = []
        level_betti_mass = []
        for complex_obj in snapshot.get("exact_levels", []):
            level_h0.append(float(complex_obj["h0"]))
            level_h1.append(float(complex_obj["h1"]))
            level_variety.append(float(complex_obj.get("variety_complex_residual", 0.0)))
            level_fitting.append(float(complex_obj.get("fitting_minor_rank_residual", 0.0)))
            level_be_rank.append(float(complex_obj.get("buchsbaum_eisenbud_rank_residual", 0.0)))
            level_be_multiplier.append(float(complex_obj.get("buchsbaum_eisenbud_multiplier_residual", 0.0)))
            level_betti_mass.append(float(complex_obj.get("multigraded_betti_mass", 0.0)))
            truncated_count += int(bool(complex_obj.get("truncated", False)))
        if level_h0:
            h0_dims.append(level_h0)
            h1_dims.append(level_h1)
            variety_residuals.append(level_variety)
            fitting_residuals.append(level_fitting)
            be_rank_residuals.append(level_be_rank)
            be_multiplier_residuals.append(level_be_multiplier)
            betti_masses.append(level_betti_mass)

    h0_map_ranks: list[list[float]] = []
    h1_map_ranks: list[list[float]] = []
    shifts: list[list[float]] = []
    edge_validities: list[list[float]] = []
    triangle_validities: list[list[float]] = []
    directed_edge_validities: list[list[float]] = []
    computed = 0
    for source_snapshot, target_snapshot in zip(snapshots[:-1], snapshots[1:]):
        source_points = source_snapshot["points"]
        target_points = target_snapshot["points"]
        distances = np.linalg.norm(source_points[:, None, :] - target_points[None, :, :], axis=-1)
        vertex_map = np.argmin(distances, axis=1).astype(int)
        row_h0 = []
        row_h1 = []
        row_shift = []
        row_edge_valid = []
        row_triangle_valid = []
        row_directed_valid = []
        for src_level in range(levels):
            source_complex = source_snapshot["exact_levels"][src_level]
            chosen = None
            chosen_data = None
            for target_level in range(src_level, levels):
                target_complex = target_snapshot["exact_levels"][target_level]
                f0, f1, edge_validity, triangle_validity = _chain_map_matrices(
                    source_complex,
                    target_complex,
                    vertex_map,
                )
                if edge_validity >= 1.0 and triangle_validity >= 1.0 and not source_complex["truncated"] and not target_complex["truncated"]:
                    chosen = target_level
                    chosen_data = (target_complex, f0, f1, edge_validity, triangle_validity)
                    break
                chosen_data = (target_complex, f0, f1, edge_validity, triangle_validity)
            if chosen is None:
                chosen = levels - 1
            target_complex, f0, f1, edge_validity, triangle_validity = chosen_data  # type: ignore[misc]
            directed_source = source_snapshot["directed_levels"][src_level] > 0.5
            directed_target = target_snapshot["directed_levels"][chosen] > 0.5
            directed_valid = 0
            directed_total = 0
            for i, j in zip(*np.nonzero(directed_source)):
                if i == j:
                    continue
                directed_total += 1
                a, b = int(vertex_map[i]), int(vertex_map[j])
                if a == b or directed_target[a, b]:
                    directed_valid += 1
            row_h0.append(float(_induced_homology_rank(source_complex, target_complex, f0, degree=0)))
            row_h1.append(float(_induced_homology_rank(source_complex, target_complex, f1, degree=1)))
            row_shift.append(float(chosen - src_level))
            row_edge_valid.append(float(edge_validity))
            row_triangle_valid.append(float(triangle_validity))
            row_directed_valid.append(float(directed_valid / max(1, directed_total)))
            computed += 1
        h0_map_ranks.append(row_h0)
        h1_map_ranks.append(row_h1)
        shifts.append(row_shift)
        edge_validities.append(row_edge_valid)
        triangle_validities.append(row_triangle_valid)
        directed_edge_validities.append(row_directed_valid)

    def mean_or_zero(values: list[list[float]]) -> float:
        return float(np.asarray(values, dtype=float).mean()) if values else 0.0

    return {
        "exact_h0_dims_by_window_radius": np.asarray(h0_dims, dtype=float),
        "exact_h1_dims_by_window_radius": np.asarray(h1_dims, dtype=float),
        "variety_complex_residual_by_window_radius": np.asarray(variety_residuals, dtype=float),
        "fitting_minor_rank_residual_by_window_radius": np.asarray(fitting_residuals, dtype=float),
        "buchsbaum_eisenbud_rank_residual_by_window_radius": np.asarray(be_rank_residuals, dtype=float),
        "buchsbaum_eisenbud_multiplier_residual_by_window_radius": np.asarray(be_multiplier_residuals, dtype=float),
        "multigraded_betti_mass_by_window_radius": np.asarray(betti_masses, dtype=float),
        "exact_h0_map_rank_by_transition_radius": np.asarray(h0_map_ranks, dtype=float),
        "exact_h1_map_rank_by_transition_radius": np.asarray(h1_map_ranks, dtype=float),
        "exact_morphism_radius_shift_by_transition": np.asarray(shifts, dtype=float),
        "exact_morphism_edge_validity_by_transition": np.asarray(edge_validities, dtype=float),
        "exact_morphism_triangle_validity_by_transition": np.asarray(triangle_validities, dtype=float),
        "exact_directed_edge_validity_by_transition": np.asarray(directed_edge_validities, dtype=float),
        "exact_h0_dim_mean": mean_or_zero(h0_dims),
        "exact_h1_dim_mean": mean_or_zero(h1_dims),
        "variety_complex_residual_mean": mean_or_zero(variety_residuals),
        "fitting_minor_rank_residual_mean": mean_or_zero(fitting_residuals),
        "buchsbaum_eisenbud_rank_residual_mean": mean_or_zero(be_rank_residuals),
        "buchsbaum_eisenbud_multiplier_residual_mean": mean_or_zero(be_multiplier_residuals),
        "multigraded_betti_mass_mean": mean_or_zero(betti_masses),
        "exact_h0_map_rank_mean": mean_or_zero(h0_map_ranks),
        "exact_h1_map_rank_mean": mean_or_zero(h1_map_ranks),
        "exact_morphism_radius_shift_mean": mean_or_zero(shifts),
        "exact_morphism_edge_validity_mean": mean_or_zero(edge_validities),
        "exact_morphism_triangle_validity_mean": mean_or_zero(triangle_validities),
        "exact_directed_edge_validity_mean": mean_or_zero(directed_edge_validities),
        "exact_morphism_computed": float(computed),
        "exact_morphism_truncated_complexes": float(truncated_count),
    }


def directed_step_filtration_stats_np(
    hidden: np.ndarray,
    *,
    config: ReasoningTopologyConfig | None = None,
) -> dict[str, Any]:
    """Audit the same directed step complexes with hard/soft NumPy summaries."""

    cfg = config or ReasoningTopologyConfig()
    if hidden.shape[0] < 4:
        radii = np.linspace(cfg.radius_min, cfg.radius_max, max(2, cfg.levels))
        zeros = np.zeros_like(radii, dtype=float)
        return {
            "radii": radii.tolist(),
            "step_window_index": [0],
            "edge_density": zeros.tolist(),
            "triangle_density": zeros.tolist(),
            "directed_edge_density": zeros.tolist(),
            "directed_asymmetry": zeros.tolist(),
            "directed_cycle_flux": zeros.tolist(),
            "directed_transitive_loss": zeros.tolist(),
            "directed_chain_commutator": zeros.tolist(),
            "inclusion_violation": zeros.tolist(),
            "boundary_residual": zeros.tolist(),
            "dirichlet_energy": zeros.tolist(),
            "dec_conservation_loss": zeros.tolist(),
            "dec_mass_residual": zeros.tolist(),
            "dec_vorticity_drift": zeros.tolist(),
            "dec_kinetic_energy": zeros.tolist(),
            "dec_kinetic_energy_drift": zeros.tolist(),
            "dec_hodge_balance": zeros.tolist(),
            "dec_wedge_interior_residual": zeros.tolist(),
            "analogical_map_loss": 0.0,
            "directed_map_loss": 0.0,
            "transport_entropy": 0.0,
            "analogical_map_by_radius": zeros.tolist(),
            "directed_map_by_radius": zeros.tolist(),
            "exact_h0_dim_mean": 0.0,
            "exact_h1_dim_mean": 0.0,
            "exact_h0_map_rank_mean": 0.0,
            "exact_h1_map_rank_mean": 0.0,
            "variety_complex_residual_mean": 0.0,
            "fitting_minor_rank_residual_mean": 0.0,
            "buchsbaum_eisenbud_rank_residual_mean": 0.0,
            "buchsbaum_eisenbud_multiplier_residual_mean": 0.0,
            "multigraded_betti_mass_mean": 0.0,
            "exact_morphism_radius_shift_mean": 0.0,
            "exact_morphism_edge_validity_mean": 0.0,
            "exact_morphism_triangle_validity_mean": 0.0,
            "exact_directed_edge_validity_mean": 0.0,
            "exact_morphism_computed": 0.0,
            "exact_morphism_truncated_complexes": 0.0,
            "betti0": zeros.tolist(),
            "cycle_rank": zeros.tolist(),
            "hdbscan_cluster_count": zeros.tolist(),
            "hdbscan_noise_fraction": zeros.tolist(),
            "hdbscan_stability": zeros.tolist(),
            "hdbscan_persistent_edge_density": zeros.tolist(),
            "hdbscan_core_radius": 0.0,
            "skew_norm": 0.0,
            "distance": np.zeros((1, 1), dtype=float),
            "mutual_reachability": np.zeros((1, 1), dtype=float),
            "hdbscan_persistence_adjacency": np.zeros((1, 1), dtype=float),
            "skew": np.zeros((1, 1), dtype=float),
            "directed_adjacency": [],
            "step_radius_heatmaps": {},
            "simplex_tree_summary": [],
        }

    x = hidden.astype(np.float32)
    starts = _window_starts(int(x.shape[0]), cfg)
    radii = np.linspace(float(cfg.radius_min), float(cfg.radius_max), max(2, int(cfg.levels)))
    aggregate = {name: [] for name in (
        "edge_density",
        "triangle_density",
        "directed_edge_density",
        "directed_asymmetry",
        "directed_cycle_flux",
        "directed_transitive_loss",
        "directed_chain_commutator",
        "inclusion_violation",
        "boundary_residual",
        "dirichlet_energy",
        "dec_conservation_loss",
        "dec_mass_residual",
        "dec_vorticity_drift",
        "dec_kinetic_energy",
        "dec_kinetic_energy_drift",
        "dec_hodge_balance",
        "dec_wedge_interior_residual",
        "betti0",
        "cycle_rank",
        "hdbscan_cluster_count",
        "hdbscan_noise_fraction",
        "hdbscan_stability",
        "hdbscan_persistent_edge_density",
    )}
    heat_edge = []
    heat_cycle = []
    heat_betti0 = []
    transition_map_heat = []
    transition_directed_map_heat = []
    simplex_tree_summary: list[dict[str, float]] = []
    snapshots: list[dict[str, Any]] = []
    best_matrices: dict[str, Any] | None = None
    for window_id, start in enumerate(starts):
        points = x[start : start + max(4, min(int(cfg.window_size), x.shape[0]))]
        if points.shape[0] > int(cfg.max_points):
            idx = np.linspace(0, points.shape[0] - 1, num=int(cfg.max_points)).round().astype(int)
            points = points[idx]
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / np.maximum(norms, 1e-8)
        n = int(points.shape[0])
        if n < 4:
            continue
        diff = points[:, None, :] - points[None, :, :]
        dist = np.linalg.norm(diff, axis=-1)
        positive = dist[dist > 1e-8]
        dist = dist / max(float(np.median(positive)) if positive.size else 1.0, 1e-6)
        eye = np.eye(n, dtype=float)
        half = max(1, points.shape[1] // 2)
        left = points[:, :half]
        right = points[:, half : half + half]
        if right.shape[1] < left.shape[1]:
            right = np.pad(right, ((0, 0), (0, left.shape[1] - right.shape[1])), mode="constant")
        skew = left @ right.T - right @ left.T
        skew_abs = np.abs(skew[np.triu_indices(n, k=1)])
        skew = np.clip(skew / max(float(np.median(skew_abs)) if skew_abs.size else 1.0, 1e-6), -4.0, 4.0) * float(cfg.skew_scale)
        time_index = np.arange(n, dtype=float)
        time_orientation = np.sign(time_index[None, :] - time_index[:, None]) * float(cfg.time_bias)
        masked_dist = dist + eye * 1.0e6
        core_k = min(max(1, int(cfg.hdbscan_min_samples)), n - 1)
        core_radius = np.partition(masked_dist, kth=core_k - 1, axis=1)[:, core_k - 1]
        mutual = np.maximum(dist, np.maximum(core_radius[:, None], core_radius[None, :])) + eye * 1.0e6
        hdbscan_persistence = np.zeros((n, n), dtype=float)
        prev_sym = None
        prev_dir = None
        sym_levels = []
        directed_adjacencies = []
        exact_levels = []
        per_window = {name: [] for name in aggregate}
        prev_vorticity = None
        prev_kinetic = None
        for radius in radii:
            sym = sigmoid_np((radius - dist) / max(float(cfg.temperature), 1e-6)) * (1.0 - eye)
            directed = sigmoid_np((radius - dist + skew + time_orientation) / max(float(cfg.temperature), 1e-6)) * (1.0 - eye)
            density = sigmoid_np((radius - mutual) / max(float(cfg.temperature), 1e-6)) * (1.0 - eye)
            hdbscan_persistence += density / max(1, len(radii))
            hard = (sym > 0.5)
            hard_sym = hard | hard.T
            components, sizes = _component_count(hard_sym)
            edges = float(np.triu(hard_sym, k=1).sum())
            tri_count = 0
            for i in range(n):
                for j in range(i + 1, n):
                    if not hard_sym[i, j]:
                        continue
                    for k in range(j + 1, n):
                        if hard_sym[i, k] and hard_sym[j, k]:
                            tri_count += 1
            cycle_rank = max(0.0, edges - float(n) + float(components) - float(tri_count))
            stable_components = [size for size in sizes if size >= max(2, int(cfg.hdbscan_min_cluster_size))]
            stable_nodes = sum(stable_components)
            rows = directed + eye
            rows = rows / np.maximum(rows.sum(axis=1, keepdims=True), 1e-8)
            if prev_dir is None:
                chain = 0.0
            else:
                prev_rows = prev_dir + eye
                prev_rows = prev_rows / np.maximum(prev_rows.sum(axis=1, keepdims=True), 1e-8)
                chain = float(np.mean((prev_rows @ rows - rows @ prev_rows) ** 2))
            inclusion = 0.0 if prev_sym is None else float(np.mean(np.maximum(0.0, prev_sym - sym) ** 2))
            directed_square = directed @ directed
            directed_transitive = float(np.mean(np.maximum(0.0, directed_square - directed) ** 2))
            cycle_forward = float(np.einsum("ij,jk,ki->", directed, directed, directed) / max(1, n * (n - 1) * (n - 2)))
            cycle_backward = float(np.einsum("ji,kj,ik->", directed, directed, directed) / max(1, n * (n - 1) * (n - 2)))
            boundary = float(np.mean(np.maximum(0.0, np.einsum("ij,jk,ik->ijk", directed, directed, directed) - directed[:, :, None] - directed[:, None, :] - directed[None, :, :]) ** 2))
            weights = (sym + sym.T) * 0.5
            dirichlet = float((weights * dist**2).sum() / max(float(weights.sum()), 1e-8))
            flow = directed - directed.T
            divergence = flow.sum(axis=-1) / max(1, n - 1)
            dec_mass = float(np.mean(divergence**2))
            degree_weight = np.maximum(sym.sum(axis=-1), 1e-6)
            vorticity_node = (flow * sym).sum(axis=-1) / degree_weight
            edge_mass = max(float(sym.sum()), 1e-8)
            dec_kinetic = float((sym * flow**2).sum() / edge_mass)
            if prev_vorticity is None:
                dec_vorticity_drift = 0.0
                dec_kinetic_drift = 0.0
            else:
                dec_vorticity_drift = float(np.mean((vorticity_node - prev_vorticity) ** 2))
                dec_kinetic_drift = float((dec_kinetic - prev_kinetic) ** 2)
            hodge_edge = (1.0 / np.maximum(core_radius[:, None] + core_radius[None, :], 1e-6)) * (1.0 - eye)
            positive_hodge = hodge_edge[hodge_edge > 0]
            hodge_edge = hodge_edge / max(float(positive_hodge.mean()) if positive_hodge.size else 1.0, 1e-6)
            hodge_energy = float((hodge_edge * sym * flow**2).sum() / max(float((hodge_edge * sym).sum()), 1e-8))
            dec_hodge = float((hodge_energy / max(abs(dec_kinetic), 1e-8) - 1.0) ** 2)
            avg_vorticity = 0.5 * (vorticity_node[:, None] + vorticity_node[None, :])
            wedge_form = flow * avg_vorticity
            interior_form = directed * vorticity_node[None, :] - directed.T * vorticity_node[:, None]
            dec_wedge = float((sym * (wedge_form - interior_form) ** 2).sum() / edge_mass)
            dec_conservation = (
                dec_mass
                + 0.25 * dec_vorticity_drift
                + 0.10 * dec_kinetic_drift
                + 0.05 * dec_hodge
                + 0.05 * dec_wedge
            )
            prev_vorticity = vorticity_node.copy()
            prev_kinetic = dec_kinetic
            per_window["edge_density"].append(float(sym.sum() / max(1, n * (n - 1))))
            per_window["triangle_density"].append(float(np.einsum("ij,jk,ik->", sym, sym, sym) / max(1, n * (n - 1) * (n - 2))))
            per_window["directed_edge_density"].append(float(directed.sum() / max(1, n * (n - 1))))
            per_window["directed_asymmetry"].append(float(np.mean(np.abs(directed - directed.T))))
            per_window["directed_cycle_flux"].append(abs(cycle_forward - cycle_backward))
            per_window["directed_transitive_loss"].append(directed_transitive)
            per_window["directed_chain_commutator"].append(chain)
            per_window["inclusion_violation"].append(inclusion)
            per_window["boundary_residual"].append(boundary)
            per_window["dirichlet_energy"].append(dirichlet)
            per_window["dec_conservation_loss"].append(float(dec_conservation))
            per_window["dec_mass_residual"].append(float(dec_mass))
            per_window["dec_vorticity_drift"].append(float(dec_vorticity_drift))
            per_window["dec_kinetic_energy"].append(float(dec_kinetic))
            per_window["dec_kinetic_energy_drift"].append(float(dec_kinetic_drift))
            per_window["dec_hodge_balance"].append(float(dec_hodge))
            per_window["dec_wedge_interior_residual"].append(float(dec_wedge))
            per_window["betti0"].append(float(components))
            per_window["cycle_rank"].append(float(cycle_rank))
            per_window["hdbscan_cluster_count"].append(float(len(stable_components)))
            per_window["hdbscan_noise_fraction"].append(float(1.0 - stable_nodes / max(1, n)))
            per_window["hdbscan_stability"].append(float(stable_nodes / max(1, n)))
            per_window["hdbscan_persistent_edge_density"].append(float(density.sum() / max(1, n * (n - 1))))
            sym_levels.append(sym)
            directed_adjacencies.append(directed)
            exact_levels.append(_build_flag_complex_f2(hard_sym))
            simplex_tree_summary.append(
                {
                    "window": float(window_id),
                    "radius": float(radius),
                    "vertices": float(n),
                    "edges": edges,
                    "triangles": float(tri_count),
                    "betti0": float(components),
                    "cycle_rank": float(cycle_rank),
                }
            )
            prev_sym = sym
            prev_dir = directed
        for name, values in per_window.items():
            aggregate[name].append(values)
        heat_edge.append(per_window["edge_density"])
        heat_cycle.append(per_window["cycle_rank"])
        heat_betti0.append(per_window["betti0"])
        snapshots.append(
            {
                "points": points,
                "sym_levels": sym_levels,
                "directed_levels": directed_adjacencies,
                "exact_levels": exact_levels,
            }
        )
        if best_matrices is None:
            best_matrices = {
                "distance": dist,
                "mutual_reachability": mutual,
                "hdbscan_persistence_adjacency": hdbscan_persistence,
                "skew": skew,
                "directed_adjacency": directed_adjacencies,
                "hdbscan_core_radius": float(np.mean(core_radius)),
                "skew_norm": float(np.mean(np.abs(skew))),
            }
    if best_matrices is None:
        return directed_step_filtration_stats_np(np.zeros((0, max(1, hidden.shape[-1])), dtype=np.float32), config=cfg)
    result: dict[str, Any] = {"radii": radii.tolist(), "step_window_index": list(range(len(heat_edge)))}
    for name, values in aggregate.items():
        matrix = np.asarray(values, dtype=float)
        result[name] = matrix.mean(axis=0).tolist()
    map_values = []
    directed_map_values = []
    entropy_values = []
    for source, target in zip(snapshots[:-1], snapshots[1:]):
        src = source["points"]
        tgt = target["points"]
        logits = -np.linalg.norm(src[:, None, :] - tgt[None, :, :], axis=-1) / max(float(cfg.temperature), 1e-6)
        logits = logits - logits.max(axis=-1, keepdims=True)
        transport = np.exp(logits)
        transport = transport / np.maximum(transport.sum(axis=-1, keepdims=True), 1e-8)
        entropy = -np.sum(transport * np.log(np.maximum(transport, 1e-8)), axis=-1).mean()
        entropy_values.append(float(entropy / math.log(max(2, tgt.shape[0]))))
        per_level = []
        per_level_directed = []
        for sym_source, sym_target in zip(source["sym_levels"], target["sym_levels"]):
            pushed = transport.T @ sym_source @ transport
            per_level.append(float(np.mean((pushed - sym_target) ** 2)))
        for directed_source, directed_target in zip(source["directed_levels"], target["directed_levels"]):
            pushed_dir = transport.T @ directed_source @ transport
            per_level_directed.append(float(np.mean((pushed_dir - directed_target) ** 2)))
        if per_level:
            map_values.append(per_level)
            transition_map_heat.append(per_level)
        if per_level_directed:
            directed_map_values.append(per_level_directed)
            transition_directed_map_heat.append(per_level_directed)
    if map_values:
        map_matrix = np.asarray(map_values, dtype=float)
        directed_map_matrix = np.asarray(directed_map_values, dtype=float)
        result["analogical_map_by_radius"] = map_matrix.mean(axis=0).tolist()
        result["directed_map_by_radius"] = directed_map_matrix.mean(axis=0).tolist()
        result["analogical_map_loss"] = float(map_matrix.mean())
        result["directed_map_loss"] = float(directed_map_matrix.mean())
    else:
        result["analogical_map_by_radius"] = np.zeros_like(radii, dtype=float).tolist()
        result["directed_map_by_radius"] = np.zeros_like(radii, dtype=float).tolist()
        result["analogical_map_loss"] = 0.0
        result["directed_map_loss"] = 0.0
    result["transport_entropy"] = float(np.mean(entropy_values)) if entropy_values else 0.0
    result.update(_exact_persistence_morphism_audit(snapshots, levels=max(2, int(cfg.levels))))
    result.update(best_matrices)
    result["step_radius_heatmaps"] = {
        "edge_density": np.asarray(heat_edge, dtype=float),
        "cycle_rank": np.asarray(heat_cycle, dtype=float),
        "betti0": np.asarray(heat_betti0, dtype=float),
        "dec_conservation_loss": np.asarray(aggregate["dec_conservation_loss"], dtype=float),
        "dec_mass_residual": np.asarray(aggregate["dec_mass_residual"], dtype=float),
        "dec_vorticity_drift": np.asarray(aggregate["dec_vorticity_drift"], dtype=float),
        "dec_kinetic_energy": np.asarray(aggregate["dec_kinetic_energy"], dtype=float),
        "dec_kinetic_energy_drift": np.asarray(aggregate["dec_kinetic_energy_drift"], dtype=float),
        "dec_hodge_balance": np.asarray(aggregate["dec_hodge_balance"], dtype=float),
        "dec_wedge_interior_residual": np.asarray(aggregate["dec_wedge_interior_residual"], dtype=float),
        "analogical_map_loss": np.asarray(transition_map_heat, dtype=float),
        "directed_map_loss": np.asarray(transition_directed_map_heat, dtype=float),
        "exact_h0_dims": result["exact_h0_dims_by_window_radius"],
        "exact_h1_dims": result["exact_h1_dims_by_window_radius"],
        "variety_complex_residual": result["variety_complex_residual_by_window_radius"],
        "fitting_minor_rank_residual": result["fitting_minor_rank_residual_by_window_radius"],
        "buchsbaum_eisenbud_rank_residual": result["buchsbaum_eisenbud_rank_residual_by_window_radius"],
        "buchsbaum_eisenbud_multiplier_residual": result[
            "buchsbaum_eisenbud_multiplier_residual_by_window_radius"
        ],
        "multigraded_betti_mass": result["multigraded_betti_mass_by_window_radius"],
        "exact_h0_map_rank": result["exact_h0_map_rank_by_transition_radius"],
        "exact_h1_map_rank": result["exact_h1_map_rank_by_transition_radius"],
        "exact_radius_shift": result["exact_morphism_radius_shift_by_transition"],
        "exact_edge_validity": result["exact_morphism_edge_validity_by_transition"],
        "exact_triangle_validity": result["exact_morphism_triangle_validity_by_transition"],
        "exact_directed_edge_validity": result["exact_directed_edge_validity_by_transition"],
    }
    result["simplex_tree_summary"] = simplex_tree_summary
    return result
