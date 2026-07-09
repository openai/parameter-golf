"""Differentiable losses that consume exact CAS certificates.

The functions here do not compute algebraic facts.  They only consume exact
targets from a certificate and turn them into PyTorch losses.  Missing exact
targets are errors by default.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from .cas_certificates import validate_certificate_payload


def _zero_like(reference: torch.Tensor) -> torch.Tensor:
    return reference.float().sum() * 0.0


def _certificate_payload(certificate: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(certificate, dict):
        raise TypeError("certificate must be a dict")
    provenance = certificate.get("provenance")
    if provenance in {"surrogate_torch", None}:
        raise ValueError(f"certificate provenance is not exact enough for CAS-backed loss: {provenance}")
    return certificate


def _get_path(payload: dict[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = payload
    for key in path:
        if not isinstance(cursor, dict) or key not in cursor:
            raise KeyError("missing certificate field: " + ".".join(path))
        cursor = cursor[key]
    return cursor


def load_exact_certificate(path: str | Path) -> dict[str, Any]:
    """Load and validate an exact CAS/closed-form certificate from JSON."""

    certificate_path = Path(path)
    payload = json.loads(certificate_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"certificate JSON must contain an object: {certificate_path}")
    errors = validate_certificate_payload(payload)
    if errors:
        raise ValueError(f"invalid certificate {certificate_path}: {'; '.join(errors)}")
    return _certificate_payload(payload)


def load_exact_certificates(paths: list[str | Path] | tuple[str | Path, ...]) -> list[dict[str, Any]]:
    """Load multiple exact certificates with strict validation."""

    return [load_exact_certificate(path) for path in paths]


def relation_tensor_from_certificate(
    certificate: dict[str, Any],
    *,
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """Return integer binomial relations from an exact toric-ideal certificate.

    The accepted row formats are either:

    - `[i, j, k, l]`, representing `ell_i + ell_j = ell_k + ell_l`;
    - `{"positive": [[idx, coeff], ...], "negative": [[idx, coeff], ...]}`.

    The dict form is returned as a dense signed relation matrix `[R, C]`.
    The four-column form is returned as `LongTensor[R, 4]`.
    """

    payload = _certificate_payload(certificate)
    algebra = _get_path(payload, ("commutative_algebra",))
    rows = algebra.get("toric_ideal_relations") or algebra.get("binomial_relations")
    if not rows:
        raise KeyError("certificate has no commutative_algebra.toric_ideal_relations")
    if all(isinstance(row, (list, tuple)) and len(row) == 4 for row in rows):
        return torch.tensor(rows, dtype=torch.long, device=device)
    max_index = -1
    parsed: list[dict[str, list[tuple[int, float]]]] = []
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError("mixed or unsupported binomial relation row format")
        positive = [(int(i), float(c)) for i, c in row.get("positive", [])]
        negative = [(int(i), float(c)) for i, c in row.get("negative", [])]
        for index, _coeff in positive + negative:
            max_index = max(max_index, index)
        parsed.append({"positive": positive, "negative": negative})
    if max_index < 0:
        raise ValueError("empty binomial relation rows")
    matrix = torch.zeros(len(parsed), max_index + 1, dtype=torch.float32, device=device)
    for row_idx, row in enumerate(parsed):
        for index, coeff in row["positive"]:
            matrix[row_idx, index] += coeff
        for index, coeff in row["negative"]:
            matrix[row_idx, index] -= coeff
    return matrix


def toric_binomial_loss_from_certificate(
    logits: torch.Tensor,
    certificate: dict[str, Any],
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Binomial consistency loss with exact relations from a certificate.

    `logits` is `[B, L, C]` or `[N, C]`.  Four-column relations compute
    `ell_i + ell_j - ell_k - ell_l`.  Dense signed relation matrices compute
    `logits @ relation.T`.
    """

    if logits.ndim not in {2, 3}:
        raise ValueError("logits must be [N,C] or [B,L,C]")
    rel = relation_tensor_from_certificate(certificate, device=logits.device)
    values: torch.Tensor
    if rel.ndim == 2 and rel.dtype == torch.long and rel.shape[-1] == 4:
        max_col = int(rel.max().item()) if rel.numel() else -1
        if max_col >= logits.shape[-1]:
            raise ValueError(f"relation index {max_col} exceeds logits width {logits.shape[-1]}")
        values = logits[..., rel[:, 0]] + logits[..., rel[:, 1]] - logits[..., rel[:, 2]] - logits[..., rel[:, 3]]
    elif rel.ndim == 2:
        if rel.shape[-1] > logits.shape[-1]:
            raise ValueError(f"relation width {rel.shape[-1]} exceeds logits width {logits.shape[-1]}")
        rel_float = rel.to(device=logits.device, dtype=logits.dtype)
        values = torch.matmul(logits[..., : rel.shape[-1]], rel_float.transpose(0, 1))
    else:
        raise ValueError("relation tensor must be rank 2")
    loss = values.float().square().mean()
    if normalize:
        loss = loss / logits.detach().float().square().mean().clamp_min(1e-6)
    return loss


def balanced_cycle_loss_from_certificate(
    predicted_multiplicities: torch.Tensor,
    certificate: dict[str, Any],
) -> torch.Tensor:
    """Tropical balancing loss from exact incidence and primitive normals.

    Certificate field:

    ```json
    tropical.balancing_stars = [
      {
        "facets": [0, 1, 2],
        "primitive_normals": [[1,0], [-1,1], [0,-1]]
      }
    ]
    ```
    """

    payload = _certificate_payload(certificate)
    stars = _get_path(payload, ("tropical",)).get("balancing_stars")
    if not stars:
        raise KeyError("certificate has no tropical.balancing_stars")
    pred = predicted_multiplicities.float()
    losses: list[torch.Tensor] = []
    for star in stars:
        facets = torch.tensor(star["facets"], dtype=torch.long, device=pred.device)
        normals = torch.tensor(star["primitive_normals"], dtype=pred.dtype, device=pred.device)
        if facets.numel() != normals.shape[0]:
            raise ValueError("balancing star facets and normals have inconsistent lengths")
        if int(facets.max().item()) >= pred.shape[-1]:
            raise ValueError("balancing star facet index exceeds predicted multiplicity width")
        weights = pred.index_select(-1, facets)
        residual = (weights.unsqueeze(-1) * normals).sum(dim=-2)
        losses.append(residual.square().mean())
    return torch.stack(losses).mean() if losses else _zero_like(predicted_multiplicities)


def cartier_bend_loss_from_certificate(
    predicted_slopes: torch.Tensor,
    certificate: dict[str, Any],
) -> torch.Tensor:
    """Cartier bend loss from exact wall normals and bend targets.

    Certificate field:

    ```json
    toric.cartier_wall_bends = [
      {"left": 0, "right": 1, "normal": [1, -1], "bend": 2}
    ]
    ```

    `predicted_slopes` is `[C, D]` or `[B, C, D]`.
    """

    payload = _certificate_payload(certificate)
    rows = _get_path(payload, ("toric",)).get("cartier_wall_bends")
    if not rows:
        raise KeyError("certificate has no toric.cartier_wall_bends")
    slopes = predicted_slopes.float()
    if slopes.ndim not in {2, 3}:
        raise ValueError("predicted_slopes must be [C,D] or [B,C,D]")
    losses: list[torch.Tensor] = []
    for row in rows:
        left = int(row["left"])
        right = int(row["right"])
        normal = torch.tensor(row["normal"], dtype=slopes.dtype, device=slopes.device)
        bend = torch.as_tensor(float(row["bend"]), dtype=slopes.dtype, device=slopes.device)
        if max(left, right) >= slopes.shape[-2]:
            raise ValueError("Cartier bend cell index exceeds predicted slope count")
        if normal.numel() != slopes.shape[-1]:
            raise ValueError("Cartier wall normal dimension does not match slope dimension")
        actual = ((slopes[..., left, :] - slopes[..., right, :]) * normal).sum(dim=-1)
        losses.append((actual - bend).square().mean())
    return torch.stack(losses).mean() if losses else _zero_like(predicted_slopes)


def cone_label_loss_from_certificate(
    cone_logits: torch.Tensor,
    cone_labels: torch.Tensor,
    certificate: dict[str, Any],
) -> torch.Tensor:
    """Cross entropy for cone labels certified by an exact fan certificate."""

    _certificate_payload(certificate)
    expected_cones = _get_path(certificate, ("toric",)).get("maximal_cones")
    if not expected_cones:
        raise KeyError("certificate has no toric.maximal_cones")
    if cone_logits.shape[-1] < len(expected_cones):
        raise ValueError("cone_logits has fewer classes than certified maximal cones")
    return F.cross_entropy(cone_logits.reshape(-1, cone_logits.shape[-1]), cone_labels.reshape(-1).long())


def koszul_betti_vector_from_certificate(
    certificate: dict[str, Any],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor:
    """Return exact free ranks/Betti ranks from a Koszul/free-resolution cert."""

    payload = _certificate_payload(certificate)
    algebra = _get_path(payload, ("commutative_algebra",))
    ranks = algebra.get("free_module_ranks")
    if ranks is None:
        rows = algebra.get("betti_rows")
        if not rows:
            raise KeyError("certificate has no commutative_algebra.free_module_ranks or betti_rows")
        max_degree = max(int(row.get("homological_degree", -1)) for row in rows if isinstance(row, dict))
        if max_degree < 0:
            raise ValueError("certificate betti_rows do not contain homological_degree fields")
        ranks = [0.0 for _ in range(max_degree + 1)]
        for row in rows:
            if not isinstance(row, dict):
                continue
            degree = int(row["homological_degree"])
            ranks[degree] += float(row.get("rank", 0.0))
    if not isinstance(ranks, list) or not ranks:
        raise ValueError("certificate free_module_ranks must be a nonempty list")
    return torch.tensor([float(value) for value in ranks], dtype=dtype, device=device)


def koszul_betti_loss_from_certificate(
    predicted_betti: torch.Tensor,
    certificate: dict[str, Any],
    *,
    normalize: bool = True,
) -> torch.Tensor:
    """Profile loss against exact CAS free ranks/Betti ranks.

    `predicted_betti` may be `[D]`, `[B,D]`, or any tensor whose final axis is
    homological degree.  The certificate supplies exact ranks, for example
    `[1, 3, 3, 1]` for the Koszul resolution of `QQ[x,y,z]/(x,y,z)`.
    """

    pred = predicted_betti.float()
    target = koszul_betti_vector_from_certificate(certificate, device=pred.device, dtype=pred.dtype)
    if pred.shape[-1] < target.numel():
        raise ValueError(f"predicted_betti width {pred.shape[-1]} is smaller than target width {target.numel()}")
    residual = pred[..., : target.numel()] - target
    loss = residual.square().mean()
    if normalize:
        loss = loss / target.square().mean().clamp_min(1e-6)
    return loss
