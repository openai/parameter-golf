"""Exact CAS certificate schemas and cache helpers.

This module is intentionally independent from PyTorch.  It stores finite
certificates emitted by exact backends such as SageMath, Macaulay2, gfan, or a
closed-form toy certificate implemented in this repository.  Differentiable
training losses may consume these certificates, but the certificates themselves
are provenance-carrying algebraic data.
"""

from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal


CertificateProvenance = Literal[
    "exact_cas/macaulay2",
    "exact_cas/sage",
    "exact_cas/gfan",
    "exact_closed_form",
    "surrogate_torch",
    "surrogate_torch_validated_against_cas",
    "cas_unavailable",
]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items(), key=lambda kv: str(kv[0]))}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return str(value)


def canonical_json(value: Any) -> str:
    """Serialize a value with stable ordering for certificate hashes."""

    return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"), ensure_ascii=True)


def stable_hash(value: Any, *, digest_size: int = 20) -> str:
    payload = canonical_json(value).encode("utf-8")
    return hashlib.blake2b(payload, digest_size=digest_size).hexdigest()


def utc_timestamp() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


@dataclass(frozen=True)
class CASBackendInfo:
    """Backend provenance embedded in every exact certificate."""

    backend: str
    executable: str | None
    version: str | None
    packages: tuple[str, ...] = ()
    package_status: dict[str, bool] = field(default_factory=dict)
    available: bool = False
    provenance: CertificateProvenance = "cas_unavailable"
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ToricTropicalCertificate:
    """Machine-readable finite toric/tropical/CCA certificate."""

    kind: str
    input_hash: str
    provenance: CertificateProvenance
    created_at_utc: str = field(default_factory=utc_timestamp)
    source: dict[str, Any] = field(default_factory=dict)
    cas: dict[str, Any] = field(default_factory=dict)
    toric: dict[str, Any] = field(default_factory=dict)
    tropical: dict[str, Any] = field(default_factory=dict)
    commutative_algebra: dict[str, Any] = field(default_factory=dict)
    tensors: dict[str, str] = field(default_factory=dict)
    diagnostics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _json_safe(asdict(self))

    @property
    def certificate_hash(self) -> str:
        payload = self.to_dict()
        payload.pop("created_at_utc", None)
        return stable_hash(payload)

    def with_hash(self) -> dict[str, Any]:
        payload = self.to_dict()
        payload["certificate_hash"] = self.certificate_hash
        return payload


class CertificateCache:
    """JSON certificate cache with deterministic filenames."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def path_for_hash(self, certificate_hash: str) -> Path:
        prefix = certificate_hash[:2] if len(certificate_hash) >= 2 else "xx"
        return self.root / prefix / f"{certificate_hash}.json"

    def write(self, certificate: ToricTropicalCertificate) -> Path:
        certificate_hash = certificate.certificate_hash
        path = self.path_for_hash(certificate_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(certificate.with_hash(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        return path

    def read_hash(self, certificate_hash: str) -> dict[str, Any]:
        path = self.path_for_hash(certificate_hash)
        return json.loads(path.read_text(encoding="utf-8"))

    def find_by_input_hash(self, input_hash: str, *, kind: str | None = None) -> list[Path]:
        matches: list[Path] = []
        for path in sorted(self.root.glob("*/*.json")):
            try:
                payload = json.loads(path.read_text(encoding="utf-8"))
            except Exception:
                continue
            if payload.get("input_hash") != input_hash:
                continue
            if kind is not None and payload.get("kind") != kind:
                continue
            matches.append(path)
        return matches


def validate_certificate_payload(payload: dict[str, Any]) -> list[str]:
    """Return validation errors for a certificate payload."""

    errors: list[str] = []
    required = ("kind", "input_hash", "provenance", "created_at_utc", "source", "cas")
    for key in required:
        if key not in payload:
            errors.append(f"missing required field: {key}")
    provenance = payload.get("provenance")
    allowed = set(CertificateProvenance.__args__)  # type: ignore[attr-defined]
    if provenance not in allowed:
        errors.append(f"invalid provenance: {provenance}")
    if provenance == "surrogate_torch":
        errors.append("surrogate_torch payload is not an exact certificate")
    if payload.get("certificate_hash"):
        copied = dict(payload)
        claimed = str(copied.pop("certificate_hash"))
        copied.pop("created_at_utc", None)
        actual = stable_hash(copied)
        if claimed != actual:
            errors.append(f"certificate_hash mismatch: claimed {claimed}, actual {actual}")
    if payload.get("kind") == "sage_normal_fan_certificate" and provenance == "exact_cas/sage":
        toric = payload.get("toric", {})
        if not isinstance(toric, dict):
            errors.append("sage_normal_fan_certificate missing toric payload")
        else:
            for key in (
                "num_one_dimensional_cones",
                "fan_one_dimensional_cones",
                "maximal_cones",
                "cone_dimensions",
                "face_incidence",
                "orbit_strata",
                "fan_properties",
                "cone_containment_checks",
                "fan_refinement_checks",
            ):
                if key not in toric:
                    errors.append(f"sage_normal_fan_certificate missing toric.{key}")
            if toric.get("num_one_dimensional_cones") != toric.get("num_rays"):
                errors.append("sage_normal_fan_certificate one-dimensional cone count differs from compatibility ray count")
            if toric.get("fan_one_dimensional_cones") != toric.get("fan_rays"):
                errors.append("sage_normal_fan_certificate one-dimensional cone list differs from compatibility ray list")
            containment = toric.get("cone_containment_checks", {})
            if isinstance(containment, dict) and containment.get("all_maximal_indices_are_one_dimensional_cones") is not True:
                errors.append("sage_normal_fan_certificate cone containment check failed")
            properties = toric.get("fan_properties", {})
            if isinstance(properties, dict) and properties.get("is_complete") is not True:
                errors.append("sage_normal_fan_certificate normal fan is not complete")
    if payload.get("kind") == "macaulay2_toric_vector_bundle_certificate" and provenance == "exact_cas/macaulay2":
        toric = payload.get("toric", {})
        algebra = payload.get("commutative_algebra", {})
        if not isinstance(toric, dict):
            errors.append("macaulay2_toric_vector_bundle_certificate missing toric payload")
            toric = {}
        if not isinstance(algebra, dict):
            errors.append("macaulay2_toric_vector_bundle_certificate missing commutative_algebra payload")
            algebra = {}
        if algebra.get("is_vector_bundle") is not True:
            errors.append("macaulay2_toric_vector_bundle_certificate is_vector_bundle check failed")
        for key in ("one_dimensional_cones", "maximal_cones"):
            if key not in toric:
                errors.append(f"macaulay2_toric_vector_bundle_certificate missing toric.{key}")
        for key in (
            "structured_certificate_provenance",
            "one_dimensional_cone_filtrations",
            "chart_weights",
            "transition_matrices",
            "cech_cocycle_checks",
            "chart_overlap_checks",
            "cohomology_summary",
        ):
            if key not in algebra:
                errors.append(f"macaulay2_toric_vector_bundle_certificate missing commutative_algebra.{key}")
        cocycles = algebra.get("cech_cocycle_checks", {})
        if isinstance(cocycles, dict) and any(value is not True for value in cocycles.values()):
            errors.append("macaulay2_toric_vector_bundle_certificate Cech cocycle check failed")
    return errors


def exact_closed_form_certificate(
    *,
    kind: str,
    source: dict[str, Any],
    toric: dict[str, Any] | None = None,
    tropical: dict[str, Any] | None = None,
    commutative_algebra: dict[str, Any] | None = None,
    diagnostics: dict[str, Any] | None = None,
) -> ToricTropicalCertificate:
    """Build an exact certificate from a finite closed-form construction."""

    payload_for_hash = {
        "kind": kind,
        "source": source,
        "toric": toric or {},
        "tropical": tropical or {},
        "commutative_algebra": commutative_algebra or {},
        "diagnostics": diagnostics or {},
    }
    return ToricTropicalCertificate(
        kind=kind,
        input_hash=stable_hash(payload_for_hash),
        provenance="exact_closed_form",
        source=source,
        cas={
            "backend": "closed_form",
            "available": True,
            "version": "toricgt.symbolic_multigraded_resolution",
            "packages": [],
        },
        toric=toric or {},
        tropical=tropical or {},
        commutative_algebra=commutative_algebra or {},
        diagnostics=diagnostics or {},
    )
