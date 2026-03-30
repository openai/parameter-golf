from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class RunnerConfig:
    workdir: Path
    script_path: Path
    python_bin: Path
    activate_script: Path | None
    gpus: int
    logs_dir: Path


@dataclass(frozen=True)
class ParamSpec:
    distribution: str
    min: float
    max: float
    scale: float | str = "auto"
    round_to: float | None = None


@dataclass(frozen=True)
class SearchSettings:
    seed: int
    max_runs: int
    warm_start_suggestions: int
    candidate_samples: int
    max_observations: int
    prune_pareto: bool
    suggestions_per_center: int
    gp_alpha: float
    default_proxy_int6_gap: float
    target_cost_ratios: tuple[float, ...]
    output_root: Path
    run_id_prefix: str


@dataclass(frozen=True)
class SearchConfig:
    path: Path
    name: str
    runner: RunnerConfig
    fixed_env: dict[str, Any]
    search_space: dict[str, ParamSpec]
    search: SearchSettings


def _require_mapping(data: Any, label: str) -> dict[str, Any]:
    if not isinstance(data, dict):
        raise TypeError(f"{label} must be a mapping")
    return data


def _resolve_path(raw: str | None, *, base: Path) -> Path | None:
    if raw in (None, ""):
        return None
    path = Path(raw)
    return path if path.is_absolute() else (base / path).resolve()


def load_search_config(path: str | Path) -> SearchConfig:
    config_path = Path(path).resolve()
    config_dir = config_path.parent
    data = _require_mapping(
        yaml.safe_load(config_path.read_text(encoding="utf-8")),
        "search config",
    )

    runner_raw = _require_mapping(data.get("runner"), "runner")
    workdir = _resolve_path(str(runner_raw["workdir"]), base=config_dir)
    if workdir is None:
        raise ValueError("runner.workdir is required")
    script_path = Path(str(runner_raw["script_path"]))
    if not script_path.is_absolute():
        script_path = (workdir / script_path).resolve()
    python_bin = _resolve_path(str(runner_raw["python_bin"]), base=config_dir)
    if python_bin is None:
        raise ValueError("runner.python_bin is required")
    activate_script = _resolve_path(runner_raw.get("activate_script"), base=config_dir)
    logs_dir = Path(str(runner_raw.get("logs_dir", "logs")))
    if not logs_dir.is_absolute():
        logs_dir = (workdir / logs_dir).resolve()
    runner = RunnerConfig(
        workdir=workdir,
        script_path=script_path,
        python_bin=python_bin,
        activate_script=activate_script,
        gpus=int(runner_raw.get("gpus", 1)),
        logs_dir=logs_dir,
    )

    fixed_env = _require_mapping(data.get("fixed_env", {}), "fixed_env")

    search_space_raw = _require_mapping(data.get("search_space"), "search_space")
    search_space: dict[str, ParamSpec] = {}
    for name, raw_spec in search_space_raw.items():
        spec = _require_mapping(raw_spec, f"search_space.{name}")
        search_space[name] = ParamSpec(
            distribution=str(spec["distribution"]),
            min=float(spec["min"]),
            max=float(spec["max"]),
            scale=spec.get("scale", "auto"),
            round_to=float(spec["round_to"]) if spec.get("round_to") is not None else None,
        )

    search_raw = _require_mapping(data.get("search"), "search")
    output_root = Path(str(search_raw.get("output_root", f"search_runs/{config_path.stem}")))
    if not output_root.is_absolute():
        output_root = (runner.workdir / output_root).resolve()
    search = SearchSettings(
        seed=int(search_raw.get("seed", 1337)),
        max_runs=int(search_raw.get("max_runs", 8)),
        warm_start_suggestions=int(search_raw.get("warm_start_suggestions", 8)),
        candidate_samples=int(search_raw.get("candidate_samples", 128)),
        max_observations=int(search_raw.get("max_observations", 200)),
        prune_pareto=bool(search_raw.get("prune_pareto", True)),
        suggestions_per_center=int(search_raw.get("suggestions_per_center", 32)),
        gp_alpha=float(search_raw.get("gp_alpha", 1e-6)),
        default_proxy_int6_gap=float(search_raw.get("default_proxy_int6_gap", 0.010)),
        target_cost_ratios=tuple(float(e) for e in search_raw.get("target_cost_ratios", [0.16, 0.32, 0.48, 0.64, 0.80, 1.0])),
        output_root=output_root,
        run_id_prefix=str(search_raw.get("run_id_prefix", config_path.stem)),
    )

    return SearchConfig(
        path=config_path,
        name=config_path.stem,
        runner=runner,
        fixed_env=fixed_env,
        search_space=search_space,
        search=search,
    )

