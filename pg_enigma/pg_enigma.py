#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shlex
import shutil
import sys
from itertools import combinations
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from search_harness import (  # noqa: E402
    HarnessError,
    SearchHarness,
    load_json,
    load_text,
    nested_get,
    normalize_json_text,
    prompt_compact_text,
    relative_or_absolute,
    resolve_path,
    run_process,
    sha256_text,
    truncate_text,
    write_json,
    write_text,
)


DEFAULT_REFERENCE_DOCS = [
    "findings.md",
    "search_strategy_catalog.md",
    "search_strategy_meta_prompt.md",
    "search_verifier_codex.md",
    "pg_enigma/self_check_rubric.md",
]

DEFAULT_PROMPTS = {
    "explorer": "explorer_prompt.md",
    "verifier": "verifier_prompt.md",
    "distiller": "distiller_prompt.md",
    "compiler": "compiler_prompt.md",
    "analyst": "analyst_prompt.md",
}

DEFAULT_SCHEMAS = {
    "explorer": "exploration_slate_schema.json",
    "verifier": "verification_report_schema.json",
    "distiller": "campaign_schema.json",
    "compiler": "family_compile_schema.json",
    "analyst": "postmortem_schema.json",
}

ALLOWED_SEARCH_LEVELS = {
    "metric_lane",
    "base_contract",
    "program_family",
    "composition",
    "reset",
}

ALLOWED_IMPLEMENTATION_MODES = {
    "catalog_executable_now",
    "needs_new_primitive",
    "needs_new_base_cycle",
}

ALLOWED_REALIZATION_TARGETS = {
    "current_search_harness_catalog",
    "needs_new_primitive",
    "needs_new_base_cycle",
}

ALLOWED_CONSEQUENCE_AXES = {
    "representation_contract",
    "compute_allocation",
    "regime_transition",
    "artifact_selection",
    "deployment_path",
    "model_contract",
    "search_strategy_reset",
}

ALLOWED_VERIFIER_VERDICTS = {"PASS", "RETRY", "FAIL"}
ALLOWED_REVIEW_VERDICTS = {"KEEP", "REWRITE", "DROP"}
ALLOWED_CAMPAIGN_VERDICTS = {"READY", "RETRY", "FAIL"}
ALLOWED_COMPILE_VERDICTS = {"READY", "RETRY", "FAIL"}
ALLOWED_POSTMORTEM_VERDICTS = {"READY", "RETRY", "FAIL"}
ALLOWED_POSTMORTEM_OUTCOMES = {"confirmed", "weakened", "rejected", "inconclusive", "pending"}
ALLOWED_POSTMORTEM_ACTIONS = {"promote", "hold", "drop", "reframe", "compose_later", "wait_for_data"}
ALLOWED_MODEL_BACKENDS = {"codex", "copilot"}

ALLOWED_LANES = {
    "base_frontier",
    "training",
    "selector",
    "deployment",
    "composition",
}

ALLOWED_PHASE_WINDOWS = {
    "frontier",
    "early",
    "mid",
    "late",
    "post_train",
    "pairwise",
    "hybrid",
}

ALLOWED_PACK_KINDS = {
    "base_frontier_pack",
    "early_training_pack",
    "mid_training_pack",
    "late_training_pack",
    "selector_pack",
    "deployment_pack",
    "pairwise_composition_pack",
    "hybrid_pack",
}

INITIAL_EXECUTABLE_PACK_KINDS = {
    "base_frontier_pack",
    "early_training_pack",
    "mid_training_pack",
    "late_training_pack",
    "selector_pack",
    "deployment_pack",
}

SCORE_FIELDS = (
    "consequence",
    "novelty",
    "falsifiability",
    "lane_integrity",
    "implementation_honesty",
)


def dedupe_strings(items: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        ordered.append(item)
    return ordered


def canonical_candidate_id(trajectory_id: str, hypothesis_id: str) -> str:
    return f"{trajectory_id}/{hypothesis_id}"


def safe_slug(text: str, limit: int = 64) -> str:
    slug = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_.-")
    if not slug:
        slug = "item"
    return slug[:limit]


def read_excerpt(path: Path, limit: int = 4000) -> str:
    if path.suffix.lower() == ".json":
        try:
            return truncate_text(json.dumps(load_json(path), indent=2), limit)
        except Exception:  # noqa: BLE001
            pass
    try:
        return truncate_text(load_text(path), limit)
    except UnicodeDecodeError:
        return f"[binary file omitted: {path.name}]"


def extract_last_json_object_text(text: str) -> str:
    candidate: str | None = None
    start: int | None = None
    depth = 0
    in_string = False
    escape = False
    for index, char in enumerate(text):
        if start is None:
            if char == "{":
                start = index
                depth = 1
                in_string = False
                escape = False
            continue

        if in_string:
            if escape:
                escape = False
            elif char == "\\":
                escape = True
            elif char == '"':
                in_string = False
            continue

        if char == '"':
            in_string = True
        elif char == "{":
            depth += 1
        elif char == "}":
            depth -= 1
            if depth == 0:
                raw = text[start : index + 1]
                try:
                    json.loads(raw)
                except json.JSONDecodeError:
                    pass
                else:
                    candidate = raw
                start = None

    if candidate is None:
        raise HarnessError("could not locate a parseable JSON object in model output")
    return candidate


def strip_markdown_preamble(text: str, start_patterns: list[str]) -> str:
    stripped = text.lstrip()
    for pattern in start_patterns:
        match = re.search(pattern, stripped, flags=re.MULTILINE)
        if match:
            return stripped[match.start() :].lstrip()
    return stripped


def strip_breadth_markdown_output(text: str) -> str:
    return strip_markdown_preamble(
        text,
        start_patterns=[
            r"^===\s*PATHS\s*===\s*$",
            r"^#\s*Path\s+1\b",
        ],
    )


def strip_depth_markdown_output(text: str) -> str:
    return strip_markdown_preamble(
        text,
        start_patterns=[
            r"^===\s*WORLDVIEW\s*===\s*$",
            r"^#\s*Worldview\b",
        ],
    )


def average(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def better(goal: str, left: float, right: float) -> bool:
    if goal == "maximize":
        return left > right
    return left < right


def best_value(goal: str, values: list[float]) -> float | None:
    if not values:
        return None
    return max(values) if goal == "maximize" else min(values)


class EnigmaHarness:
    def __init__(self, config_path: Path):
        self.module_root = Path(__file__).resolve().parent
        self.repo_root = REPO_ROOT
        self.config_path = config_path.resolve()
        self.config_dir = self.config_path.parent
        self.config = load_json(self.config_path)
        self.cycle_id = str(self.config["cycle_id"])
        workspace_root = resolve_path(self.config_dir, self.config.get("workspace", {}).get("root", "runs"))
        if workspace_root is None:
            raise HarnessError("workspace.root must resolve")
        self.workspace_root = workspace_root
        self.cycle_root = self.workspace_root / self.cycle_id

    def round_dir(self, round_index: int) -> Path:
        return self.cycle_root / f"round_{round_index:03d}"

    def round_paths(self, round_index: int) -> dict[str, Path]:
        round_dir = self.round_dir(round_index)
        campaign_path = round_dir / "campaign.json"
        return {
            "round_dir": round_dir,
            "request": round_dir / "request.json",
            "explorations_dir": round_dir / "explorations",
            "verification_report": round_dir / "verification_report.json",
            "campaign": campaign_path,
            "distilled_strategy": campaign_path,
            "family_dossiers_dir": round_dir / "family_dossiers",
            "compile_queue": round_dir / "compile_queue.json",
            "family_compiles_dir": round_dir / "family_compiles",
            "pack_queue": round_dir / "pack_queue.json",
            "pack_handoffs": round_dir / "pack_handoffs.json",
            "pack_handoffs_dir": round_dir / "pack_handoffs",
            "promotion_report": round_dir / "promotion_report.json",
            "composition_queue": round_dir / "composition_queue.json",
            "hybrid_queue": round_dir / "hybrid_queue.json",
            "postmortem_report": round_dir / "postmortem_report.json",
            "search_handoff_md": round_dir / "SEARCH_HANDOFF.md",
            "search_handoff_prompt": round_dir / "SEARCH_HANDOFF_PROMPT.txt",
            "round_summary": round_dir / "round_summary.json",
            "family_status_report": round_dir / "family_status_report.json",
            "runnable_families": round_dir / "runnable_families.json",
            "focus_dir": round_dir / "focus_files",
            "evidence_dir": round_dir / "evidence_files",
            "codex_dir": round_dir / "codex",
            "codex_summary": round_dir / "codex" / "summary.json",
            "copilot_dir": round_dir / "copilot",
            "copilot_summary": round_dir / "copilot" / "summary.json",
            "agent_dir": round_dir / "agent",
            "agent_instructions": round_dir / "agent" / "AGENT_INSTRUCTIONS.md",
            "agent_prompt": round_dir / "agent" / "AGENT_PROMPT.txt",
        }

    def codex_settings(self) -> dict[str, Any]:
        return self.config.get("codex", {})

    def normalize_backend(self, backend: str) -> str:
        selected = str(backend).strip().lower()
        if selected not in ALLOWED_MODEL_BACKENDS:
            raise HarnessError(f"backend must be one of {sorted(ALLOWED_MODEL_BACKENDS)}, got: {backend}")
        return selected

    def backend_attempt_root(self, paths: dict[str, Path], backend: str) -> Path:
        selected = self.normalize_backend(backend)
        return paths["copilot_dir"] if selected == "copilot" else paths["codex_dir"]

    def backend_summary_path(self, paths: dict[str, Path], backend: str) -> Path:
        selected = self.normalize_backend(backend)
        return paths["copilot_summary"] if selected == "copilot" else paths["codex_summary"]

    def objective_settings(self) -> dict[str, Any]:
        return dict(self.config.get("objective", {}))

    def exploration_settings(self) -> dict[str, int]:
        exploration = self.config.get("exploration", {})
        trajectories = int(exploration.get("trajectories", 4))
        families_per_trajectory = int(exploration.get("families_per_trajectory", 4))
        if trajectories < 1:
            raise HarnessError("exploration.trajectories must be >= 1")
        if families_per_trajectory < 1:
            raise HarnessError("exploration.families_per_trajectory must be >= 1")
        return {
            "trajectories": trajectories,
            "families_per_trajectory": families_per_trajectory,
        }

    def verification_settings(self) -> dict[str, int]:
        verification = self.config.get("verification", {})
        min_keep = int(verification.get("min_keep", 2))
        min_consequence_score = int(verification.get("min_consequence_score", 4))
        min_falsifiability_score = int(verification.get("min_falsifiability_score", 3))
        min_implementation_honesty_score = int(verification.get("min_implementation_honesty_score", 3))
        if min_keep < 2:
            raise HarnessError("verification.min_keep must be >= 2")
        return {
            "min_keep": min_keep,
            "min_consequence_score": min_consequence_score,
            "min_falsifiability_score": min_falsifiability_score,
            "min_implementation_honesty_score": min_implementation_honesty_score,
        }

    def compiler_settings(self) -> dict[str, int]:
        compiler = self.config.get("compiler", {})
        realizations_per_family = int(compiler.get("realizations_per_family", 3))
        min_ready_realizations = int(compiler.get("min_ready_realizations", 1))
        if realizations_per_family < 2:
            raise HarnessError("compiler.realizations_per_family must be >= 2")
        if min_ready_realizations < 1:
            raise HarnessError("compiler.min_ready_realizations must be >= 1")
        return {
            "realizations_per_family": realizations_per_family,
            "min_ready_realizations": min_ready_realizations,
        }

    def packing_settings(self) -> dict[str, int]:
        packing = self.config.get("packing", {})
        controls_per_pack = int(packing.get("controls_per_pack", 2))
        max_candidates_per_pack = int(packing.get("max_candidates_per_pack", 4))
        max_realizations_per_family_per_pack = int(packing.get("max_realizations_per_family_per_pack", 2))
        if controls_per_pack < 2:
            raise HarnessError("packing.controls_per_pack must be >= 2")
        if max_candidates_per_pack < 1:
            raise HarnessError("packing.max_candidates_per_pack must be >= 1")
        if max_realizations_per_family_per_pack < 1:
            raise HarnessError("packing.max_realizations_per_family_per_pack must be >= 1")
        return {
            "controls_per_pack": controls_per_pack,
            "max_candidates_per_pack": max_candidates_per_pack,
            "max_realizations_per_family_per_pack": max_realizations_per_family_per_pack,
        }

    def promotion_settings(self) -> dict[str, Any]:
        promotion = self.config.get("promotion", {})
        min_positive_realizations = int(promotion.get("min_positive_realizations", 2))
        min_surviving_realizations = int(promotion.get("min_surviving_realizations", 1))
        require_directional_support = bool(promotion.get("require_directional_support", True))
        if min_positive_realizations < 1:
            raise HarnessError("promotion.min_positive_realizations must be >= 1")
        if min_surviving_realizations < 0:
            raise HarnessError("promotion.min_surviving_realizations must be >= 0")
        return {
            "min_positive_realizations": min_positive_realizations,
            "min_surviving_realizations": min_surviving_realizations,
            "require_directional_support": require_directional_support,
        }

    def composition_settings(self) -> dict[str, int]:
        composition = self.config.get("composition", {})
        max_pairwise_candidates = int(composition.get("max_pairwise_candidates", 6))
        max_hybrid_candidates = int(composition.get("max_hybrid_candidates", 4))
        if max_pairwise_candidates < 1:
            raise HarnessError("composition.max_pairwise_candidates must be >= 1")
        if max_hybrid_candidates < 1:
            raise HarnessError("composition.max_hybrid_candidates must be >= 1")
        return {
            "max_pairwise_candidates": max_pairwise_candidates,
            "max_hybrid_candidates": max_hybrid_candidates,
        }

    def handoff_settings(self) -> dict[str, Any]:
        return dict(self.config.get("handoff", {}))

    def resolve_runtime_path(self, raw: str | None, default_to_repo_root: bool = True) -> Path | None:
        if raw is None:
            return self.repo_root if default_to_repo_root else None
        path = Path(raw)
        if path.is_absolute():
            return path.resolve()
        repo_candidate = (self.repo_root / path).resolve()
        config_candidate = (self.config_dir / path).resolve()
        if repo_candidate.exists() or not config_candidate.exists():
            return repo_candidate
        return config_candidate

    def prompt_path(self, key: str) -> Path:
        raw = self.config.get("prompts", {}).get(key)
        if raw:
            path = self.resolve_runtime_path(str(raw), default_to_repo_root=False)
            if path is None:
                raise HarnessError(f"Could not resolve prompt path for {key}")
            return path
        return (self.module_root / DEFAULT_PROMPTS[key]).resolve()

    def prompt_text(self, key: str) -> str:
        path = self.prompt_path(key)
        if not path.exists():
            return ""
        return load_text(path)

    def schema_path(self, key: str) -> Path:
        raw = self.config.get("schemas", {}).get(key)
        if raw:
            path = self.resolve_runtime_path(str(raw), default_to_repo_root=False)
            if path is None:
                raise HarnessError(f"Could not resolve schema path for {key}")
            return path
        return (self.module_root / DEFAULT_SCHEMAS[key]).resolve()

    def agent_root(self) -> Path:
        codex = self.codex_settings()
        return (self.resolve_runtime_path(codex.get("cd", ".")) or self.repo_root).resolve()

    def project_relative(self, path: Path) -> str:
        return relative_or_absolute(path, self.agent_root())

    def resolve_input_files(self, raw_files: list[str] | None) -> list[Path]:
        if not raw_files:
            return []
        resolved: list[Path] = []
        seen: set[Path] = set()
        agent_root = self.agent_root()
        for raw in raw_files:
            candidates: list[Path] = []
            first = resolve_path(self.config_dir, raw)
            if first is not None:
                candidates.append(first)
            second = resolve_path(agent_root, raw)
            if second is not None and second not in candidates:
                candidates.append(second)
            path = next((candidate for candidate in candidates if candidate.exists() and candidate.is_file()), None)
            if path is None:
                raise HarnessError(f"input file does not exist: {raw}")
            actual = path.resolve()
            try:
                actual.relative_to(agent_root)
            except ValueError as exc:
                raise HarnessError(f"input file is outside the agent root {agent_root}: {raw}") from exc
            if actual in seen:
                continue
            seen.add(actual)
            resolved.append(actual)
        return resolved

    def family_file_name(self, candidate_id: str) -> str:
        return f"{safe_slug(candidate_id)}.json"

    def expected_trajectory_ids(self, request: dict[str, Any]) -> list[str]:
        trajectories = int(request["exploration"]["trajectories"])
        return [f"trajectory_{index:03d}" for index in range(1, trajectories + 1)]

    def build_reference_docs(self) -> list[dict[str, str]]:
        docs = self.config.get("reference_docs", DEFAULT_REFERENCE_DOCS)
        payload: list[dict[str, str]] = []
        for raw in docs:
            path = self.resolve_runtime_path(str(raw), default_to_repo_root=False)
            if path is None or not path.exists():
                continue
            payload.append(
                {
                    "path": self.project_relative(path),
                    "excerpt": read_excerpt(path, 4000),
                }
            )
        return payload

    def build_history_context(self, round_index: int) -> dict[str, Any]:
        history_window = int(self.config.get("history_window", 3))
        history: list[dict[str, Any]] = []
        for idx in range(max(0, round_index - history_window), round_index):
            round_paths = self.round_paths(idx)
            entry: dict[str, Any] = {"round": idx}
            for key in ("round_summary", "promotion_report", "postmortem_report", "composition_queue"):
                path = round_paths[key]
                if path.exists():
                    entry[key] = load_json(path)
            if len(entry) > 1:
                history.append(entry)
        return {
            "cycle_id": self.cycle_id,
            "round": round_index,
            "history": history,
        }

    def build_file_context(self, paths: list[Path], limit: int = 4000) -> list[dict[str, str]]:
        return [
            {
                "path": self.project_relative(path),
                "excerpt": read_excerpt(path, limit),
            }
            for path in paths
        ]

    def build_round_request(
        self,
        round_index: int,
        instructions: str,
        focus_paths: list[Path],
        evidence_paths: list[Path],
    ) -> dict[str, Any]:
        cleaned = instructions.strip()
        if not cleaned:
            raise HarnessError("instructions must not be empty")
        return {
            "cycle_id": self.cycle_id,
            "round": round_index,
            "user_instructions": cleaned,
            "objective": self.objective_settings(),
            "exploration": self.exploration_settings(),
            "verification": self.verification_settings(),
            "compiler": self.compiler_settings(),
            "packing": self.packing_settings(),
            "promotion": self.promotion_settings(),
            "composition": self.composition_settings(),
            "handoff": self.handoff_settings(),
            "focus_files": [self.project_relative(path) for path in focus_paths],
            "focus_file_excerpts": self.build_file_context(focus_paths),
            "evidence_files": [self.project_relative(path) for path in evidence_paths],
            "evidence_file_excerpts": self.build_file_context(evidence_paths),
            "history_context": self.build_history_context(round_index),
            "reference_docs": self.build_reference_docs(),
        }

    def copy_inputs(self, paths: list[Path], target_dir: Path) -> list[str]:
        copied: list[str] = []
        for path in paths:
            rel = self.project_relative(path)
            destination = target_dir / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied.append(rel)
        return copied

    def prepare_round(
        self,
        round_index: int,
        instructions: str,
        repo_files: list[str] | None = None,
        evidence_files: list[str] | None = None,
    ) -> tuple[Path, dict[str, Any]]:
        paths = self.round_paths(round_index)
        paths["round_dir"].mkdir(parents=True, exist_ok=True)

        focus_paths = self.resolve_input_files(repo_files)
        evidence_paths = self.resolve_input_files(evidence_files)
        request = self.build_round_request(round_index, instructions, focus_paths, evidence_paths)
        write_json(paths["request"], request)

        for key in ("focus_dir", "evidence_dir"):
            if paths[key].exists():
                shutil.rmtree(paths[key])
            paths[key].mkdir(parents=True, exist_ok=True)
        self.copy_inputs(focus_paths, paths["focus_dir"])
        self.copy_inputs(evidence_paths, paths["evidence_dir"])
        return paths["round_dir"], request

    def build_round_agent_instructions(self, round_index: int, request: dict[str, Any]) -> str:
        paths = self.round_paths(round_index)
        relative_config = relative_or_absolute(self.config_path, self.agent_root())
        relative_request = self.project_relative(paths["request"])
        relative_round_dir = self.project_relative(paths["round_dir"])
        relative_agent_dir = self.project_relative(paths["agent_dir"])
        relative_focus_dir = self.project_relative(paths["focus_dir"])
        relative_evidence_dir = self.project_relative(paths["evidence_dir"])
        validate_command = (
            f"python3 pg_enigma/pg_enigma.py validate-round "
            f"--config {shlex.quote(relative_config)} --round {round_index}"
        )
        trajectory_outputs = "\n".join(
            f"- `{self.project_relative(paths['explorations_dir'] / f'{trajectory_id}.json')}`"
            for trajectory_id in self.expected_trajectory_ids(request)
        )
        focus_block = "\n".join(f"- `{path}`" for path in request["focus_files"]) or "- None provided"
        evidence_block = "\n".join(f"- `{path}`" for path in request["evidence_files"]) or "- None provided"
        return "\n".join(
            [
                "# Agent Instructions",
                "",
                "Work in the repository root. Do not modify unrelated files.",
                f"Your task is to complete pg_enigma round `{round_index}` inside `{relative_round_dir}`.",
                "",
                "## Inputs to read first",
                f"- Round request: `{relative_request}`",
                f"- Prompt and schema copies in: `{relative_agent_dir}`",
                f"- Focus snapshots: `{relative_focus_dir}`",
                f"- Evidence snapshots: `{relative_evidence_dir}`",
                "",
                "## Outputs to create",
                trajectory_outputs,
                f"- `{self.project_relative(paths['verification_report'])}`",
                f"- `{self.project_relative(paths['campaign'])}`",
                "",
                "## Focus repo files",
                focus_block,
                "",
                "## Evidence files",
                evidence_block,
                "",
                "## Hard rules",
                "1. Exploration outputs must follow the explorer schema exactly.",
                "2. Verification output must review every candidate exactly once.",
                "3. Campaign output must preserve multiple consequential families; do not collapse to one final answer.",
                "4. Campaign families must define lane, phase window, pack kind, and compiler pass@k.",
                "5. Do not write code in this stage. Produce campaign structure only.",
                "6. If a line needs a new primitive or a new base cycle, say so explicitly instead of pretending it is executable now.",
                "",
                "## Validation command",
                f"- Run: `{validate_command}`",
                "",
                "## Success condition",
                f"- `{self.project_relative(paths['search_handoff_md'])}` exists",
                f"- `{self.project_relative(paths['compile_queue'])}` exists",
                f"- `{self.project_relative(paths['round_summary'])}` exists",
                "- the validation command succeeds without edits afterwards",
                "",
                "## User instructions",
                request["user_instructions"],
            ]
        ) + "\n"

    def prepare_agent_folder(
        self,
        round_index: int,
        instructions: str,
        repo_files: list[str] | None = None,
        evidence_files: list[str] | None = None,
    ) -> Path:
        _, request = self.prepare_round(
            round_index=round_index,
            instructions=instructions,
            repo_files=repo_files,
            evidence_files=evidence_files,
        )
        paths = self.round_paths(round_index)
        if paths["agent_dir"].exists():
            shutil.rmtree(paths["agent_dir"])
        paths["agent_dir"].mkdir(parents=True, exist_ok=True)

        for key in ("explorer", "verifier", "distiller", "compiler", "analyst"):
            source = self.prompt_path(key)
            if source.exists():
                shutil.copy2(source, paths["agent_dir"] / source.name)
        for key in ("explorer", "verifier", "distiller", "compiler", "analyst"):
            source = self.schema_path(key)
            if source.exists():
                shutil.copy2(source, paths["agent_dir"] / source.name)
        rubric = (self.module_root / "self_check_rubric.md").resolve()
        if rubric.exists():
            shutil.copy2(rubric, paths["agent_dir"] / rubric.name)

        write_text(paths["agent_instructions"], self.build_round_agent_instructions(round_index, request))
        write_text(
            paths["agent_prompt"],
            (
                f"Read `{self.project_relative(paths['agent_instructions'])}` and complete the task fully. "
                "Write the required JSON outputs, run the validation command, and leave the round folder ready for compilation.\n"
            ),
        )
        return paths["agent_dir"]

    def codex_stage_model(self, stage: str) -> str | None:
        codex = self.codex_settings()
        models = codex.get("models", {})
        model = models.get(stage) if isinstance(models, dict) else None
        fallback = codex.get("model")
        selected = model or fallback
        return str(selected) if selected else None

    def codex_stage_reasoning_effort(self, stage: str) -> str | None:
        codex = self.codex_settings()
        efforts = codex.get("reasoning_efforts", {})
        effort = efforts.get(stage) if isinstance(efforts, dict) else None
        fallback = codex.get("reasoning_effort")
        selected = effort or fallback
        return str(selected) if selected else None

    def invoke_codex(
        self,
        prompt: str,
        schema_path: Path,
        output_path: Path,
        attempt_dir: Path,
        label: str,
        stage: str,
    ) -> dict[str, Any]:
        codex = self.codex_settings()
        codex_bin = str(codex.get("bin", "codex"))
        codex_cwd = self.agent_root()
        args = [
            codex_bin,
            "exec",
            "-",
            "-C",
            str(codex_cwd),
            "--sandbox",
            str(codex.get("sandbox", "workspace-write")),
            "--output-schema",
            str(schema_path),
            "-o",
            str(output_path),
        ]
        model = self.codex_stage_model(stage)
        reasoning_effort = self.codex_stage_reasoning_effort(stage)
        profile = codex.get("profile")
        if model:
            args.extend(["-m", model])
        if reasoning_effort:
            args.extend(["-c", f'model_reasoning_effort="{reasoning_effort}"'])
        if profile:
            args.extend(["-p", str(profile)])
        if codex.get("skip_git_repo_check"):
            args.append("--skip-git-repo-check")
        if codex.get("full_auto"):
            args.append("--full-auto")
        for extra in codex.get("extra_args", []):
            args.append(str(extra))

        completed = run_process(args, cwd=codex_cwd, env=os.environ.copy(), input_text=prompt)
        write_text(attempt_dir / f"{label}_stdout.log", completed.stdout)
        write_text(attempt_dir / f"{label}_stderr.log", completed.stderr)
        if completed.returncode != 0:
            raise HarnessError(
                f"Codex {label} failed ({completed.returncode}).\n"
                f"STDOUT:\n{truncate_text(completed.stdout)}\n"
                f"STDERR:\n{truncate_text(completed.stderr)}"
            )
        if not output_path.exists():
            raise HarnessError(f"Codex {label} did not write {output_path}")
        raw = normalize_json_text(load_text(output_path))
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise HarnessError(f"Codex {label} returned invalid JSON: {exc}") from exc

    def invoke_copilot_json(
        self,
        prompt: str,
        output_path: Path,
        work_dir: Path,
        model: str,
        reasoning_effort: str,
        label: str = "copilot",
    ) -> dict[str, Any]:
        args = [
            "copilot",
            "-p",
            prompt,
            "--model",
            model,
            "--reasoning-effort",
            reasoning_effort,
            "--allow-all-tools",
            "--allow-all-paths",
            "--silent",
            "--output-format",
            "text",
            "--no-custom-instructions",
        ]
        completed = run_process(args, cwd=self.agent_root(), env=os.environ.copy())
        write_text(work_dir / f"{label}_stdout.log", completed.stdout)
        write_text(work_dir / f"{label}_stderr.log", completed.stderr)
        if completed.returncode != 0:
            raise HarnessError(
                f"Copilot {label} failed ({completed.returncode}).\n"
                f"STDOUT:\n{truncate_text(completed.stdout)}\n"
                f"STDERR:\n{truncate_text(completed.stderr)}"
            )
        write_text(output_path, completed.stdout)
        raw = normalize_json_text(completed.stdout)
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
        try:
            extracted = extract_last_json_object_text(completed.stdout)
            write_text(work_dir / f"{label}_extracted_json.txt", extracted)
            return json.loads(extracted)
        except Exception as exc:
            raise HarnessError(f"Copilot {label} returned invalid JSON: {exc}") from exc

    def invoke_backend_json(
        self,
        backend: str,
        prompt: str,
        schema_path: Path,
        output_path: Path,
        attempt_dir: Path,
        label: str,
        stage: str,
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> dict[str, Any]:
        selected = self.normalize_backend(backend)
        if selected == "copilot":
            chosen_model = str(model or self.codex_stage_model(stage) or "gpt-5.4")
            chosen_reasoning = str(reasoning_effort or self.codex_stage_reasoning_effort(stage) or "xhigh")
            return self.invoke_copilot_json(
                prompt=prompt,
                output_path=output_path,
                work_dir=attempt_dir,
                model=chosen_model,
                reasoning_effort=chosen_reasoning,
                label=label,
            )
        return self.invoke_codex(
            prompt=prompt,
            schema_path=schema_path,
            output_path=output_path,
            attempt_dir=attempt_dir,
            label=label,
            stage=stage,
        )

    def flatten_candidates(self, trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        flattened: list[dict[str, Any]] = []
        for slate in trajectories:
            worldview = str(slate["worldview"])
            for hypothesis in slate["hypotheses"]:
                flattened.append(
                    {
                        "candidate_id": hypothesis["candidate_id"],
                        "trajectory_id": slate["trajectory_id"],
                        "hypothesis_id": hypothesis["id"],
                        "worldview": worldview,
                        "name": hypothesis["name"],
                        "search_level": hypothesis["search_level"],
                        "lane": hypothesis["lane"],
                        "family_group": hypothesis["family_group"],
                        "implementation_mode": hypothesis["implementation_mode"],
                        "mechanism": hypothesis["mechanism"],
                        "consequence_axes": hypothesis["consequence_axes"],
                        "broken_invariant": hypothesis["broken_invariant"],
                        "not_local_tuning_reason": hypothesis["not_local_tuning_reason"],
                        "measurement_plan": hypothesis["measurement_plan"],
                        "smallest_decisive_probe": hypothesis["smallest_decisive_probe"],
                        "expected_observable": hypothesis["expected_observable"],
                        "falsifier": hypothesis["falsifier"],
                        "novelty_vs_history": hypothesis["novelty_vs_history"],
                        "implementation_basis": hypothesis["implementation_basis"],
                        "self_check_summary": hypothesis["self_check_summary"],
                        "self_scores": hypothesis["self_scores"],
                    }
                )
        return flattened

    def verifier_request_summary(self, request: dict[str, Any]) -> dict[str, Any]:
        exploration = request["exploration"]
        summary = {
            "cycle_id": request["cycle_id"],
            "round": request["round"],
            "user_instructions": prompt_compact_text(request["user_instructions"], 500),
            "objective": prompt_compact_text(request["objective"], 320),
            "exploration": {
                "trajectory_count": int(exploration.get("trajectory_count", 0)),
                "families_per_trajectory": int(exploration.get("families_per_trajectory", 0)),
            },
            "verification": request["verification"],
            "focus_files": list(request.get("focus_files", [])),
            "evidence_files": list(request.get("evidence_files", [])),
            "reference_doc_paths": [
                str(item.get("path", "")).strip()
                for item in request.get("reference_docs", [])
                if isinstance(item, dict) and str(item.get("path", "")).strip()
            ],
        }
        history_context = request.get("history_context")
        if isinstance(history_context, dict):
            history_entries = history_context.get("history", [])
            if isinstance(history_entries, list) and history_entries:
                summary["history_context"] = {
                    "cycle_id": history_context.get("cycle_id"),
                    "round": history_context.get("round"),
                    "available_rounds": [
                        entry.get("round")
                        for entry in history_entries
                        if isinstance(entry, dict) and entry.get("round") is not None
                    ],
                }
        return summary

    def prompt_request_summary(self, request: dict[str, Any]) -> dict[str, Any]:
        summary = {
            "cycle_id": request["cycle_id"],
            "round": request["round"],
            "user_instructions": request["user_instructions"],
            "objective": request["objective"],
            "exploration": request["exploration"],
            "verification": request["verification"],
            "compiler": request["compiler"],
            "packing": request["packing"],
            "promotion": request["promotion"],
            "composition": request["composition"],
            "handoff": request["handoff"],
            "focus_files": list(request.get("focus_files", [])),
            "evidence_files": list(request.get("evidence_files", [])),
            "reference_doc_paths": [
                str(item.get("path", "")).strip()
                for item in request.get("reference_docs", [])
                if isinstance(item, dict) and str(item.get("path", "")).strip()
            ],
        }
        history_context = request.get("history_context")
        if isinstance(history_context, dict):
            history_entries = history_context.get("history", [])
            if isinstance(history_entries, list) and history_entries:
                summary["history_context"] = {
                    "cycle_id": history_context.get("cycle_id"),
                    "round": history_context.get("round"),
                    "available_rounds": [
                        entry.get("round")
                        for entry in history_entries
                        if isinstance(entry, dict) and entry.get("round") is not None
                    ],
                }
        return summary

    def breadth_target_file(self, request: dict[str, Any]) -> str:
        raw_candidates: list[str] = []
        objective = request.get("objective", {})
        target_script = objective.get("target_script")
        if isinstance(target_script, str) and target_script.strip():
            raw_candidates.append(target_script.strip())
        for raw in request.get("focus_files", []):
            if isinstance(raw, str) and raw.strip():
                raw_candidates.append(raw.strip())
        for raw in dedupe_strings(raw_candidates):
            resolved = self.resolve_runtime_path(raw, default_to_repo_root=False)
            if resolved is not None and resolved.exists():
                return self.project_relative(resolved)
            parent_candidate = (self.repo_root.parent / raw).resolve()
            if parent_candidate.exists():
                return relative_or_absolute(parent_candidate, self.agent_root())
        if raw_candidates:
            raw = raw_candidates[0]
            return raw if raw.startswith("../") else f"../{raw}"
        return "../frontier_rebase/pr1394/train_gpt_human.py"

    def previous_postmortem_summary(self, round_index: int) -> str | None:
        for idx in range(round_index - 1, -1, -1):
            path = self.round_paths(idx)["postmortem_report"]
            if not path.exists():
                continue
            payload = load_json(path)
            compact_payload = {
                "round": idx,
                "summary": payload.get("summary"),
                "generation_instruction_delta": payload.get("generation_instruction_delta"),
                "family_assessments": (
                    payload.get("family_assessments", [])[:4]
                    if isinstance(payload.get("family_assessments"), list)
                    else []
                ),
            }
            return prompt_compact_text(json.dumps(compact_payload, indent=2), 1600)
        return None

    def build_breadth_explorer_prompt(
        self,
        round_index: int,
        request: dict[str, Any],
        previous_feedback: str | None,
    ) -> str:
        target_file = self.breadth_target_file(request)
        objective = request["objective"]
        postmortem = self.previous_postmortem_summary(round_index)
        sections = [
            "You are the breadth-stage explorer for pg_enigma.",
            "",
            "Your job is to propose exactly 10 distinct search paths for this target, not to fully develop them yet.",
            "Sample from the entire distribution; your proposals should be distinct.",
            "",
            "Start the answer with exactly `=== PATHS ===`.",
            "Do not narrate your process.",
            "Do not mention locating files, reading files, checking contracts, or validating paths.",
            "Treat the listed target file as canonical. Do not search for alternate copies.",
            "Read only the listed target file for this breadth pass.",
            "Return markdown only. Do not return JSON.",
            "",
            "CHALLENGE",
            f"- User instructions: {prompt_compact_text(request['user_instructions'], 900)}",
            f"- Goal: {objective.get('goal', '')}",
            f"- Primary metric: {objective.get('primary_metric', '')}",
            f"- Secondary metrics: {prompt_compact_text(objective.get('secondary_metrics', []), 200)}",
            f"- Constraint notes: {prompt_compact_text(objective.get('notes', []), 360)}",
            "",
            "CONSTRAINTS",
            "- Prefer first-order, consequential paths over local tuning.",
            "- If a path sounds runnable with the current surface, mark it `likely executable now`; otherwise mark it `likely blocked`.",
            "- Do not output pack plans, tournament schedules, or implementation details for every patch.",
            "- Do not collapse multiple paths into one vague umbrella.",
            "- For each path, say what should be explored next inside that path in a bit of detail.",
            "",
            "TARGET FILE TO READ",
            f"- `{target_file}`",
        ]
        if postmortem:
            sections.extend(["", "PREVIOUS POSTMORTEM SUMMARY", postmortem])
        if previous_feedback:
            sections.extend(["", "PREVIOUS ATTEMPT FEEDBACK", previous_feedback])
        sections.extend(
            [
                "",
                "OUTPUT FORMAT",
                "=== PATHS ===",
                "",
                "# Path 1",
                "- Name:",
                "- Core path:",
                "- Broken invariant:",
                "- Why this is first-order:",
                "- Why this is distinct:",
                "- Status: likely executable now | likely blocked",
                "- What to explore next:",
                "- Smallest decisive probe:",
                "",
                "Repeat through `# Path 10` with the same fields.",
            ]
        )
        return "\n".join(sections) + "\n"

    def trajectory_summary(self, trajectories: list[dict[str, Any]]) -> list[dict[str, Any]]:
        return [
            {
                "trajectory_id": slate["trajectory_id"],
                "worldview": slate["worldview"],
                "candidate_ids": [hypothesis["candidate_id"] for hypothesis in slate["hypotheses"]],
            }
            for slate in trajectories
        ]

    def candidate_lookup(self, trajectories: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
        return {item["candidate_id"]: item for item in self.flatten_candidates(trajectories)}

    def verifier_alignment_warnings(self, candidate: dict[str, Any]) -> list[str]:
        warnings: list[str] = []
        axes = set(str(axis) for axis in candidate["consequence_axes"])
        search_level = str(candidate["search_level"])
        has_reset_axis = "search_strategy_reset" in axes
        if search_level == "reset" and not has_reset_axis:
            warnings.append("search_level=reset but consequence_axes omits search_strategy_reset")
        if search_level != "reset" and has_reset_axis:
            warnings.append("consequence_axes includes search_strategy_reset outside search_level=reset")
        return warnings

    def verifier_claim_cards(
        self,
        trajectories: list[dict[str, Any]],
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        verification = request["verification"]
        cards: list[dict[str, Any]] = []
        for candidate in self.flatten_candidates(trajectories):
            scores = candidate["self_scores"]
            cards.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "family_group": candidate["family_group"],
                    "search_level": candidate["search_level"],
                    "lane": candidate["lane"],
                    "implementation_mode": candidate["implementation_mode"],
                    "mechanism": prompt_compact_text(candidate["mechanism"], 220),
                    "broken_invariant": prompt_compact_text(candidate["broken_invariant"], 180),
                    "consequence_axes": candidate["consequence_axes"],
                    "not_local_tuning_reason": prompt_compact_text(candidate["not_local_tuning_reason"], 220),
                    "measurement_plan": prompt_compact_text(candidate["measurement_plan"], 220),
                    "smallest_decisive_probe": prompt_compact_text(candidate["smallest_decisive_probe"], 220),
                    "expected_observable": prompt_compact_text(candidate["expected_observable"], 200),
                    "falsifier": prompt_compact_text(candidate["falsifier"], 180),
                    "novelty_vs_history": prompt_compact_text(candidate["novelty_vs_history"], 220),
                    "implementation_basis": prompt_compact_text(candidate["implementation_basis"], 220),
                    "self_scores": scores,
                    "self_check_summary": prompt_compact_text(candidate["self_check_summary"], 220),
                    "static_audit": {
                        "self_scores_clear_keep_floor": {
                            "consequence": int(scores["consequence"]) >= int(verification["min_consequence_score"]),
                            "falsifiability": int(scores["falsifiability"])
                            >= int(verification["min_falsifiability_score"]),
                            "implementation_honesty": int(scores["implementation_honesty"])
                            >= int(verification["min_implementation_honesty_score"]),
                        },
                        "alignment_warnings": self.verifier_alignment_warnings(candidate),
                    },
                }
            )
        return cards

    def candidate_reuse_fingerprint(self, candidate: dict[str, Any]) -> str:
        payload = {
            "search_level": str(candidate["search_level"]),
            "lane": str(candidate["lane"]),
            "family_group": str(candidate["family_group"]),
            "implementation_mode": str(candidate["implementation_mode"]),
            "mechanism": prompt_compact_text(candidate["mechanism"], 4000),
            "broken_invariant": prompt_compact_text(candidate["broken_invariant"], 4000),
            "consequence_axes": sorted(str(axis) for axis in candidate["consequence_axes"]),
            "not_local_tuning_reason": prompt_compact_text(candidate["not_local_tuning_reason"], 4000),
            "measurement_plan": prompt_compact_text(candidate["measurement_plan"], 4000),
            "smallest_decisive_probe": prompt_compact_text(candidate["smallest_decisive_probe"], 4000),
            "expected_observable": prompt_compact_text(candidate["expected_observable"], 4000),
            "falsifier": prompt_compact_text(candidate["falsifier"], 4000),
            "novelty_vs_history": prompt_compact_text(candidate["novelty_vs_history"], 4000),
            "implementation_basis": prompt_compact_text(candidate["implementation_basis"], 4000),
            "self_scores": dict(candidate["self_scores"]),
            "self_check_summary": prompt_compact_text(candidate["self_check_summary"], 4000),
        }
        return sha256_text(json.dumps(payload, sort_keys=True))

    def build_reused_review_cards(
        self,
        reused_entries: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        for entry in reused_entries:
            candidate = entry["candidate"]
            review = entry["review"]
            cards.append(
                {
                    "candidate_id": candidate["candidate_id"],
                    "reused_from_candidate_id": entry["source_candidate_id"],
                    "family_group": candidate["family_group"],
                    "search_level": candidate["search_level"],
                    "lane": candidate["lane"],
                    "mechanism": prompt_compact_text(candidate["mechanism"], 120),
                    "broken_invariant": prompt_compact_text(candidate["broken_invariant"], 120),
                    "prior_review": {
                        "verdict": review["verdict"],
                        "consequence_score": review["consequence_score"],
                        "novelty_score": review["novelty_score"],
                        "falsifiability_score": review["falsifiability_score"],
                        "lane_integrity_score": review["lane_integrity_score"],
                        "implementation_honesty_score": review["implementation_honesty_score"],
                    },
                }
            )
        return cards

    def load_attempt_trajectories(
        self,
        attempt_dir: Path,
        request: dict[str, Any],
    ) -> list[dict[str, Any]]:
        exploration_settings = request["exploration"]
        trajectories: list[dict[str, Any]] = []
        for trajectory_id in self.expected_trajectory_ids(request):
            normalized_path = attempt_dir / f"{trajectory_id}.json"
            raw_path = attempt_dir / f"{trajectory_id}_raw.json"
            payload_path = normalized_path if normalized_path.exists() else raw_path
            if not payload_path.exists():
                raise HarnessError(f"missing attempt trajectory file: {payload_path}")
            normalized = self.normalize_exploration_slate(
                load_json(payload_path),
                expected_trajectory_id=trajectory_id,
                expected_count=int(exploration_settings["families_per_trajectory"]),
            )
            trajectories.append(normalized)
        return trajectories

    def load_attempt_verification_state(
        self,
        attempt_dir: Path,
        request: dict[str, Any],
    ) -> dict[str, Any] | None:
        verification_path = attempt_dir / "verification_raw.json"
        if not verification_path.exists():
            return None
        trajectories = self.load_attempt_trajectories(attempt_dir, request)
        allowed_ids = {item["candidate_id"] for item in self.flatten_candidates(trajectories)}
        verification = self.normalize_verification_report(load_json(verification_path), allowed_ids)
        return {
            "trajectories": trajectories,
            "verification": verification,
        }

    def model_attempt_dir(self, round_index: int, attempt: int, backend: str = "codex") -> Path:
        paths = self.round_paths(round_index)
        return self.backend_attempt_root(paths, backend) / f"attempt_{attempt:03d}"

    def codex_attempt_dir(self, round_index: int, attempt: int) -> Path:
        return self.model_attempt_dir(round_index, attempt, backend="codex")

    def find_prior_verification_state(
        self,
        paths: dict[str, Path],
        request: dict[str, Any],
        attempt: int,
        backend: str = "codex",
    ) -> dict[str, Any] | None:
        attempt_root = self.backend_attempt_root(paths, backend)
        for previous_attempt in range(attempt - 1, 0, -1):
            attempt_dir = attempt_root / f"attempt_{previous_attempt:03d}"
            if not attempt_dir.exists():
                continue
            try:
                state = self.load_attempt_verification_state(attempt_dir, request)
            except Exception:  # noqa: BLE001
                continue
            if state:
                return state
        return None

    def load_verification_state_for_debug(
        self,
        round_index: int,
        request: dict[str, Any],
        attempt: int | None,
    ) -> dict[str, Any]:
        if attempt is None:
            trajectories = self.load_round_trajectories(round_index, request)
            verification_path = self.round_paths(round_index)["verification_report"]
            allowed_ids = {item["candidate_id"] for item in self.flatten_candidates(trajectories)}
            verification = None
            if verification_path.exists():
                verification = self.normalize_verification_report(load_json(verification_path), allowed_ids)
            return {
                "trajectories": trajectories,
                "verification": verification,
            }
        attempt_dir = self.codex_attempt_dir(round_index, attempt)
        if not attempt_dir.exists():
            raise HarnessError(f"missing attempt directory: {attempt_dir}")
        state = self.load_attempt_verification_state(attempt_dir, request)
        if state is None:
            raise HarnessError(f"attempt {attempt} does not contain verifier artifacts: {attempt_dir}")
        return state

    def plan_verification_reuse(
        self,
        request: dict[str, Any],
        trajectories: list[dict[str, Any]],
        prior_state: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if prior_state is None:
            return None
        current_candidates = self.flatten_candidates(trajectories)
        prior_candidates = self.flatten_candidates(prior_state["trajectories"])
        prior_review_lookup = self.review_lookup(prior_state["verification"])
        reusable_by_fingerprint: dict[str, list[dict[str, Any]]] = {}
        for candidate in prior_candidates:
            candidate_id = candidate["candidate_id"]
            if candidate_id not in prior_review_lookup:
                continue
            fingerprint = self.candidate_reuse_fingerprint(candidate)
            reusable_by_fingerprint.setdefault(fingerprint, []).append(
                {
                    "candidate": candidate,
                    "review": prior_review_lookup[candidate_id],
                }
            )

        reused_entries: list[dict[str, Any]] = []
        changed_candidates: list[dict[str, Any]] = []
        for candidate in current_candidates:
            fingerprint = self.candidate_reuse_fingerprint(candidate)
            bucket = reusable_by_fingerprint.get(fingerprint, [])
            if bucket:
                matched = bucket.pop(0)
                review = copy.deepcopy(matched["review"])
                review["candidate_id"] = candidate["candidate_id"]
                reused_entries.append(
                    {
                        "candidate": candidate,
                        "review": review,
                        "source_candidate_id": matched["candidate"]["candidate_id"],
                        "fingerprint": fingerprint,
                    }
                )
            else:
                changed_candidates.append(candidate)

        current_ids = [candidate["candidate_id"] for candidate in current_candidates]
        reused_ids = [entry["candidate"]["candidate_id"] for entry in reused_entries]
        changed_ids = [candidate["candidate_id"] for candidate in changed_candidates]
        reused_keep_entries = [
            entry for entry in reused_entries if entry["review"]["verdict"] == "KEEP"
        ]
        reused_keep_ids = [entry["candidate"]["candidate_id"] for entry in reused_keep_entries]
        return {
            "all_candidate_ids": current_ids,
            "reused_entries": reused_entries,
            "reused_review_cards": self.build_reused_review_cards(reused_keep_entries),
            "reused_candidate_ids": reused_ids,
            "reused_keep_ids": reused_keep_ids,
            "override_candidate_ids": reused_keep_ids,
            "changed_claim_cards": [
                card
                for card in self.verifier_claim_cards(trajectories, request)
                if card["candidate_id"] in set(changed_ids)
            ],
            "changed_candidate_ids": changed_ids,
            "reused_candidate_count": len(reused_ids),
            "changed_candidate_count": len(changed_ids),
            "all_reused": bool(current_ids) and not changed_ids,
            "prior_verification": copy.deepcopy(prior_state["verification"]),
        }

    def reuse_verification_report(self, reuse_plan: dict[str, Any]) -> dict[str, Any]:
        verification = copy.deepcopy(reuse_plan["prior_verification"])
        review_lookup = {
            entry["candidate"]["candidate_id"]: copy.deepcopy(entry["review"])
            for entry in reuse_plan["reused_entries"]
        }
        ordered_reviews = [review_lookup[candidate_id] for candidate_id in reuse_plan["all_candidate_ids"]]
        verification["family_reviews"] = ordered_reviews
        verification["keep_ids"] = [
            candidate_id
            for candidate_id in reuse_plan["all_candidate_ids"]
            if review_lookup[candidate_id]["verdict"] == "KEEP"
        ]
        verification["global_warnings"] = list(verification.get("global_warnings", []))
        verification["global_warnings"].append(
            f"Reused verifier reviews for all {len(reuse_plan['all_candidate_ids'])} unchanged candidates."
        )
        return verification

    def merge_verification_patch(
        self,
        reuse_plan: dict[str, Any],
        patch: dict[str, Any],
    ) -> dict[str, Any]:
        review_lookup = {
            entry["candidate"]["candidate_id"]: copy.deepcopy(entry["review"])
            for entry in reuse_plan["reused_entries"]
        }
        for review in patch["family_reviews"]:
            review_lookup[review["candidate_id"]] = copy.deepcopy(review)
        ordered_reviews = [review_lookup[candidate_id] for candidate_id in reuse_plan["all_candidate_ids"]]
        return {
            "verdict": patch["verdict"],
            "summary": patch["summary"],
            "global_errors": [str(item) for item in patch.get("global_errors", [])],
            "global_warnings": [
                *[str(item) for item in patch.get("global_warnings", [])],
                (
                    f"Reused {reuse_plan['reused_candidate_count']} unchanged reviews and "
                    f"re-reviewed {len(patch['family_reviews'])} candidate(s)."
                ),
            ],
            "feedback_to_generator": str(patch.get("feedback_to_generator", "")).strip(),
            "family_reviews": ordered_reviews,
            "keep_ids": [
                candidate_id
                for candidate_id in reuse_plan["all_candidate_ids"]
                if review_lookup[candidate_id]["verdict"] == "KEEP"
            ],
        }

    def review_lookup(self, verification: dict[str, Any]) -> dict[str, dict[str, Any]]:
        return {item["candidate_id"]: item for item in verification["family_reviews"]}

    def verification_runnable_split(
        self,
        trajectories: list[dict[str, Any]],
        verification: dict[str, Any],
    ) -> dict[str, Any]:
        candidate_lookup = self.candidate_lookup(trajectories)
        review_lookup = self.review_lookup(verification)
        runnable_keep: list[dict[str, Any]] = []
        blocked_keep: list[dict[str, Any]] = []
        for candidate_id in verification["keep_ids"]:
            candidate = candidate_lookup.get(candidate_id)
            if candidate is None:
                continue
            review = review_lookup.get(candidate_id, {})
            entry = {
                "candidate_id": candidate_id,
                "family_group": candidate["family_group"],
                "name": candidate["name"],
                "search_level": candidate["search_level"],
                "lane": candidate["lane"],
                "phase_window": candidate["phase_window"],
                "implementation_mode": candidate["implementation_mode"],
                "verifier_review_verdict": review.get("verdict"),
                "consequence": candidate["consequence"],
            }
            if candidate["implementation_mode"] == "catalog_executable_now":
                runnable_keep.append(entry)
            else:
                blocked_keep.append(entry)
        runnable_keep.sort(key=lambda item: item["candidate_id"])
        blocked_keep.sort(key=lambda item: item["candidate_id"])
        return {
            "runnable_keep_ids": [item["candidate_id"] for item in runnable_keep],
            "blocked_keep_ids": [item["candidate_id"] for item in blocked_keep],
            "runnable_keep_families": runnable_keep,
            "blocked_keep_families": blocked_keep,
        }

    def build_explorer_prompt(
        self,
        request: dict[str, Any],
        trajectory_id: str,
        previous_feedback: str | None,
    ) -> str:
        sections = [
            "You are one independent exploration trajectory in pg_enigma.",
            "Return JSON only.",
            "Do not write code. Do not output prose outside the JSON object.",
            "",
            f"TRAJECTORY ID: {trajectory_id}",
            "",
            "STAGE RULES",
            f"- Emit exactly {request['exploration']['families_per_trajectory']} hypotheses.",
            "- Every hypothesis must be consequential, not a local threshold or scalar retune.",
            "- Use different family_group values within this trajectory.",
            "- Use the self-check rubric before finalizing each idea.",
            "- Keep justification fields compact; the verifier will audit them directly.",
            "- Every hypothesis must include measurement_plan and implementation_basis.",
            "- If a line is not strong enough, move it to rejected_lines instead of hypotheses.",
            "- Be honest about implementation_mode: catalog_executable_now, needs_new_primitive, or needs_new_base_cycle.",
            "",
            "ROUND REQUEST JSON",
            json.dumps(request, indent=2),
        ]
        doctrine = self.prompt_text("explorer").strip()
        if doctrine:
            sections.extend(["", "EXPLORER DOCTRINE", doctrine])
        if previous_feedback:
            sections.extend(["", "PREVIOUS ATTEMPT FEEDBACK", previous_feedback])
        return "\n".join(sections) + "\n"

    def build_verifier_prompt(
        self,
        request: dict[str, Any],
        trajectories: list[dict[str, Any]],
        previous_feedback: str | None,
    ) -> str:
        verification = request["verification"]
        claim_cards = self.verifier_claim_cards(trajectories, request)
        sections = [
            "You are the adversarial verifier for pg_enigma.",
            "Return JSON only.",
            "",
            "AUDIT MODE",
            "- The explorer already supplied structured claim cards with justifications.",
            "- The harness already enforced schema validity, required fields, candidate-id uniqueness, and within-trajectory family_group uniqueness.",
            "- Do not rediscover the ideas from scratch. Audit contradictions, weak premises, fake novelty, lane mismatch, and dishonest implementation claims.",
            "",
            "VERDICT RULES",
            "- PASS: enough consequential ideas survive to justify a multi-family campaign",
            "- RETRY: the exploration is repairable, but the generator should try again",
            "- FAIL: there is no credible consequential direction in the current material",
            "",
            "HARD RULES",
            "- Review every candidate exactly once.",
            f"- keep_ids should contain at least {verification['min_keep']} candidates only if they are truly strong keeps.",
            (
                f"- Any keep should plausibly score at least {verification['min_consequence_score']} on consequence, "
                f"{verification['min_falsifiability_score']} on falsifiability, and "
                f"{verification['min_implementation_honesty_score']} on implementation honesty."
            ),
            "- If an idea is promising but too local, use REWRITE and explain how to move it up one search level.",
            "- Do not keep local retunes just to satisfy a quota.",
            "",
            "VERIFICATION REQUEST SUMMARY JSON",
            json.dumps(self.verifier_request_summary(request), indent=2),
            "",
            "CANDIDATE CLAIM CARDS JSON",
            json.dumps(claim_cards, indent=2),
        ]
        doctrine = self.prompt_text("verifier").strip()
        if doctrine:
            sections.extend(["", "VERIFIER DOCTRINE", doctrine])
        if previous_feedback:
            sections.extend(["", "PREVIOUS ATTEMPT FEEDBACK", previous_feedback])
        return "\n".join(sections) + "\n"

    def build_verifier_patch_prompt(
        self,
        request: dict[str, Any],
        reuse_plan: dict[str, Any],
        previous_feedback: str | None,
    ) -> str:
        verification = request["verification"]
        sections = [
            "You are the adversarial verifier for pg_enigma.",
            "Return JSON only.",
            "",
            "DIFF REVIEW MODE",
            "- Reused candidate reviews below were already verified against identical claim fingerprints.",
            "- Carry reused reviews forward as-is unless a changed candidate creates a direct contradiction, duplicate, or stronger replacement that should override one of them.",
            "- Review every candidate in CHANGED CANDIDATE CLAIM CARDS exactly once.",
            "- You may also review any candidate in REUSED VERIFIED REVIEWS if you need to override its prior verdict.",
            "- keep_ids should include only KEEP candidate_ids that you reviewed in this pass.",
            "- The final merged report will preserve reused reviews you do not override and combine them with your reviewed candidates.",
            "",
            "VERDICT RULES",
            "- PASS: enough consequential ideas survive in the merged slate to justify a multi-family campaign",
            "- RETRY: the changed material is repairable, but the generator should try again",
            "- FAIL: there is no credible consequential direction in the merged material",
            "",
            "HARD RULES",
            (
                f"- The merged round still needs at least {verification['min_keep']} total KEEP candidates, "
                "counting reused keeps plus any new keeps from this pass."
            ),
            (
                f"- Any reviewed keep should plausibly score at least {verification['min_consequence_score']} on consequence, "
                f"{verification['min_falsifiability_score']} on falsifiability, and "
                f"{verification['min_implementation_honesty_score']} on implementation honesty."
            ),
            "- Do not re-review reused candidates unless you are explicitly overriding them.",
            "- If a changed idea is promising but too local, use REWRITE and explain how to move it up one search level.",
            "",
            "VERIFICATION REQUEST SUMMARY JSON",
            json.dumps(self.verifier_request_summary(request), indent=2),
            "",
            "REVIEW REUSE SUMMARY JSON",
            json.dumps(
                {
                    "reused_candidate_count": reuse_plan["reused_candidate_count"],
                    "changed_candidate_count": reuse_plan["changed_candidate_count"],
                    "reused_keep_count": len(reuse_plan["reused_keep_ids"]),
                    "override_candidate_ids": reuse_plan["override_candidate_ids"],
                },
                indent=2,
            ),
            "",
            "REUSED VERIFIED REVIEWS JSON",
            json.dumps(reuse_plan["reused_review_cards"], indent=2),
            "",
            "CHANGED CANDIDATE CLAIM CARDS JSON",
            json.dumps(reuse_plan["changed_claim_cards"], indent=2),
        ]
        doctrine = self.prompt_text("verifier").strip()
        if doctrine:
            sections.extend(["", "VERIFIER DOCTRINE", doctrine])
        if previous_feedback:
            sections.extend(["", "PREVIOUS ATTEMPT FEEDBACK", previous_feedback])
        return "\n".join(sections) + "\n"

    def build_campaign_prompt(
        self,
        request: dict[str, Any],
        trajectories: list[dict[str, Any]],
        verification: dict[str, Any],
        previous_feedback: str | None,
    ) -> str:
        candidate_lookup = self.candidate_lookup(trajectories)
        review_lookup = self.review_lookup(verification)
        kept_candidates = [
            {
                "candidate": candidate_lookup[candidate_id],
                "verifier_review": review_lookup[candidate_id],
            }
            for candidate_id in verification["keep_ids"]
            if candidate_id in candidate_lookup and candidate_id in review_lookup
        ]
        sections = [
            "You are the campaign distillation stage for pg_enigma.",
            "Return JSON only.",
            "",
            "CAMPAIGN RULES",
            "- Do not collapse the round to one final answer.",
            f"- Keep at least {request['verification']['min_keep']} families for a READY campaign.",
            "- Every family in the campaign must define lane, phase_window, pack_kind, and compiler_pass_k.",
            "- compiler_pass_k should be at least 2 for any READY family.",
            "- pack_strategy must define how solo packs are organized before composition.",
            "- promotion_policy and composition_policy must make lineage of confidence explicit.",
            "- campaign_handoff must describe how downstream agents should execute the campaign.",
            "",
            "ROUND REQUEST SUMMARY JSON",
            json.dumps(self.prompt_request_summary(request), indent=2),
            "",
            "VERIFIER REPORT JSON",
            json.dumps(verification, indent=2),
            "",
            "KEPT CANDIDATES JSON",
            json.dumps(kept_candidates, indent=2),
        ]
        doctrine = self.prompt_text("distiller").strip()
        if doctrine:
            sections.extend(["", "CAMPAIGN DISTILLER DOCTRINE", doctrine])
        if previous_feedback:
            sections.extend(["", "PREVIOUS ATTEMPT FEEDBACK", previous_feedback])
        return "\n".join(sections) + "\n"

    def build_compiler_prompt(
        self,
        request: dict[str, Any],
        campaign: dict[str, Any],
        dossier: dict[str, Any],
        previous_feedback: str | None,
    ) -> str:
        sections = [
            "You are the family compiler stage for pg_enigma.",
            "Return JSON only.",
            "",
            "COMPILER RULES",
            f"- Emit exactly {dossier['compiler_pass_k']} realizations when verdict=READY.",
            "- Preserve the family mechanism; vary the executable realization.",
            "- Do not emit threshold variants disguised as different realizations.",
            "- Keep lane, phase_window, and pack_kind coherent with the family dossier.",
            "- If the family is catalog_executable_now, every realization should target current_search_harness_catalog.",
            "- If the family is blocked on a primitive or a base cycle, do not pretend the current search harness can execute it now.",
            "- The instructions field for each realization should be detailed enough for a downstream search_harness agent or coding agent to act on.",
            "",
            "ROUND REQUEST JSON",
            json.dumps(request, indent=2),
            "",
            "CAMPAIGN JSON",
            json.dumps(campaign, indent=2),
            "",
            "FAMILY DOSSIER JSON",
            json.dumps(dossier, indent=2),
        ]
        doctrine = self.prompt_text("compiler").strip()
        if doctrine:
            sections.extend(["", "COMPILER DOCTRINE", doctrine])
        if previous_feedback:
            sections.extend(["", "PREVIOUS ATTEMPT FEEDBACK", previous_feedback])
        return "\n".join(sections) + "\n"

    def summarize_pack_evidence(self, round_index: int) -> list[dict[str, Any]]:
        paths = self.round_paths(round_index)
        if not paths["pack_handoffs"].exists():
            return []
        payload = load_json(paths["pack_handoffs"])
        evidence: list[dict[str, Any]] = []
        for handoff in payload.get("handoffs", []):
            if not isinstance(handoff, dict):
                continue
            item = {
                "pack_id": str(handoff.get("pack_id", "")),
                "pack_kind": str(handoff.get("pack_kind", "")),
                "lane": str(handoff.get("lane", "")),
                "phase_window": str(handoff.get("phase_window", "")),
                "generation": handoff.get("generation"),
                "status": "pending",
                "candidate_results": [],
            }
            config_path = handoff.get("search_harness_config")
            generation = handoff.get("generation")
            if not config_path or generation is None:
                evidence.append(item)
                continue
            try:
                harness = SearchHarness(Path(str(config_path)).resolve())
                generation_paths = harness.generation_paths(int(generation))
                summary_path = generation_paths["generation_summary"]
                results_path = generation_paths["results_index"]
                if not summary_path.exists() or not results_path.exists():
                    evidence.append(item)
                    continue
                summary = load_json(summary_path)
                results_index = load_json(results_path)
                metric_path = str(summary.get("primary_metric_path", "metrics.score_bpb"))
                goal = str(summary.get("goal", "minimize"))
                control_slots = [str(slot) for slot in summary.get("control_slots", [])]
                control_values: list[float] = []
                for slot_id in control_slots:
                    try:
                        control_values.append(float(nested_get(results_index["slots"][slot_id], metric_path)))
                    except Exception:  # noqa: BLE001
                        continue
                item.update(
                    {
                        "status": "evaluated",
                        "metric_path": metric_path,
                        "goal": goal,
                        "control_slots": control_slots,
                        "control_values": control_values,
                        "control_spread": summary.get("control_spread"),
                        "invalid_reasons": [str(reason) for reason in summary.get("invalid_reasons", [])],
                        "survivors": [str(slot) for slot in summary.get("survivors", [])],
                    }
                )
                for slot_id, meta in handoff.get("candidate_slot_map", {}).items():
                    result = {
                        "slot_id": slot_id,
                        "candidate_id": str(meta.get("candidate_id", "")),
                        "family_group": str(meta.get("family_group", "")),
                        "realization_id": str(meta.get("realization_id", "")),
                        "title": str(meta.get("title", "")),
                    }
                    try:
                        slot_payload = results_index["slots"][slot_id]
                        result["metric"] = float(nested_get(slot_payload, metric_path))
                        result["survived"] = slot_id in item["survivors"]
                    except Exception as exc:  # noqa: BLE001
                        result["error"] = str(exc)
                    item["candidate_results"].append(result)
            except Exception as exc:  # noqa: BLE001
                item["status"] = "error"
                item["error"] = str(exc)
            evidence.append(item)
        return evidence

    def build_postmortem_prompt(
        self,
        request: dict[str, Any],
        campaign: dict[str, Any],
        promotion_report: dict[str, Any],
        pack_evidence: list[dict[str, Any]],
        previous_feedback: str | None,
    ) -> str:
        sections = [
            "You are the round analyst for pg_enigma.",
            "Return JSON only.",
            "",
            "ANALYST RULES",
            "- Compare actual pack evidence against the original family hypotheses.",
            "- Review every campaign family exactly once.",
            "- Distinguish confirmed, weakened, rejected, inconclusive, and pending outcomes honestly.",
            "- Focus on what the evidence says about the mechanism, not just whether a slot ranked first.",
            "- generation_instruction_delta must help the next round move, not merely restate the failure.",
            "",
            "ROUND REQUEST JSON",
            json.dumps(request, indent=2),
            "",
            "CAMPAIGN JSON",
            json.dumps(campaign, indent=2),
            "",
            "PROMOTION REPORT JSON",
            json.dumps(promotion_report, indent=2),
            "",
            "PACK EVIDENCE JSON",
            json.dumps(pack_evidence, indent=2),
        ]
        doctrine = self.prompt_text("analyst").strip()
        if doctrine:
            sections.extend(["", "ANALYST DOCTRINE", doctrine])
        if previous_feedback:
            sections.extend(["", "PREVIOUS ATTEMPT FEEDBACK", previous_feedback])
        return "\n".join(sections) + "\n"

    def normalize_exploration_slate(
        self,
        payload: dict[str, Any],
        expected_trajectory_id: str,
        expected_count: int,
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HarnessError("exploration payload must be a JSON object")
        trajectory_id = str(payload.get("trajectory_id", "")).strip()
        if trajectory_id != expected_trajectory_id:
            raise HarnessError(
                f"exploration trajectory_id must be {expected_trajectory_id}, got: {trajectory_id or '<empty>'}"
            )
        worldview = str(payload.get("worldview", "")).strip()
        if not worldview:
            raise HarnessError(f"{trajectory_id} must define a non-empty worldview")
        raw_hypotheses = payload.get("hypotheses")
        if not isinstance(raw_hypotheses, list) or len(raw_hypotheses) != expected_count:
            raise HarnessError(f"{trajectory_id} must contain exactly {expected_count} hypotheses")
        seen_ids: set[str] = set()
        seen_groups: set[str] = set()
        normalized_hypotheses: list[dict[str, Any]] = []
        for raw in raw_hypotheses:
            if not isinstance(raw, dict):
                raise HarnessError(f"{trajectory_id} hypotheses must be JSON objects")
            hypothesis_id = str(raw.get("id", "")).strip()
            if not hypothesis_id:
                raise HarnessError(f"{trajectory_id} contains a hypothesis with an empty id")
            if hypothesis_id in seen_ids:
                raise HarnessError(f"{trajectory_id} contains duplicate hypothesis id: {hypothesis_id}")
            seen_ids.add(hypothesis_id)

            name = str(raw.get("name", "")).strip()
            search_level = str(raw.get("search_level", "")).strip()
            lane = str(raw.get("lane", "")).strip()
            mechanism = str(raw.get("mechanism", "")).strip()
            broken_invariant = str(raw.get("broken_invariant", "")).strip()
            not_local_tuning_reason = str(raw.get("not_local_tuning_reason", "")).strip()
            measurement_plan = str(raw.get("measurement_plan", raw.get("expected_observable", ""))).strip()
            smallest_decisive_probe = str(raw.get("smallest_decisive_probe", "")).strip()
            expected_observable = str(raw.get("expected_observable", "")).strip()
            falsifier = str(raw.get("falsifier", "")).strip()
            novelty_vs_history = str(raw.get("novelty_vs_history", "")).strip()
            implementation_mode = str(raw.get("implementation_mode", "")).strip()
            implementation_basis = str(raw.get("implementation_basis", implementation_mode)).strip()
            family_group = str(raw.get("family_group", "")).strip()
            self_check_summary = str(raw.get("self_check_summary", "")).strip()

            if not name:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} must define a non-empty name")
            if search_level not in ALLOWED_SEARCH_LEVELS:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} has invalid search_level: {search_level}")
            if not lane:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} must define a non-empty lane")
            if not mechanism:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} must define a non-empty mechanism")
            if not broken_invariant:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} must define broken_invariant")
            if not not_local_tuning_reason:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} must explain why it is not local tuning")
            if not measurement_plan:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} must define measurement_plan")
            if not smallest_decisive_probe:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} must define smallest_decisive_probe")
            if not expected_observable:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} must define expected_observable")
            if not falsifier:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} must define a falsifier")
            if not novelty_vs_history:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} must define novelty_vs_history")
            if implementation_mode not in ALLOWED_IMPLEMENTATION_MODES:
                raise HarnessError(
                    f"{trajectory_id}/{hypothesis_id} has invalid implementation_mode: {implementation_mode}"
                )
            if not implementation_basis:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} must define implementation_basis")
            if not family_group:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} must define family_group")
            if family_group in seen_groups:
                raise HarnessError(f"{trajectory_id} contains duplicate family_group values: {family_group}")
            seen_groups.add(family_group)
            if not self_check_summary:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} must define self_check_summary")

            raw_axes = raw.get("consequence_axes")
            if not isinstance(raw_axes, list) or not raw_axes:
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} must define non-empty consequence_axes")
            consequence_axes = dedupe_strings([str(item) for item in raw_axes])
            invalid_axes = [axis for axis in consequence_axes if axis not in ALLOWED_CONSEQUENCE_AXES]
            if invalid_axes:
                raise HarnessError(
                    f"{trajectory_id}/{hypothesis_id} has invalid consequence axes: {invalid_axes}"
                )

            raw_scores = raw.get("self_scores")
            if not isinstance(raw_scores, dict):
                raise HarnessError(f"{trajectory_id}/{hypothesis_id} self_scores must be an object")
            scores: dict[str, int] = {}
            for key in SCORE_FIELDS:
                if key not in raw_scores:
                    raise HarnessError(f"{trajectory_id}/{hypothesis_id} self_scores missing {key}")
                value = int(raw_scores[key])
                if value < 1 or value > 5:
                    raise HarnessError(f"{trajectory_id}/{hypothesis_id} self_scores.{key} must be 1..5")
                scores[key] = value

            normalized_hypotheses.append(
                {
                    "id": hypothesis_id,
                    "candidate_id": canonical_candidate_id(trajectory_id, hypothesis_id),
                    "name": name,
                    "search_level": search_level,
                    "lane": lane,
                    "mechanism": mechanism,
                    "broken_invariant": broken_invariant,
                    "consequence_axes": consequence_axes,
                    "not_local_tuning_reason": not_local_tuning_reason,
                    "measurement_plan": measurement_plan,
                    "smallest_decisive_probe": smallest_decisive_probe,
                    "expected_observable": expected_observable,
                    "falsifier": falsifier,
                    "novelty_vs_history": novelty_vs_history,
                    "implementation_mode": implementation_mode,
                    "implementation_basis": implementation_basis,
                    "family_group": family_group,
                    "self_scores": scores,
                    "self_check_summary": self_check_summary,
                }
            )

        rejected_lines = [str(item) for item in payload.get("rejected_lines", [])]
        return {
            "trajectory_id": trajectory_id,
            "worldview": worldview,
            "hypotheses": normalized_hypotheses,
            "rejected_lines": rejected_lines,
        }

    def normalize_verification_report(
        self,
        payload: dict[str, Any],
        allowed_ids: set[str],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HarnessError("verification report must be a JSON object")
        verdict = str(payload.get("verdict", "")).strip()
        if verdict not in ALLOWED_VERIFIER_VERDICTS:
            raise HarnessError(f"verification verdict must be one of {sorted(ALLOWED_VERIFIER_VERDICTS)}")
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            raise HarnessError("verification summary must be non-empty")
        feedback = str(payload.get("feedback_to_generator", "")).strip()
        raw_reviews = payload.get("family_reviews")
        if not isinstance(raw_reviews, list) or not raw_reviews:
            raise HarnessError("verification family_reviews must be a non-empty list")

        reviews: list[dict[str, Any]] = []
        seen_review_ids: set[str] = set()
        for raw in raw_reviews:
            if not isinstance(raw, dict):
                raise HarnessError("each family review must be a JSON object")
            candidate_id = str(raw.get("candidate_id", "")).strip()
            if candidate_id not in allowed_ids:
                raise HarnessError(f"verification reviewed unknown candidate_id: {candidate_id}")
            if candidate_id in seen_review_ids:
                raise HarnessError(f"verification duplicated candidate_id: {candidate_id}")
            seen_review_ids.add(candidate_id)
            review_verdict = str(raw.get("verdict", "")).strip()
            if review_verdict not in ALLOWED_REVIEW_VERDICTS:
                raise HarnessError(f"verification review has invalid verdict: {review_verdict}")
            review = {
                "candidate_id": candidate_id,
                "verdict": review_verdict,
                "consequence_score": int(raw.get("consequence_score")),
                "novelty_score": int(raw.get("novelty_score")),
                "falsifiability_score": int(raw.get("falsifiability_score")),
                "lane_integrity_score": int(raw.get("lane_integrity_score")),
                "implementation_honesty_score": int(raw.get("implementation_honesty_score")),
                "reasons": [str(item) for item in raw.get("reasons", [])],
                "rewrite_instructions": [str(item) for item in raw.get("rewrite_instructions", [])],
            }
            for score_key in (
                "consequence_score",
                "novelty_score",
                "falsifiability_score",
                "lane_integrity_score",
                "implementation_honesty_score",
            ):
                value = int(review[score_key])
                if value < 1 or value > 5:
                    raise HarnessError(f"verification {candidate_id} {score_key} must be 1..5")
            reviews.append(review)

        if seen_review_ids != allowed_ids:
            missing = sorted(allowed_ids.difference(seen_review_ids))
            raise HarnessError(f"verification must review every candidate exactly once; missing: {missing}")

        keep_ids = dedupe_strings([str(item) for item in payload.get("keep_ids", [])])
        unknown_keep_ids = [candidate_id for candidate_id in keep_ids if candidate_id not in allowed_ids]
        if unknown_keep_ids:
            raise HarnessError(f"verification keep_ids include unknown candidates: {unknown_keep_ids}")
        return {
            "verdict": verdict,
            "summary": summary,
            "global_errors": [str(item) for item in payload.get("global_errors", [])],
            "global_warnings": [str(item) for item in payload.get("global_warnings", [])],
            "feedback_to_generator": feedback,
            "family_reviews": reviews,
            "keep_ids": keep_ids,
        }

    def normalize_verification_patch(
        self,
        payload: dict[str, Any],
        required_ids: set[str],
        reusable_ids: set[str],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HarnessError("verification patch must be a JSON object")
        verdict = str(payload.get("verdict", "")).strip()
        if verdict not in ALLOWED_VERIFIER_VERDICTS:
            raise HarnessError(f"verification patch verdict must be one of {sorted(ALLOWED_VERIFIER_VERDICTS)}")
        summary = str(payload.get("summary", "")).strip()
        if not summary:
            raise HarnessError("verification patch summary must be non-empty")
        feedback = str(payload.get("feedback_to_generator", "")).strip()
        raw_reviews = payload.get("family_reviews")
        if not isinstance(raw_reviews, list) or not raw_reviews:
            raise HarnessError("verification patch family_reviews must be a non-empty list")

        allowed_patch_ids = set(required_ids).union(reusable_ids)
        reviews: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for raw in raw_reviews:
            if not isinstance(raw, dict):
                raise HarnessError("each verification patch family review must be a JSON object")
            candidate_id = str(raw.get("candidate_id", "")).strip()
            if candidate_id not in allowed_patch_ids:
                raise HarnessError(f"verification patch reviewed unknown candidate_id: {candidate_id}")
            if candidate_id in seen_ids:
                raise HarnessError(f"verification patch duplicated candidate_id: {candidate_id}")
            seen_ids.add(candidate_id)
            review_verdict = str(raw.get("verdict", "")).strip()
            if review_verdict not in ALLOWED_REVIEW_VERDICTS:
                raise HarnessError(f"verification patch review has invalid verdict: {review_verdict}")
            review = {
                "candidate_id": candidate_id,
                "verdict": review_verdict,
                "consequence_score": int(raw.get("consequence_score")),
                "novelty_score": int(raw.get("novelty_score")),
                "falsifiability_score": int(raw.get("falsifiability_score")),
                "lane_integrity_score": int(raw.get("lane_integrity_score")),
                "implementation_honesty_score": int(raw.get("implementation_honesty_score")),
                "reasons": [str(item) for item in raw.get("reasons", [])],
                "rewrite_instructions": [str(item) for item in raw.get("rewrite_instructions", [])],
            }
            for score_key in (
                "consequence_score",
                "novelty_score",
                "falsifiability_score",
                "lane_integrity_score",
                "implementation_honesty_score",
            ):
                value = int(review[score_key])
                if value < 1 or value > 5:
                    raise HarnessError(f"verification patch {candidate_id} {score_key} must be 1..5")
            reviews.append(review)

        missing = sorted(required_ids.difference(seen_ids))
        if missing:
            raise HarnessError(f"verification patch must review every changed candidate exactly once; missing: {missing}")

        keep_ids = dedupe_strings([str(item) for item in payload.get("keep_ids", [])])
        unknown_keep_ids = [candidate_id for candidate_id in keep_ids if candidate_id not in seen_ids]
        if unknown_keep_ids:
            raise HarnessError(f"verification patch keep_ids include non-reviewed candidates: {unknown_keep_ids}")
        return {
            "verdict": verdict,
            "summary": summary,
            "global_errors": [str(item) for item in payload.get("global_errors", [])],
            "global_warnings": [str(item) for item in payload.get("global_warnings", [])],
            "feedback_to_generator": feedback,
            "family_reviews": reviews,
            "keep_ids": keep_ids,
        }

    def enforce_verification_thresholds(self, verification: dict[str, Any], request: dict[str, Any]) -> None:
        settings = request["verification"]
        if verification["verdict"] != "PASS":
            raise HarnessError(f"Verifier returned {verification['verdict']}: {verification['summary']}")
        if len(verification["keep_ids"]) < settings["min_keep"]:
            raise HarnessError(
                f"Verifier kept only {len(verification['keep_ids'])} candidates; "
                f"need at least {settings['min_keep']}"
            )
        review_lookup = self.review_lookup(verification)
        for candidate_id in verification["keep_ids"]:
            review = review_lookup[candidate_id]
            if review["verdict"] != "KEEP":
                raise HarnessError(f"Verifier keep_ids includes non-KEEP review: {candidate_id}")
            if review["consequence_score"] < settings["min_consequence_score"]:
                raise HarnessError(
                    f"Verifier kept {candidate_id} below consequence threshold "
                    f"{settings['min_consequence_score']}"
                )
            if review["falsifiability_score"] < settings["min_falsifiability_score"]:
                raise HarnessError(
                    f"Verifier kept {candidate_id} below falsifiability threshold "
                    f"{settings['min_falsifiability_score']}"
                )
            if review["implementation_honesty_score"] < settings["min_implementation_honesty_score"]:
                raise HarnessError(
                    f"Verifier kept {candidate_id} below implementation honesty threshold "
                    f"{settings['min_implementation_honesty_score']}"
                )

    def normalize_campaign(
        self,
        payload: dict[str, Any],
        verifier_keep_ids: set[str],
        request: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HarnessError("campaign must be a JSON object")
        verdict = str(payload.get("verdict", "")).strip()
        if verdict not in ALLOWED_CAMPAIGN_VERDICTS:
            raise HarnessError(f"campaign verdict must be one of {sorted(ALLOWED_CAMPAIGN_VERDICTS)}")
        summary = str(payload.get("summary", "")).strip()
        campaign_goal = str(payload.get("campaign_goal", "")).strip()
        dominant_search_level = str(payload.get("dominant_search_level", "")).strip()
        if not summary:
            raise HarnessError("campaign summary must be non-empty")
        if not campaign_goal:
            raise HarnessError("campaign campaign_goal must be non-empty")
        if dominant_search_level not in ALLOWED_SEARCH_LEVELS:
            raise HarnessError(f"campaign dominant_search_level is invalid: {dominant_search_level}")

        keep_ids = dedupe_strings([str(item) for item in payload.get("keep_ids", [])])
        unknown_keep_ids = [candidate_id for candidate_id in keep_ids if candidate_id not in verifier_keep_ids]
        if unknown_keep_ids:
            raise HarnessError(f"campaign keep_ids must be a subset of verifier keep_ids: {unknown_keep_ids}")

        raw_families = payload.get("families")
        if not isinstance(raw_families, list):
            raise HarnessError("campaign families must be a list")
        families: list[dict[str, Any]] = []
        seen_family_ids: set[str] = set()
        for raw in raw_families:
            if not isinstance(raw, dict):
                raise HarnessError("each campaign family must be a JSON object")
            candidate_id = str(raw.get("candidate_id", "")).strip()
            if not candidate_id:
                raise HarnessError("campaign family candidate_id must be non-empty")
            if candidate_id in seen_family_ids:
                raise HarnessError(f"campaign duplicated family candidate_id: {candidate_id}")
            seen_family_ids.add(candidate_id)
            if candidate_id not in keep_ids:
                raise HarnessError(f"campaign family candidate_id must be listed in keep_ids: {candidate_id}")

            family = {
                "candidate_id": candidate_id,
                "name": str(raw.get("name", "")).strip(),
                "family_group": str(raw.get("family_group", "")).strip(),
                "search_level": str(raw.get("search_level", "")).strip(),
                "lane": str(raw.get("lane", "")).strip(),
                "phase_window": str(raw.get("phase_window", "")).strip(),
                "pack_kind": str(raw.get("pack_kind", "")).strip(),
                "implementation_mode": str(raw.get("implementation_mode", "")).strip(),
                "compiler_pass_k": int(raw.get("compiler_pass_k", 0)),
                "why_keep": str(raw.get("why_keep", "")).strip(),
                "minimal_probe": str(raw.get("minimal_probe", "")).strip(),
                "expected_signal": str(raw.get("expected_signal", "")).strip(),
                "composition_tags": dedupe_strings([str(item) for item in raw.get("composition_tags", [])]),
                "incompatible_with": dedupe_strings([str(item) for item in raw.get("incompatible_with", [])]),
                "mutate_after_survival": [str(item) for item in raw.get("mutate_after_survival", [])],
                "pack_rationale": str(raw.get("pack_rationale", "")).strip(),
            }
            if not family["name"]:
                raise HarnessError(f"campaign family {candidate_id} must define name")
            if not family["family_group"]:
                raise HarnessError(f"campaign family {candidate_id} must define family_group")
            if family["search_level"] not in ALLOWED_SEARCH_LEVELS:
                raise HarnessError(f"campaign family {candidate_id} has invalid search_level")
            if family["lane"] not in ALLOWED_LANES:
                raise HarnessError(f"campaign family {candidate_id} has invalid lane: {family['lane']}")
            if family["phase_window"] not in ALLOWED_PHASE_WINDOWS:
                raise HarnessError(
                    f"campaign family {candidate_id} has invalid phase_window: {family['phase_window']}"
                )
            if family["pack_kind"] not in ALLOWED_PACK_KINDS:
                raise HarnessError(f"campaign family {candidate_id} has invalid pack_kind")
            if family["implementation_mode"] not in ALLOWED_IMPLEMENTATION_MODES:
                raise HarnessError(
                    f"campaign family {candidate_id} has invalid implementation_mode: "
                    f"{family['implementation_mode']}"
                )
            if family["compiler_pass_k"] < 2:
                raise HarnessError(f"campaign family {candidate_id} compiler_pass_k must be >= 2")
            if not family["why_keep"]:
                raise HarnessError(f"campaign family {candidate_id} must define why_keep")
            if not family["minimal_probe"]:
                raise HarnessError(f"campaign family {candidate_id} must define minimal_probe")
            if not family["expected_signal"]:
                raise HarnessError(f"campaign family {candidate_id} must define expected_signal")
            if not family["pack_rationale"]:
                raise HarnessError(f"campaign family {candidate_id} must define pack_rationale")
            families.append(family)

        raw_pack_strategy = payload.get("pack_strategy")
        if not isinstance(raw_pack_strategy, dict):
            raise HarnessError("campaign pack_strategy must be an object")
        pack_strategy = {
            "controls_per_pack": int(raw_pack_strategy.get("controls_per_pack", 0)),
            "max_candidates_per_pack": int(raw_pack_strategy.get("max_candidates_per_pack", 0)),
            "pack_order": [str(item) for item in raw_pack_strategy.get("pack_order", [])],
            "notes": [str(item) for item in raw_pack_strategy.get("notes", [])],
        }
        if pack_strategy["controls_per_pack"] < 2:
            raise HarnessError("campaign pack_strategy.controls_per_pack must be >= 2")
        if pack_strategy["max_candidates_per_pack"] < 1:
            raise HarnessError("campaign pack_strategy.max_candidates_per_pack must be >= 1")

        raw_promotion = payload.get("promotion_policy")
        if not isinstance(raw_promotion, dict):
            raise HarnessError("campaign promotion_policy must be an object")
        promotion_policy = {
            "min_positive_realizations": int(raw_promotion.get("min_positive_realizations", 0)),
            "require_directional_support": bool(raw_promotion.get("require_directional_support", False)),
            "family_survival_rule": str(raw_promotion.get("family_survival_rule", "")).strip(),
            "composition_entry_rule": str(raw_promotion.get("composition_entry_rule", "")).strip(),
        }
        if promotion_policy["min_positive_realizations"] < 1:
            raise HarnessError("campaign promotion_policy.min_positive_realizations must be >= 1")
        if not promotion_policy["family_survival_rule"]:
            raise HarnessError("campaign promotion_policy.family_survival_rule must be non-empty")
        if not promotion_policy["composition_entry_rule"]:
            raise HarnessError("campaign promotion_policy.composition_entry_rule must be non-empty")

        raw_composition = payload.get("composition_policy")
        if not isinstance(raw_composition, dict):
            raise HarnessError("campaign composition_policy must be an object")
        composition_policy = {
            "pairwise_after_solo_survival": bool(raw_composition.get("pairwise_after_solo_survival", False)),
            "max_pairwise_candidates": int(raw_composition.get("max_pairwise_candidates", 0)),
            "max_hybrid_candidates": int(raw_composition.get("max_hybrid_candidates", 0)),
            "hybrid_entry_rule": str(raw_composition.get("hybrid_entry_rule", "")).strip(),
            "notes": [str(item) for item in raw_composition.get("notes", [])],
        }
        if composition_policy["max_pairwise_candidates"] < 1:
            raise HarnessError("campaign composition_policy.max_pairwise_candidates must be >= 1")
        if composition_policy["max_hybrid_candidates"] < 1:
            raise HarnessError("campaign composition_policy.max_hybrid_candidates must be >= 1")
        if not composition_policy["hybrid_entry_rule"]:
            raise HarnessError("campaign composition_policy.hybrid_entry_rule must be non-empty")

        raw_handoff = payload.get("campaign_handoff")
        if not isinstance(raw_handoff, dict):
            raise HarnessError("campaign campaign_handoff must be an object")
        campaign_handoff = {
            "global_instructions": str(raw_handoff.get("global_instructions", "")).strip(),
            "control_principles": [str(item) for item in raw_handoff.get("control_principles", [])],
            "admission_principles": [str(item) for item in raw_handoff.get("admission_principles", [])],
            "metric_principles": [str(item) for item in raw_handoff.get("metric_principles", [])],
            "implementation_principles": [str(item) for item in raw_handoff.get("implementation_principles", [])],
        }
        if not campaign_handoff["global_instructions"]:
            raise HarnessError("campaign campaign_handoff.global_instructions must be non-empty")

        normalized = {
            "verdict": verdict,
            "summary": summary,
            "campaign_goal": campaign_goal,
            "dominant_search_level": dominant_search_level,
            "keep_ids": keep_ids,
            "families": families,
            "stop_doing": [str(item) for item in payload.get("stop_doing", [])],
            "pack_strategy": pack_strategy,
            "promotion_policy": promotion_policy,
            "composition_policy": composition_policy,
            "campaign_handoff": campaign_handoff,
        }

        if verdict == "READY":
            if len(keep_ids) < request["verification"]["min_keep"]:
                raise HarnessError(
                    f"READY campaign must keep at least {request['verification']['min_keep']} families"
                )
            if not families:
                raise HarnessError("READY campaign must contain families")
            family_ids = {item["candidate_id"] for item in families}
            if family_ids != set(keep_ids):
                raise HarnessError("campaign families must match keep_ids exactly for READY verdict")
        return normalized

    def normalize_family_compile(
        self,
        payload: dict[str, Any],
        dossier: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HarnessError("family compile must be a JSON object")
        candidate_id = str(payload.get("candidate_id", "")).strip()
        family_group = str(payload.get("family_group", "")).strip()
        lane = str(payload.get("lane", "")).strip()
        phase_window = str(payload.get("phase_window", "")).strip()
        pack_kind = str(payload.get("pack_kind", "")).strip()
        verdict = str(payload.get("verdict", "")).strip()
        summary = str(payload.get("summary", "")).strip()
        compiler_pass_k = int(payload.get("compiler_pass_k", 0))

        if candidate_id != dossier["candidate_id"]:
            raise HarnessError(
                f"family compile candidate_id must be {dossier['candidate_id']}, got: {candidate_id}"
            )
        if family_group != dossier["family_group"]:
            raise HarnessError(
                f"family compile family_group must be {dossier['family_group']}, got: {family_group}"
            )
        if lane != dossier["lane"]:
            raise HarnessError(f"family compile lane must be {dossier['lane']}, got: {lane}")
        if phase_window != dossier["phase_window"]:
            raise HarnessError(
                f"family compile phase_window must be {dossier['phase_window']}, got: {phase_window}"
            )
        if pack_kind != dossier["pack_kind"]:
            raise HarnessError(f"family compile pack_kind must be {dossier['pack_kind']}, got: {pack_kind}")
        if verdict not in ALLOWED_COMPILE_VERDICTS:
            raise HarnessError(f"family compile verdict must be one of {sorted(ALLOWED_COMPILE_VERDICTS)}")
        if not summary:
            raise HarnessError("family compile summary must be non-empty")
        if compiler_pass_k != int(dossier["compiler_pass_k"]):
            raise HarnessError(
                f"family compile compiler_pass_k must be {dossier['compiler_pass_k']}, got: {compiler_pass_k}"
            )

        raw_realizations = payload.get("realizations", [])
        if not isinstance(raw_realizations, list):
            raise HarnessError("family compile realizations must be a list")
        if verdict == "READY" and len(raw_realizations) != compiler_pass_k:
            raise HarnessError(
                f"READY family compile must contain exactly {compiler_pass_k} realizations"
            )

        realizations: list[dict[str, Any]] = []
        seen_realization_ids: set[str] = set()
        for raw in raw_realizations:
            if not isinstance(raw, dict):
                raise HarnessError("each realization must be a JSON object")
            realization_id = str(raw.get("realization_id", "")).strip()
            if not realization_id:
                raise HarnessError(f"{candidate_id} has a realization with an empty realization_id")
            if realization_id in seen_realization_ids:
                raise HarnessError(f"{candidate_id} duplicated realization_id: {realization_id}")
            seen_realization_ids.add(realization_id)
            realization = {
                "realization_id": realization_id,
                "realization_key": f"{candidate_id}/{realization_id}",
                "title": str(raw.get("title", "")).strip(),
                "mechanism_preservation": str(raw.get("mechanism_preservation", "")).strip(),
                "implementation_target": str(raw.get("implementation_target", "")).strip(),
                "lane": str(raw.get("lane", "")).strip(),
                "phase_window": str(raw.get("phase_window", "")).strip(),
                "pack_kind": str(raw.get("pack_kind", "")).strip(),
                "instructions": str(raw.get("instructions", "")).strip(),
                "control_plan": [str(item) for item in raw.get("control_plan", [])],
                "admission_rules": [str(item) for item in raw.get("admission_rules", [])],
                "metric_plan": [str(item) for item in raw.get("metric_plan", [])],
                "implementation_notes": [str(item) for item in raw.get("implementation_notes", [])],
                "composition_tags": dedupe_strings([str(item) for item in raw.get("composition_tags", [])]),
                "incompatible_with": dedupe_strings([str(item) for item in raw.get("incompatible_with", [])]),
                "mutate_after_survival": [str(item) for item in raw.get("mutate_after_survival", [])],
            }
            if not realization["title"]:
                raise HarnessError(f"{candidate_id}/{realization_id} must define title")
            if not realization["mechanism_preservation"]:
                raise HarnessError(f"{candidate_id}/{realization_id} must define mechanism_preservation")
            if realization["implementation_target"] not in ALLOWED_REALIZATION_TARGETS:
                raise HarnessError(
                    f"{candidate_id}/{realization_id} has invalid implementation_target: "
                    f"{realization['implementation_target']}"
                )
            if realization["lane"] != lane:
                raise HarnessError(f"{candidate_id}/{realization_id} lane must match top-level lane {lane}")
            if realization["phase_window"] != phase_window:
                raise HarnessError(
                    f"{candidate_id}/{realization_id} phase_window must match top-level phase_window {phase_window}"
                )
            if realization["pack_kind"] != pack_kind:
                raise HarnessError(
                    f"{candidate_id}/{realization_id} pack_kind must match top-level pack_kind {pack_kind}"
                )
            if not realization["instructions"]:
                raise HarnessError(f"{candidate_id}/{realization_id} must define instructions")

            if dossier["implementation_mode"] == "catalog_executable_now":
                if realization["implementation_target"] != "current_search_harness_catalog":
                    raise HarnessError(
                        f"{candidate_id}/{realization_id} must target current_search_harness_catalog"
                    )
            else:
                if realization["implementation_target"] == "current_search_harness_catalog":
                    raise HarnessError(
                        f"{candidate_id}/{realization_id} cannot target current_search_harness_catalog "
                        "when the family is blocked"
                    )
            realizations.append(realization)

        dropped_realizations = [str(item) for item in payload.get("dropped_realizations", [])]
        if verdict == "READY" and not realizations:
            raise HarnessError("READY family compile must contain realizations")
        return {
            "candidate_id": candidate_id,
            "family_group": family_group,
            "lane": lane,
            "phase_window": phase_window,
            "pack_kind": pack_kind,
            "verdict": verdict,
            "summary": summary,
            "compiler_pass_k": compiler_pass_k,
            "realizations": realizations,
            "dropped_realizations": dropped_realizations,
        }

    def normalize_postmortem_report(
        self,
        payload: dict[str, Any],
        campaign: dict[str, Any],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HarnessError("postmortem report must be a JSON object")
        verdict = str(payload.get("verdict", "")).strip()
        if verdict not in ALLOWED_POSTMORTEM_VERDICTS:
            raise HarnessError(f"postmortem verdict must be one of {sorted(ALLOWED_POSTMORTEM_VERDICTS)}")
        summary = str(payload.get("summary", "")).strip()
        round_assessment = str(payload.get("round_assessment", "")).strip()
        search_level_diagnosis = str(payload.get("search_level_diagnosis", "")).strip()
        lane_diagnosis = str(payload.get("lane_diagnosis", "")).strip()
        generation_instruction_delta = str(payload.get("generation_instruction_delta", "")).strip()
        if not summary:
            raise HarnessError("postmortem summary must be non-empty")
        if not round_assessment:
            raise HarnessError("postmortem round_assessment must be non-empty")
        if not search_level_diagnosis:
            raise HarnessError("postmortem search_level_diagnosis must be non-empty")
        if not lane_diagnosis:
            raise HarnessError("postmortem lane_diagnosis must be non-empty")
        if not generation_instruction_delta:
            raise HarnessError("postmortem generation_instruction_delta must be non-empty")

        campaign_lookup = {family["candidate_id"]: family for family in campaign["families"]}
        raw_reviews = payload.get("family_reviews")
        if not isinstance(raw_reviews, list) or not raw_reviews:
            raise HarnessError("postmortem family_reviews must be a non-empty list")
        reviews: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for raw in raw_reviews:
            if not isinstance(raw, dict):
                raise HarnessError("each postmortem family review must be a JSON object")
            candidate_id = str(raw.get("candidate_id", "")).strip()
            if candidate_id not in campaign_lookup:
                raise HarnessError(f"postmortem reviewed unknown family candidate_id: {candidate_id}")
            if candidate_id in seen_ids:
                raise HarnessError(f"postmortem duplicated family candidate_id: {candidate_id}")
            seen_ids.add(candidate_id)
            outcome = str(raw.get("outcome", "")).strip()
            next_action = str(raw.get("next_action", "")).strip()
            if outcome not in ALLOWED_POSTMORTEM_OUTCOMES:
                raise HarnessError(f"postmortem {candidate_id} has invalid outcome: {outcome}")
            if next_action not in ALLOWED_POSTMORTEM_ACTIONS:
                raise HarnessError(f"postmortem {candidate_id} has invalid next_action: {next_action}")
            reviews.append(
                {
                    "candidate_id": candidate_id,
                    "family_group": str(raw.get("family_group", "")).strip(),
                    "outcome": outcome,
                    "evidence": str(raw.get("evidence", "")).strip(),
                    "metric_takeaway": str(raw.get("metric_takeaway", "")).strip(),
                    "hypothesis_takeaway": str(raw.get("hypothesis_takeaway", "")).strip(),
                    "next_action": next_action,
                    "next_generation_bias": [str(item) for item in raw.get("next_generation_bias", [])],
                }
            )
            if not reviews[-1]["family_group"]:
                raise HarnessError(f"postmortem {candidate_id} must define family_group")
            if not reviews[-1]["evidence"]:
                raise HarnessError(f"postmortem {candidate_id} must define evidence")
            if not reviews[-1]["metric_takeaway"]:
                raise HarnessError(f"postmortem {candidate_id} must define metric_takeaway")
            if not reviews[-1]["hypothesis_takeaway"]:
                raise HarnessError(f"postmortem {candidate_id} must define hypothesis_takeaway")
        if seen_ids != set(campaign_lookup):
            missing = sorted(set(campaign_lookup).difference(seen_ids))
            raise HarnessError(f"postmortem must review every campaign family exactly once; missing: {missing}")

        keep_for_next_round = dedupe_strings([str(item) for item in payload.get("keep_for_next_round", [])])
        drop_for_next_round = dedupe_strings([str(item) for item in payload.get("drop_for_next_round", [])])
        for label, values in (
            ("keep_for_next_round", keep_for_next_round),
            ("drop_for_next_round", drop_for_next_round),
        ):
            unknown = [candidate_id for candidate_id in values if candidate_id not in campaign_lookup]
            if unknown:
                raise HarnessError(f"postmortem {label} includes unknown candidate ids: {unknown}")

        return {
            "verdict": verdict,
            "summary": summary,
            "round_assessment": round_assessment,
            "family_reviews": reviews,
            "search_level_diagnosis": search_level_diagnosis,
            "lane_diagnosis": lane_diagnosis,
            "keep_for_next_round": keep_for_next_round,
            "drop_for_next_round": drop_for_next_round,
            "new_family_requests": [str(item) for item in payload.get("new_family_requests", [])],
            "compile_biases": [str(item) for item in payload.get("compile_biases", [])],
            "pack_biases": [str(item) for item in payload.get("pack_biases", [])],
            "stop_doing": [str(item) for item in payload.get("stop_doing", [])],
            "generation_instruction_delta": generation_instruction_delta,
        }

    def load_round_request(self, round_index: int) -> dict[str, Any]:
        paths = self.round_paths(round_index)
        if not paths["request"].exists():
            raise HarnessError(f"missing round request: {paths['request']}")
        return load_json(paths["request"])

    def resolve_round_request(
        self,
        round_index: int,
        instructions: str | None,
        repo_files: list[str] | None,
        evidence_files: list[str] | None,
        resume: bool,
    ) -> dict[str, Any]:
        if resume:
            if repo_files:
                raise HarnessError("resume reuses the stored round request; omit --repo-file")
            if evidence_files:
                raise HarnessError("resume reuses the stored round request; omit --evidence-file")
            request = self.load_round_request(round_index)
            provided = str(instructions or "").strip()
            stored = str(request.get("user_instructions", "")).strip()
            if provided and stored and provided != stored:
                raise HarnessError(
                    "resume requested but provided instructions differ from the stored round request"
                )
            return request

        cleaned = str(instructions or "").strip()
        if not cleaned:
            raise HarnessError("instructions are required unless --resume is used")
        _, request = self.prepare_round(
            round_index=round_index,
            instructions=cleaned,
            repo_files=repo_files,
            evidence_files=evidence_files,
        )
        return request

    def load_attempt_feedback(self, path: Path) -> str:
        if not path.exists():
            return ""
        return load_text(path).strip()

    def load_round_trajectories(self, round_index: int, request: dict[str, Any]) -> list[dict[str, Any]]:
        paths = self.round_paths(round_index)
        exploration_settings = request["exploration"]
        trajectories: list[dict[str, Any]] = []
        for trajectory_id in self.expected_trajectory_ids(request):
            trajectory_path = paths["explorations_dir"] / f"{trajectory_id}.json"
            if not trajectory_path.exists():
                raise HarnessError(f"missing exploration trajectory file: {trajectory_path}")
            normalized = self.normalize_exploration_slate(
                load_json(trajectory_path),
                expected_trajectory_id=trajectory_id,
                expected_count=int(exploration_settings["families_per_trajectory"]),
            )
            write_json(trajectory_path, normalized)
            trajectories.append(normalized)
        return trajectories

    def build_family_dossiers(
        self,
        round_index: int,
        request: dict[str, Any],
        trajectories: list[dict[str, Any]],
        verification: dict[str, Any],
        campaign: dict[str, Any],
    ) -> list[dict[str, Any]]:
        paths = self.round_paths(round_index)
        if paths["family_dossiers_dir"].exists():
            shutil.rmtree(paths["family_dossiers_dir"])
        paths["family_dossiers_dir"].mkdir(parents=True, exist_ok=True)

        candidate_lookup = self.candidate_lookup(trajectories)
        review_lookup = self.review_lookup(verification)
        dossiers: list[dict[str, Any]] = []
        for family in campaign["families"]:
            candidate_id = family["candidate_id"]
            dossier_path = paths["family_dossiers_dir"] / self.family_file_name(candidate_id)
            dossier = {
                "cycle_id": request["cycle_id"],
                "round": request["round"],
                "campaign_goal": campaign["campaign_goal"],
                "dominant_search_level": campaign["dominant_search_level"],
                "candidate_id": candidate_id,
                "name": family["name"],
                "family_group": family["family_group"],
                "search_level": family["search_level"],
                "lane": family["lane"],
                "phase_window": family["phase_window"],
                "pack_kind": family["pack_kind"],
                "implementation_mode": family["implementation_mode"],
                "compiler_pass_k": family["compiler_pass_k"],
                "why_keep": family["why_keep"],
                "minimal_probe": family["minimal_probe"],
                "expected_signal": family["expected_signal"],
                "composition_tags": family["composition_tags"],
                "incompatible_with": family["incompatible_with"],
                "mutate_after_survival": family["mutate_after_survival"],
                "pack_rationale": family["pack_rationale"],
                "source_candidate": candidate_lookup.get(candidate_id, {}),
                "verifier_review": review_lookup.get(candidate_id, {}),
                "dossier_path": self.project_relative(dossier_path),
            }
            write_json(dossier_path, dossier)
            dossiers.append(dossier)
        return dossiers

    def build_compile_queue(
        self,
        round_index: int,
        request: dict[str, Any],
        campaign: dict[str, Any],
        dossiers: list[dict[str, Any]],
    ) -> dict[str, Any]:
        paths = self.round_paths(round_index)
        queue = {
            "cycle_id": request["cycle_id"],
            "round": request["round"],
            "compiler_defaults": request["compiler"],
            "families": [
                {
                    "candidate_id": dossier["candidate_id"],
                    "name": dossier["name"],
                    "family_group": dossier["family_group"],
                    "lane": dossier["lane"],
                    "phase_window": dossier["phase_window"],
                    "pack_kind": dossier["pack_kind"],
                    "implementation_mode": dossier["implementation_mode"],
                    "compiler_pass_k": dossier["compiler_pass_k"],
                    "dossier_path": dossier["dossier_path"],
                    "compile_status": "queued",
                }
                for dossier in dossiers
            ],
            "campaign_goal": campaign["campaign_goal"],
        }
        write_json(paths["compile_queue"], queue)
        return queue

    def build_round_summary(
        self,
        request: dict[str, Any],
        verification: dict[str, Any],
        campaign: dict[str, Any],
    ) -> dict[str, Any]:
        runnable_now_ids = [
            item["candidate_id"]
            for item in campaign["families"]
            if item["implementation_mode"] == "catalog_executable_now"
        ]
        blocked_backlog_ids = [
            item["candidate_id"]
            for item in campaign["families"]
            if item["implementation_mode"] != "catalog_executable_now"
        ]
        return {
            "cycle_id": request["cycle_id"],
            "round": request["round"],
            "objective": request["objective"],
            "verification_verdict": verification["verdict"],
            "verification_summary": verification["summary"],
            "campaign_verdict": campaign["verdict"],
            "campaign_summary": campaign["summary"],
            "campaign_goal": campaign["campaign_goal"],
            "dominant_search_level": campaign["dominant_search_level"],
            "keep_ids": campaign["keep_ids"],
            "runnable_now_ids": runnable_now_ids,
            "blocked_backlog_ids": blocked_backlog_ids,
            "family_groups": [item["family_group"] for item in campaign["families"]],
            "pack_order": campaign["pack_strategy"]["pack_order"],
            "stop_doing": campaign["stop_doing"],
        }

    def build_campaign_handoff_instructions(
        self,
        request: dict[str, Any],
        campaign: dict[str, Any],
    ) -> str:
        lines = [
            f"Use pg_enigma round {request['round']} as a campaign contract, not a single-answer mutation batch.",
            f"Objective: {request['objective'].get('goal', '')}",
            f"Campaign goal: {campaign['campaign_goal']}",
            f"Dominant search level: {campaign['dominant_search_level']}",
            "",
            "Families to keep alive:",
        ]
        for family in campaign["families"]:
            lines.append(
                f"- {family['candidate_id']} ({family['name']}; family_group={family['family_group']}; "
                f"lane={family['lane']}; phase={family['phase_window']}; pack={family['pack_kind']}; "
                f"compiler_pass_k={family['compiler_pass_k']}) - {family['why_keep']}"
            )
        lines.extend(
            [
                "",
                "Stop doing:",
                *[f"- {item}" for item in campaign["stop_doing"]],
                "",
                "Pack strategy:",
                f"- controls_per_pack={campaign['pack_strategy']['controls_per_pack']}",
                f"- max_candidates_per_pack={campaign['pack_strategy']['max_candidates_per_pack']}",
                *[f"- pack_order: {item}" for item in campaign["pack_strategy"]["pack_order"]],
                *[f"- note: {item}" for item in campaign["pack_strategy"]["notes"]],
                "",
                "Promotion policy:",
                f"- min_positive_realizations={campaign['promotion_policy']['min_positive_realizations']}",
                (
                    f"- require_directional_support={campaign['promotion_policy']['require_directional_support']}"
                ),
                f"- family_survival_rule: {campaign['promotion_policy']['family_survival_rule']}",
                f"- composition_entry_rule: {campaign['promotion_policy']['composition_entry_rule']}",
                "",
                "Composition policy:",
                (
                    f"- pairwise_after_solo_survival="
                    f"{campaign['composition_policy']['pairwise_after_solo_survival']}"
                ),
                f"- max_pairwise_candidates={campaign['composition_policy']['max_pairwise_candidates']}",
                f"- max_hybrid_candidates={campaign['composition_policy']['max_hybrid_candidates']}",
                f"- hybrid_entry_rule: {campaign['composition_policy']['hybrid_entry_rule']}",
                *[f"- note: {item}" for item in campaign["composition_policy"]["notes"]],
                "",
                "Global instructions:",
                campaign["campaign_handoff"]["global_instructions"],
                "",
                "Control principles:",
                *[f"- {item}" for item in campaign["campaign_handoff"]["control_principles"]],
                "",
                "Admission principles:",
                *[f"- {item}" for item in campaign["campaign_handoff"]["admission_principles"]],
                "",
                "Metric principles:",
                *[f"- {item}" for item in campaign["campaign_handoff"]["metric_principles"]],
                "",
                "Implementation principles:",
                *[f"- {item}" for item in campaign["campaign_handoff"]["implementation_principles"]],
            ]
        )
        return "\n".join(lines).strip() + "\n"

    def render_campaign_handoff(
        self,
        round_index: int,
        request: dict[str, Any],
        verification: dict[str, Any],
        campaign: dict[str, Any],
    ) -> None:
        paths = self.round_paths(round_index)
        prompt_text = self.build_campaign_handoff_instructions(request, campaign)
        markdown_lines = [
            "# Enigma Campaign Handoff",
            "",
            f"- **Cycle**: `{request['cycle_id']}`",
            f"- **Round**: `{request['round']}`",
            f"- **Objective**: {request['objective'].get('goal', '')}",
            f"- **Campaign goal**: {campaign['campaign_goal']}",
            f"- **Dominant search level**: `{campaign['dominant_search_level']}`",
            "",
            "## Summary",
            "",
            campaign["summary"],
            "",
            "## Verifier summary",
            "",
            verification["summary"],
            "",
            "## Campaign handoff",
            "",
            "```text",
            prompt_text.rstrip(),
            "```",
        ]
        write_text(paths["search_handoff_md"], "\n".join(markdown_lines) + "\n")
        write_text(paths["search_handoff_prompt"], prompt_text)
        write_json(paths["round_summary"], self.build_round_summary(request, verification, campaign))

    def validate_round(self, round_index: int) -> Path:
        paths = self.round_paths(round_index)
        request = self.load_round_request(round_index)
        trajectories = self.load_round_trajectories(round_index, request)
        allowed_ids = {item["candidate_id"] for item in self.flatten_candidates(trajectories)}

        if not paths["verification_report"].exists():
            raise HarnessError(f"missing verification report: {paths['verification_report']}")
        verification = self.normalize_verification_report(load_json(paths["verification_report"]), allowed_ids)
        write_json(paths["verification_report"], verification)
        self.enforce_verification_thresholds(verification, request)

        if not paths["campaign"].exists():
            raise HarnessError(f"missing campaign file: {paths['campaign']}")
        campaign = self.normalize_campaign(
            load_json(paths["campaign"]),
            verifier_keep_ids=set(verification["keep_ids"]),
            request=request,
        )
        write_json(paths["campaign"], campaign)
        if campaign["verdict"] != "READY":
            raise HarnessError(f"Campaign returned {campaign['verdict']}: {campaign['summary']}")

        dossiers = self.build_family_dossiers(round_index, request, trajectories, verification, campaign)
        self.build_compile_queue(round_index, request, campaign, dossiers)
        self.render_campaign_handoff(round_index, request, verification, campaign)
        self.build_family_status_reports(round_index, request, trajectories, verification, campaign)
        return paths["round_dir"]

    def debug_verifier(
        self,
        round_index: int,
        mode: str = "auto",
        attempt: int | None = None,
        reuse_from_attempt: int | None = None,
        model: str = "gpt-5.4",
        reasoning_effort: str = "xhigh",
        label: str | None = None,
    ) -> Path:
        if mode not in {"auto", "full", "diff"}:
            raise HarnessError(f"debug verifier mode must be one of auto/full/diff, got: {mode}")

        request = self.load_round_request(round_index)
        current_state = self.load_verification_state_for_debug(round_index, request, attempt)
        trajectories = current_state["trajectories"]
        allowed_ids = {item["candidate_id"] for item in self.flatten_candidates(trajectories)}

        compare_attempt = reuse_from_attempt
        if compare_attempt is None and mode in {"auto", "diff"} and attempt is not None and attempt > 1:
            compare_attempt = attempt - 1

        prior_state = None
        if compare_attempt is not None:
            prior_state = self.load_verification_state_for_debug(round_index, request, compare_attempt)
        reuse_plan = self.plan_verification_reuse(request, trajectories, prior_state) if prior_state else None

        effective_mode = mode
        if mode == "auto":
            if reuse_plan is not None and 0 < reuse_plan["changed_candidate_count"] < len(reuse_plan["all_candidate_ids"]):
                effective_mode = "diff"
            else:
                effective_mode = "full"

        if effective_mode == "diff":
            if reuse_plan is None:
                raise HarnessError("diff mode requires --reuse-from-attempt or --attempt > 1")
            if reuse_plan["changed_candidate_count"] == 0:
                raise HarnessError("diff mode found no changed candidates; use --mode full or a different attempt comparison")

        source_label = f"attempt_{attempt:03d}" if attempt is not None else "current_round"
        compare_label = f"_from_attempt_{compare_attempt:03d}" if compare_attempt is not None else ""
        debug_label = safe_slug(label or f"verifier_{effective_mode}_{source_label}{compare_label}", limit=96)
        debug_dir = self.round_paths(round_index)["round_dir"] / "debug" / debug_label
        if debug_dir.exists():
            shutil.rmtree(debug_dir)
        debug_dir.mkdir(parents=True, exist_ok=True)

        write_json(debug_dir / "request_summary.json", self.verifier_request_summary(request))
        write_json(debug_dir / "trajectories.json", {"trajectories": trajectories})
        if current_state.get("verification"):
            write_json(debug_dir / "baseline_verification.json", current_state["verification"])
        if prior_state is not None:
            write_json(debug_dir / "prior_trajectories.json", {"trajectories": prior_state["trajectories"]})
            if prior_state.get("verification"):
                write_json(debug_dir / "prior_verification.json", prior_state["verification"])
        if reuse_plan is not None:
            write_json(
                debug_dir / "verification_reuse.json",
                {
                    "reused_candidate_ids": reuse_plan["reused_candidate_ids"],
                    "reused_keep_ids": reuse_plan["reused_keep_ids"],
                    "override_candidate_ids": reuse_plan["override_candidate_ids"],
                    "changed_candidate_ids": reuse_plan["changed_candidate_ids"],
                    "reused_candidate_count": reuse_plan["reused_candidate_count"],
                    "changed_candidate_count": reuse_plan["changed_candidate_count"],
                    "all_reused": reuse_plan["all_reused"],
                    "reused_review_cards": reuse_plan["reused_review_cards"],
                    "changed_claim_cards": reuse_plan["changed_claim_cards"],
                },
            )

        if effective_mode == "diff":
            prompt = self.build_verifier_patch_prompt(request, reuse_plan, None)
        else:
            prompt = self.build_verifier_prompt(request, trajectories, None)
        write_text(debug_dir / "prompt.txt", prompt)

        raw_payload = self.invoke_copilot_json(
            prompt=prompt,
            output_path=debug_dir / "copilot_raw.txt",
            work_dir=debug_dir,
            model=model,
            reasoning_effort=reasoning_effort,
        )

        if effective_mode == "diff":
            patch = self.normalize_verification_patch(
                raw_payload,
                required_ids=set(reuse_plan["changed_candidate_ids"]),
                reusable_ids=set(reuse_plan["override_candidate_ids"]),
            )
            write_json(debug_dir / "normalized_patch.json", patch)
            merged = self.merge_verification_patch(reuse_plan, patch)
            write_json(debug_dir / "merged_verification.json", merged)
            verification = self.normalize_verification_report(merged, allowed_ids)
        else:
            verification = self.normalize_verification_report(raw_payload, allowed_ids)
        write_json(debug_dir / "normalized_verification.json", verification)
        runnable_split = self.verification_runnable_split(trajectories, verification)
        write_json(debug_dir / "runnable_keep_families.json", runnable_split)

        validation_error = ""
        try:
            self.enforce_verification_thresholds(verification, request)
        except Exception as exc:  # noqa: BLE001
            validation_error = str(exc)

        summary = {
            "round": round_index,
            "mode_requested": mode,
            "mode_used": effective_mode,
            "source": source_label,
            "compare_source": f"attempt_{compare_attempt:03d}" if compare_attempt is not None else None,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "prompt_chars": len(prompt),
            "candidate_count": len(allowed_ids),
            "reused_candidate_count": reuse_plan["reused_candidate_count"] if reuse_plan is not None else 0,
            "changed_candidate_count": reuse_plan["changed_candidate_count"] if reuse_plan is not None else len(allowed_ids),
            "keep_ids": verification["keep_ids"],
            "runnable_keep_ids": runnable_split["runnable_keep_ids"],
            "blocked_keep_ids": runnable_split["blocked_keep_ids"],
            "verdict": verification["verdict"],
            "summary": verification["summary"],
            "validation_error": validation_error,
            "artifacts": {
                "prompt": str(debug_dir / "prompt.txt"),
                "copilot_raw": str(debug_dir / "copilot_raw.txt"),
                "normalized_verification": str(debug_dir / "normalized_verification.json"),
                "runnable_keep_families": str(debug_dir / "runnable_keep_families.json"),
            },
        }
        if effective_mode == "diff":
            summary["artifacts"]["normalized_patch"] = str(debug_dir / "normalized_patch.json")
            summary["artifacts"]["merged_verification"] = str(debug_dir / "merged_verification.json")
        write_json(debug_dir / "summary.json", summary)

        if validation_error:
            raise HarnessError(
                f"Debug verifier output failed validation: {validation_error}. "
                f"Inspect artifacts under {debug_dir}"
            )
        return debug_dir

    def run_model_round(
        self,
        round_index: int,
        instructions: str | None,
        repo_files: list[str] | None = None,
        evidence_files: list[str] | None = None,
        max_attempts: int | None = None,
        resume: bool = False,
        backend: str = "codex",
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> Path:
        selected_backend = self.normalize_backend(backend)
        request = self.resolve_round_request(
            round_index=round_index,
            instructions=instructions,
            repo_files=repo_files,
            evidence_files=evidence_files,
            resume=resume,
        )
        paths = self.round_paths(round_index)
        attempt_root = self.backend_attempt_root(paths, selected_backend)
        summary_path = self.backend_summary_path(paths, selected_backend)
        if resume:
            paths["explorations_dir"].mkdir(parents=True, exist_ok=True)
            attempt_root.mkdir(parents=True, exist_ok=True)
            if paths["verification_report"].exists() and paths["campaign"].exists():
                try:
                    self.validate_round(round_index)
                    return paths["round_dir"]
                except Exception:  # noqa: BLE001
                    pass
        else:
            for key in ("explorations_dir", "family_dossiers_dir", "family_compiles_dir", "codex_dir", "copilot_dir"):
                if paths[key].exists():
                    shutil.rmtree(paths[key])
            for key in (
                "verification_report",
                "campaign",
                "compile_queue",
                "pack_queue",
                "pack_handoffs",
                "promotion_report",
                "composition_queue",
                "hybrid_queue",
                "postmortem_report",
                "search_handoff_md",
                "search_handoff_prompt",
                "round_summary",
                "family_status_report",
                "runnable_families",
            ):
                if paths[key].exists():
                    paths[key].unlink()
            paths["explorations_dir"].mkdir(parents=True, exist_ok=True)
            attempt_root.mkdir(parents=True, exist_ok=True)

        attempts = int(max_attempts or self.codex_settings().get("max_attempts", 2))
        if attempts < 1:
            raise HarnessError("max_attempts must be >= 1")
        previous_feedback = ""

        for attempt in range(1, attempts + 1):
            attempt_dir = attempt_root / f"attempt_{attempt:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            failure_path = attempt_dir / "failure.txt"
            attempt_feedback = self.load_attempt_feedback(failure_path) if resume else previous_feedback
            if resume and not attempt_feedback:
                attempt_feedback = previous_feedback
            try:
                trajectories: list[dict[str, Any]] = []
                for trajectory_id in self.expected_trajectory_ids(request):
                    raw_path = attempt_dir / f"{trajectory_id}_raw.json"
                    if resume and raw_path.exists():
                        raw = load_json(raw_path)
                    else:
                        prompt = self.build_explorer_prompt(request, trajectory_id, attempt_feedback or None)
                        write_text(attempt_dir / f"{trajectory_id}_prompt.txt", prompt)
                        raw = self.invoke_backend_json(
                            backend=selected_backend,
                            prompt=prompt,
                            schema_path=self.schema_path("explorer"),
                            output_path=raw_path,
                            attempt_dir=attempt_dir,
                            label=trajectory_id,
                            stage="explorer",
                            model=model,
                            reasoning_effort=reasoning_effort,
                        )
                    normalized = self.normalize_exploration_slate(
                        raw,
                        expected_trajectory_id=trajectory_id,
                        expected_count=int(request["exploration"]["families_per_trajectory"]),
                    )
                    write_json(attempt_dir / f"{trajectory_id}.json", normalized)
                    write_json(paths["explorations_dir"] / f"{trajectory_id}.json", normalized)
                    trajectories.append(normalized)

                allowed_ids = {item["candidate_id"] for item in self.flatten_candidates(trajectories)}
                verifier_raw_path = attempt_dir / "verification_raw.json"
                if resume and verifier_raw_path.exists():
                    verifier_raw = load_json(verifier_raw_path)
                else:
                    prior_state = self.find_prior_verification_state(paths, request, attempt, backend=selected_backend)
                    reuse_plan = self.plan_verification_reuse(request, trajectories, prior_state)
                    if reuse_plan is not None:
                        write_json(
                            attempt_dir / "verification_reuse.json",
                            {
                                "reused_candidate_ids": reuse_plan["reused_candidate_ids"],
                                "changed_candidate_ids": reuse_plan["changed_candidate_ids"],
                                "reused_keep_ids": reuse_plan["reused_keep_ids"],
                                "all_reused": reuse_plan["all_reused"],
                            },
                        )
                    if reuse_plan is not None and reuse_plan["all_reused"]:
                        write_text(
                            attempt_dir / "verifier_prompt.txt",
                            "Verifier skipped: all candidate reviews were reused from a prior identical claim set.\n",
                        )
                        verifier_raw = self.reuse_verification_report(reuse_plan)
                        write_json(verifier_raw_path, verifier_raw)
                    elif reuse_plan is not None and reuse_plan["reused_candidate_count"] > 0:
                        verifier_prompt = self.build_verifier_patch_prompt(
                            request=request,
                            reuse_plan=reuse_plan,
                            previous_feedback=attempt_feedback or None,
                        )
                        write_text(attempt_dir / "verifier_prompt.txt", verifier_prompt)
                        verifier_patch_raw_path = attempt_dir / "verification_patch_raw.json"
                        verifier_patch_raw = self.invoke_backend_json(
                            backend=selected_backend,
                            prompt=verifier_prompt,
                            schema_path=self.schema_path("verifier"),
                            output_path=verifier_patch_raw_path,
                            attempt_dir=attempt_dir,
                            label="verifier",
                            stage="verifier",
                            model=model,
                            reasoning_effort=reasoning_effort,
                        )
                        verifier_patch = self.normalize_verification_patch(
                            verifier_patch_raw,
                            required_ids=set(reuse_plan["changed_candidate_ids"]),
                            reusable_ids=set(reuse_plan["reused_candidate_ids"]),
                        )
                        verifier_raw = self.merge_verification_patch(reuse_plan, verifier_patch)
                        write_json(verifier_raw_path, verifier_raw)
                    else:
                        verifier_prompt = self.build_verifier_prompt(request, trajectories, attempt_feedback or None)
                        write_text(attempt_dir / "verifier_prompt.txt", verifier_prompt)
                        verifier_raw = self.invoke_backend_json(
                            backend=selected_backend,
                            prompt=verifier_prompt,
                            schema_path=self.schema_path("verifier"),
                            output_path=verifier_raw_path,
                            attempt_dir=attempt_dir,
                            label="verifier",
                            stage="verifier",
                            model=model,
                            reasoning_effort=reasoning_effort,
                        )
                verification = self.normalize_verification_report(verifier_raw, allowed_ids)
                write_json(paths["verification_report"], verification)
                self.enforce_verification_thresholds(verification, request)

                campaign_raw_path = attempt_dir / "campaign_raw.json"
                if resume and campaign_raw_path.exists():
                    campaign_raw = load_json(campaign_raw_path)
                else:
                    campaign_prompt = self.build_campaign_prompt(
                        request=request,
                        trajectories=trajectories,
                        verification=verification,
                        previous_feedback=attempt_feedback or None,
                    )
                    write_text(attempt_dir / "campaign_prompt.txt", campaign_prompt)
                    campaign_raw = self.invoke_backend_json(
                        backend=selected_backend,
                        prompt=campaign_prompt,
                        schema_path=self.schema_path("distiller"),
                        output_path=campaign_raw_path,
                        attempt_dir=attempt_dir,
                        label="campaign",
                        stage="distiller",
                        model=model,
                        reasoning_effort=reasoning_effort,
                    )
                campaign = self.normalize_campaign(
                    campaign_raw,
                    verifier_keep_ids=set(verification["keep_ids"]),
                    request=request,
                )
                write_json(paths["campaign"], campaign)
                if campaign["verdict"] != "READY":
                    raise HarnessError(f"Campaign returned {campaign['verdict']}: {campaign['summary']}")
                self.validate_round(round_index)
                write_json(
                    summary_path,
                    {
                        "status": "ready",
                        "backend": selected_backend,
                        "model": model,
                        "reasoning_effort": reasoning_effort,
                        "attempts_used": attempt,
                        "round": round_index,
                        "verification_report": str(paths["verification_report"]),
                        "campaign": str(paths["campaign"]),
                        "compile_queue": str(paths["compile_queue"]),
                        "search_handoff": str(paths["search_handoff_md"]),
                        "resumed": resume,
                    },
                )
                return paths["round_dir"]
            except Exception as exc:  # noqa: BLE001
                previous_feedback = str(exc)
                write_text(failure_path, previous_feedback + "\n")

        write_json(
            summary_path,
            {
                "status": "failed",
                "backend": selected_backend,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "attempts_used": attempts,
                "round": round_index,
                "last_error": previous_feedback,
                "resumed": resume,
            },
        )
        raise HarnessError(
            f"pg_enigma could not prepare round {round_index} after {attempts} attempts: {previous_feedback}"
        )

    def run_codex_round(
        self,
        round_index: int,
        instructions: str | None,
        repo_files: list[str] | None = None,
        evidence_files: list[str] | None = None,
        max_attempts: int | None = None,
        resume: bool = False,
    ) -> Path:
        return self.run_model_round(
            round_index=round_index,
            instructions=instructions,
            repo_files=repo_files,
            evidence_files=evidence_files,
            max_attempts=max_attempts,
            resume=resume,
            backend="codex",
        )

    def load_campaign(self, round_index: int, request: dict[str, Any], verification: dict[str, Any]) -> dict[str, Any]:
        paths = self.round_paths(round_index)
        if not paths["campaign"].exists():
            raise HarnessError(f"missing campaign file: {paths['campaign']}")
        campaign = self.normalize_campaign(
            load_json(paths["campaign"]),
            verifier_keep_ids=set(verification["keep_ids"]),
            request=request,
        )
        write_json(paths["campaign"], campaign)
        return campaign

    def load_dossiers(self, round_index: int, campaign: dict[str, Any]) -> dict[str, dict[str, Any]]:
        paths = self.round_paths(round_index)
        dossiers: dict[str, dict[str, Any]] = {}
        for family in campaign["families"]:
            dossier_path = paths["family_dossiers_dir"] / self.family_file_name(family["candidate_id"])
            if not dossier_path.exists():
                raise HarnessError(f"missing family dossier: {dossier_path}")
            dossier = load_json(dossier_path)
            dossiers[family["candidate_id"]] = dossier
        return dossiers

    def build_family_status_reports(
        self,
        round_index: int,
        request: dict[str, Any],
        trajectories: list[dict[str, Any]],
        verification: dict[str, Any],
        campaign: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        paths = self.round_paths(round_index)
        candidate_lookup = self.candidate_lookup(trajectories)
        review_lookup = self.review_lookup(verification)
        dossiers: dict[str, dict[str, Any]] = {}
        if paths["family_dossiers_dir"].exists():
            try:
                dossiers = self.load_dossiers(round_index, campaign)
            except HarnessError:
                dossiers = {}

        compile_statuses: dict[str, dict[str, Any]] = {}
        for family in campaign["families"]:
            candidate_id = family["candidate_id"]
            compile_path = paths["family_compiles_dir"] / self.family_file_name(candidate_id)
            status = {
                "compile_path": self.project_relative(compile_path),
                "verdict": "MISSING",
                "realization_count": 0,
            }
            if compile_path.exists():
                try:
                    if candidate_id in dossiers:
                        compile_payload = self.normalize_family_compile(
                            load_json(compile_path), dossiers[candidate_id]
                        )
                        write_json(compile_path, compile_payload)
                    else:
                        compile_payload = load_json(compile_path)
                    realizations = compile_payload.get("realizations", [])
                    status = {
                        "compile_path": self.project_relative(compile_path),
                        "verdict": str(compile_payload.get("verdict", "UNKNOWN")).strip() or "UNKNOWN",
                        "realization_count": len(realizations) if isinstance(realizations, list) else 0,
                    }
                except (HarnessError, json.JSONDecodeError, OSError) as exc:
                    status = {
                        "compile_path": self.project_relative(compile_path),
                        "verdict": "INVALID",
                        "realization_count": 0,
                        "error": str(exc),
                    }
            compile_statuses[candidate_id] = status

        pack_ids_by_candidate: dict[str, list[str]] = {}
        deferred_reasons: dict[str, list[str]] = {}
        if paths["pack_queue"].exists():
            pack_queue = load_json(paths["pack_queue"])
            for pack in pack_queue.get("packs", []):
                pack_id = str(pack.get("pack_id", "")).strip()
                if not pack_id:
                    continue
                for slot in pack.get("candidate_slots", []):
                    candidate_id = str(slot.get("candidate_id", "")).strip()
                    if not candidate_id:
                        continue
                    pack_ids_by_candidate.setdefault(candidate_id, []).append(pack_id)
            for item in pack_queue.get("deferred_families", []):
                candidate_id = str(item.get("candidate_id", "")).strip()
                reason = str(item.get("reason", "")).strip()
                if not candidate_id or not reason:
                    continue
                deferred_reasons.setdefault(candidate_id, []).append(reason)

        verifier_keep_ids = set(verification["keep_ids"])
        family_reports: list[dict[str, Any]] = []
        runnable_now: list[dict[str, Any]] = []
        blocked_backlog: list[dict[str, Any]] = []
        ready_for_pack: list[dict[str, Any]] = []
        packed: list[dict[str, Any]] = []

        for family in campaign["families"]:
            candidate_id = family["candidate_id"]
            candidate = candidate_lookup.get(candidate_id, {})
            review = review_lookup.get(candidate_id, {})
            compile_status = compile_statuses.get(candidate_id, {})
            family_pack_ids = dedupe_strings(pack_ids_by_candidate.get(candidate_id, []))
            family_deferred_reasons = sorted(deferred_reasons.get(candidate_id, []))
            is_runnable_now = family["implementation_mode"] == "catalog_executable_now"
            initial_pack_kind = family["pack_kind"] in INITIAL_EXECUTABLE_PACK_KINDS
            compile_ready = compile_status.get("verdict") == "READY"
            is_ready_for_pack = is_runnable_now and initial_pack_kind and compile_ready
            if family_pack_ids:
                current_state = "packed"
            elif not is_runnable_now:
                current_state = "blocked_backlog"
            elif not initial_pack_kind:
                current_state = "reserved_for_later_stage"
            elif compile_status.get("verdict") == "MISSING":
                current_state = "compile_needed"
            elif not compile_ready:
                current_state = "compile_blocked"
            elif family_deferred_reasons:
                current_state = "deferred"
            else:
                current_state = "ready_for_pack"

            entry = {
                "candidate_id": candidate_id,
                "name": family["name"],
                "family_group": family["family_group"],
                "search_level": family["search_level"],
                "lane": family["lane"],
                "phase_window": family["phase_window"],
                "pack_kind": family["pack_kind"],
                "implementation_mode": family["implementation_mode"],
                "verifier_review_verdict": review.get("verdict"),
                "kept_by_verifier": candidate_id in verifier_keep_ids,
                "consequence": candidate.get("consequence"),
                "minimal_probe": family["minimal_probe"],
                "expected_signal": family["expected_signal"],
                "compile_status": compile_status,
                "pack_ids": family_pack_ids,
                "deferred_reasons": family_deferred_reasons,
                "runnable_now": is_runnable_now,
                "ready_for_pack": is_ready_for_pack,
                "current_state": current_state,
            }
            family_reports.append(entry)
            if is_runnable_now:
                runnable_now.append(entry)
            else:
                blocked_backlog.append(entry)
            if is_ready_for_pack:
                ready_for_pack.append(entry)
            if family_pack_ids:
                packed.append(entry)

        family_reports.sort(key=lambda item: (item["current_state"], item["candidate_id"]))
        runnable_now.sort(key=lambda item: item["candidate_id"])
        blocked_backlog.sort(key=lambda item: item["candidate_id"])
        ready_for_pack.sort(key=lambda item: item["candidate_id"])
        packed.sort(key=lambda item: item["candidate_id"])

        family_status_report = {
            "cycle_id": request["cycle_id"],
            "round": request["round"],
            "campaign_goal": campaign["campaign_goal"],
            "summary": {
                "keep_count": len(campaign["keep_ids"]),
                "runnable_now_count": len(runnable_now),
                "blocked_backlog_count": len(blocked_backlog),
                "ready_for_pack_count": len(ready_for_pack),
                "packed_count": len(packed),
            },
            "runnable_now_ids": [item["candidate_id"] for item in runnable_now],
            "blocked_backlog_ids": [item["candidate_id"] for item in blocked_backlog],
            "ready_for_pack_ids": [item["candidate_id"] for item in ready_for_pack],
            "packed_family_ids": [item["candidate_id"] for item in packed],
            "family_reports": family_reports,
        }

        selected_families = packed or ready_for_pack or runnable_now
        selection_basis = "packed" if packed else "ready_for_pack" if ready_for_pack else "runnable_now"
        runnable_families = {
            "cycle_id": request["cycle_id"],
            "round": request["round"],
            "campaign_goal": campaign["campaign_goal"],
            "selection_basis": selection_basis,
            "selected_family_ids": [item["candidate_id"] for item in selected_families],
            "selected_families": selected_families,
            "runnable_now_ids": family_status_report["runnable_now_ids"],
            "ready_for_pack_ids": family_status_report["ready_for_pack_ids"],
            "packed_family_ids": family_status_report["packed_family_ids"],
            "blocked_backlog_ids": family_status_report["blocked_backlog_ids"],
            "artifacts": {
                "family_status_report": self.project_relative(paths["family_status_report"]),
                "runnable_families": self.project_relative(paths["runnable_families"]),
            },
        }
        write_json(paths["family_status_report"], family_status_report)
        write_json(paths["runnable_families"], runnable_families)
        return family_status_report, runnable_families

    def compile_families(
        self,
        round_index: int,
        family_ids: list[str] | None = None,
        max_attempts: int | None = None,
        resume: bool = False,
        backend: str = "codex",
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> Path:
        selected_backend = self.normalize_backend(backend)
        self.validate_round(round_index)
        paths = self.round_paths(round_index)
        request = self.load_round_request(round_index)
        verification = load_json(paths["verification_report"])
        campaign = self.load_campaign(round_index, request, verification)
        compile_queue = load_json(paths["compile_queue"])
        dossiers = self.load_dossiers(round_index, campaign)

        selected_ids = dedupe_strings([str(item) for item in (family_ids or [])])
        available_ids = {item["candidate_id"] for item in compile_queue.get("families", [])}
        if selected_ids:
            unknown = [candidate_id for candidate_id in selected_ids if candidate_id not in available_ids]
            if unknown:
                raise HarnessError(f"Unknown family ids requested for compilation: {unknown}")
        else:
            selected_ids = [item["candidate_id"] for item in compile_queue.get("families", [])]

        paths["family_compiles_dir"].mkdir(parents=True, exist_ok=True)
        attempts = int(max_attempts or self.codex_settings().get("max_attempts", 2))
        if attempts < 1:
            raise HarnessError("max_attempts must be >= 1")

        compile_root = self.backend_attempt_root(paths, selected_backend)
        compile_root.mkdir(parents=True, exist_ok=True)
        failures: list[str] = []
        for candidate_id in selected_ids:
            dossier = dossiers[candidate_id]
            slug = safe_slug(candidate_id)
            output_path = paths["family_compiles_dir"] / self.family_file_name(candidate_id)
            if resume and output_path.exists():
                try:
                    normalized_existing = self.normalize_family_compile(load_json(output_path), dossier)
                    write_json(output_path, normalized_existing)
                    continue
                except Exception:  # noqa: BLE001
                    pass
            compile_attempt_dir = compile_root / f"compile_{slug}"
            if compile_attempt_dir.exists() and not resume:
                shutil.rmtree(compile_attempt_dir)
            compile_attempt_dir.mkdir(parents=True, exist_ok=True)
            previous_feedback = ""
            success = False
            for attempt in range(1, attempts + 1):
                attempt_dir = compile_attempt_dir / f"attempt_{attempt:03d}"
                attempt_dir.mkdir(parents=True, exist_ok=True)
                failure_path = attempt_dir / "failure.txt"
                attempt_feedback = self.load_attempt_feedback(failure_path) if resume else previous_feedback
                if resume and not attempt_feedback:
                    attempt_feedback = previous_feedback
                try:
                    raw_path = attempt_dir / "family_compile_raw.json"
                    if resume and raw_path.exists():
                        raw = load_json(raw_path)
                    else:
                        prompt = self.build_compiler_prompt(
                            request=request,
                            campaign=campaign,
                            dossier=dossier,
                            previous_feedback=attempt_feedback or None,
                        )
                        write_text(attempt_dir / "compiler_prompt.txt", prompt)
                        raw = self.invoke_backend_json(
                            backend=selected_backend,
                            prompt=prompt,
                            schema_path=self.schema_path("compiler"),
                            output_path=raw_path,
                            attempt_dir=attempt_dir,
                            label=f"compiler_{slug}",
                            stage="compiler",
                            model=model,
                            reasoning_effort=reasoning_effort,
                        )
                    normalized = self.normalize_family_compile(raw, dossier)
                    write_json(output_path, normalized)
                    success = True
                    break
                except Exception as exc:  # noqa: BLE001
                    previous_feedback = str(exc)
                    write_text(failure_path, previous_feedback + "\n")
            if not success:
                failures.append(f"{candidate_id}: {previous_feedback}")

        self.build_pack_queue(round_index)
        if failures:
            raise HarnessError("Some family compiles failed:\n- " + "\n- ".join(failures))
        return paths["pack_queue"]

    def pack_order_index(self, campaign: dict[str, Any], pack_kind: str) -> int:
        order = campaign["pack_strategy"].get("pack_order", [])
        try:
            return order.index(pack_kind)
        except ValueError:
            return len(order)

    def build_pack_instruction_text(
        self,
        request: dict[str, Any],
        campaign: dict[str, Any],
        pack: dict[str, Any],
    ) -> str:
        lines = [
            f"Use pg_enigma round {request['round']} pack {pack['pack_id']} as the execution contract.",
            f"Pack kind: {pack['pack_kind']}",
            f"Lane: {pack['lane']}",
            f"Phase window: {pack['phase_window']}",
            "",
            "Create exactly these control slots and keep them unpatched:",
            *[f"- {slot_id}" for slot_id in pack["control_slot_ids"]],
            "",
            "Create candidate slots using these exact slot ids:",
        ]
        for candidate in pack["candidate_slots"]:
            lines.extend(
                [
                    (
                        f"- {candidate['slot_id']}: {candidate['candidate_id']} / {candidate['realization_id']} "
                        f"({candidate['title']})"
                    ),
                    f"  family_group: {candidate['family_group']}",
                    f"  rationale: {candidate['rationale']}",
                    f"  instructions: {candidate['instructions']}",
                    f"  composition_tags: {', '.join(candidate['composition_tags']) or 'none'}",
                    (
                        f"  incompatible_with: {', '.join(candidate['incompatible_with']) or 'none'}"
                    ),
                    (
                        f"  mutate_after_survival: "
                        f"{', '.join(candidate['mutate_after_survival']) or 'none'}"
                    ),
                ]
            )

        lines.extend(
            [
                "",
                "Control principles:",
                *[f"- {item}" for item in pack["control_plan"]],
                "",
                "Admission principles:",
                *[f"- {item}" for item in pack["admission_rules"]],
                "",
                "Metric principles:",
                *[f"- {item}" for item in pack["metric_plan"]],
                "",
                "Implementation principles:",
                *[f"- {item}" for item in pack["implementation_notes"]],
                "",
                "Global campaign instruction:",
                campaign["campaign_handoff"]["global_instructions"],
                "",
                "Hard execution rules:",
                "- Preserve the requested slot ids exactly so downstream promotion can attribute outcomes.",
                "- Do not substitute filler candidates if one realization cannot be mapped honestly.",
                "- Prefer fewer executable candidates over invented or mixed-lane fillers.",
                "- Treat this as one campaign pack, not a whole-strategy rewrite.",
            ]
        )
        return "\n".join(lines).strip() + "\n"

    def build_pack_queue(self, round_index: int) -> Path:
        self.validate_round(round_index)
        paths = self.round_paths(round_index)
        request = self.load_round_request(round_index)
        verification = load_json(paths["verification_report"])
        campaign = self.load_campaign(round_index, request, verification)
        dossiers = self.load_dossiers(round_index, campaign)
        packing = request["packing"]

        compile_map: dict[str, dict[str, Any]] = {}
        compile_paths: dict[str, str] = {}
        deferred: list[dict[str, Any]] = []
        for family in campaign["families"]:
            candidate_id = family["candidate_id"]
            compile_path = paths["family_compiles_dir"] / self.family_file_name(candidate_id)
            if not compile_path.exists():
                deferred.append(
                    {
                        "candidate_id": candidate_id,
                        "family_group": family["family_group"],
                        "reason": "family compile file missing",
                    }
                )
                continue
            compile_payload = self.normalize_family_compile(load_json(compile_path), dossiers[candidate_id])
            write_json(compile_path, compile_payload)
            compile_map[candidate_id] = compile_payload
            compile_paths[candidate_id] = self.project_relative(compile_path)

        grouped: dict[tuple[str, str, str], dict[str, list[dict[str, Any]]]] = {}
        for family in campaign["families"]:
            candidate_id = family["candidate_id"]
            compile_payload = compile_map.get(candidate_id)
            if compile_payload is None:
                continue
            if compile_payload["verdict"] != "READY":
                deferred.append(
                    {
                        "candidate_id": candidate_id,
                        "family_group": family["family_group"],
                        "reason": f"family compile verdict={compile_payload['verdict']}",
                    }
                )
                continue
            if family["pack_kind"] not in INITIAL_EXECUTABLE_PACK_KINDS:
                deferred.append(
                    {
                        "candidate_id": candidate_id,
                        "family_group": family["family_group"],
                        "reason": f"pack kind {family['pack_kind']} is reserved for later campaign stages",
                    }
                )
                continue
            if family["implementation_mode"] != "catalog_executable_now":
                deferred.append(
                    {
                        "candidate_id": candidate_id,
                        "family_group": family["family_group"],
                        "reason": f"family implementation_mode={family['implementation_mode']}",
                    }
                )
                continue
            key = (family["pack_kind"], family["lane"], family["phase_window"])
            family_bucket = grouped.setdefault(key, {})
            family_entries = family_bucket.setdefault(candidate_id, [])
            for realization in compile_payload["realizations"]:
                family_entries.append(
                    {
                        "candidate_id": candidate_id,
                        "family_group": family["family_group"],
                        "name": family["name"],
                        "lane": family["lane"],
                        "phase_window": family["phase_window"],
                        "pack_kind": family["pack_kind"],
                        "rationale": family["pack_rationale"],
                        "realization_id": realization["realization_id"],
                        "realization_key": realization["realization_key"],
                        "title": realization["title"],
                        "instructions": realization["instructions"],
                        "control_plan": realization["control_plan"],
                        "admission_rules": realization["admission_rules"],
                        "metric_plan": realization["metric_plan"],
                        "implementation_notes": realization["implementation_notes"],
                        "composition_tags": realization["composition_tags"],
                        "incompatible_with": realization["incompatible_with"],
                        "mutate_after_survival": realization["mutate_after_survival"],
                        "dossier_path": dossiers[candidate_id]["dossier_path"],
                        "compile_path": compile_paths[candidate_id],
                    }
                )

        packs: list[dict[str, Any]] = []
        pack_counters: dict[tuple[str, str, str], int] = {}
        for key, family_bucket in grouped.items():
            pack_kind, lane, phase_window = key
            working = {
                candidate_id: list(sorted(items, key=lambda item: item["realization_id"]))
                for candidate_id, items in sorted(family_bucket.items())
            }
            family_ids = list(working.keys())
            while any(working[candidate_id] for candidate_id in family_ids):
                selected: list[dict[str, Any]] = []
                family_counts: dict[str, int] = {}
                progress = True
                while len(selected) < int(packing["max_candidates_per_pack"]) and progress:
                    progress = False
                    for candidate_id in family_ids:
                        if len(selected) >= int(packing["max_candidates_per_pack"]):
                            break
                        if family_counts.get(candidate_id, 0) >= int(
                            packing["max_realizations_per_family_per_pack"]
                        ):
                            continue
                        if not working[candidate_id]:
                            continue
                        selected.append(working[candidate_id].pop(0))
                        family_counts[candidate_id] = family_counts.get(candidate_id, 0) + 1
                        progress = True
                if not selected:
                    break
                pack_index = pack_counters.get(key, 0) + 1
                pack_counters[key] = pack_index
                pack_id = (
                    f"{safe_slug(pack_kind)}_"
                    f"{safe_slug(lane)}_"
                    f"{safe_slug(phase_window)}_"
                    f"{pack_index:03d}"
                )
                control_plan = dedupe_strings(
                    list(campaign["campaign_handoff"]["control_principles"])
                    + [item for candidate in selected for item in candidate["control_plan"]]
                )
                admission_rules = dedupe_strings(
                    list(campaign["campaign_handoff"]["admission_principles"])
                    + [item for candidate in selected for item in candidate["admission_rules"]]
                )
                metric_plan = dedupe_strings(
                    list(campaign["campaign_handoff"]["metric_principles"])
                    + [item for candidate in selected for item in candidate["metric_plan"]]
                )
                implementation_notes = dedupe_strings(
                    list(campaign["campaign_handoff"]["implementation_principles"])
                    + [item for candidate in selected for item in candidate["implementation_notes"]]
                )
                candidate_slots: list[dict[str, Any]] = []
                for index, candidate in enumerate(selected):
                    candidate_slots.append(
                        {
                            **candidate,
                            "slot_id": f"H{index:02d}",
                        }
                    )
                pack = {
                    "pack_id": pack_id,
                    "pack_kind": pack_kind,
                    "lane": lane,
                    "phase_window": phase_window,
                    "controls_per_pack": int(campaign["pack_strategy"]["controls_per_pack"]),
                    "control_slot_ids": [f"C{i}" for i in range(int(campaign["pack_strategy"]["controls_per_pack"]))],
                    "candidate_slots": candidate_slots,
                    "control_plan": control_plan,
                    "admission_rules": admission_rules,
                    "metric_plan": metric_plan,
                    "implementation_notes": implementation_notes,
                    "notes": list(campaign["pack_strategy"]["notes"]),
                }
                pack["search_harness_instructions"] = self.build_pack_instruction_text(request, campaign, pack)
                packs.append(pack)

        packs.sort(
            key=lambda pack: (
                self.pack_order_index(campaign, pack["pack_kind"]),
                pack["pack_kind"],
                pack["pack_id"],
            )
        )
        seen_pack_ids: set[str] = set()
        duplicate_pack_ids: list[str] = []
        for pack in packs:
            pack_id = pack["pack_id"]
            if pack_id in seen_pack_ids:
                duplicate_pack_ids.append(pack_id)
                continue
            seen_pack_ids.add(pack_id)
        if duplicate_pack_ids:
            raise HarnessError(
                "pack queue generated duplicate pack ids: " f"{sorted(set(duplicate_pack_ids))}"
            )
        payload = {
            "cycle_id": request["cycle_id"],
            "round": request["round"],
            "campaign_goal": campaign["campaign_goal"],
            "pack_strategy": campaign["pack_strategy"],
            "packs": packs,
            "deferred_families": deferred,
        }
        write_json(paths["pack_queue"], payload)
        trajectories = self.load_round_trajectories(round_index, request)
        self.build_family_status_reports(round_index, request, trajectories, verification, campaign)
        return paths["pack_queue"]

    def canonicalize_search_harness_path(self, raw: str | None, base_dir: Path) -> str | None:
        if raw is None:
            return None
        path = Path(raw)
        if path.is_absolute():
            return str(path.resolve())
        repo_candidate = (self.repo_root / path).resolve()
        config_candidate = (base_dir / path).resolve()
        path = repo_candidate if repo_candidate.exists() or not config_candidate.exists() else config_candidate
        return relative_or_absolute(path, self.repo_root)

    def build_search_harness_config(
        self,
        base_config_path: Path,
        round_index: int,
        candidate_count: int,
    ) -> dict[str, Any]:
        base_dir = base_config_path.parent
        config = copy.deepcopy(load_json(base_config_path))
        config["cycle_id"] = f"{config.get('cycle_id', 'search')}_{self.cycle_id}_r{round_index:03d}"

        workspace = config.setdefault("workspace", {})
        workspace["root"] = self.canonicalize_search_harness_path(workspace.get("root", "."), base_dir)

        prompts = config.setdefault("prompts", {})
        for key, raw in list(prompts.items()):
            prompts[key] = self.canonicalize_search_harness_path(raw, base_dir)

        base = config.setdefault("base", {})
        if "default_base_script" in base:
            base["default_base_script"] = self.canonicalize_search_harness_path(base.get("default_base_script"), base_dir)
        if base.get("default_patch_module") is not None:
            base["default_patch_module"] = self.canonicalize_search_harness_path(base.get("default_patch_module"), base_dir)

        codex = config.setdefault("codex", {})
        for key in ("cd", "catalog_root", "compiled_schema", "verifier_schema"):
            if codex.get(key) is not None:
                codex[key] = self.canonicalize_search_harness_path(codex.get(key), base_dir)

        admission = config.setdefault("admission", {})
        if admission.get("fixed_base_script") is not None:
            admission["fixed_base_script"] = self.canonicalize_search_harness_path(
                admission.get("fixed_base_script"),
                base_dir,
            )
        admission["min_candidate_slots"] = candidate_count
        admission["max_candidate_slots"] = candidate_count

        hooks = config.setdefault("hooks", {})
        for hook_name in ("generator", "verifier", "compiler", "extractor"):
            hook = hooks.get(hook_name)
            if isinstance(hook, dict) and hook.get("cwd") is not None:
                hook["cwd"] = self.canonicalize_search_harness_path(hook.get("cwd"), base_dir)

        selection = config.setdefault("selection", {})
        control_slots = [str(slot) for slot in selection.get("control_slots", ["C0", "C1"])]
        if len(control_slots) < 2:
            control_slots = ["C0", "C1"]
        selection["control_slots"] = control_slots[:2]
        current_top_k = int(selection.get("top_k", 1))
        selection["top_k"] = max(1, min(current_top_k, candidate_count))

        return config

    def select_pack_ids(
        self,
        pack_queue: dict[str, Any],
        pack_ids: list[str],
        all_packs: bool,
    ) -> list[str]:
        available = [pack["pack_id"] for pack in pack_queue.get("packs", [])]
        if all_packs:
            return available
        requested = dedupe_strings(pack_ids)
        if requested:
            unknown = [pack_id for pack_id in requested if pack_id not in available]
            if unknown:
                raise HarnessError(f"Unknown pack ids requested: {unknown}")
            return requested
        if len(available) == 1:
            return available
        raise HarnessError("Specify --pack-id or --all-packs when the pack queue contains multiple packs")

    def handoff_to_search_harness(
        self,
        round_index: int,
        harness_config: str | None,
        pack_ids: list[str],
        all_packs: bool,
        start_generation: int,
        repo_files: list[str] | None = None,
        resume: bool = False,
    ) -> Path:
        self.build_pack_queue(round_index)
        paths = self.round_paths(round_index)
        request = self.load_round_request(round_index)
        verification = load_json(paths["verification_report"])
        campaign = self.load_campaign(round_index, request, verification)
        pack_queue = load_json(paths["pack_queue"])
        selected_pack_ids = self.select_pack_ids(pack_queue, pack_ids, all_packs)

        handoff = self.handoff_settings()
        target_config_raw = harness_config or handoff.get("default_search_harness_config")
        if not target_config_raw:
            raise HarnessError("No search_harness config provided and handoff.default_search_harness_config is unset")
        target_config = self.resolve_runtime_path(str(target_config_raw), default_to_repo_root=False)
        if target_config is None or not target_config.exists():
            raise HarnessError(f"search_harness config does not exist: {target_config_raw}")

        paths["pack_handoffs_dir"].mkdir(parents=True, exist_ok=True)
        existing_handoffs: dict[str, Any] = {}
        if paths["pack_handoffs"].exists():
            payload = load_json(paths["pack_handoffs"])
            for item in payload.get("handoffs", []):
                if isinstance(item, dict) and item.get("pack_id"):
                    existing_handoffs[str(item["pack_id"])] = item

        packs_by_id = {pack["pack_id"]: pack for pack in pack_queue.get("packs", [])}
        used_generations = {
            int(item["generation"])
            for item in existing_handoffs.values()
            if isinstance(item, dict) and item.get("generation") is not None
        }
        generation = start_generation
        handoff_records: list[dict[str, Any]] = []
        for pack_id in selected_pack_ids:
            if resume and pack_id in existing_handoffs:
                handoff_records.append(existing_handoffs[pack_id])
                continue
            pack = packs_by_id[pack_id]
            candidate_count = len(pack["candidate_slots"])
            if candidate_count < 1:
                raise HarnessError(f"pack {pack_id} has no candidate slots")
            while generation in used_generations:
                generation += 1

            derived_config = self.build_search_harness_config(target_config, round_index, candidate_count)
            derived_config_path = paths["pack_handoffs_dir"] / f"{pack_id}_search_harness_config.json"
            write_json(derived_config_path, derived_config)
            harness = SearchHarness(derived_config_path)

            handoff_md_path = paths["pack_handoffs_dir"] / f"{pack_id}_SEARCH_HANDOFF.md"
            handoff_prompt_path = paths["pack_handoffs_dir"] / f"{pack_id}_SEARCH_HANDOFF_PROMPT.txt"
            markdown_lines = [
                f"# Search Harness Handoff for {pack_id}",
                "",
                f"- **Round**: `{request['round']}`",
                f"- **Pack**: `{pack_id}`",
                f"- **Pack kind**: `{pack['pack_kind']}`",
                f"- **Lane**: `{pack['lane']}`",
                f"- **Phase window**: `{pack['phase_window']}`",
                "",
                "```text",
                pack["search_harness_instructions"].rstrip(),
                "```",
            ]
            write_text(handoff_md_path, "\n".join(markdown_lines) + "\n")
            write_text(handoff_prompt_path, pack["search_harness_instructions"])

            downstream_files = list(repo_files or [])
            downstream_files.extend(str(item) for item in handoff.get("default_repo_files", []))
            target_script = request.get("objective", {}).get("target_script")
            if target_script:
                downstream_files.append(str(target_script))
            downstream_files.extend(
                [
                    harness.project_relative(paths["search_handoff_md"]),
                    harness.project_relative(paths["campaign"]),
                    harness.project_relative(paths["pack_queue"]),
                    harness.project_relative(handoff_md_path),
                ]
            )
            for candidate in pack["candidate_slots"]:
                downstream_files.append(candidate["dossier_path"])
                downstream_files.append(candidate["compile_path"])
            downstream_files = dedupe_strings(downstream_files)

            agent_dir = harness.prepare_agent_folder(
                generation=generation,
                instructions=pack["search_harness_instructions"],
                repo_files=downstream_files,
            )
            generation_paths = harness.generation_paths(generation)
            record = {
                "pack_id": pack_id,
                "round": round_index,
                "search_harness_config": str(derived_config_path),
                "base_search_harness_config": str(target_config),
                "generation": generation,
                "generation_dir": str(generation_paths["generation_dir"]),
                "agent_dir": str(agent_dir),
                "pack_kind": pack["pack_kind"],
                "lane": pack["lane"],
                "phase_window": pack["phase_window"],
                "control_slot_ids": pack["control_slot_ids"],
                "candidate_slot_map": {
                    candidate["slot_id"]: {
                        "candidate_id": candidate["candidate_id"],
                        "family_group": candidate["family_group"],
                        "realization_id": candidate["realization_id"],
                        "realization_key": candidate["realization_key"],
                        "title": candidate["title"],
                    }
                    for candidate in pack["candidate_slots"]
                },
                "focus_files": downstream_files,
                "handoff_markdown": str(handoff_md_path),
                "handoff_prompt": str(handoff_prompt_path),
                "status": "staged",
            }
            existing_handoffs[pack_id] = record
            handoff_records.append(record)
            used_generations.add(generation)
            generation += 1

        write_json(
            paths["pack_handoffs"],
            {
                "cycle_id": request["cycle_id"],
                "round": request["round"],
                "handoffs": [existing_handoffs[key] for key in sorted(existing_handoffs)],
            },
        )
        return paths["pack_handoffs"]

    def promote_families(self, round_index: int) -> Path:
        self.build_pack_queue(round_index)
        paths = self.round_paths(round_index)
        request = self.load_round_request(round_index)
        verification = load_json(paths["verification_report"])
        campaign = self.load_campaign(round_index, request, verification)
        if not paths["pack_handoffs"].exists():
            raise HarnessError(f"missing pack handoffs file: {paths['pack_handoffs']}")
        handoffs_payload = load_json(paths["pack_handoffs"])
        handoffs = [item for item in handoffs_payload.get("handoffs", []) if isinstance(item, dict)]

        family_lookup = {family["candidate_id"]: family for family in campaign["families"]}
        evidence_by_family: dict[str, list[dict[str, Any]]] = {candidate_id: [] for candidate_id in family_lookup}
        pack_reports: list[dict[str, Any]] = []

        for handoff in handoffs:
            config_path = Path(str(handoff["search_harness_config"])).resolve()
            harness = SearchHarness(config_path)
            generation = int(handoff["generation"])
            generation_paths = harness.generation_paths(generation)
            summary_path = generation_paths["generation_summary"]
            results_path = generation_paths["results_index"]
            if not summary_path.exists() or not results_path.exists():
                pack_reports.append(
                    {
                        "pack_id": handoff["pack_id"],
                        "generation": generation,
                        "status": "pending",
                        "reason": "generation_summary.json or results_index.json not available yet",
                    }
                )
                continue

            summary = load_json(summary_path)
            results_index = load_json(results_path)
            metric_path = str(summary.get("primary_metric_path", "metrics.score_bpb"))
            goal = str(summary.get("goal", "minimize"))
            control_slot_ids = [str(slot) for slot in handoff.get("control_slot_ids", summary.get("control_slots", []))]
            control_values: list[float] = []
            for slot_id in control_slot_ids:
                try:
                    control_values.append(float(nested_get(results_index["slots"][slot_id], metric_path)))
                except Exception:  # noqa: BLE001
                    continue
            control_anchor = best_value(goal, control_values)
            control_mean = average(control_values)
            invalid_reasons = [str(item) for item in summary.get("invalid_reasons", [])]
            valid_pack = not invalid_reasons and control_anchor is not None

            pack_candidate_reports: list[dict[str, Any]] = []
            for slot_id, slot_meta in handoff.get("candidate_slot_map", {}).items():
                record = {
                    "slot_id": slot_id,
                    **slot_meta,
                }
                try:
                    payload = results_index["slots"][slot_id]
                    metric = float(nested_get(payload, metric_path))
                    record["metric"] = metric
                    record["survived"] = slot_id in summary.get("survivors", [])
                    record["valid_pack"] = valid_pack
                    if valid_pack and control_anchor is not None:
                        record["positive_vs_control"] = better(goal, metric, float(control_anchor))
                    else:
                        record["positive_vs_control"] = False
                    evidence_by_family[slot_meta["candidate_id"]].append(record)
                except Exception as exc:  # noqa: BLE001
                    record["error"] = str(exc)
                    evidence_by_family[slot_meta["candidate_id"]].append(record)
                pack_candidate_reports.append(record)

            pack_reports.append(
                {
                    "pack_id": handoff["pack_id"],
                    "generation": generation,
                    "status": "evaluated",
                    "valid_pack": valid_pack,
                    "invalid_reasons": invalid_reasons,
                    "metric_path": metric_path,
                    "goal": goal,
                    "control_values": control_values,
                    "control_anchor": control_anchor,
                    "control_mean": control_mean,
                    "candidates": pack_candidate_reports,
                }
            )

        promotion = request["promotion"]
        family_reports: list[dict[str, Any]] = []
        for candidate_id, family in family_lookup.items():
            evidence = evidence_by_family.get(candidate_id, [])
            valid_evidence = [
                item for item in evidence if item.get("valid_pack") and "metric" in item
            ]
            positive_evidence = [item for item in valid_evidence if item.get("positive_vs_control")]
            surviving_evidence = [item for item in valid_evidence if item.get("survived")]
            metrics = [float(item["metric"]) for item in valid_evidence]
            goal = "minimize"
            best_metric_value = best_value(goal, metrics)
            best_realization = None
            if best_metric_value is not None:
                for item in valid_evidence:
                    if float(item["metric"]) == float(best_metric_value):
                        best_realization = item.get("realization_id")
                        break

            promoted = False
            if len(positive_evidence) >= int(promotion["min_positive_realizations"]):
                if not promotion["require_directional_support"]:
                    promoted = True
                elif len(surviving_evidence) >= int(promotion["min_surviving_realizations"]):
                    promoted = True

            status = "pending"
            if not evidence:
                status = "pending"
            elif not valid_evidence:
                status = "blocked"
            elif promoted:
                status = "promoted"
            elif positive_evidence:
                status = "watch"
            else:
                status = "retire"

            family_reports.append(
                {
                    "candidate_id": candidate_id,
                    "name": family["name"],
                    "family_group": family["family_group"],
                    "lane": family["lane"],
                    "phase_window": family["phase_window"],
                    "pack_kind": family["pack_kind"],
                    "composition_tags": family["composition_tags"],
                    "incompatible_with": family["incompatible_with"],
                    "executed_realizations": len([item for item in evidence if "metric" in item]),
                    "valid_realizations": len(valid_evidence),
                    "positive_realizations": len(positive_evidence),
                    "surviving_realizations": len(surviving_evidence),
                    "best_metric": best_metric_value,
                    "best_realization_id": best_realization,
                    "status": status,
                    "promoted": promoted,
                    "evidence": evidence,
                }
            )

        payload = {
            "cycle_id": request["cycle_id"],
            "round": request["round"],
            "promotion_policy": request["promotion"],
            "pack_reports": pack_reports,
            "family_reports": family_reports,
        }
        write_json(paths["promotion_report"], payload)
        return paths["promotion_report"]

    def families_compatible(self, left: dict[str, Any], right: dict[str, Any]) -> bool:
        left_incompat = set(str(item) for item in left.get("incompatible_with", []))
        right_incompat = set(str(item) for item in right.get("incompatible_with", []))
        left_tags = set(str(item) for item in left.get("composition_tags", []))
        right_tags = set(str(item) for item in right.get("composition_tags", []))
        left_keys = {left["candidate_id"], left["family_group"], *left_tags}
        right_keys = {right["candidate_id"], right["family_group"], *right_tags}
        return not (left_incompat.intersection(right_keys) or right_incompat.intersection(left_keys))

    def composition_pair_score(self, left: dict[str, Any], right: dict[str, Any]) -> tuple[int, int, int]:
        lane_bonus = 1 if left["lane"] != right["lane"] else 0
        phase_bonus = 1 if left["phase_window"] != right["phase_window"] else 0
        tag_bonus = len(set(left.get("composition_tags", [])).intersection(right.get("composition_tags", [])))
        evidence_bonus = int(left.get("positive_realizations", 0)) + int(right.get("positive_realizations", 0))
        return (tag_bonus + lane_bonus + phase_bonus + evidence_bonus, evidence_bonus, tag_bonus)

    def compose_survivors(self, round_index: int) -> Path:
        promotion_path = self.promote_families(round_index)
        paths = self.round_paths(round_index)
        request = self.load_round_request(round_index)
        verification = load_json(paths["verification_report"])
        campaign = self.load_campaign(round_index, request, verification)
        promotion_report = load_json(promotion_path)
        promoted = [item for item in promotion_report.get("family_reports", []) if item.get("promoted")]

        candidate_pairs: list[dict[str, Any]] = []
        for left, right in combinations(promoted, 2):
            if not self.families_compatible(left, right):
                continue
            score = self.composition_pair_score(left, right)
            candidate_pairs.append(
                {
                    "composition_id": "",
                    "families": [left["candidate_id"], right["candidate_id"]],
                    "preferred_realizations": [
                        left.get("best_realization_id"),
                        right.get("best_realization_id"),
                    ],
                    "lanes": [left["lane"], right["lane"]],
                    "phase_windows": [left["phase_window"], right["phase_window"]],
                    "family_groups": [left["family_group"], right["family_group"]],
                    "score": score[0],
                    "rationale": (
                        f"Combine {left['name']} ({left['lane']}/{left['phase_window']}) with "
                        f"{right['name']} ({right['lane']}/{right['phase_window']}) after solo survival."
                    ),
                    "instructions": (
                        f"Build a pairwise composition pack for {left['candidate_id']} and {right['candidate_id']} "
                        "using their best solo-supported realizations only."
                    ),
                }
            )

        candidate_pairs.sort(
            key=lambda item: (-int(item["score"]), item["families"][0], item["families"][1])
        )
        limited_pairs = candidate_pairs[: int(request["composition"]["max_pairwise_candidates"])]
        for index, item in enumerate(limited_pairs, start=1):
            item["composition_id"] = f"pair_{index:03d}"

        composition_payload = {
            "cycle_id": request["cycle_id"],
            "round": request["round"],
            "campaign_goal": campaign["campaign_goal"],
            "composition_policy": campaign["composition_policy"],
            "pairs": limited_pairs,
        }
        write_json(paths["composition_queue"], composition_payload)

        hybrid_payload = {
            "cycle_id": request["cycle_id"],
            "round": request["round"],
            "status": "awaiting_pairwise_results",
            "max_hybrid_candidates": int(request["composition"]["max_hybrid_candidates"]),
            "hybrid_entry_rule": campaign["composition_policy"]["hybrid_entry_rule"],
            "notes": [
                "Do not build final hybrids until pairwise composition packs have real evidence.",
                "Use pairwise winners, not all promoted solos, as the input set for hybrid assembly.",
            ],
            "candidates": [],
        }
        write_json(paths["hybrid_queue"], hybrid_payload)
        return paths["composition_queue"]

    def run_postmortem(
        self,
        round_index: int,
        max_attempts: int | None = None,
        resume: bool = False,
    ) -> Path:
        self.promote_families(round_index)
        self.compose_survivors(round_index)
        paths = self.round_paths(round_index)
        request = self.load_round_request(round_index)
        verification = load_json(paths["verification_report"])
        campaign = self.load_campaign(round_index, request, verification)
        promotion_report = load_json(paths["promotion_report"])
        pack_evidence = self.summarize_pack_evidence(round_index)
        if resume and paths["postmortem_report"].exists():
            try:
                postmortem = self.normalize_postmortem_report(load_json(paths["postmortem_report"]), campaign)
                write_json(paths["postmortem_report"], postmortem)
                if postmortem["verdict"] == "READY":
                    return paths["postmortem_report"]
            except Exception:  # noqa: BLE001
                pass

        attempts = int(max_attempts or self.codex_settings().get("max_attempts", 2))
        if attempts < 1:
            raise HarnessError("max_attempts must be >= 1")
        previous_feedback = ""
        for attempt in range(1, attempts + 1):
            attempt_dir = paths["codex_dir"] / f"postmortem_{attempt:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            failure_path = attempt_dir / "failure.txt"
            attempt_feedback = self.load_attempt_feedback(failure_path) if resume else previous_feedback
            if resume and not attempt_feedback:
                attempt_feedback = previous_feedback
            try:
                raw_path = attempt_dir / "postmortem_raw.json"
                if resume and raw_path.exists():
                    raw = load_json(raw_path)
                else:
                    prompt = self.build_postmortem_prompt(
                        request=request,
                        campaign=campaign,
                        promotion_report=promotion_report,
                        pack_evidence=pack_evidence,
                        previous_feedback=attempt_feedback or None,
                    )
                    write_text(attempt_dir / "postmortem_prompt.txt", prompt)
                    raw = self.invoke_codex(
                        prompt=prompt,
                        schema_path=self.schema_path("analyst"),
                        output_path=raw_path,
                        attempt_dir=attempt_dir,
                        label="analyst",
                        stage="analyst",
                    )
                postmortem = self.normalize_postmortem_report(raw, campaign)
                write_json(paths["postmortem_report"], postmortem)
                if postmortem["verdict"] != "READY":
                    raise HarnessError(
                        f"Postmortem returned {postmortem['verdict']}: {postmortem['summary']}"
                    )
                return paths["postmortem_report"]
            except Exception as exc:  # noqa: BLE001
                previous_feedback = str(exc)
                write_text(failure_path, previous_feedback + "\n")
        raise HarnessError(
            f"pg_enigma could not produce postmortem for round {round_index} after {attempts} attempts: "
            f"{previous_feedback}"
        )

    def next_generation_start(self, round_index: int, explicit: int | None) -> int:
        if explicit is not None:
            return int(explicit)
        paths = self.round_paths(round_index)
        if not paths["pack_handoffs"].exists():
            return 0
        payload = load_json(paths["pack_handoffs"])
        generations = [
            int(item["generation"])
            for item in payload.get("handoffs", [])
            if isinstance(item, dict) and item.get("generation") is not None
        ]
        if not generations:
            return 0
        return max(generations) + 1

    def run_round(
        self,
        round_index: int,
        instructions: str | None,
        harness_config: str | None,
        start_generation: int,
        repo_files: list[str] | None = None,
        evidence_files: list[str] | None = None,
        max_attempts: int | None = None,
        pack_ids: list[str] | None = None,
        all_packs: bool = True,
        resume: bool = False,
        backend: str = "codex",
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> Path:
        self.run_model_round(
            round_index=round_index,
            instructions=instructions,
            repo_files=repo_files,
            evidence_files=evidence_files,
            max_attempts=max_attempts,
            resume=resume,
            backend=backend,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        self.compile_families(
            round_index=round_index,
            family_ids=None,
            max_attempts=max_attempts,
            resume=resume,
            backend=backend,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        self.build_pack_queue(round_index)
        request = self.load_round_request(round_index)
        downstream_repo_files = list(repo_files or [])
        if resume and not downstream_repo_files:
            downstream_repo_files = [str(item) for item in request.get("focus_files", [])]
        return self.handoff_to_search_harness(
            round_index=round_index,
            harness_config=harness_config,
            pack_ids=pack_ids or [],
            all_packs=all_packs if not pack_ids else False,
            start_generation=start_generation,
            repo_files=downstream_repo_files,
            resume=resume,
        )

    def runnable_families_report(self, round_index: int) -> dict[str, Any]:
        self.build_pack_queue(round_index)
        paths = self.round_paths(round_index)
        if not paths["runnable_families"].exists():
            raise HarnessError(f"missing runnable families report: {paths['runnable_families']}")
        return load_json(paths["runnable_families"])

    def advance_round(
        self,
        from_round: int,
        to_round: int,
        harness_config: str | None,
        start_generation: int | None = None,
        instructions: str | None = None,
        repo_files: list[str] | None = None,
        evidence_files: list[str] | None = None,
        max_attempts: int | None = None,
        pack_ids: list[str] | None = None,
        all_packs: bool = True,
    ) -> Path:
        self.run_postmortem(from_round, max_attempts=max_attempts)
        previous_request = self.load_round_request(from_round)
        next_instructions = instructions or str(previous_request.get("user_instructions", "")).strip()
        if not next_instructions:
            raise HarnessError("advance-round requires instructions or a previous round with user_instructions")
        next_repo_files = list(repo_files) if repo_files else [str(item) for item in previous_request.get("focus_files", [])]
        next_evidence_files = (
            list(evidence_files) if evidence_files else [str(item) for item in previous_request.get("evidence_files", [])]
        )
        return self.run_round(
            round_index=to_round,
            instructions=next_instructions,
            harness_config=harness_config,
            start_generation=self.next_generation_start(from_round, start_generation),
            repo_files=next_repo_files,
            evidence_files=next_evidence_files,
            max_attempts=max_attempts,
            pack_ids=pack_ids,
            all_packs=all_packs,
        )


def init_config(path: Path) -> None:
    template = load_json((Path(__file__).resolve().parent / "reference_config.json").resolve())
    write_json(path, template)


def load_instructions_arg(args: argparse.Namespace) -> str:
    if getattr(args, "instructions", None):
        return str(args.instructions)
    raw_file = getattr(args, "instructions_file", None)
    if raw_file:
        return load_text(Path(raw_file).resolve())
    raise HarnessError("Either --instructions or --instructions-file is required")


def load_optional_instructions_arg(args: argparse.Namespace) -> str | None:
    if getattr(args, "instructions", None):
        return str(args.instructions)
    raw_file = getattr(args, "instructions_file", None)
    if raw_file:
        return load_text(Path(raw_file).resolve())
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Campaign-oriented consequential hypothesis front-end for search_harness.py.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-config", help="Write an example pg_enigma config.")
    init.add_argument("--output", required=True)

    prepare = sub.add_parser("prepare-round", help="Create a round request plus focus/evidence snapshots.")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--round", type=int, required=True)
    prepare_group = prepare.add_mutually_exclusive_group(required=True)
    prepare_group.add_argument("--instructions")
    prepare_group.add_argument("--instructions-file")
    prepare.add_argument("--repo-file", action="append", default=[], help="Repo file to emphasize. Repeatable.")
    prepare.add_argument("--evidence-file", action="append", default=[], help="Evidence file to snapshot. Repeatable.")

    agent = sub.add_parser(
        "prepare-agent-folder",
        help="Create an external-agent folder with prompts, schemas, snapshots, and instructions.",
    )
    agent.add_argument("--config", required=True)
    agent.add_argument("--round", type=int, required=True)
    agent_group = agent.add_mutually_exclusive_group(required=True)
    agent_group.add_argument("--instructions")
    agent_group.add_argument("--instructions-file")
    agent.add_argument("--repo-file", action="append", default=[], help="Repo file to emphasize. Repeatable.")
    agent.add_argument("--evidence-file", action="append", default=[], help="Evidence file to snapshot. Repeatable.")

    codex_round = sub.add_parser(
        "codex-round",
        help="Run the direct Codex explore -> verify -> campaign-distill loop for one round.",
    )
    codex_round.add_argument("--config", required=True)
    codex_round.add_argument("--round", type=int, required=True)
    codex_group = codex_round.add_mutually_exclusive_group()
    codex_group.add_argument("--instructions")
    codex_group.add_argument("--instructions-file")
    codex_round.add_argument("--repo-file", action="append", default=[], help="Repo file to emphasize. Repeatable.")
    codex_round.add_argument("--evidence-file", action="append", default=[], help="Evidence file to snapshot. Repeatable.")
    codex_round.add_argument("--max-attempts", type=int)
    codex_round.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the stored round request and any existing attempt artifacts instead of clearing the round.",
    )

    validate = sub.add_parser("validate-round", help="Validate a completed round and render campaign outputs.")
    validate.add_argument("--config", required=True)
    validate.add_argument("--round", type=int, required=True)

    debug_verifier = sub.add_parser(
        "debug-verifier",
        help="Run the round verifier in real-world debug mode via copilot CLI and write prompt/raw/normalized artifacts.",
    )
    debug_verifier.add_argument("--config", required=True)
    debug_verifier.add_argument("--round", type=int, required=True)
    debug_verifier.add_argument(
        "--mode",
        default="auto",
        choices=["auto", "full", "diff"],
        help="full = review all candidates, diff = review only changed candidates vs a prior attempt, auto = choose diff when it meaningfully applies.",
    )
    debug_verifier.add_argument(
        "--attempt",
        type=int,
        help="Use trajectories from this attempt directory instead of the current round-level trajectories.",
    )
    debug_verifier.add_argument(
        "--reuse-from-attempt",
        type=int,
        help="Prior attempt to diff against when using auto or diff mode.",
    )
    debug_verifier.add_argument("--model", default="gpt-5.4")
    debug_verifier.add_argument(
        "--reasoning-effort",
        default="xhigh",
        choices=["low", "medium", "high", "xhigh"],
    )
    debug_verifier.add_argument("--label", help="Optional debug artifact folder label.")

    runnable = sub.add_parser(
        "runnable-families",
        help="Print the current runnable-now family shortlist for a completed round.",
    )
    runnable.add_argument("--config", required=True)
    runnable.add_argument("--round", type=int, required=True)

    run_round = sub.add_parser(
        "run-round",
        help="Run the full one-command round flow: codex round, family compile, pack queue, and search_harness handoff.",
    )
    run_round.add_argument("--config", required=True)
    run_round.add_argument("--round", type=int, required=True)
    run_group = run_round.add_mutually_exclusive_group()
    run_group.add_argument("--instructions")
    run_group.add_argument("--instructions-file")
    run_round.add_argument("--harness-config", help="Override the downstream search_harness config path.")
    run_round.add_argument("--start-generation", type=int, default=0)
    run_round.add_argument("--repo-file", action="append", default=[], help="Repo file to emphasize. Repeatable.")
    run_round.add_argument("--evidence-file", action="append", default=[], help="Evidence file to snapshot. Repeatable.")
    run_round.add_argument("--max-attempts", type=int)
    run_round.add_argument("--backend", choices=sorted(ALLOWED_MODEL_BACKENDS), default="codex")
    run_round.add_argument("--model", help="Model override for Copilot-backed round execution.")
    run_round.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        help="Reasoning effort override for Copilot-backed round execution.",
    )
    run_round.add_argument(
        "--resume",
        action="store_true",
        help="Resume a halted round from the stored request and existing attempt artifacts.",
    )
    run_round_group = run_round.add_mutually_exclusive_group()
    run_round_group.add_argument("--all-packs", action="store_true")
    run_round_group.add_argument("--pack-id", action="append", default=[])

    compile_cmd = sub.add_parser(
        "compile-families",
        help="Compile kept families into multiple code realization plans.",
    )
    compile_cmd.add_argument("--config", required=True)
    compile_cmd.add_argument("--round", type=int, required=True)
    compile_cmd.add_argument("--family-id", action="append", default=[], help="Specific family candidate_id to compile. Repeatable.")
    compile_cmd.add_argument("--max-attempts", type=int)
    compile_cmd.add_argument("--backend", choices=sorted(ALLOWED_MODEL_BACKENDS), default="codex")
    compile_cmd.add_argument("--model", help="Model override for Copilot-backed family compilation.")
    compile_cmd.add_argument(
        "--reasoning-effort",
        choices=["low", "medium", "high", "xhigh"],
        help="Reasoning effort override for Copilot-backed family compilation.",
    )
    compile_cmd.add_argument(
        "--resume",
        action="store_true",
        help="Resume family compilation from existing compile artifacts and attempt logs.",
    )

    pack_cmd = sub.add_parser(
        "build-pack-queue",
        help="Validate family compiles and build the executable pack queue.",
    )
    pack_cmd.add_argument("--config", required=True)
    pack_cmd.add_argument("--round", type=int, required=True)

    handoff = sub.add_parser(
        "handoff-to-search-harness",
        help="Create downstream search_harness agent folders from executable campaign packs.",
    )
    handoff.add_argument("--config", required=True)
    handoff.add_argument("--round", type=int, required=True)
    handoff.add_argument("--harness-config", help="Override the downstream search_harness config path.")
    handoff.add_argument("--start-generation", type=int, default=0)
    handoff.add_argument(
        "--resume",
        action="store_true",
        help="Reuse already staged pack handoffs and continue with any missing packs only.",
    )
    handoff_group = handoff.add_mutually_exclusive_group()
    handoff_group.add_argument("--all-packs", action="store_true")
    handoff_group.add_argument("--pack-id", action="append", default=[])
    handoff.add_argument(
        "--repo-file",
        action="append",
        default=[],
        help="Additional repo files to include in downstream search_harness agent folders. Repeatable.",
    )

    promote = sub.add_parser(
        "promote-families",
        help="Read executed pack evidence and write a family-level promotion report.",
    )
    promote.add_argument("--config", required=True)
    promote.add_argument("--round", type=int, required=True)

    postmortem = sub.add_parser(
        "postmortem-round",
        help="Run the analyst postmortem against executed round evidence.",
    )
    postmortem.add_argument("--config", required=True)
    postmortem.add_argument("--round", type=int, required=True)
    postmortem.add_argument("--max-attempts", type=int)
    postmortem.add_argument(
        "--resume",
        action="store_true",
        help="Resume the analyst postmortem from existing attempt artifacts.",
    )

    compose = sub.add_parser(
        "compose-survivors",
        help="Build pairwise composition and hybrid queue placeholders from promoted solo families.",
    )
    compose.add_argument("--config", required=True)
    compose.add_argument("--round", type=int, required=True)

    advance = sub.add_parser(
        "advance-round",
        help="Run postmortem for round N and then stage round N+1 in the same cycle folder.",
    )
    advance.add_argument("--config", required=True)
    advance.add_argument("--from-round", type=int, required=True)
    advance.add_argument("--to-round", type=int)
    advance_group = advance.add_mutually_exclusive_group()
    advance_group.add_argument("--instructions")
    advance_group.add_argument("--instructions-file")
    advance.add_argument("--harness-config", help="Override the downstream search_harness config path.")
    advance.add_argument("--start-generation", type=int)
    advance.add_argument("--repo-file", action="append", default=[], help="Override repo files for the next round. Repeatable.")
    advance.add_argument("--evidence-file", action="append", default=[], help="Override evidence files for the next round. Repeatable.")
    advance.add_argument("--max-attempts", type=int)
    advance_handoff_group = advance.add_mutually_exclusive_group()
    advance_handoff_group.add_argument("--all-packs", action="store_true")
    advance_handoff_group.add_argument("--pack-id", action="append", default=[])

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.command == "init-config":
        init_config(Path(args.output).resolve())
        return

    config_path = Path(args.config).resolve()
    harness = EnigmaHarness(config_path)

    if args.command == "prepare-round":
        harness.prepare_round(
            round_index=args.round,
            instructions=load_instructions_arg(args),
            repo_files=args.repo_file,
            evidence_files=args.evidence_file,
        )
        return

    if args.command == "prepare-agent-folder":
        harness.prepare_agent_folder(
            round_index=args.round,
            instructions=load_instructions_arg(args),
            repo_files=args.repo_file,
            evidence_files=args.evidence_file,
        )
        return

    if args.command == "codex-round":
        harness.run_codex_round(
            round_index=args.round,
            instructions=load_optional_instructions_arg(args),
            repo_files=args.repo_file,
            evidence_files=args.evidence_file,
            max_attempts=args.max_attempts,
            resume=args.resume,
        )
        return

    if args.command == "validate-round":
        harness.validate_round(args.round)
        return

    if args.command == "debug-verifier":
        harness.debug_verifier(
            round_index=args.round,
            mode=args.mode,
            attempt=args.attempt,
            reuse_from_attempt=args.reuse_from_attempt,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            label=args.label,
        )
        return

    if args.command == "runnable-families":
        payload = harness.runnable_families_report(args.round)
        print(json.dumps(payload, indent=2))
        return

    if args.command == "run-round":
        harness.run_round(
            round_index=args.round,
            instructions=load_optional_instructions_arg(args),
            harness_config=args.harness_config,
            start_generation=args.start_generation,
            repo_files=args.repo_file,
            evidence_files=args.evidence_file,
            max_attempts=args.max_attempts,
            pack_ids=args.pack_id,
            all_packs=True if not args.pack_id else False,
            resume=args.resume,
            backend=args.backend,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
        )
        return

    if args.command == "compile-families":
        harness.compile_families(
            round_index=args.round,
            family_ids=args.family_id,
            max_attempts=args.max_attempts,
            resume=args.resume,
            backend=args.backend,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
        )
        return

    if args.command == "build-pack-queue":
        harness.build_pack_queue(args.round)
        return

    if args.command == "handoff-to-search-harness":
        harness.handoff_to_search_harness(
            round_index=args.round,
            harness_config=args.harness_config,
            pack_ids=args.pack_id,
            all_packs=args.all_packs,
            start_generation=args.start_generation,
            repo_files=args.repo_file,
            resume=args.resume,
        )
        return

    if args.command == "promote-families":
        harness.promote_families(args.round)
        return

    if args.command == "postmortem-round":
        harness.run_postmortem(args.round, max_attempts=args.max_attempts, resume=args.resume)
        return

    if args.command == "compose-survivors":
        harness.compose_survivors(args.round)
        return

    if args.command == "advance-round":
        harness.advance_round(
            from_round=args.from_round,
            to_round=args.to_round if args.to_round is not None else args.from_round + 1,
            harness_config=args.harness_config,
            start_generation=args.start_generation,
            instructions=load_optional_instructions_arg(args),
            repo_files=args.repo_file,
            evidence_files=args.evidence_file,
            max_attempts=args.max_attempts,
            pack_ids=args.pack_id,
            all_packs=True if not args.pack_id else False,
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
