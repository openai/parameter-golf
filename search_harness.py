#!/usr/bin/env python3
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import re
import shlex
import shutil
import stat
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class HarnessError(RuntimeError):
    pass


DEFAULT_FREEZE_ENV_KEYS = [
    "DATA_PATH",
    "TOKENIZER_PATH",
    "VOCAB_SIZE",
]

PREDICATE_AUDIT_FIELDS = (
    "target_predicate",
    "baseline_truth",
    "score_path_trace",
    "operator_claim",
)

PREDICATE_BASELINE_TRUTH_VALUES = frozenset({"false", "true", "unknown"})


FAMILY_GROUP_OVERRIDES = {
    "best_checkpoint": "checkpoint_selection",
    "checkpoint_selection": "checkpoint_selection",
    "event_branch_tournament": "late_branch_portfolio",
    "late_branch_finishers": "late_branch_portfolio",
    "state_controller": "state_conditioned_control",
    "velocity_gate": "state_conditioned_control",
}


def infer_family_group(family: str) -> str:
    return FAMILY_GROUP_OVERRIDES.get(family, family)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_path(base_dir: Path, raw: str | None) -> Path | None:
    if raw is None:
        return None
    path = Path(raw)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def ensure_executable(path: Path) -> None:
    mode = path.stat().st_mode
    path.chmod(mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)


def nested_get(payload: dict[str, Any], dotted_path: str) -> Any:
    cursor: Any = payload
    for part in dotted_path.split("."):
        if not isinstance(cursor, dict) or part not in cursor:
            raise KeyError(dotted_path)
        cursor = cursor[part]
    return cursor


def merge_dicts(*parts: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for part in parts:
        out.update(part)
    return out


def normalize_slot_metadata(
    metadata_payload: dict[str, Any],
    *,
    slot_id: str,
    role: str,
    require_predicate_audit: bool,
) -> dict[str, str]:
    metadata = {
        str(key): ("" if value is None else str(value).strip())
        for key, value in metadata_payload.items()
    }
    baseline_truth = metadata.get("baseline_truth", "")
    if baseline_truth:
        baseline_truth = baseline_truth.lower()
        if baseline_truth not in PREDICATE_BASELINE_TRUTH_VALUES:
            allowed = ", ".join(sorted(PREDICATE_BASELINE_TRUTH_VALUES))
            raise HarnessError(
                f"candidate slot {slot_id} has invalid metadata.baseline_truth={baseline_truth!r}; "
                f"expected one of: {allowed}"
            )
        metadata["baseline_truth"] = baseline_truth
    if role != "candidate":
        return metadata

    missing_core = [
        field for field in ("purpose", "broken_invariant", "probe")
        if not metadata.get(field, "")
    ]
    if missing_core:
        raise HarnessError(
            f"candidate slot {slot_id} is missing required metadata fields: {', '.join(missing_core)}"
        )
    if not require_predicate_audit:
        return metadata

    missing_predicate_fields = [field for field in PREDICATE_AUDIT_FIELDS if not metadata.get(field, "")]
    if missing_predicate_fields:
        raise HarnessError(
            f"candidate slot {slot_id} is missing predicate-audit metadata fields: "
            f"{', '.join(missing_predicate_fields)}"
        )
    if metadata["baseline_truth"] != "false":
        raise HarnessError(
            f"candidate slot {slot_id} failed predicate audit: metadata.baseline_truth must be "
            f"'false' before admission, got {metadata['baseline_truth']!r}"
        )
    return metadata


def substitute(template: str, values: dict[str, str]) -> str:
    env_placeholders: dict[str, str] = {}

    def preserve_env_placeholder(match: re.Match[str]) -> str:
        token = f"__SEARCH_HARNESS_ENV_{len(env_placeholders)}__"
        env_placeholders[token] = match.group(0)
        return token

    masked = re.sub(r"\$\{[A-Za-z_][A-Za-z0-9_]*\}", preserve_env_placeholder, template)
    try:
        rendered = masked.format(**values)
    except KeyError as exc:
        missing = exc.args[0]
        raise HarnessError(f"Missing template value '{missing}' in command: {template}") from exc
    for token, placeholder in env_placeholders.items():
        rendered = rendered.replace(token, placeholder)
    return rendered


def run_command(command: str, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    completed = subprocess.run(command, shell=True, cwd=cwd, env=env, check=False)
    if completed.returncode != 0:
        raise HarnessError(f"Command failed ({completed.returncode}): {command}")


def run_process(
    args: list[str],
    cwd: Path | None = None,
    env: dict[str, str] | None = None,
    input_text: str | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        args,
        cwd=cwd,
        env=env,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def parse_rsync_target(raw: str) -> tuple[str | None, str]:
    if ":" in raw and not raw.startswith("/"):
        host, remote_path = raw.split(":", 1)
        return host, remote_path
    return None, raw


def relative_or_absolute(path: Path, root: Path) -> str:
    resolved = path.resolve()
    try:
        return str(resolved.relative_to(root.resolve()))
    except ValueError:
        return str(resolved)


def truncate_text(text: str, limit: int = 4000) -> str:
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def prompt_compact_text(text: str, limit: int = 280) -> str:
    normalized = re.sub(r"\s+", " ", str(text).strip())
    return truncate_text(normalized, limit)


def normalize_json_text(raw: str) -> str:
    text = raw.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].startswith("```"):
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    return text


def load_patch_module(module_path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(module_path.stem, module_path)
    if spec is None or spec.loader is None:
        raise HarnessError(f"Cannot import patch module: {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def apply_patches(source: str, module: Any, patch_names: list[str]) -> str:
    patched = source
    for patch_name in patch_names:
        fn_name = patch_name if patch_name.startswith("patch_") else f"patch_{patch_name}"
        fn = getattr(module, fn_name, None)
        if fn is None:
            raise HarnessError(f"Patch function not found: {fn_name}")
        patched = fn(patched)
    return patched


def default_run_script(env: dict[str, str], command: str) -> str:
    exports = "\n".join(
        f"export {key}={shlex.quote(str(value))}" for key, value in sorted(env.items())
    )
    return "\n".join(
        [
            "#!/usr/bin/env bash",
            "set -euo pipefail",
            'cd "$(dirname "$0")"',
            "mkdir -p results",
            exports,
            "set +e",
            f"{command} > results/train.log 2>&1",
            "rc=$?",
            "set -e",
            'printf \'{"returncode": %s, "finished_at": "%s"}\\n\' "$rc" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" > results/runner_status.json',
            "exit $rc",
        ]
    ) + "\n"


@dataclass(frozen=True)
class SlotBundle:
    slot: str
    role: str
    family: str
    slot_dir: Path
    results_dir: Path
    nproc_per_slot: int
    gpu_spec: list[str] | None
    command: str
    metric_path: str | None


def build_waves(manifest: dict[str, Any]) -> list[list[dict[str, Any]]]:
    gpu_pool = [str(item) for item in manifest.get("gpu_pool", [])]
    pending = list(manifest["slots"])
    waves: list[list[dict[str, Any]]] = []
    if not gpu_pool:
        return [[{**slot, "assigned_gpus": []} for slot in pending]]

    while pending:
        free = set(gpu_pool)
        wave: list[dict[str, Any]] = []
        deferred: list[dict[str, Any]] = []
        for slot in pending:
            explicit = slot.get("gpu_spec")
            if explicit:
                requested = [piece.strip() for piece in str(explicit).split(",") if piece.strip()]
                if all(gpu in free for gpu in requested):
                    free.difference_update(requested)
                    wave.append({**slot, "assigned_gpus": requested})
                else:
                    deferred.append(slot)
                continue
            needed = int(slot.get("nproc_per_slot", 1))
            if len(free) >= needed:
                assigned = sorted(list(free))[:needed]
                for gpu in assigned:
                    free.remove(gpu)
                wave.append({**slot, "assigned_gpus": assigned})
            else:
                deferred.append(slot)
        if not wave:
            forced = deferred.pop(0)
            explicit = forced.get("gpu_spec")
            if explicit:
                assigned = [piece.strip() for piece in str(explicit).split(",") if piece.strip()]
            else:
                needed = int(forced.get("nproc_per_slot", 1))
                assigned = gpu_pool[:needed]
            wave = [{**forced, "assigned_gpus": assigned}]
        waves.append(wave)
        pending = deferred
    return waves


def execute_bundle_manifest(bundle_manifest_path: Path) -> dict[str, Any]:
    manifest = load_json(bundle_manifest_path)
    bundle_dir = bundle_manifest_path.parent
    report: dict[str, Any] = {
        "cycle_id": manifest.get("cycle_id", "unknown"),
        "generation": manifest.get("generation", -1),
        "waves": [],
    }
    waves = build_waves(manifest)
    for wave_index, wave in enumerate(waves):
        wave_payload: dict[str, Any] = {"index": wave_index, "slots": []}
        procs: list[tuple[subprocess.Popen[str], dict[str, Any], Path]] = []
        for assigned in wave:
            slot_dir = bundle_dir / assigned["path"]
            env = os.environ.copy()
            env["CUDA_VISIBLE_DEVICES"] = ",".join(assigned["assigned_gpus"])
            env["SEARCH_HARNESS_SLOT"] = assigned["slot"]
            started_at = time.time()
            proc = subprocess.Popen(
                ["bash", str(slot_dir / "run.sh")],
                cwd=slot_dir,
                env=env,
                text=True,
            )
            procs.append(
                (
                    proc,
                    {
                        "slot": assigned["slot"],
                        "assigned_gpus": assigned["assigned_gpus"],
                        "started_at": started_at,
                    },
                    slot_dir,
                )
            )
        for proc, slot_meta, slot_dir in procs:
            rc = proc.wait()
            finished_at = time.time()
            slot_report = {
                **slot_meta,
                "returncode": rc,
                "finished_at": finished_at,
                "duration_seconds": round(finished_at - float(slot_meta["started_at"]), 3),
                "results_dir": str((slot_dir / "results").relative_to(bundle_dir)),
            }
            write_json(slot_dir / "results" / "executor_status.json", slot_report)
            wave_payload["slots"].append(slot_report)
        report["waves"].append(wave_payload)
    write_json(bundle_dir / "execution_report.json", report)
    return report


class SearchHarness:
    def __init__(self, config_path: Path):
        self.config_path = config_path.resolve()
        self.config_dir = self.config_path.parent
        self.repo_root = Path(__file__).resolve().parent
        self.config = load_json(self.config_path)
        self.cycle_id = str(self.config["cycle_id"])
        workspace = self.config.get("workspace", {})
        workspace_root = resolve_path(self.config_dir, workspace.get("root", "search_cycles"))
        if workspace_root is None:
            raise HarnessError("workspace.root must resolve")
        self.workspace_root = workspace_root
        self.cycle_root = self.workspace_root / self.cycle_id
        self.prompts = self.config.get("prompts", {})
        self.hooks = self.config.get("hooks", {})
        self.remote = self.config.get("remote", {})
        self.selection_defaults = self.config.get("selection", {})
        self.gpu_pool = [str(gpu) for gpu in self.config.get("gpu_pool", [])]
        self.base = self.config.get("base", {})
        self.admission = self.config.get("admission", {})

    def generation_dir(self, generation: int) -> Path:
        return self.cycle_root / f"gen_{generation:03d}"

    def generation_paths(self, generation: int) -> dict[str, Path]:
        gen_dir = self.generation_dir(generation)
        return {
            "generation_dir": gen_dir,
            "context": gen_dir / "context.json",
            "track_spec": gen_dir / "track_spec.json",
            "verifier_report": gen_dir / "verifier_report.json",
            "compiled_spec": gen_dir / "compiled_generation.json",
            "results_index": gen_dir / "results_index.json",
            "generation_summary": gen_dir / "generation_summary.json",
            "bundle_dir": gen_dir / "bundle",
            "bundle_manifest": gen_dir / "bundle" / "bundle_manifest.json",
            "codex_dir": gen_dir / "codex",
            "codex_catalog": gen_dir / "codex" / "catalog.json",
            "codex_request": gen_dir / "codex" / "request.json",
            "codex_summary": gen_dir / "codex" / "summary.json",
            "agent_dir": gen_dir / "agent",
            "agent_catalog": gen_dir / "agent" / "catalog.json",
            "agent_request": gen_dir / "agent" / "request.json",
            "agent_memory": gen_dir / "agent" / "postmortem_memory.json",
            "agent_instructions": gen_dir / "agent" / "AGENT_INSTRUCTIONS.md",
            "agent_prompt": gen_dir / "agent" / "AGENT_PROMPT.txt",
            "agent_focus_dir": gen_dir / "agent" / "focus_files",
        }

    def prompt_path(self, key: str) -> Path | None:
        raw = self.prompts.get(key)
        return self.resolve_runtime_path(str(raw), default_to_repo_root=False) if raw else None

    def prompt_text(self, key: str) -> str:
        path = self.prompt_path(key)
        if path is None or not path.exists():
            return ""
        return load_text(path)

    def build_context(self, generation: int) -> dict[str, Any]:
        history_window = int(self.config.get("history_window", 3))
        history: list[dict[str, Any]] = []
        start = max(0, generation - history_window)
        for idx in range(start, generation):
            paths = self.generation_paths(idx)
            item: dict[str, Any] = {"generation": idx}
            for key in ("track_spec", "verifier_report", "results_index", "generation_summary"):
                path = paths[key]
                if path.exists():
                    item[key] = load_json(path)
            if len(item) > 1:
                history.append(item)
        return {
            "cycle_id": self.cycle_id,
            "generation": generation,
            "history": history,
        }

    def codex_settings(self) -> dict[str, Any]:
        return self.config.get("codex", {})

    def codex_reasoning_effort(self) -> str | None:
        value = self.codex_settings().get("reasoning_effort")
        return str(value) if value else None

    def default_base_path(self) -> Path | None:
        default_base_raw = self.base.get("default_base_script")
        if default_base_raw is None:
            return None
        path = self.resolve_repo_or_config_path(str(default_base_raw))
        if path is None or not path.exists():
            raise HarnessError(f"base.default_base_script could not be resolved: {default_base_raw}")
        return path

    def configured_frozen_base_path(self) -> Path | None:
        fixed_base_raw = self.admission.get("fixed_base_script")
        if fixed_base_raw is None:
            return None
        path = self.resolve_repo_or_config_path(str(fixed_base_raw))
        if path is None or not path.exists():
            raise HarnessError(f"admission.fixed_base_script could not be resolved: {fixed_base_raw}")
        return path

    def base_default_env(self) -> dict[str, str]:
        return {str(key): str(value) for key, value in self.base.get("defaults", {}).items()}

    def admission_settings(self) -> dict[str, Any]:
        control_slots = [str(slot) for slot in self.selection_defaults.get("control_slots", ["C0", "C1"])]
        default_max_candidates = max(1, len(self.gpu_pool) - len(control_slots)) if self.gpu_pool else 6
        freeze_env_keys = [str(key) for key in self.admission.get("freeze_env_keys", DEFAULT_FREEZE_ENV_KEYS)]
        min_candidate_slots = int(self.admission.get("min_candidate_slots", 1))
        max_candidate_slots = int(self.admission.get("max_candidate_slots", default_max_candidates))
        fixed_base_path = self.configured_frozen_base_path()
        if max_candidate_slots < 1:
            raise HarnessError("admission.max_candidate_slots must be >= 1")
        if min_candidate_slots < 1:
            raise HarnessError("admission.min_candidate_slots must be >= 1")
        if min_candidate_slots > max_candidate_slots:
            raise HarnessError("admission.min_candidate_slots cannot exceed admission.max_candidate_slots")
        return {
            "mode": str(self.admission.get("mode", "survival")),
            "require_single_base": bool(self.admission.get("require_single_base", True)),
            "require_unique_family_groups": bool(self.admission.get("require_unique_family_groups", True)),
            "require_predicate_audit": bool(self.admission.get("require_predicate_audit", True)),
            "freeze_env_keys": freeze_env_keys,
            "min_candidate_slots": min_candidate_slots,
            "max_candidate_slots": max_candidate_slots,
            "fixed_base_script": (
                self.project_relative(fixed_base_path)
                if fixed_base_path is not None
                else None
            ),
        }

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

    def agent_root(self) -> Path:
        codex = self.codex_settings()
        return (self.resolve_runtime_path(codex.get("cd", ".")) or self.repo_root).resolve()

    def project_relative(self, path: Path) -> str:
        return relative_or_absolute(path, self.agent_root())

    def resolve_repo_or_config_path(self, raw: str | None) -> Path | None:
        if raw is None:
            return None
        candidates: list[Path] = []
        first = resolve_path(self.config_dir, raw)
        if first is not None:
            candidates.append(first)
        second = resolve_path(self.agent_root(), raw)
        if second is not None and second not in candidates:
            candidates.append(second)
        for candidate in candidates:
            if candidate.exists():
                return candidate
        return candidates[0] if candidates else None

    def schema_path(self, key: str) -> Path:
        codex = self.codex_settings()
        override = codex.get(f"{key}_schema")
        if override:
            path = self.resolve_runtime_path(str(override), default_to_repo_root=False)
            if path is None:
                raise HarnessError(f"codex.{key}_schema could not be resolved")
            return path
        default_name = {
            "compiled": "search_harness_compiled_generation_schema.json",
            "verifier": "search_harness_verifier_report_schema.json",
        }[key]
        return Path(__file__).resolve().with_name(default_name)

    def discover_executable_families(self) -> list[dict[str, Any]]:
        codex = self.codex_settings()
        search_root = self.resolve_runtime_path(codex.get("catalog_root", "."))
        if search_root is None:
            raise HarnessError("codex.catalog_root could not be resolved")
        default_base = self.default_base_path()
        fixed_base = self.configured_frozen_base_path()
        entries: list[dict[str, Any]] = []
        seen: set[tuple[str, str, str]] = set()
        for patch_path in sorted(search_root.rglob("patches.py")):
            sibling_base = patch_path.with_name("base_train_gpt.py")
            base_script = sibling_base if sibling_base.exists() else default_base
            if base_script is None or not base_script.exists():
                continue
            if fixed_base is not None and base_script.resolve() != fixed_base.resolve():
                continue
            module = load_patch_module(patch_path)
            for attr in sorted(dir(module)):
                if not attr.startswith("patch_"):
                    continue
                family = attr[len("patch_") :]
                base_rel = self.project_relative(base_script)
                patch_rel = self.project_relative(patch_path)
                key = (family, base_rel, patch_rel)
                if key in seen:
                    continue
                seen.add(key)
                entries.append(
                    {
                        "family": family,
                        "family_group": infer_family_group(family),
                        "base_script": base_rel,
                        "patch_module": patch_rel,
                        "patches": [family],
                        "stage": patch_path.parent.name,
                        "patch_function": attr,
                    }
                )
        return entries

    def build_catalog_summary(self, catalog: list[dict[str, Any]]) -> list[dict[str, Any]]:
        summary_by_base: dict[str, dict[str, Any]] = {}
        for entry in catalog:
            base_script = str(entry["base_script"])
            bucket = summary_by_base.setdefault(
                base_script,
                {
                    "base_script": base_script,
                    "family_count": 0,
                    "family_groups": set(),
                    "families": [],
                },
            )
            bucket["family_count"] += 1
            bucket["family_groups"].add(str(entry.get("family_group") or entry["family"]))
            bucket["families"].append(
                {
                    "family": str(entry["family"]),
                    "family_group": str(entry.get("family_group") or entry["family"]),
                    "patch_module": str(entry["patch_module"]),
                    "stage": str(entry.get("stage", "")),
                }
            )

        summary: list[dict[str, Any]] = []
        for item in summary_by_base.values():
            families = sorted(
                item["families"],
                key=lambda family_entry: (
                    family_entry["family_group"],
                    family_entry["family"],
                    family_entry["patch_module"],
                ),
            )
            family_groups = sorted(str(group) for group in item["family_groups"])
            summary.append(
                {
                    "base_script": item["base_script"],
                    "family_count": item["family_count"],
                    "family_group_count": len(family_groups),
                    "family_groups": family_groups,
                    "families": families,
                }
            )
        return sorted(
            summary,
            key=lambda item: (
                -int(item["family_group_count"]),
                -int(item["family_count"]),
                item["base_script"],
            ),
        )

    def resolve_focus_files(self, raw_files: list[str] | None) -> list[Path]:
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
                raise HarnessError(f"focus file does not exist: {raw}")
            actual = path.resolve()
            try:
                actual.relative_to(agent_root)
            except ValueError as exc:
                raise HarnessError(f"focus file is outside the agent root {agent_root}: {raw}") from exc
            if actual in seen:
                continue
            seen.add(actual)
            resolved.append(actual)
        return resolved

    def build_postmortem_memory(self, generation: int) -> dict[str, Any]:
        history_context = self.build_context(generation)
        prior_generations: list[dict[str, Any]] = []
        for item in history_context["history"]:
            summary = item.get("generation_summary", {}) if isinstance(item.get("generation_summary"), dict) else {}
            verifier = item.get("verifier_report", {}) if isinstance(item.get("verifier_report"), dict) else {}
            track = item.get("track_spec", {}) if isinstance(item.get("track_spec"), dict) else {}
            slots_overview: list[dict[str, Any]] = []
            slots_payload = track.get("slots")
            if isinstance(slots_payload, list):
                for slot in slots_payload[:8]:
                    if not isinstance(slot, dict):
                        continue
                    slots_overview.append(
                        {
                            "slot": slot.get("slot"),
                            "role": slot.get("role"),
                            "family": slot.get("family"),
                            "patches": slot.get("patches", []),
                        }
                    )
            prior_generations.append(
                {
                    "generation": item.get("generation"),
                    "verdict": verifier.get("verdict"),
                    "verifier_summary": verifier.get("summary") or verifier.get("notes"),
                    "survivors": summary.get("survivors", []),
                    "invalid_reasons": summary.get("invalid_reasons", []),
                    "control_spread": summary.get("control_spread"),
                    "slots": slots_overview,
                }
            )

        reference_docs: list[dict[str, Any]] = []
        for rel_path in (
            "findings.md",
            "search_reset.md",
            "search_reset_agent_prompt.md",
            "search_verifier_codex.md",
        ):
            path = self.resolve_runtime_path(rel_path)
            if path is None or not path.exists():
                continue
            reference_docs.append(
                {
                    "path": self.project_relative(path),
                    "excerpt": truncate_text(load_text(path), 2000),
                }
            )

        return {
            "history_window": history_context["history"],
            "prior_generations": prior_generations,
            "reference_docs": reference_docs,
        }

    def prompt_postmortem_memory_summary(self, memory: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(memory, dict):
            return {}
        prior_generations: list[dict[str, Any]] = []
        for item in memory.get("prior_generations", []):
            if not isinstance(item, dict):
                continue
            prior_generations.append(
                {
                    "generation": item.get("generation"),
                    "verdict": item.get("verdict"),
                    "verifier_summary": prompt_compact_text(item.get("verifier_summary", ""), 240),
                    "survivors": item.get("survivors", []),
                    "invalid_reasons": item.get("invalid_reasons", []),
                    "control_spread": item.get("control_spread"),
                    "slots": item.get("slots", []),
                }
            )
            if len(prior_generations) >= 6:
                break
        reference_doc_paths = [
            str(item.get("path", "")).strip()
            for item in memory.get("reference_docs", [])
            if isinstance(item, dict) and str(item.get("path", "")).strip()
        ]
        return {
            "prior_generations": prior_generations,
            "reference_doc_paths": reference_doc_paths,
        }

    def normalize_codex_request(
        self,
        request: dict[str, Any],
        generation: int,
        catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(request, dict):
            raise HarnessError("Codex request must be a JSON object")
        default_base = self.configured_frozen_base_path() or self.default_base_path()
        default_base_script = request.get("default_base_script")
        if not default_base_script and default_base is not None and default_base.exists():
            default_base_script = self.project_relative(default_base)
        base_defaults_raw = request.get("base_defaults")
        if isinstance(base_defaults_raw, dict):
            base_defaults = {str(key): str(value) for key, value in base_defaults_raw.items()}
        else:
            base_defaults = self.base_default_env()
        admission_raw = request.get("admission_policy")
        admission = (
            merge_dicts(self.admission_settings(), admission_raw)
            if isinstance(admission_raw, dict)
            else self.admission_settings()
        )
        frozen_env_raw = request.get("frozen_env_contract")
        if isinstance(frozen_env_raw, dict):
            frozen_env_contract = {
                str(key): (str(value) if value is not None else None)
                for key, value in frozen_env_raw.items()
            }
        else:
            frozen_env_contract = {
                key: base_defaults.get(key) for key in admission.get("freeze_env_keys", DEFAULT_FREEZE_ENV_KEYS)
            }
        catalog_raw = request.get("catalog")
        catalog_entries = [item for item in catalog_raw if isinstance(item, dict)] if isinstance(catalog_raw, list) else catalog
        catalog_summary_raw = request.get("catalog_summary")
        catalog_summary = (
            catalog_summary_raw
            if isinstance(catalog_summary_raw, list)
            else self.build_catalog_summary(catalog_entries)
        )
        normalized = {
            "cycle_id": str(request.get("cycle_id", self.cycle_id)),
            "generation": int(request.get("generation", generation)),
            "user_instructions": str(request.get("user_instructions", "")).strip(),
            "default_base_script": default_base_script,
            "default_entrypoint": str(
                request.get("default_entrypoint", self.base.get("default_entrypoint", "python3 {script}"))
            ),
            "base_defaults": base_defaults,
            "frozen_env_contract": frozen_env_contract,
            "primary_metric_path": str(
                request.get("primary_metric_path", self.selection_defaults.get("primary_metric_path", "metrics.score_bpb"))
            ),
            "control_slots": [
                str(slot) for slot in request.get("control_slots", self.selection_defaults.get("control_slots", ["C0", "C1"]))
            ],
            "focus_files": [str(path) for path in request.get("focus_files", [])],
            "history_context": request.get("history_context", self.build_context(generation)),
            "postmortem_memory": request.get("postmortem_memory", self.build_postmortem_memory(generation)),
            "admission_policy": admission,
            "catalog_summary": catalog_summary,
            "catalog": catalog_entries,
        }
        if not normalized["user_instructions"]:
            raise HarnessError("Codex request must include non-empty user_instructions")
        return normalized

    def build_codex_request(
        self,
        generation: int,
        instructions: str,
        catalog: list[dict[str, Any]],
        focus_files: list[Path] | None = None,
    ) -> dict[str, Any]:
        return self.normalize_codex_request(
            {
                "cycle_id": self.cycle_id,
                "generation": generation,
                "user_instructions": instructions.strip(),
                "default_base_script": (
                    self.project_relative(self.configured_frozen_base_path() or self.default_base_path())
                    if (self.configured_frozen_base_path() or self.default_base_path()) is not None
                    else None
                ),
                "default_entrypoint": str(self.base.get("default_entrypoint", "python3 {script}")),
                "base_defaults": self.base_default_env(),
                "frozen_env_contract": {
                    key: self.base_default_env().get(key) for key in self.admission_settings()["freeze_env_keys"]
                },
                "primary_metric_path": str(
                    self.selection_defaults.get("primary_metric_path", "metrics.score_bpb")
                ),
                "control_slots": [str(slot) for slot in self.selection_defaults.get("control_slots", ["C0", "C1"])],
                "focus_files": [self.project_relative(path) for path in (focus_files or [])],
                "history_context": self.build_context(generation),
                "postmortem_memory": self.build_postmortem_memory(generation),
                "admission_policy": self.admission_settings(),
                "catalog_summary": self.build_catalog_summary(catalog),
                "catalog": catalog,
            },
            generation=generation,
            catalog=catalog,
        )

    def build_codex_generator_prompt(
        self,
        request: dict[str, Any],
        previous_error: str | None,
    ) -> str:
        doctrine = self.prompt_text("generator").strip()
        admission = request["admission_policy"]
        min_candidates = int(admission["min_candidate_slots"])
        max_candidates = int(admission["max_candidate_slots"])
        freeze_env_keys = [str(key) for key in admission.get("freeze_env_keys", [])]
        fixed_base = admission.get("fixed_base_script")
        require_predicate_audit = bool(admission.get("require_predicate_audit", True))
        candidate_rule = (
            f"{min_candidates} to {max_candidates}"
            if min_candidates != max_candidates
            else str(min_candidates)
        )
        base_rule = (
            f"5. Use the fixed frozen base `{fixed_base}` for every control and candidate. Do not choose a different base."
            if fixed_base
            else "5. Choose exactly one frozen base from BASE_CATALOG_SUMMARY and keep every control and candidate on that same base_script."
        )
        sections = [
            "You are preparing one executable search generation for this repository.",
            "Return JSON only. Do not return prose outside the JSON object.",
            "The JSON must be directly usable as compiled_generation.json by search_harness.py.",
            "",
            "USER INSTRUCTIONS",
            request["user_instructions"],
            "",
            "HARD RULES",
            "1. Use only entries from EXECUTABLE_FAMILY_CATALOG.",
            "2. Do not invent new patch modules, patch names, or base scripts.",
            "3. Produce one family-admission pack that is runnable now.",
            "4. Include two control slots named C0 and C1 with no patches. Control family labels like base_root and base_root_repeat are valid anchor labels and do not need catalog entries.",
            base_rule,
            f"6. Include {candidate_rule} candidate slots named H0, H1, H2, ... as needed. Fewer candidates are better than mixed-base filler.",
            "7. Prefer first-order families over compounds.",
            "8. Every candidate must use the exact base_script, patch_module, and patches values from the catalog.",
            "9. Use at most one candidate per family_group.",
            f"10. BASE_DEFAULT_ENV is the shared frozen env contract. Keep slot env empty unless a non-frozen per-slot override is truly necessary, and never change these frozen keys: {freeze_env_keys}.",
            "11. For each candidate, fill metadata.purpose, metadata.broken_invariant, metadata.target_predicate, metadata.baseline_truth, metadata.score_path_trace, metadata.operator_claim, metadata.probe, and metadata.family_group.",
            (
                "12. Only admit candidates whose metadata.baseline_truth is exactly \"false\" on the baseline score path. "
                "Kill already-satisfied or unknown predicates instead of spending slots on them."
                if require_predicate_audit
                else "12. If you provide predicate-audit fields, keep metadata.baseline_truth consistent with the current baseline score path."
            ),
            "13. metadata.operator_claim must say how the chosen family flips metadata.target_predicate on metadata.score_path_trace, not just nearby behavior.",
            "14. Keep selection.control_slots aligned with the actual controls.",
            f"15. Use {request['primary_metric_path']} as the primary metric unless the instructions force a different executable metric.",
            "16. If no single base can support the full candidate budget, return fewer candidates instead of mixing bases or near-duplicate families.",
            "17. If the instructions ask for something impossible with the current catalog, choose the closest executable pack that still obeys the frozen-base rules and explain the gap in metadata.",
            "",
            "FOCUS FILES",
            json.dumps(request.get("focus_files", []), indent=2),
            "",
            "DEFAULTS",
            json.dumps(
                {
                    "cycle_id": request["cycle_id"],
                    "generation": request["generation"],
                    "default_base_script": request["default_base_script"],
                    "default_entrypoint": request["default_entrypoint"],
                    "primary_metric_path": request["primary_metric_path"],
                    "control_slots": request["control_slots"],
                },
                indent=2,
            ),
            "",
            "BASE DEFAULT ENV",
            json.dumps(request["base_defaults"], indent=2),
            "",
            "FROZEN ENV CONTRACT",
            json.dumps(request["frozen_env_contract"], indent=2),
            "",
            "ADMISSION POLICY",
            json.dumps(request["admission_policy"], indent=2),
            "",
            "BASE_CATALOG_SUMMARY",
            json.dumps(request["catalog_summary"], indent=2),
            "",
            "POSTMORTEM MEMORY JSON",
            json.dumps(request.get("postmortem_memory", {}), indent=2),
            "",
            "HISTORY CONTEXT JSON",
            json.dumps(request["history_context"], indent=2),
            "",
            "EXECUTABLE_FAMILY_CATALOG",
            json.dumps(request["catalog"], indent=2),
        ]
        if doctrine:
            sections.extend(["", "REFERENCE DOCTRINE", doctrine])
        if previous_error:
            sections.extend(["", "PREVIOUS ATTEMPT FAILED", previous_error])
        return "\n".join(sections) + "\n"

    def build_codex_verifier_prompt(
        self,
        request: dict[str, Any],
        compiled: dict[str, Any],
        previous_error: str | None,
    ) -> str:
        doctrine = self.prompt_text("verifier").strip()
        admission = request["admission_policy"]
        slot_cards = self.verifier_slot_cards(compiled)
        static_audit = self.verifier_pack_static_audit(request, compiled)
        require_predicate_audit = bool(admission.get("require_predicate_audit", True))
        sections = [
            "You are verifying one executable family-admission pack before GPU execution.",
            "Return JSON only.",
            "",
            "AUDIT MODE",
            "- normalize_compiled_generation already enforced schema validity, catalog membership, first-order candidate slots, frozen-base rules, unique family_group rules, frozen-env alignment, and candidate predicate-audit field presence.",
            "- Do not spend time rebuilding those checks from scratch.",
            "- Use the slot cards and static audit below to decide whether the pack is distinct, useful, instruction-aligned, and worth GPU budget.",
            "",
            "VERDICT RULES",
            "- PASS: ready to materialize and run now",
            "- PASS_WITH_WARNINGS: runnable now, but there are non-blocking concerns",
            "- RETRY: structurally repairable; the generator should try again",
            "- FAIL: cannot satisfy the request with the current executable family catalog",
            "",
            "CHECKLIST",
            "- controls are still stable anchors for this question",
            "- candidate families are distinct enough to teach us something",
            "- candidates are not hidden compounds in spirit",
            (
                "- each candidate's target_predicate is still false on the actual baseline score path"
                if require_predicate_audit
                else "- if predicate-audit fields are present, they are internally coherent with the stated score path"
            ),
            "- each metadata.operator_claim would flip its metadata.target_predicate rather than merely touch adjacent behavior",
            "- the pack matches the user instructions closely enough to be useful",
            "- the metric/selection plan is coherent",
            "",
            "VERIFICATION REQUEST SUMMARY JSON",
            json.dumps(
                {
                    "cycle_id": request["cycle_id"],
                    "generation": request["generation"],
                    "user_instructions": prompt_compact_text(request["user_instructions"], 500),
                    "default_base_script": request["default_base_script"],
                    "primary_metric_path": request["primary_metric_path"],
                    "control_slots": request["control_slots"],
                    "focus_files": request.get("focus_files", []),
                    "admission_policy": {
                        "min_candidate_slots": admission.get("min_candidate_slots"),
                        "max_candidate_slots": admission.get("max_candidate_slots"),
                        "freeze_env_keys": admission.get("freeze_env_keys", []),
                        "fixed_base_script": admission.get("fixed_base_script"),
                        "require_single_base": admission.get("require_single_base"),
                        "require_unique_family_groups": admission.get("require_unique_family_groups"),
                        "require_predicate_audit": admission.get("require_predicate_audit"),
                    },
                },
                indent=2,
            ),
            "",
            "POSTMORTEM MEMORY SUMMARY JSON",
            json.dumps(self.prompt_postmortem_memory_summary(request.get("postmortem_memory", {})), indent=2),
            "",
            "PACK SLOT CARDS JSON",
            json.dumps(slot_cards, indent=2),
            "",
            "STATIC PACK AUDIT JSON",
            json.dumps(static_audit, indent=2),
        ]
        if doctrine:
            sections.extend(["", "REFERENCE VERIFIER DOCTRINE", doctrine])
        if previous_error:
            sections.extend(["", "PREVIOUS ATTEMPT FAILED", previous_error])
        return "\n".join(sections) + "\n"

    def verifier_slot_cards(self, compiled: dict[str, Any]) -> list[dict[str, Any]]:
        cards: list[dict[str, Any]] = []
        for slot in compiled["slots"]:
            metadata = slot.get("metadata", {}) if isinstance(slot.get("metadata"), dict) else {}
            cards.append(
                {
                    "slot": slot["slot"],
                    "role": slot["role"],
                    "family": slot["family"],
                    "base_script": slot["base_script"],
                    "patches": slot.get("patches", []),
                    "family_group": metadata.get("family_group"),
                    "purpose": prompt_compact_text(metadata.get("purpose", ""), 220),
                    "broken_invariant": prompt_compact_text(metadata.get("broken_invariant", ""), 180),
                    "target_predicate": prompt_compact_text(metadata.get("target_predicate", ""), 180),
                    "baseline_truth": metadata.get("baseline_truth", ""),
                    "score_path_trace": prompt_compact_text(metadata.get("score_path_trace", ""), 220),
                    "operator_claim": prompt_compact_text(metadata.get("operator_claim", ""), 220),
                    "probe": prompt_compact_text(metadata.get("probe", ""), 220),
                    "metric_path": slot["metric_path"],
                    "selector_metric_path": slot["selector_metric_path"],
                    "env_override_keys": sorted(str(key) for key in slot.get("env", {}).keys()),
                }
            )
        return cards

    def verifier_pack_static_audit(self, request: dict[str, Any], compiled: dict[str, Any]) -> dict[str, Any]:
        admission = request["admission_policy"]
        slots = compiled["slots"]
        control_slots = [str(slot["slot"]) for slot in slots if str(slot["role"]) == "control"]
        candidate_slots = [str(slot["slot"]) for slot in slots if str(slot["role"]) == "candidate"]
        family_groups = sorted(
            {
                str((slot.get("metadata", {}) or {}).get("family_group", ""))
                for slot in slots
                if str(slot["role"]) == "candidate"
            }
        )
        return {
            "hard_checks_already_enforced": [
                "schema_valid",
                "catalog_membership_validated",
                "single_patch_per_candidate",
                "single_frozen_base" if admission.get("require_single_base") else "mixed_base_allowed",
                "unique_family_group" if admission.get("require_unique_family_groups") else "repeated_family_group_allowed",
                "frozen_env_alignment",
                "broken_predicate_gate" if admission.get("require_predicate_audit") else "predicate_audit_optional",
            ],
            "control_slots": control_slots,
            "candidate_slots": candidate_slots,
            "candidate_family_groups": family_groups,
            "candidate_predicate_audit": [
                {
                    "slot": str(slot["slot"]),
                    "target_predicate": prompt_compact_text(
                        str((slot.get("metadata", {}) or {}).get("target_predicate", "")),
                        180,
                    ),
                    "baseline_truth": str((slot.get("metadata", {}) or {}).get("baseline_truth", "")),
                    "score_path_trace": prompt_compact_text(
                        str((slot.get("metadata", {}) or {}).get("score_path_trace", "")),
                        220,
                    ),
                    "operator_claim": prompt_compact_text(
                        str((slot.get("metadata", {}) or {}).get("operator_claim", "")),
                        220,
                    ),
                }
                for slot in slots
                if str(slot["role"]) == "candidate"
            ],
            "base_scripts": sorted({str(slot["base_script"]) for slot in slots}),
            "freeze_env_keys": [str(key) for key in admission.get("freeze_env_keys", [])],
            "selection": compiled["selection"],
        }

    def invoke_codex(
        self,
        prompt: str,
        schema_path: Path,
        output_path: Path,
        attempt_dir: Path,
        label: str,
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
        model = codex.get("model")
        reasoning_effort = self.codex_reasoning_effort()
        profile = codex.get("profile")
        if model:
            args.extend(["-m", str(model)])
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

    def normalize_compiled_generation(
        self,
        payload: dict[str, Any],
        catalog: list[dict[str, Any]],
    ) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HarnessError("compiled_generation payload must be a JSON object")
        slots_payload = payload.get("slots")
        if not isinstance(slots_payload, list) or not slots_payload:
            raise HarnessError("compiled_generation must contain a non-empty slots list")

        catalog_lookup = {
            (entry["family"], entry["base_script"], entry["patch_module"]): entry for entry in catalog
        }
        admission = self.admission_settings()
        selection = merge_dicts(self.selection_defaults, payload.get("selection", {}))
        normalized_slots: list[dict[str, Any]] = []
        slot_ids: set[str] = set()
        control_slots: list[str] = []
        candidate_count = 0
        candidate_groups: dict[str, str] = {}
        all_base_scripts: set[str] = set()
        effective_envs: list[tuple[str, dict[str, str]]] = []
        default_env = self.base_default_env()
        default_primary_metric = str(
            selection.get("primary_metric_path", self.selection_defaults.get("primary_metric_path", "metrics.score_bpb"))
        )

        for raw_slot in slots_payload:
            if not isinstance(raw_slot, dict):
                raise HarnessError("each slot entry must be a JSON object")
            slot_id = str(raw_slot.get("slot", "")).strip()
            if not slot_id:
                raise HarnessError("each slot must define a non-empty slot id")
            if slot_id in slot_ids:
                raise HarnessError(f"duplicate slot id: {slot_id}")
            slot_ids.add(slot_id)

            role = str(raw_slot.get("role", "candidate")).strip()
            if role not in {"control", "candidate"}:
                raise HarnessError(f"slot {slot_id} has invalid role: {role}")

            base_raw = raw_slot.get("base_script") or self.base.get("default_base_script")
            base_path = self.resolve_repo_or_config_path(str(base_raw)) if base_raw is not None else None
            if base_path is None or not base_path.exists():
                raise HarnessError(f"slot {slot_id} references missing base_script: {base_raw}")
            base_script = self.project_relative(base_path)

            metadata_payload = raw_slot.get("metadata", {})
            if metadata_payload is None:
                metadata_payload = {}
            if not isinstance(metadata_payload, dict):
                raise HarnessError(f"slot {slot_id} metadata must be an object")
            metadata = normalize_slot_metadata(
                dict(metadata_payload),
                slot_id=slot_id,
                role=role,
                require_predicate_audit=bool(admission.get("require_predicate_audit", True)),
            )

            slot_env = raw_slot.get("env", {})
            if slot_env is None:
                slot_env = {}
            if not isinstance(slot_env, dict):
                raise HarnessError(f"slot {slot_id} env must be an object")

            normalized = dict(raw_slot)
            normalized["slot"] = slot_id
            normalized["role"] = role
            normalized["base_script"] = base_script
            normalized["metric_path"] = str(raw_slot.get("metric_path") or default_primary_metric)
            normalized["selector_metric_path"] = str(
                raw_slot.get("selector_metric_path") or normalized["metric_path"]
            )
            normalized["metadata"] = metadata
            normalized["env"] = {str(key): str(value) for key, value in slot_env.items()}
            all_base_scripts.add(base_script)
            effective_envs.append((slot_id, merge_dicts(default_env, normalized["env"])))

            if role == "control":
                if raw_slot.get("patch_module") or raw_slot.get("patches"):
                    raise HarnessError(f"control slot {slot_id} must not include patches")
                normalized.pop("patch_module", None)
                normalized.pop("patches", None)
                normalized["family"] = str(raw_slot.get("family") or f"control_{slot_id.lower()}")
                normalized["metadata"].setdefault("family_group", "control")
                control_slots.append(slot_id)
            else:
                patch_names = [str(name) for name in raw_slot.get("patches", [])]
                if len(patch_names) != 1:
                    raise HarnessError(
                        f"candidate slot {slot_id} must contain exactly one patch name for a first-order family"
                    )
                patch_raw = raw_slot.get("patch_module")
                patch_path = self.resolve_repo_or_config_path(str(patch_raw)) if patch_raw is not None else None
                if patch_path is None or not patch_path.exists():
                    raise HarnessError(f"candidate slot {slot_id} references missing patch_module: {patch_raw}")
                patch_module = self.project_relative(patch_path)
                family = patch_names[0]
                if (family, base_script, patch_module) not in catalog_lookup:
                    raise HarnessError(
                        f"candidate slot {slot_id} is not in the executable family catalog: "
                        f"{family} @ {patch_module} on {base_script}"
                    )
                module = load_patch_module(patch_path)
                fn_name = family if family.startswith("patch_") else f"patch_{family}"
                if getattr(module, fn_name, None) is None:
                    raise HarnessError(f"candidate slot {slot_id} references unknown patch function: {fn_name}")
                catalog_entry = catalog_lookup[(family, base_script, patch_module)]
                normalized["family"] = family
                normalized["metadata"].setdefault(
                    "family_group",
                    str(catalog_entry.get("family_group") or family),
                )
                normalized["patch_module"] = patch_module
                normalized["patches"] = patch_names
                family_group = str(normalized["metadata"]["family_group"])
                if admission["require_unique_family_groups"]:
                    prior_slot = candidate_groups.get(family_group)
                    if prior_slot is not None:
                        raise HarnessError(
                            f"candidate slot {slot_id} duplicates family_group '{family_group}' already used by {prior_slot}"
                        )
                    candidate_groups[family_group] = slot_id
                candidate_count += 1

            normalized_slots.append(normalized)

        if len(control_slots) < 2:
            raise HarnessError("compiled_generation must contain at least two control slots")
        if candidate_count < int(admission["min_candidate_slots"]):
            raise HarnessError(
                f"compiled_generation must contain at least {admission['min_candidate_slots']} candidate slots"
            )
        if candidate_count > int(admission["max_candidate_slots"]):
            raise HarnessError(
                f"compiled_generation must contain no more than {admission['max_candidate_slots']} candidate slots"
            )
        if admission["require_single_base"] and len(all_base_scripts) != 1:
            raise HarnessError(
                f"compiled_generation must keep all slots on one frozen base, got: {sorted(all_base_scripts)}"
            )
        for env_key in admission["freeze_env_keys"]:
            values = {env.get(env_key) for _, env in effective_envs}
            if len(values) > 1:
                details = ", ".join(
                    f"{slot_id}={env.get(env_key)!r}"
                    for slot_id, env in effective_envs
                )
                raise HarnessError(
                    f"compiled_generation must keep {env_key} aligned across all slots, got: {details}"
                )

        primary_metric = str(selection.get("primary_metric_path") or default_primary_metric)
        goal = str(selection.get("goal", self.selection_defaults.get("goal", "minimize")))
        if goal not in {"minimize", "maximize"}:
            raise HarnessError(f"selection.goal must be 'minimize' or 'maximize', got: {goal}")
        control_refs = [str(slot_id) for slot_id in selection.get("control_slots", control_slots)]
        unknown_controls = [slot_id for slot_id in control_refs if slot_id not in slot_ids]
        if unknown_controls:
            raise HarnessError(f"selection.control_slots reference unknown slots: {unknown_controls}")
        non_control_refs = [slot_id for slot_id in control_refs if slot_id not in control_slots]
        if non_control_refs:
            raise HarnessError(f"selection.control_slots must only reference control slots: {non_control_refs}")

        top_k = int(selection.get("top_k", self.selection_defaults.get("top_k", 1)))
        if top_k < 1:
            raise HarnessError("selection.top_k must be >= 1")
        max_control_spread = selection.get("max_control_spread")
        if max_control_spread is not None:
            max_control_spread = float(max_control_spread)

        return {
            "slots": normalized_slots,
            "selection": {
                "primary_metric_path": primary_metric,
                "goal": goal,
                "control_slots": control_refs,
                "max_control_spread": max_control_spread,
                "top_k": min(top_k, candidate_count),
            },
        }

    def normalize_verifier_report(self, payload: dict[str, Any]) -> dict[str, Any]:
        if not isinstance(payload, dict):
            raise HarnessError("verifier report must be a JSON object")
        verdict = str(payload.get("verdict", "")).strip()
        allowed = {"PASS", "PASS_WITH_WARNINGS", "RETRY", "FAIL"}
        if verdict not in allowed:
            raise HarnessError(f"verifier verdict must be one of {sorted(allowed)}, got: {verdict}")
        summary = str(payload.get("summary") or payload.get("notes") or "").strip()
        return {
            "verdict": verdict,
            "summary": summary,
            "errors": [str(item) for item in payload.get("errors", [])],
            "warnings": [str(item) for item in payload.get("warnings", [])],
            "fixes": [str(item) for item in payload.get("fixes", [])],
        }

    def copy_focus_files(self, focus_files: list[Path], target_dir: Path) -> list[str]:
        copied: list[str] = []
        for path in focus_files:
            rel = self.project_relative(path)
            destination = target_dir / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied.append(rel)
        return copied

    def build_agent_instructions(
        self,
        generation: int,
        request: dict[str, Any],
        focus_files: list[str],
    ) -> str:
        paths = self.generation_paths(generation)
        relative_gen_dir = self.project_relative(paths["generation_dir"])
        relative_track = self.project_relative(paths["track_spec"])
        relative_compiled = self.project_relative(paths["compiled_spec"])
        relative_verifier = self.project_relative(paths["verifier_report"])
        relative_bundle = self.project_relative(paths["bundle_dir"])
        relative_config = relative_or_absolute(self.config_path, self.agent_root())
        relative_request = self.project_relative(paths["agent_request"])
        relative_catalog = self.project_relative(paths["agent_catalog"])
        relative_memory = self.project_relative(paths["agent_memory"])
        relative_focus_dir = self.project_relative(paths["agent_focus_dir"])
        compiled_schema = self.project_relative(self.schema_path("compiled"))
        verifier_schema = self.project_relative(self.schema_path("verifier"))
        admission = request["admission_policy"]
        fixed_base = admission.get("fixed_base_script")
        compiled_example = self.project_relative(
            Path(__file__).resolve().with_name("search_harness_compiled_generation_example.json"),
        )
        dry_run_command = (
            f"python3 search_harness.py run-generation --config {shlex.quote(relative_config)} "
            f"--generation {generation} --dry-run"
        )
        focus_block = "\n".join(f"- `{path}`" for path in focus_files) if focus_files else "- None provided"
        return "\n".join(
            [
                "# Agent Instructions",
                "",
                "Work in the repository root. Do not modify unrelated source files.",
                f"Your task is to prepare generation `{generation}` so `{relative_bundle}` exists and is ready to run on the GPU setup later.",
                "",
                "## Inputs to read first",
                f"- Request: `{relative_request}`",
                f"- Family catalog: `{relative_catalog}`",
                f"- Postmortem memory: `{relative_memory}`",
                f"- Compiled schema: `{compiled_schema}`",
                f"- Verifier schema: `{verifier_schema}`",
                f"- Compiled example: `{compiled_example}`",
                f"- Focus-file snapshots: `{relative_focus_dir}`",
                "",
                "## Focus repo files",
                focus_block,
                "",
                "## Required outputs to create",
                f"- `{relative_compiled}`",
                f"- `{relative_track}` (same content as compiled_generation.json is acceptable)",
                f"- `{relative_verifier}`",
                "",
                "## Hard rules",
                "1. Use only executable families from the family catalog.",
                "2. Keep controls unpatched.",
                (
                    f"3. Use the fixed frozen base `{fixed_base}` for every control and candidate."
                    if fixed_base
                    else "3. Choose one frozen base from the catalog summary in the request and keep every control and candidate on it."
                ),
                "4. Keep candidates first-order; do not create compounds.",
                "5. Use at most one candidate per family_group.",
                f"6. BASE_DEFAULT_ENV in the request is the shared frozen env contract. Leave slot env empty unless a non-frozen override is truly needed, and never change these frozen keys: {admission.get('freeze_env_keys', [])}.",
                "7. For each candidate, fill metadata.purpose, metadata.broken_invariant, metadata.target_predicate, metadata.baseline_truth, metadata.score_path_trace, metadata.operator_claim, metadata.probe, and metadata.family_group.",
                (
                    "8. Do not admit a candidate unless metadata.baseline_truth is exactly \"false\" on the actual baseline score path."
                    if admission.get("require_predicate_audit", True)
                    else "8. If predicate-audit metadata is present, keep it consistent with the actual baseline score path."
                ),
                "9. metadata.operator_claim must describe how the chosen family flips metadata.target_predicate on metadata.score_path_trace.",
                "10. If a single base cannot support the full slot budget, return fewer candidates instead of mixing bases or near-duplicate families.",
                "11. Use postmortem memory from previous runs to avoid repeating mistakes.",
                "12. If the user instructions ask for something impossible, choose the nearest executable pack that still obeys the frozen-base rules and explain the gap in metadata plus verifier warnings.",
                "",
                "## After writing the JSON files",
                f"Run: `{dry_run_command}`",
                "",
                "## Success condition",
                f"- `{relative_bundle}/bundle_manifest.json` exists",
                f"- slot folders exist under `{relative_bundle}/slots/`",
                "- the generation folder is ready for later remote GPU execution",
                "",
                "## User instructions",
                request["user_instructions"],
                "",
                "Finish by leaving the generation folder populated and dry-run materialized.",
            ]
        ) + "\n"

    def prepare_agent_folder(
        self,
        generation: int,
        instructions: str,
        repo_files: list[str] | None = None,
    ) -> Path:
        paths = self.generation_paths(generation)
        paths["generation_dir"].mkdir(parents=True, exist_ok=True)
        write_json(paths["context"], self.build_context(generation))
        if paths["agent_dir"].exists():
            shutil.rmtree(paths["agent_dir"])
        paths["agent_dir"].mkdir(parents=True, exist_ok=True)

        catalog = self.discover_executable_families()
        if not catalog:
            raise HarnessError("No executable patch families were discovered for agent handoff")
        focus_paths = self.resolve_focus_files(repo_files)
        request = self.build_codex_request(generation, instructions, catalog, focus_paths)
        write_json(paths["agent_request"], request)
        write_json(paths["agent_catalog"], {"families": catalog})
        write_json(paths["agent_memory"], request["postmortem_memory"])
        self.copy_focus_files(focus_paths, paths["agent_focus_dir"])

        compiled_schema = self.schema_path("compiled")
        verifier_schema = self.schema_path("verifier")
        shutil.copy2(compiled_schema, paths["agent_dir"] / compiled_schema.name)
        shutil.copy2(verifier_schema, paths["agent_dir"] / verifier_schema.name)
        compiled_example = Path(__file__).resolve().with_name("search_harness_compiled_generation_example.json")
        if compiled_example.exists():
            shutil.copy2(compiled_example, paths["agent_dir"] / compiled_example.name)

        focus_rel = [self.project_relative(path) for path in focus_paths]
        instructions_text = self.build_agent_instructions(generation, request, focus_rel)
        write_text(paths["agent_instructions"], instructions_text)
        write_text(
            paths["agent_prompt"],
            (
                f"Read `{self.project_relative(paths['agent_instructions'])}` and complete the task fully. "
                "Write the required JSON files, run the dry-run command, and leave the generation folder ready to execute.\n"
            ),
        )
        return paths["agent_dir"]

    def prepare_generation_with_codex(
        self,
        generation: int,
        instructions: str | None,
        focus_files: list[str] | None = None,
        max_attempts: int | None = None,
        resume: bool = False,
    ) -> Path:
        codex = self.codex_settings()
        attempts = int(max_attempts or codex.get("max_attempts", 4))
        if attempts < 1:
            raise HarnessError("max_attempts must be >= 1")

        paths = self.generation_paths(generation)
        paths["generation_dir"].mkdir(parents=True, exist_ok=True)
        write_json(paths["context"], self.build_context(generation))
        if paths["codex_dir"].exists() and not resume:
            shutil.rmtree(paths["codex_dir"])
        paths["codex_dir"].mkdir(parents=True, exist_ok=True)

        catalog_payload = load_json(paths["codex_catalog"]) if resume and paths["codex_catalog"].exists() else {}
        catalog = (
            [item for item in catalog_payload.get("families", []) if isinstance(item, dict)]
            if isinstance(catalog_payload, dict)
            else []
        )
        if not catalog:
            catalog = self.discover_executable_families()
        if not catalog:
            raise HarnessError("No executable patch families were discovered for Codex generation")
        write_json(paths["codex_catalog"], {"families": catalog})

        if resume:
            if focus_files:
                raise HarnessError("resume reuses the stored Codex request; omit --repo-file")
            if not paths["codex_request"].exists():
                raise HarnessError(f"missing Codex request for generation {generation}: {paths['codex_request']}")
            request = self.normalize_codex_request(load_json(paths["codex_request"]), generation, catalog)
            write_json(paths["codex_request"], request)
            provided = str(instructions or "").strip()
            stored = str(request.get("user_instructions", "")).strip()
            if provided and stored and provided != stored:
                raise HarnessError(
                    "resume requested but provided instructions differ from the stored generation request"
                )
        else:
            cleaned = str(instructions or "").strip()
            if not cleaned:
                raise HarnessError("instructions are required unless --resume is used")
            request = self.build_codex_request(
                generation,
                cleaned,
                catalog,
                self.resolve_focus_files(focus_files),
            )
            write_json(paths["codex_request"], request)
        compiled_schema = self.schema_path("compiled")
        verifier_schema = self.schema_path("verifier")
        if not compiled_schema.exists():
            raise HarnessError(f"Missing compiled schema file: {compiled_schema}")
        if not verifier_schema.exists():
            raise HarnessError(f"Missing verifier schema file: {verifier_schema}")

        allowed_pass = set(self.selection_defaults.get("pass_verdicts", ["PASS", "PASS_WITH_WARNINGS"]))
        if resume and paths["compiled_spec"].exists() and paths["verifier_report"].exists():
            try:
                compiled = self.normalize_compiled_generation(load_json(paths["compiled_spec"]), catalog)
                write_json(paths["compiled_spec"], compiled)
                write_json(paths["track_spec"], compiled)
                verifier = self.normalize_verifier_report(load_json(paths["verifier_report"]))
                write_json(paths["verifier_report"], verifier)
                if verifier["verdict"] in allowed_pass:
                    bundle_manifest = self.materialize_generation(generation)
                    write_json(
                        paths["codex_summary"],
                        {
                            "status": "ready",
                            "attempts_used": 0,
                            "generation": generation,
                            "compiled_spec": str(paths["compiled_spec"]),
                            "verifier_report": str(paths["verifier_report"]),
                            "bundle_manifest": str(bundle_manifest),
                            "resumed": True,
                        },
                    )
                    return bundle_manifest
            except Exception:  # noqa: BLE001
                pass
        last_error = ""

        for attempt in range(1, attempts + 1):
            attempt_dir = paths["codex_dir"] / f"attempt_{attempt:03d}"
            attempt_dir.mkdir(parents=True, exist_ok=True)
            failure_path = attempt_dir / "failure.txt"
            attempt_error = load_text(failure_path).strip() if resume and failure_path.exists() else last_error
            try:
                compiled_raw_path = attempt_dir / "compiled_generation_raw.json"
                if resume and compiled_raw_path.exists():
                    compiled_payload = load_json(compiled_raw_path)
                else:
                    generator_prompt = self.build_codex_generator_prompt(request, attempt_error or None)
                    write_text(attempt_dir / "generator_prompt.txt", generator_prompt)
                    compiled_payload = self.invoke_codex(
                        prompt=generator_prompt,
                        schema_path=compiled_schema,
                        output_path=compiled_raw_path,
                        attempt_dir=attempt_dir,
                        label="generator",
                    )
                compiled = self.normalize_compiled_generation(compiled_payload, catalog)
                write_json(paths["compiled_spec"], compiled)
                write_json(paths["track_spec"], compiled)

                verifier_raw_path = attempt_dir / "verifier_report_raw.json"
                if resume and verifier_raw_path.exists():
                    verifier_payload = load_json(verifier_raw_path)
                else:
                    verifier_prompt = self.build_codex_verifier_prompt(request, compiled, attempt_error or None)
                    write_text(attempt_dir / "verifier_prompt.txt", verifier_prompt)
                    verifier_payload = self.invoke_codex(
                        prompt=verifier_prompt,
                        schema_path=verifier_schema,
                        output_path=verifier_raw_path,
                        attempt_dir=attempt_dir,
                        label="verifier",
                    )
                verifier = self.normalize_verifier_report(verifier_payload)
                write_json(paths["verifier_report"], verifier)
                if verifier["verdict"] not in allowed_pass:
                    last_error = (
                        f"Verifier returned {verifier['verdict']}: {verifier['summary']}\n"
                        f"Errors: {verifier['errors']}\n"
                        f"Fixes: {verifier['fixes']}"
                    )
                    write_text(failure_path, last_error + "\n")
                    continue

                bundle_manifest = self.materialize_generation(generation)
                write_json(
                    paths["codex_summary"],
                    {
                        "status": "ready",
                        "attempts_used": attempt,
                        "generation": generation,
                        "compiled_spec": str(paths["compiled_spec"]),
                        "verifier_report": str(paths["verifier_report"]),
                        "bundle_manifest": str(bundle_manifest),
                        "resumed": resume,
                    },
                )
                return bundle_manifest
            except Exception as exc:  # noqa: BLE001
                last_error = str(exc)
                write_text(failure_path, last_error + "\n")

        write_json(
            paths["codex_summary"],
            {
                "status": "failed",
                "attempts_used": attempts,
                "generation": generation,
                "last_error": last_error,
                "resumed": resume,
            },
        )
        raise HarnessError(
            f"Codex could not prepare generation {generation} after {attempts} attempts: {last_error}"
        )

    def run_hook(self, hook_name: str, generation: int) -> None:
        hook = self.hooks.get(hook_name, {})
        command = hook.get("command")
        if not command:
            return
        paths = self.generation_paths(generation)
        values = {
            "config": str(self.config_path),
            "cycle_root": str(self.cycle_root),
            "generation": str(generation),
            "generation_dir": str(paths["generation_dir"]),
            "context_json": str(paths["context"]),
            "track_spec_json": str(paths["track_spec"]),
            "verifier_report_json": str(paths["verifier_report"]),
            "compiled_spec_json": str(paths["compiled_spec"]),
            "results_index_json": str(paths["results_index"]),
            "generation_summary_json": str(paths["generation_summary"]),
            "generator_prompt": str(self.prompt_path("generator") or ""),
            "verifier_prompt": str(self.prompt_path("verifier") or ""),
            "compiler_prompt": str(self.prompt_path("compiler") or ""),
        }
        rendered = substitute(str(command), values)
        cwd = resolve_path(self.config_dir, hook.get("cwd")) or self.config_dir
        env = os.environ.copy()
        env.update({str(k): str(v) for k, v in hook.get("env", {}).items()})
        run_command(rendered, cwd=cwd, env=env)

    def ensure_generation_inputs(self, generation: int) -> dict[str, Path]:
        paths = self.generation_paths(generation)
        paths["generation_dir"].mkdir(parents=True, exist_ok=True)
        if not paths["context"].exists():
            write_json(paths["context"], self.build_context(generation))
        if not paths["track_spec"].exists():
            self.run_hook("generator", generation)
        if not paths["track_spec"].exists():
            raise HarnessError(f"Missing track spec for generation {generation}: {paths['track_spec']}")
        if not paths["verifier_report"].exists():
            self.run_hook("verifier", generation)
        if not paths["verifier_report"].exists():
            raise HarnessError(
                f"Missing verifier report for generation {generation}: {paths['verifier_report']}"
            )
        verifier = load_json(paths["verifier_report"])
        allowed = set(self.selection_defaults.get("pass_verdicts", ["PASS", "PASS_WITH_WARNINGS"]))
        verdict = verifier.get("verdict")
        if verdict not in allowed:
            raise HarnessError(f"Verifier blocked generation {generation} with verdict: {verdict}")
        if not paths["compiled_spec"].exists():
            self.run_hook("compiler", generation)
        if not paths["compiled_spec"].exists():
            track = load_json(paths["track_spec"])
            if "slots" in track:
                write_json(paths["compiled_spec"], track)
            else:
                raise HarnessError(
                    f"Missing compiled generation spec for generation {generation}: {paths['compiled_spec']}"
                )
        return paths

    def materialize_generation(self, generation: int) -> Path:
        paths = self.ensure_generation_inputs(generation)
        bundle_dir = paths["bundle_dir"]
        if bundle_dir.exists():
            shutil.rmtree(bundle_dir)
        bundle_dir.mkdir(parents=True, exist_ok=True)
        compiled = load_json(paths["compiled_spec"])
        default_base_raw = self.base.get("default_base_script")
        base_script_default = (
            self.resolve_repo_or_config_path(str(default_base_raw))
            if default_base_raw is not None
            else None
        )
        default_patch_raw = self.base.get("default_patch_module")
        patch_module_default = (
            self.resolve_repo_or_config_path(str(default_patch_raw))
            if default_patch_raw is not None
            else None
        )
        default_env = self.base_default_env()
        default_entrypoint = str(self.base.get("default_entrypoint", "python train_gpt.py"))

        slots_payload = compiled.get("slots")
        if not isinstance(slots_payload, list) or not slots_payload:
            raise HarnessError("compiled_generation.json must contain a non-empty slots list")

        slot_bundles: list[dict[str, Any]] = []
        copied_tool = bundle_dir / "search_harness.py"
        shutil.copy2(Path(__file__).resolve(), copied_tool)

        for slot in slots_payload:
            slot_id = str(slot["slot"])
            slot_dir = bundle_dir / "slots" / slot_id
            results_dir = slot_dir / "results"
            results_dir.mkdir(parents=True, exist_ok=True)

            slot_base_raw = slot.get("base_script")
            base_script = (
                self.resolve_repo_or_config_path(str(slot_base_raw))
                if slot_base_raw is not None
                else None
            ) or base_script_default
            if base_script is None:
                raise HarnessError(f"No base_script configured for slot {slot_id}")
            source = base_script.read_text(encoding="utf-8")

            slot_patch_raw = slot.get("patch_module")
            patch_module = (
                self.resolve_repo_or_config_path(str(slot_patch_raw))
                if slot_patch_raw is not None
                else None
            ) or patch_module_default
            patch_names = [str(name) for name in slot.get("patches", [])]
            if patch_names:
                if patch_module is None:
                    raise HarnessError(f"Slot {slot_id} requests patches but no patch_module is configured")
                module = load_patch_module(patch_module)
                source = apply_patches(source, module, patch_names)

            output_script_name = str(slot.get("output_script_name", base_script.name))
            output_script_path = slot_dir / output_script_name
            output_script_path.write_text(source, encoding="utf-8")

            for raw_copy in slot.get("copy_files", []):
                src = self.resolve_repo_or_config_path(str(raw_copy))
                if src is None or not src.exists():
                    raise HarnessError(f"copy_files entry does not exist for slot {slot_id}: {raw_copy}")
                dst = slot_dir / src.name
                if src.is_dir():
                    shutil.copytree(src, dst)
                else:
                    shutil.copy2(src, dst)

            env = merge_dicts(default_env, {str(k): str(v) for k, v in slot.get("env", {}).items()})
            env.setdefault("PYTHONUNBUFFERED", "1")
            nproc_per_slot = int(slot.get("nproc_per_slot", 1))
            command_template = str(slot.get("entrypoint", default_entrypoint))
            command = substitute(
                command_template,
                {
                    "script": output_script_name,
                    "nproc_per_slot": str(nproc_per_slot),
                    "slot": slot_id,
                },
            )

            run_script = default_run_script(env, command)
            run_script_path = slot_dir / "run.sh"
            run_script_path.write_text(run_script, encoding="utf-8")
            ensure_executable(run_script_path)

            manifest = {
                "cycle_id": self.cycle_id,
                "generation": generation,
                "slot": slot_id,
                "role": slot.get("role", "candidate"),
                "family": slot.get("family", "unknown"),
                "patches": patch_names,
                "base_script": str(base_script),
                "base_sha256": sha256_file(base_script),
                "patched_sha256": sha256_text(source),
                "nproc_per_slot": nproc_per_slot,
                "gpu_spec": slot.get("gpu_spec"),
                "metric_path": slot.get("metric_path"),
                "selector_metric_path": slot.get("selector_metric_path"),
                "metadata": slot.get("metadata", {}),
                "env": env,
            }
            write_json(slot_dir / "manifest.json", manifest)

            slot_bundles.append(
                {
                    "slot": slot_id,
                    "role": manifest["role"],
                    "family": manifest["family"],
                    "path": str(slot_dir.relative_to(bundle_dir)),
                    "results_path": str(results_dir.relative_to(bundle_dir)),
                    "nproc_per_slot": nproc_per_slot,
                    "gpu_spec": slot.get("gpu_spec"),
                    "command": command,
                    "metric_path": slot.get("metric_path"),
                }
            )

        bundle_manifest = {
            "cycle_id": self.cycle_id,
            "generation": generation,
            "gpu_pool": self.gpu_pool,
            "selection": merge_dicts(self.selection_defaults, compiled.get("selection", {})),
            "slots": slot_bundles,
        }
        write_json(paths["bundle_manifest"], bundle_manifest)
        return paths["bundle_manifest"]

    def execute_bundle(self, bundle_manifest_path: Path) -> dict[str, Any]:
        return execute_bundle_manifest(bundle_manifest_path)

    def sync_to_remote(self, generation: int) -> tuple[str, str]:
        if not self.remote.get("enabled"):
            raise HarnessError("remote.enabled is false")
        target = self.remote.get("rsync_target")
        ssh_target = self.remote.get("ssh_target")
        if not target:
            raise HarnessError("remote.rsync_target is required")
        parsed_host, remote_root = parse_rsync_target(str(target))
        resolved_ssh_target = str(ssh_target or parsed_host or "")
        if not resolved_ssh_target:
            raise HarnessError("remote ssh_target is required when rsync_target has no host prefix")
        paths = self.generation_paths(generation)
        local_dir = paths["generation_dir"]
        remote_dir = f"{str(remote_root).rstrip('/')}/gen_{generation:03d}"
        mkdir_cmd = f"mkdir -p {shlex.quote(remote_dir)}"
        run_command(f"ssh {shlex.quote(resolved_ssh_target)} {shlex.quote(mkdir_cmd)}")
        rsync_dest = f"{resolved_ssh_target}:{remote_dir}/"
        run_command(f"rsync -az {shlex.quote(str(local_dir))}/ {shlex.quote(rsync_dest)}")
        return resolved_ssh_target, remote_dir

    def execute_remote(self, generation: int, remote_dir: str, ssh_target: str) -> None:
        bundle_manifest = f"{remote_dir}/bundle/bundle_manifest.json"
        runner = f"{remote_dir}/bundle/search_harness.py"
        remote_python = str(self.remote.get("remote_python", "python"))
        pre = str(self.remote.get("pre_command", "")).strip()
        pieces = [f"{remote_python} {shlex.quote(runner)} execute-bundle --bundle {shlex.quote(bundle_manifest)}"]
        command = pieces[0]
        if pre:
            command = f"{pre} && {command}"
        run_command(f"ssh {shlex.quote(ssh_target)} {shlex.quote(command)}")

    def collect_from_remote(self, generation: int, remote_dir: str) -> None:
        target = self.remote.get("rsync_target")
        if not target:
            raise HarnessError("remote.rsync_target is required")
        ssh_target = self.remote.get("ssh_target")
        parsed_host, remote_root = parse_rsync_target(str(target))
        resolved_ssh_target = str(ssh_target or parsed_host or "")
        if not resolved_ssh_target:
            raise HarnessError("remote ssh_target is required when rsync_target has no host prefix")
        local_dir = self.generation_paths(generation)["generation_dir"]
        remote_generation_dir = f"{str(remote_root).rstrip('/')}/gen_{generation:03d}"
        sync_mode = str(self.config.get("workspace", {}).get("sync_mode", "bundle"))
        if sync_mode == "results_only":
            cmd = (
                "rsync -az --prune-empty-dirs "
                "--include '*/' --include '*.json' --include '*.log' "
                "--exclude '*' "
                f"{shlex.quote(f'{resolved_ssh_target}:{remote_generation_dir}/bundle/')}"
                f" {shlex.quote(str(local_dir / 'bundle'))}/"
            )
            run_command(cmd)
            return
        run_command(
            f"rsync -az {shlex.quote(f'{resolved_ssh_target}:{remote_generation_dir}/')} {shlex.quote(str(local_dir))}/"
        )

    def extract_results(self, generation: int) -> Path:
        paths = self.generation_paths(generation)
        hook = self.hooks.get("extractor", {})
        if hook.get("command"):
            self.run_hook("extractor", generation)
            if paths["results_index"].exists():
                return paths["results_index"]
            raise HarnessError(f"Extractor hook did not produce {paths['results_index']}")

        bundle = load_json(paths["bundle_manifest"])
        bundle_dir = paths["bundle_dir"]
        slots_payload: dict[str, Any] = {}
        for slot in bundle["slots"]:
            slot_dir = bundle_dir / slot["path"]
            results_dir = bundle_dir / slot["results_path"]
            item: dict[str, Any] = {
                "role": slot["role"],
                "family": slot["family"],
                "command": slot["command"],
                "results_dir": str(results_dir.relative_to(bundle_dir)),
            }
            for name in ("runner_status", "executor_status", "metrics", "selection", "failure", "summary"):
                candidate = results_dir / f"{name}.json"
                if candidate.exists():
                    item[name] = load_json(candidate)
            log_path = results_dir / "train.log"
            if log_path.exists():
                item["train_log"] = str(log_path.relative_to(bundle_dir))
            slots_payload[slot["slot"]] = item
        write_json(
            paths["results_index"],
            {
                "cycle_id": self.cycle_id,
                "generation": generation,
                "slots": slots_payload,
            },
        )
        return paths["results_index"]

    def select_survivors(self, generation: int) -> Path:
        paths = self.generation_paths(generation)
        bundle_manifest = load_json(paths["bundle_manifest"])
        results_index = load_json(paths["results_index"])
        selection = bundle_manifest.get("selection", {})
        primary_metric_path = selection.get("primary_metric_path", "metrics.score_bpb")
        top_k = int(selection.get("top_k", 2))
        goal = str(selection.get("goal", "minimize"))
        control_slots = [str(slot) for slot in selection.get("control_slots", [])]
        max_control_spread = selection.get("max_control_spread")

        slot_metrics: dict[str, float] = {}
        metric_errors: dict[str, str] = {}
        for slot_id, payload in results_index["slots"].items():
            try:
                value = nested_get(payload, primary_metric_path)
                slot_metrics[slot_id] = float(value)
            except Exception as exc:  # noqa: BLE001
                metric_errors[slot_id] = str(exc)

        invalid_reasons: list[str] = []
        control_values = [slot_metrics[slot] for slot in control_slots if slot in slot_metrics]
        control_spread = None
        if control_values:
            control_spread = max(control_values) - min(control_values)
            if max_control_spread is not None and control_spread > float(max_control_spread):
                invalid_reasons.append(
                    f"control spread {control_spread:.6f} exceeded max_control_spread {float(max_control_spread):.6f}"
                )
        else:
            invalid_reasons.append("no control metrics available")

        non_controls = [
            slot_id
            for slot_id, payload in results_index["slots"].items()
            if payload.get("role") != "control" and slot_id in slot_metrics
        ]
        reverse = goal == "maximize"
        ranked = sorted(non_controls, key=lambda slot_id: slot_metrics[slot_id], reverse=reverse)
        survivors = ranked[:top_k]

        summary = {
            "cycle_id": self.cycle_id,
            "generation": generation,
            "primary_metric_path": primary_metric_path,
            "goal": goal,
            "control_slots": control_slots,
            "control_spread": control_spread,
            "invalid_reasons": invalid_reasons,
            "metric_errors": metric_errors,
            "ranked_slots": [
                {"slot": slot_id, "metric": slot_metrics[slot_id]} for slot_id in ranked
            ],
            "survivors": survivors,
            "should_advance": not invalid_reasons and bool(survivors),
        }
        write_json(paths["generation_summary"], summary)
        return paths["generation_summary"]

    def execute_materialized_generation(
        self,
        generation: int,
        bundle_manifest: Path,
        execute: bool,
        remote: bool,
        dry_run: bool,
    ) -> None:
        if dry_run:
            return
        if not execute:
            return
        if remote:
            ssh_target, remote_dir = self.sync_to_remote(generation)
            self.execute_remote(generation, remote_dir, ssh_target)
            self.collect_from_remote(generation, remote_dir)
        else:
            self.execute_bundle(bundle_manifest)
        self.extract_results(generation)
        self.select_survivors(generation)

    def run_generation(self, generation: int, execute: bool, remote: bool, dry_run: bool) -> None:
        bundle_manifest = self.materialize_generation(generation)
        self.execute_materialized_generation(
            generation=generation,
            bundle_manifest=bundle_manifest,
            execute=execute,
            remote=remote,
            dry_run=dry_run,
        )

    def run_codex_generation(
        self,
        generation: int,
        instructions: str | None,
        execute: bool,
        remote: bool,
        repo_files: list[str] | None = None,
        max_attempts: int | None = None,
        resume: bool = False,
    ) -> None:
        bundle_manifest = self.prepare_generation_with_codex(
            generation=generation,
            instructions=instructions,
            focus_files=repo_files,
            max_attempts=max_attempts,
            resume=resume,
        )
        self.execute_materialized_generation(
            generation=generation,
            bundle_manifest=bundle_manifest,
            execute=execute,
            remote=remote,
            dry_run=False,
        )

    def run_prepare_agent_folder(
        self,
        generation: int,
        instructions: str,
        repo_files: list[str] | None = None,
    ) -> Path:
        return self.prepare_agent_folder(
            generation=generation,
            instructions=instructions,
            repo_files=repo_files,
        )

    def run_cycle(self, start: int, generations: int, execute: bool, remote: bool, dry_run: bool) -> None:
        for generation in range(start, start + generations):
            self.run_generation(generation, execute=execute, remote=remote, dry_run=dry_run)
            if dry_run or not execute:
                continue
            summary = load_json(self.generation_paths(generation)["generation_summary"])
            if not summary.get("should_advance", False):
                break


def init_config(path: Path) -> None:
    example = {
        "cycle_id": "example_cycle",
        "history_window": 3,
        "max_generations": 3,
        "gpu_pool": ["0", "1", "2", "3", "4", "5", "6", "7"],
        "workspace": {
            "root": "search_cycles",
            "sync_mode": "bundle",
        },
        "prompts": {
            "generator": "search_reset_agent_prompt.md",
            "verifier": "search_verifier_codex.md",
            "compiler": None,
        },
        "base": {
            "default_base_script": "train_gpt.py",
            "default_patch_module": None,
            "default_entrypoint": "python3 {script}",
            "defaults": {
                "PYTHONUNBUFFERED": "1",
            },
        },
        "codex": {
            "bin": "codex",
            "model": None,
            "reasoning_effort": None,
            "profile": None,
            "sandbox": "workspace-write",
            "cd": ".",
            "catalog_root": ".",
            "max_attempts": 4,
            "compiled_schema": "search_harness_compiled_generation_schema.json",
            "verifier_schema": "search_harness_verifier_report_schema.json",
            "full_auto": True,
            "skip_git_repo_check": True,
            "extra_args": [],
        },
        "admission": {
            "mode": "survival",
            "require_single_base": True,
            "require_unique_family_groups": True,
            "require_predicate_audit": True,
            "freeze_env_keys": ["DATA_PATH", "TOKENIZER_PATH", "VOCAB_SIZE"],
            "fixed_base_script": None,
            "min_candidate_slots": 2,
            "max_candidate_slots": 6,
        },
        "hooks": {
            "generator": {
                "command": None,
                "cwd": ".",
                "env": {},
            },
            "verifier": {
                "command": None,
                "cwd": ".",
                "env": {},
            },
            "compiler": {
                "command": None,
                "cwd": ".",
                "env": {},
            },
            "extractor": {
                "command": None,
                "cwd": ".",
                "env": {},
            },
        },
        "selection": {
            "pass_verdicts": ["PASS", "PASS_WITH_WARNINGS"],
            "primary_metric_path": "metrics.score_bpb",
            "goal": "minimize",
            "control_slots": ["C0", "C1"],
            "max_control_spread": 0.02,
            "top_k": 2,
        },
        "remote": {
            "enabled": False,
            "ssh_target": "user@host",
            "rsync_target": "user@host:/remote/search_cycles/example_cycle",
            "remote_python": "python3",
            "pre_command": "source ~/venv/bin/activate",
        },
    }
    write_json(path, example)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Automated multi-generation search harness.")
    sub = parser.add_subparsers(dest="command", required=True)

    init = sub.add_parser("init-config", help="Write an example harness config.")
    init.add_argument("--output", required=True)

    run_gen = sub.add_parser("run-generation", help="Run one generation locally or remotely.")
    run_gen.add_argument("--config", required=True)
    run_gen.add_argument("--generation", type=int, required=True)
    run_gen.add_argument("--execute", action="store_true")
    run_gen.add_argument("--remote", action="store_true")
    run_gen.add_argument("--dry-run", action="store_true")

    run_cycle = sub.add_parser("run-cycle", help="Run multiple generations.")
    run_cycle.add_argument("--config", required=True)
    run_cycle.add_argument("--start", type=int, default=0)
    run_cycle.add_argument("--generations", type=int, default=1)
    run_cycle.add_argument("--execute", action="store_true")
    run_cycle.add_argument("--remote", action="store_true")
    run_cycle.add_argument("--dry-run", action="store_true")

    codex_gen = sub.add_parser(
        "codex-generation",
        help="Use Codex to create a ready-to-run generation folder, optionally execute it.",
    )
    codex_gen.add_argument("--config", required=True)
    codex_gen.add_argument("--generation", type=int, required=True)
    codex_group = codex_gen.add_mutually_exclusive_group()
    codex_group.add_argument("--instructions", help="Natural-language instructions for Codex.")
    codex_group.add_argument("--instructions-file", help="Path to a text file containing Codex instructions.")
    codex_gen.add_argument(
        "--repo-file",
        action="append",
        default=[],
        help="Repo file to emphasize in the generation prompt. Repeatable.",
    )
    codex_gen.add_argument("--max-attempts", type=int)
    codex_gen.add_argument("--execute", action="store_true")
    codex_gen.add_argument("--remote", action="store_true")
    codex_gen.add_argument(
        "--resume",
        action="store_true",
        help="Resume from the stored generation request and any existing Codex attempt artifacts.",
    )

    agent_folder = sub.add_parser(
        "prepare-agent-folder",
        help="Create an agent-ready folder with instructions, schemas, focus files, and prior-run memory.",
    )
    agent_folder.add_argument("--config", required=True)
    agent_folder.add_argument("--generation", type=int, required=True)
    agent_group = agent_folder.add_mutually_exclusive_group(required=True)
    agent_group.add_argument("--instructions", help="Natural-language instructions for the external agent.")
    agent_group.add_argument(
        "--instructions-file",
        help="Path to a text file containing the external-agent instructions.",
    )
    agent_folder.add_argument(
        "--repo-file",
        action="append",
        default=[],
        help="Repo file to snapshot into the agent folder and emphasize in the instructions. Repeatable.",
    )

    exec_bundle = sub.add_parser("execute-bundle", help="Execute a staged bundle (for remote/local workers).")
    exec_bundle.add_argument("--bundle", required=True)

    extract = sub.add_parser("extract-results", help="Extract results into a normalized results_index.json.")
    extract.add_argument("--config", required=True)
    extract.add_argument("--generation", type=int, required=True)

    select = sub.add_parser("select-survivors", help="Select survivors from results_index.json.")
    select.add_argument("--config", required=True)
    select.add_argument("--generation", type=int, required=True)

    return parser.parse_args()


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


def main() -> None:
    args = parse_args()
    if args.command == "init-config":
        init_config(Path(args.output).resolve())
        return

    if args.command == "execute-bundle":
        bundle = Path(args.bundle).resolve()
        execute_bundle_manifest(bundle)
        return

    config_path = Path(args.config).resolve()
    harness = SearchHarness(config_path)

    if args.command == "run-generation":
        harness.run_generation(
            args.generation,
            execute=args.execute,
            remote=args.remote,
            dry_run=args.dry_run,
        )
        return

    if args.command == "run-cycle":
        harness.run_cycle(
            start=args.start,
            generations=args.generations,
            execute=args.execute,
            remote=args.remote,
            dry_run=args.dry_run,
        )
        return

    if args.command == "codex-generation":
        harness.run_codex_generation(
            generation=args.generation,
            instructions=load_optional_instructions_arg(args),
            execute=args.execute,
            remote=args.remote,
            repo_files=args.repo_file,
            max_attempts=args.max_attempts,
            resume=args.resume,
        )
        return

    if args.command == "prepare-agent-folder":
        harness.run_prepare_agent_folder(
            generation=args.generation,
            instructions=load_instructions_arg(args),
            repo_files=args.repo_file,
        )
        return

    if args.command == "extract-results":
        harness.extract_results(args.generation)
        return

    if args.command == "select-survivors":
        harness.select_survivors(args.generation)
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
