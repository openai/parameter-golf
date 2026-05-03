#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import re
import shutil
import sys
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pg_enigma.pg_enigma import EnigmaHarness  # noqa: E402
from search_harness import (  # noqa: E402
    HarnessError,
    load_json,
    load_text,
    prompt_compact_text,
    resolve_path,
    run_process,
    truncate_text,
    write_text,
)


CONTROL_IDS = ("C0", "C1")
CANDIDATE_IDS = tuple(f"H{index}" for index in range(6))
FAMILY_IDS = tuple(f"F{index}" for index in range(8))
ALLOWED_FAMILY_VERDICTS = {"KEEP", "REWRITE", "DROP"}


def markdown_bullets(items: list[str], empty_message: str = "_None._") -> str:
    cleaned = [str(item).strip() for item in items if str(item).strip()]
    if not cleaned:
        return empty_message
    return "\n".join(f"- {item}" for item in cleaned)


def text_file_ready(path: Path) -> bool:
    return path.exists() and path.is_file() and bool(load_text(path).strip())


def extract_markdown_section(text: str, heading: str) -> str:
    pattern = rf"^## {re.escape(heading)}\s*$\n(.*?)(?=^## |\Z)"
    match = re.search(pattern, text, flags=re.MULTILINE | re.DOTALL)
    if not match:
        raise HarnessError(f"missing markdown section: {heading}")
    return match.group(1).strip()


def extract_diff_block(text: str) -> str:
    match = re.search(r"```diff\n(.*?)```", text, flags=re.DOTALL)
    if not match:
        raise HarnessError("missing fenced diff block")
    return match.group(1)


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


class FileRoundHarness:
    def __init__(self, config_path: Path) -> None:
        self.base = EnigmaHarness(config_path.resolve())
        self.module_root = Path(__file__).resolve().parent

    @property
    def cycle_id(self) -> str:
        return self.base.cycle_id

    @property
    def repo_root(self) -> Path:
        return self.base.repo_root

    @property
    def config_dir(self) -> Path:
        return self.base.config_dir

    def agent_root(self) -> Path:
        return self.base.agent_root()

    def project_relative(self, path: Path) -> str:
        return self.base.project_relative(path)

    def round_root(self) -> Path:
        return (self.module_root / "runs" / self.cycle_id).resolve()

    def round_paths(self, round_index: int) -> dict[str, Path]:
        round_dir = self.round_root() / f"round_{round_index:03d}"
        copilot_dir = round_dir / "copilot"
        return {
            "round_dir": round_dir,
            "round_brief": round_dir / "ROUND_BRIEF.md",
            "postmortem_context": round_dir / "POSTMORTEM_CONTEXT.md",
            "focus_dir": round_dir / "focus_files",
            "evidence_dir": round_dir / "evidence_files",
            "step0_dir": round_dir / "step_0",
            "step0_output": round_dir / "step_0" / "STEP_0.md",
            "families_dir": round_dir / "families",
            "family_index": round_dir / "FAMILY_INDEX.md",
            "controls_dir": round_dir / "controls",
            "candidates_dir": round_dir / "candidates",
            "patch_index": round_dir / "PATCH_INDEX.md",
            "round_summary": round_dir / "ROUND_SUMMARY.md",
            "copilot_dir": copilot_dir,
            "step0_prompt": copilot_dir / "STEP_0_PROMPT.md",
            "families_prompt": copilot_dir / "FAMILIES_PROMPT.md",
            "patches_prompt": copilot_dir / "PATCHES_PROMPT.md",
            "patch_repair_prompt": copilot_dir / "PATCH_REPAIR_PROMPT.md",
            "patch_validation_feedback": copilot_dir / "PATCH_VALIDATION_FEEDBACK.txt",
        }

    def family_paths(self, round_index: int) -> list[Path]:
        paths = self.round_paths(round_index)
        return [paths["families_dir"] / f"{family_id}.md" for family_id in FAMILY_IDS]

    def control_paths(self, round_index: int) -> list[Path]:
        paths = self.round_paths(round_index)
        return [paths["controls_dir"] / f"{control_id}.md" for control_id in CONTROL_IDS]

    def candidate_paths(self, round_index: int) -> list[Path]:
        paths = self.round_paths(round_index)
        return [paths["candidates_dir"] / f"{candidate_id}.md" for candidate_id in CANDIDATE_IDS]

    def expected_patch_outputs(self, round_index: int) -> list[Path]:
        paths = self.round_paths(round_index)
        return [paths["patch_index"], *self.control_paths(round_index), *self.candidate_paths(round_index)]

    def expected_family_outputs(self, round_index: int) -> list[Path]:
        paths = self.round_paths(round_index)
        return [paths["family_index"], *self.family_paths(round_index)]

    def resolve_input_files(self, raw_files: list[str] | None) -> list[Path]:
        if not raw_files:
            return []
        resolved: list[Path] = []
        seen: set[Path] = set()
        roots = [self.config_dir, self.repo_root, self.agent_root()]
        agent_root = self.agent_root()
        for raw in raw_files:
            candidates: list[Path] = []
            raw_path = Path(raw)
            if raw_path.is_absolute():
                candidates.append(raw_path.resolve())
            else:
                for root in roots:
                    candidate = resolve_path(root, raw)
                    if candidate is not None and candidate not in candidates:
                        candidates.append(candidate)
            actual = next((candidate for candidate in candidates if candidate.exists() and candidate.is_file()), None)
            if actual is None:
                raise HarnessError(f"input file does not exist: {raw}")
            try:
                actual.relative_to(agent_root)
            except ValueError as exc:
                raise HarnessError(f"input file is outside the agent root {agent_root}: {raw}") from exc
            actual = actual.resolve()
            if actual in seen:
                continue
            seen.add(actual)
            resolved.append(actual)
        return resolved

    def copy_inputs(self, source_paths: list[Path], target_dir: Path) -> list[str]:
        copied: list[str] = []
        for path in source_paths:
            rel = self.project_relative(path)
            destination = target_dir / rel
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
            copied.append(rel)
        return copied

    def build_history_markdown(self, request: dict[str, Any]) -> str:
        history_context = request.get("history_context", {})
        if not isinstance(history_context, dict):
            return "_No prior round history in configured window._"
        history = history_context.get("history", [])
        if not isinstance(history, list) or not history:
            return "_No prior round history in configured window._"

        lines: list[str] = []
        for entry in history:
            if not isinstance(entry, dict):
                continue
            round_id = entry.get("round")
            parts: list[str] = []
            for key, label in (
                ("round_summary", "round summary"),
                ("promotion_report", "promotion"),
                ("postmortem_report", "postmortem"),
            ):
                payload = entry.get(key)
                if isinstance(payload, dict):
                    summary = str(payload.get("summary", "")).strip()
                    if summary:
                        parts.append(f"{label}: {prompt_compact_text(summary, 180)}")
            composition_payload = entry.get("composition_queue")
            if isinstance(composition_payload, dict):
                status = str(composition_payload.get("status", "")).strip()
                if status:
                    parts.append(f"composition: {status}")
            if parts:
                lines.append(f"- round {round_id}: " + "; ".join(parts))
            else:
                lines.append(f"- round {round_id}: artifacts present but unsummarized")
        return "\n".join(lines) if lines else "_No prior round history in configured window._"

    def build_postmortem_context(self, round_index: int) -> str:
        for previous_round in range(round_index - 1, -1, -1):
            postmortem_path = self.base.round_paths(previous_round)["postmortem_report"]
            if not postmortem_path.exists():
                continue
            payload = load_json(postmortem_path)
            summary = str(payload.get("summary", "")).strip()
            round_assessment = str(payload.get("round_assessment", "")).strip()
            generation_delta = str(payload.get("generation_instruction_delta", "")).strip()
            reviews = payload.get("family_reviews", [])

            lines = [
                "# Previous Postmortem Context",
                "",
                f"- Source round: `{previous_round}`",
                f"- Source file: `{self.project_relative(postmortem_path)}`",
                "",
                "## Summary",
                summary or "_No summary captured._",
                "",
                "## Round assessment",
                round_assessment or "_No round assessment captured._",
                "",
                "## Carry-forward instruction delta",
                generation_delta or "_No carry-forward delta captured._",
            ]

            if isinstance(reviews, list) and reviews:
                lines.extend(["", "## Family review highlights"])
                for review in reviews[:6]:
                    if not isinstance(review, dict):
                        continue
                    candidate_id = str(review.get("candidate_id", "")).strip() or "<unknown>"
                    outcome = str(review.get("outcome", "")).strip() or "unknown"
                    next_action = str(review.get("next_action", "")).strip() or "unknown"
                    why = prompt_compact_text(str(review.get("why", "")).strip(), 220)
                    lines.append(f"- `{candidate_id}`: outcome={outcome}, next_action={next_action}; {why or 'no rationale captured'}")
            return "\n".join(lines) + "\n"
        return ""

    def build_round_brief(
        self,
        round_index: int,
        request: dict[str, Any],
        copied_focus: list[str],
        copied_evidence: list[str],
        has_postmortem: bool,
    ) -> str:
        paths = self.round_paths(round_index)
        objective = request["objective"]
        reference_docs = [
            f"`{item['path']}`"
            for item in request.get("reference_docs", [])
            if isinstance(item, dict) and str(item.get("path", "")).strip()
        ]
        output_paths = [
            f"`{self.project_relative(paths['step0_output'])}`",
            f"`{self.project_relative(paths['family_index'])}`",
            *[f"`{self.project_relative(path)}`" for path in self.family_paths(round_index)],
            *[f"`{self.project_relative(path)}`" for path in self.control_paths(round_index)],
            *[f"`{self.project_relative(path)}`" for path in self.candidate_paths(round_index)],
            f"`{self.project_relative(paths['patch_index'])}`",
        ]
        copied_focus_paths = [f"`{self.project_relative(paths['focus_dir'] / rel)}`" for rel in copied_focus]
        copied_evidence_paths = [f"`{self.project_relative(paths['evidence_dir'] / rel)}`" for rel in copied_evidence]
        secondary_metrics = [f"`{item}`" for item in objective.get("secondary_metrics", [])]
        notes = [str(item) for item in objective.get("notes", [])]

        return "\n".join(
            [
                "# Markdown File Round Brief",
                "",
                "## Identity",
                f"- Cycle: `{request['cycle_id']}`",
                f"- Round: `{round_index}`",
                f"- Target script: `{objective.get('target_script', '')}`",
                f"- Goal: {objective.get('goal', '')}",
                "",
                "## User instructions",
                request["user_instructions"],
                "",
                "## Metric contract",
                f"- Primary metric: `{objective.get('primary_metric', '')}`",
                "- Secondary metrics:",
                markdown_bullets(secondary_metrics),
                "",
                "## Hard constraints",
                "- This lane writes markdown files only. Do not emit JSON artifacts.",
                "- Run the stages in order: Step 0 -> family slate -> final patch slate.",
                "- Families come before patches: do not jump from Step 0 directly to final candidate diffs.",
                "- The final slate must contain exactly 6 candidate files (`H0`..`H5`) and 2 control files (`C0`, `C1`).",
                "- The family slate must preserve multiple mechanism families and allocate the six candidate slots across surviving families.",
                "- Preserve broken-predicate honesty: do not sell a candidate that solves an already-satisfied predicate on the score path.",
                "- Prefer consequential program mutations over local knob nudges, env churn, or plausible-but-unfalsified narratives.",
                "",
                "## Output contract",
                markdown_bullets(output_paths),
                "",
                "## Canonical repo files",
                markdown_bullets([f"`{item}`" for item in request.get("focus_files", [])]),
                "",
                "## Focus snapshots for this round",
                markdown_bullets(copied_focus_paths),
                "",
                "## Evidence snapshots for this round",
                markdown_bullets(copied_evidence_paths),
                "",
                "## Reference docs",
                markdown_bullets(reference_docs),
                "",
                "## Objective notes",
                markdown_bullets(notes),
                "",
                "## Prior round context",
                self.build_history_markdown(request),
                "",
                "## Prior postmortem context",
                (
                    f"Read `{self.project_relative(paths['postmortem_context'])}` before Step 0."
                    if has_postmortem
                    else "_No prior postmortem context was available._"
                ),
            ]
        ) + "\n"

    def build_step0_prompt(self, round_index: int) -> str:
        paths = self.round_paths(round_index)
        reads = [
            f"`{self.project_relative(paths['round_brief'])}`",
            f"`{self.project_relative(paths['focus_dir'])}`",
            f"`{self.project_relative(paths['evidence_dir'])}`",
        ]
        if paths["postmortem_context"].exists():
            reads.append(f"`{self.project_relative(paths['postmortem_context'])}`")
        return "\n".join(
            [
                "# Step 0 Copilot Prompt",
                "",
                "Read the markdown brief and round snapshots, then write the Step 0 artifact directly to disk.",
                "",
                "## Required reads",
                markdown_bullets(reads),
                "",
                "## Required write",
                f"- Create or update `{self.project_relative(paths['step0_output'])}`",
                "",
                "## File contract",
                "Write markdown only. Use these sections exactly:",
                "1. `# Step 0`",
                "2. `## Measurement contract`",
                "3. `## Score path to the deployed metric`",
                "4. `## Mutation map`",
                "5. `## Broken-predicate shortlist`",
                "6. `## Final candidate lanes`",
                "7. `## Controls`",
                "8. `## Kill list`",
                "9. `## Family-stage instructions`",
                "",
                "## Candidate/Control contract",
                "- The final candidate lanes section must name exactly `H0`, `H1`, `H2`, `H3`, `H4`, and `H5` as seed lanes, not final patch files.",
                "- The controls section must name exactly `C0` and `C1`.",
                "- Each candidate lane must state: target surface, target predicate, why baseline truth is false, score path trace, cheap signal, and likely failure mode.",
                "- The family stage will regroup or split these seed lanes into `F0`..`F7` families before any final patch slate is written.",
                "- Controls must be explicit anchors, not disguised candidates.",
                "- Do not include diffs yet.",
                "- Do not print the full artifact to stdout; write the file and return only a short completion note.",
            ]
        ) + "\n"

    def build_families_prompt(self, round_index: int) -> str:
        paths = self.round_paths(round_index)
        writes = [
            f"`{self.project_relative(paths['family_index'])}`",
            *[f"`{self.project_relative(path)}`" for path in self.family_paths(round_index)],
        ]
        reads = [
            f"`{self.project_relative(paths['round_brief'])}`",
            f"`{self.project_relative(paths['step0_output'])}`",
            f"`{self.project_relative(paths['focus_dir'])}`",
            f"`{self.project_relative(paths['evidence_dir'])}`",
        ]
        if paths["postmortem_context"].exists():
            reads.append(f"`{self.project_relative(paths['postmortem_context'])}`")
        return "\n".join(
            [
                "# Family Slate Copilot Prompt",
                "",
                "Read the brief plus Step 0 artifact, then write the family slate directly to disk.",
                "",
                "## Required reads",
                markdown_bullets(reads),
                "",
                "## Required writes",
                markdown_bullets(writes),
                "",
                "## Operating mode",
                "- First act as a wild proposer: widen the mechanism space into families instead of jumping to final diffs.",
                "- Then act as a ruthless empiricist: mark each family `KEEP`, `REWRITE`, or `DROP` by predicate truth, score-path relevance, and continuation value.",
                "- The purpose of this stage is to preserve families, not collapse immediately to one winner.",
                "",
                "## Family file contract",
                "- Write one markdown file for each of `F0`..`F7`.",
                "- Each family file must use these sections exactly:",
                "  1. `# F? - ...`",
                "  2. `## Title`",
                "  3. `## Mechanism family`",
                "  4. `## Search level`",
                "  5. `## Operator family`",
                "  6. `## Target surface`",
                "  7. `## Broken predicate`",
                "  8. `## Why baseline truth is false`",
                "  9. `## Score path trace`",
                "  10. `## Mutation grammar`",
                "  11. `## Cheap falsifier`",
                "  12. `## Preserved contracts`",
                "  13. `## Verdict`",
                "  14. `## Candidate budget`",
                "  15. `## Family compile instructions`",
                "  16. `## Likely failure mode`",
                "- `## Verdict` must be exactly one of `KEEP`, `REWRITE`, or `DROP`.",
                "- `## Candidate budget` must be exactly `0`, `1`, or `2`.",
                "- A `DROP` family must have budget `0`.",
                "- A `KEEP` or `REWRITE` family must have budget `1` or `2`.",
                "",
                "## FAMILY_INDEX.md contract",
                "- Use these sections exactly:",
                "  1. `# Family index`",
                "  2. `## Proposed families`",
                "  3. `## Surviving families`",
                "  4. `## Candidate budget allocation`",
                "  5. `## Drop/rewrite rationale`",
                "  6. `## Patch-stage instructions`",
                "- The index must mention every family id `F0`..`F7`.",
                "- Allocate exactly 6 total candidate slots across surviving families.",
                "- Keep at least 3 families alive with positive budget so the final slate cannot collapse to one lineage.",
                "- Do not write final candidate diffs in this stage.",
                "",
                "## Output rules",
                "- Markdown only. No JSON.",
                "- Do not print the family contents to stdout; write the files and return only a short completion note.",
            ]
        ) + "\n"

    def build_patches_prompt(self, round_index: int) -> str:
        paths = self.round_paths(round_index)
        writes = [
            f"`{self.project_relative(path)}`" for path in self.control_paths(round_index)
        ] + [
            f"`{self.project_relative(path)}`" for path in self.candidate_paths(round_index)
        ] + [
            f"`{self.project_relative(paths['patch_index'])}`"
        ]
        reads = [
            f"`{self.project_relative(paths['round_brief'])}`",
            f"`{self.project_relative(paths['step0_output'])}`",
            f"`{self.project_relative(paths['family_index'])}`",
            f"`{self.project_relative(paths['families_dir'])}`",
            f"`{self.project_relative(paths['focus_dir'])}`",
            f"`{self.project_relative(paths['evidence_dir'])}`",
        ]
        if paths["postmortem_context"].exists():
            reads.append(f"`{self.project_relative(paths['postmortem_context'])}`")
        return "\n".join(
            [
                "# Patch Slate Copilot Prompt",
                "",
                "Read the markdown brief plus Step 0 artifact, then materialize the final patch slate directly to files.",
                "",
                "## Required reads",
                markdown_bullets(reads),
                "",
                "## Required writes",
                markdown_bullets(writes),
                "",
                "## Family-first rule",
                "- Use only families with positive candidate budget from `FAMILY_INDEX.md`.",
                "- Respect the family budgets when assigning `H0`..`H5`.",
                "- Represent at least 3 distinct source families across the six candidates.",
                "- Do not invent a candidate that has no surviving family lineage.",
                "",
                "## Candidate file contract",
                "- Write one markdown file for each of `H0`..`H5`.",
                "- Each candidate file must include: title, source family, mechanism, target surface, broken predicate, score path trace, operator claim, cheap signal, preserved contracts, likely failure mode, and an `Exact diff` section with a fenced ```diff block.",
                "- Use a `## Source family` section that names exactly one surviving `F#` lineage.",
                "- If a candidate is a selector or tournament, the diff must persist the selected winner as the final exported artifact and rerun the canonical final eval labels on that winner.",
                "- If a candidate changes export encoding or recipe state, the deserialize/read path must be updated symmetrically so the chosen artifact can actually be loaded.",
                "- Candidate diffs should be first-order and local enough to execute, but large enough to change program behavior meaningfully.",
                "- Stay honest about why the diff should matter; no env churn, no pure literal retuning, no fake weirdness.",
                "",
                "## Control file contract",
                "- Write one markdown file for each of `C0` and `C1`.",
                "- Controls must be explicit anchors for the tournament and should not introduce novel mechanisms.",
                "- Controls may describe no-op/baseline behavior or replay behavior, but they must still state the intended comparison role.",
                "",
                "## Patch index contract",
                "- `PATCH_INDEX.md` must list all eight files, one-line rationale for each, and the intended tournament reading order.",
                "",
                "## Output rules",
                "- Markdown only. No JSON.",
                "- Do not print the patch contents to stdout; write the files and return only a short completion note.",
            ]
        ) + "\n"

    def build_patch_repair_prompt(self, round_index: int, validation_error: str) -> str:
        paths = self.round_paths(round_index)
        reads = [
            f"`{self.project_relative(paths['round_brief'])}`",
            f"`{self.project_relative(paths['step0_output'])}`",
            f"`{self.project_relative(paths['family_index'])}`",
            f"`{self.project_relative(paths['families_dir'])}`",
            f"`{self.project_relative(paths['controls_dir'])}`",
            f"`{self.project_relative(paths['candidates_dir'])}`",
            f"`{self.project_relative(paths['patch_index'])}`",
        ]
        if paths["postmortem_context"].exists():
            reads.append(f"`{self.project_relative(paths['postmortem_context'])}`")
        writes = [
            f"`{self.project_relative(path)}`" for path in self.control_paths(round_index)
        ] + [
            f"`{self.project_relative(path)}`" for path in self.candidate_paths(round_index)
        ] + [
            f"`{self.project_relative(paths['patch_index'])}`"
        ]
        compact_error = prompt_compact_text(validation_error, 1200)
        return "\n".join(
            [
                "# Patch Repair Copilot Prompt",
                "",
                "The current patch slate failed harness validation. Read the existing files, then repair only what is necessary.",
                "",
                "## Validation failure to fix",
                compact_error,
                "",
                "## Required reads",
                markdown_bullets(reads),
                "",
                "## Required writes",
                markdown_bullets(writes),
                "",
                "## Repair priorities",
                "- If a candidate is a selector or tournament, the diff must materialize the winner as the final exported artifact.",
                "- Selector candidates must rerun canonical final eval labels (`quantized`, `quantized_sliding_window`) on the selected winner, not only log per-option evals.",
                "- If export encoding, compressor, or recipe state changes, serialize and deserialize paths must stay symmetric.",
                "- Do not change family budgets or source-family lineage unless the validation feedback proves the current lineage is invalid.",
                "- Keep markdown only and return a short completion note.",
            ]
        ) + "\n"

    def refresh_stage_prompts(self, round_index: int) -> None:
        paths = self.round_paths(round_index)
        if not paths["round_brief"].exists():
            return
        write_text(paths["step0_prompt"], self.build_step0_prompt(round_index))
        write_text(paths["families_prompt"], self.build_families_prompt(round_index))
        write_text(paths["patches_prompt"], self.build_patches_prompt(round_index))

    def validate_step0_output(self, round_index: int) -> None:
        paths = self.round_paths(round_index)
        if not text_file_ready(paths["step0_output"]):
            raise HarnessError(f"missing or empty Step 0 output: {paths['step0_output']}")
        text = load_text(paths["step0_output"])
        required_markers = [
            "# Step 0",
            "## Measurement contract",
            "## Score path to the deployed metric",
            "## Mutation map",
            "## Broken-predicate shortlist",
            "## Final candidate lanes",
            "## Controls",
            "## Family-stage instructions",
        ]
        missing = [marker for marker in required_markers if marker not in text]
        missing.extend(identifier for identifier in (*CANDIDATE_IDS, *CONTROL_IDS) if identifier not in text)
        if missing:
            raise HarnessError(f"Step 0 output is missing required markers: {missing}")

    def validate_family_outputs(self, round_index: int) -> dict[str, int]:
        paths = self.round_paths(round_index)
        missing = [path for path in self.expected_family_outputs(round_index) if not text_file_ready(path)]
        if missing:
            missing_text = ", ".join(self.project_relative(path) for path in missing)
            raise HarnessError(f"family slate is incomplete: {missing_text}")

        family_index_text = load_text(paths["family_index"])
        required_markers = [
            "# Family index",
            "## Proposed families",
            "## Surviving families",
            "## Candidate budget allocation",
            "## Drop/rewrite rationale",
            "## Patch-stage instructions",
        ]
        missing_markers = [marker for marker in required_markers if marker not in family_index_text]
        missing_markers.extend(family_id for family_id in FAMILY_IDS if family_id not in family_index_text)
        if missing_markers:
            raise HarnessError(f"FAMILY_INDEX.md is missing required markers: {missing_markers}")

        positive_budget_count = 0
        total_budget = 0
        budgets: dict[str, int] = {}
        for family_id, family_path in zip(FAMILY_IDS, self.family_paths(round_index), strict=True):
            text = load_text(family_path)
            problems: list[str] = []
            if family_id not in text:
                problems.append("missing family id")
            verdict_text = extract_markdown_section(text, "Verdict").splitlines()[0].strip().upper()
            if verdict_text not in ALLOWED_FAMILY_VERDICTS:
                problems.append(f"invalid verdict {verdict_text!r}")
            budget_section = extract_markdown_section(text, "Candidate budget")
            budget_match = re.search(r"\b([0-2])\b", budget_section)
            if not budget_match:
                problems.append("missing candidate budget")
                budget_value = -1
            else:
                budget_value = int(budget_match.group(1))

            if verdict_text == "DROP" and budget_value != 0:
                problems.append("drop family must have budget 0")
            if verdict_text in {"KEEP", "REWRITE"} and budget_value not in {1, 2}:
                problems.append("surviving family must have budget 1 or 2")
            if problems:
                raise HarnessError(f"{self.project_relative(family_path)} invalid: {', '.join(problems)}")

            budgets[family_id] = budget_value
            total_budget += budget_value
            if budget_value > 0:
                positive_budget_count += 1

        if total_budget != len(CANDIDATE_IDS):
            raise HarnessError(
                f"family candidate budgets must sum to {len(CANDIDATE_IDS)}, got {total_budget}"
            )
        if positive_budget_count < 3:
            raise HarnessError("family slate must keep at least 3 families with positive candidate budget")
        return budgets

    def patch_closure_errors(self, text: str) -> list[str]:
        errors: list[str] = []
        diff_text = extract_diff_block(text)
        lowered = text.lower()
        selector_like = any(token in lowered for token in ("selector", "tournament", "bakeoff", "best-of"))
        if selector_like:
            if diff_text.count("serialize(") < 2:
                errors.append("selector candidate does not persist the winning artifact after comparison")
            if 'timed_eval("quantized"' not in diff_text and "timed_eval('quantized'" not in diff_text:
                errors.append("selector candidate missing canonical final quantized eval on winner")
            if "quantized_sliding_window" in lowered:
                has_final_sliding = (
                    'timed_eval("quantized_sliding_window"' in diff_text
                    or "timed_eval('quantized_sliding_window'" in diff_text
                )
                if not has_final_sliding:
                    errors.append("selector candidate missing canonical final sliding eval on winner")

        compressor_recipe_like = 'recipe["compressor"]' in diff_text or "recipe['compressor']" in diff_text
        if compressor_recipe_like:
            has_recipe_deserialize_signature = "def deserialize(" in diff_text and "recipe: dict[str, object]" in diff_text
            if not has_recipe_deserialize_signature:
                errors.append("export recipe candidate changes compressor state without recipe-aware deserialize")
            has_recipe_decompress = (
                '_decompress(quant_blob_disk, recipe["compressor"])' in diff_text
                or "_decompress(quant_blob_disk, recipe['compressor'])" in diff_text
            )
            if not has_recipe_decompress:
                errors.append("export recipe candidate does not decode with the chosen recipe compressor")
            has_recipe_callsite = ", recipe)" in diff_text or ", best_recipe)" in diff_text
            if not has_recipe_callsite:
                errors.append("export recipe candidate missing recipe-aware serialize/deserialize callsite")
        return errors

    def validate_patch_outputs(self, round_index: int) -> None:
        paths = self.round_paths(round_index)
        family_budgets = self.validate_family_outputs(round_index)
        missing = [path for path in self.expected_patch_outputs(round_index) if not text_file_ready(path)]
        if missing:
            missing_text = ", ".join(self.project_relative(path) for path in missing)
            raise HarnessError(f"patch slate is incomplete: {missing_text}")

        family_counts = {family_id: 0 for family_id, budget in family_budgets.items() if budget > 0}
        for candidate_id, candidate_path in zip(CANDIDATE_IDS, self.candidate_paths(round_index), strict=True):
            text = load_text(candidate_path)
            problems: list[str] = []
            if candidate_id not in text:
                problems.append("missing candidate id")
            if "```diff" not in text:
                problems.append("missing diff block")
            if "## Source family" not in text:
                problems.append("missing source family section")
                source_family = None
            else:
                source_section = extract_markdown_section(text, "Source family")
                source_family = next((family_id for family_id in FAMILY_IDS if family_id in source_section), None)
                if source_family is None:
                    problems.append("missing valid source family id")
                elif family_budgets.get(source_family, 0) <= 0:
                    problems.append(f"source family {source_family} has no surviving budget")
                else:
                    family_counts[source_family] += 1
            if "```diff" in text:
                problems.extend(self.patch_closure_errors(text))
            if problems:
                raise HarnessError(f"{self.project_relative(candidate_path)} invalid: {', '.join(problems)}")

        for control_id, control_path in zip(CONTROL_IDS, self.control_paths(round_index), strict=True):
            text = load_text(control_path)
            if control_id not in text:
                raise HarnessError(f"{self.project_relative(control_path)} is missing control id {control_id}")

        index_text = load_text(paths["patch_index"])
        index_missing = [identifier for identifier in (*CONTROL_IDS, *CANDIDATE_IDS) if identifier not in index_text]
        if index_missing:
            raise HarnessError(f"PATCH_INDEX.md is missing entries for: {index_missing}")

        over_budget = [
            family_id
            for family_id, used in family_counts.items()
            if used > family_budgets.get(family_id, 0)
        ]
        if over_budget:
            raise HarnessError(f"candidate slate exceeds family budgets for: {over_budget}")
        under_budget = [
            family_id
            for family_id, allowed in family_budgets.items()
            if allowed > 0 and family_counts.get(family_id, 0) != allowed
        ]
        if under_budget:
            raise HarnessError(f"candidate slate does not match family budgets for: {under_budget}")
        represented_families = [family_id for family_id, used in family_counts.items() if used > 0]
        if len(represented_families) < 3:
            raise HarnessError("candidate slate must represent at least 3 distinct source families")

    def update_round_summary(self, round_index: int) -> Path:
        paths = self.round_paths(round_index)
        step0_ready = text_file_ready(paths["step0_output"])
        family_ready = all(text_file_ready(path) for path in self.expected_family_outputs(round_index))
        patch_ready = all(text_file_ready(path) for path in self.expected_patch_outputs(round_index))
        lines = [
            "# File Round Summary",
            "",
            f"- Cycle: `{self.cycle_id}`",
            f"- Round: `{round_index}`",
            f"- Round brief: `{self.project_relative(paths['round_brief'])}`",
            f"- Step 0 status: {('ready' if step0_ready else 'pending')}",
            f"- Family slate status: {('ready' if family_ready else 'pending')}",
            f"- Patch slate status: {('ready' if patch_ready else 'pending')}",
            "",
            "## Key artifacts",
            f"- Brief: `{self.project_relative(paths['round_brief'])}`",
            f"- Step 0 prompt: `{self.project_relative(paths['step0_prompt'])}`",
            f"- Step 0 artifact: `{self.project_relative(paths['step0_output'])}`",
            f"- Families prompt: `{self.project_relative(paths['families_prompt'])}`",
            f"- Family index: `{self.project_relative(paths['family_index'])}`",
            f"- Patch prompt: `{self.project_relative(paths['patches_prompt'])}`",
            f"- Patch index: `{self.project_relative(paths['patch_index'])}`",
            "",
            "## Family files",
            markdown_bullets(
                [
                    f"`{self.project_relative(path)}` - {'ready' if text_file_ready(path) else 'pending'}"
                    for path in self.family_paths(round_index)
                ]
            ),
            "",
            "## Control files",
            markdown_bullets(
                [
                    f"`{self.project_relative(path)}` - {'ready' if text_file_ready(path) else 'pending'}"
                    for path in self.control_paths(round_index)
                ]
            ),
            "",
            "## Candidate files",
            markdown_bullets(
                [
                    f"`{self.project_relative(path)}` - {'ready' if text_file_ready(path) else 'pending'}"
                    for path in self.candidate_paths(round_index)
                ]
            ),
        ]
        write_text(paths["round_summary"], "\n".join(lines) + "\n")
        return paths["round_summary"]

    def prepare_round(
        self,
        round_index: int,
        instructions: str,
        repo_files: list[str] | None = None,
        evidence_files: list[str] | None = None,
    ) -> Path:
        paths = self.round_paths(round_index)
        if paths["round_dir"].exists():
            shutil.rmtree(paths["round_dir"])
        for key in (
            "round_dir",
            "focus_dir",
            "evidence_dir",
            "step0_dir",
            "families_dir",
            "controls_dir",
            "candidates_dir",
            "copilot_dir",
        ):
            paths[key].mkdir(parents=True, exist_ok=True)

        focus_paths = self.resolve_input_files(repo_files)
        evidence_paths = self.resolve_input_files(evidence_files)
        request = self.base.build_round_request(round_index, instructions, focus_paths, evidence_paths)
        copied_focus = self.copy_inputs(focus_paths, paths["focus_dir"])
        copied_evidence = self.copy_inputs(evidence_paths, paths["evidence_dir"])

        postmortem_context = self.build_postmortem_context(round_index)
        if postmortem_context:
            write_text(paths["postmortem_context"], postmortem_context)

        write_text(
            paths["round_brief"],
            self.build_round_brief(
                round_index=round_index,
                request=request,
                copied_focus=copied_focus,
                copied_evidence=copied_evidence,
                has_postmortem=bool(postmortem_context),
            ),
        )
        self.refresh_stage_prompts(round_index)
        self.update_round_summary(round_index)
        return paths["round_dir"]

    def run_copilot_stage(
        self,
        *,
        round_index: int,
        prompt_path: Path,
        label: str,
        stage: str,
        expected_outputs: list[Path],
        model: str | None = None,
        reasoning_effort: str | None = None,
    ) -> None:
        if not prompt_path.exists():
            raise HarnessError(f"missing prompt file: {prompt_path}")

        chosen_model = str(model or self.base.codex_stage_model(stage) or "gpt-5.4")
        chosen_reasoning = str(reasoning_effort or self.base.codex_stage_reasoning_effort(stage) or "xhigh")
        prompt = load_text(prompt_path)
        paths = self.round_paths(round_index)
        args = [
            "copilot",
            "-p",
            prompt,
            "--model",
            chosen_model,
            "--reasoning-effort",
            chosen_reasoning,
            "--allow-all-tools",
            "--allow-all-paths",
            "--silent",
            "--output-format",
            "text",
            "--no-custom-instructions",
        ]
        completed = run_process(args, cwd=self.agent_root(), env=os.environ.copy())
        write_text(paths["copilot_dir"] / f"{label}_stdout.log", completed.stdout)
        write_text(paths["copilot_dir"] / f"{label}_stderr.log", completed.stderr)
        if completed.returncode != 0:
            raise HarnessError(
                f"Copilot {label} failed ({completed.returncode}).\n"
                f"STDOUT:\n{truncate_text(completed.stdout)}\n"
                f"STDERR:\n{truncate_text(completed.stderr)}"
            )
        missing = [path for path in expected_outputs if not text_file_ready(path)]
        if missing:
            missing_text = ", ".join(self.project_relative(path) for path in missing)
            raise HarnessError(
                f"Copilot {label} completed but did not write all required files: {missing_text}\n"
                f"STDOUT:\n{truncate_text(completed.stdout)}"
            )

    def run_step0(
        self,
        round_index: int,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        resume: bool = False,
    ) -> Path:
        paths = self.round_paths(round_index)
        if not paths["round_brief"].exists():
            raise HarnessError(f"round is not prepared: {paths['round_brief']}")
        self.refresh_stage_prompts(round_index)
        if resume and text_file_ready(paths["step0_output"]):
            self.validate_step0_output(round_index)
            self.update_round_summary(round_index)
            return paths["step0_output"]
        self.run_copilot_stage(
            round_index=round_index,
            prompt_path=paths["step0_prompt"],
            label="step0",
            stage="step0",
            expected_outputs=[paths["step0_output"]],
            model=model,
            reasoning_effort=reasoning_effort,
        )
        self.validate_step0_output(round_index)
        self.update_round_summary(round_index)
        return paths["step0_output"]

    def run_families(
        self,
        round_index: int,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        resume: bool = False,
    ) -> Path:
        paths = self.round_paths(round_index)
        if not paths["round_brief"].exists():
            raise HarnessError(f"round is not prepared: {paths['round_brief']}")
        self.refresh_stage_prompts(round_index)
        self.validate_step0_output(round_index)
        if resume and all(text_file_ready(path) for path in self.expected_family_outputs(round_index)):
            self.validate_family_outputs(round_index)
            self.update_round_summary(round_index)
            return paths["family_index"]
        self.run_copilot_stage(
            round_index=round_index,
            prompt_path=paths["families_prompt"],
            label="families",
            stage="families",
            expected_outputs=self.expected_family_outputs(round_index),
            model=model,
            reasoning_effort=reasoning_effort,
        )
        self.validate_family_outputs(round_index)
        self.update_round_summary(round_index)
        return paths["family_index"]

    def run_patches(
        self,
        round_index: int,
        *,
        model: str | None = None,
        reasoning_effort: str | None = None,
        resume: bool = False,
    ) -> Path:
        paths = self.round_paths(round_index)
        if not paths["round_brief"].exists():
            raise HarnessError(f"round is not prepared: {paths['round_brief']}")
        self.refresh_stage_prompts(round_index)
        self.validate_step0_output(round_index)
        self.validate_family_outputs(round_index)
        prior_error = ""
        if resume and all(text_file_ready(path) for path in self.expected_patch_outputs(round_index)):
            try:
                self.validate_patch_outputs(round_index)
                self.update_round_summary(round_index)
                return paths["round_dir"]
            except Exception as exc:  # noqa: BLE001
                prior_error = str(exc)
                write_text(paths["patch_validation_feedback"], prior_error + "\n")

        attempts = int(self.base.codex_settings().get("max_attempts", 2))
        if attempts < 1:
            raise HarnessError("max_attempts must be >= 1")
        for attempt in range(1, attempts + 1):
            if attempt == 1 and not prior_error:
                prompt_path = paths["patches_prompt"]
                label = "patches"
            else:
                write_text(paths["patch_repair_prompt"], self.build_patch_repair_prompt(round_index, prior_error))
                prompt_path = paths["patch_repair_prompt"]
                label = f"patch_repair_{attempt:03d}"
            self.run_copilot_stage(
                round_index=round_index,
                prompt_path=prompt_path,
                label=label,
                stage="patches",
                expected_outputs=self.expected_patch_outputs(round_index),
                model=model,
                reasoning_effort=reasoning_effort,
            )
            try:
                self.validate_patch_outputs(round_index)
                self.update_round_summary(round_index)
                return paths["round_dir"]
            except Exception as exc:  # noqa: BLE001
                prior_error = str(exc)
                write_text(paths["patch_validation_feedback"], prior_error + "\n")
        raise HarnessError(f"patch slate failed validation after {attempts} attempts: {prior_error}")

    def run_round(
        self,
        round_index: int,
        *,
        instructions: str | None,
        repo_files: list[str] | None = None,
        evidence_files: list[str] | None = None,
        model: str | None = None,
        reasoning_effort: str | None = None,
        resume: bool = False,
    ) -> Path:
        if resume:
            if not self.round_paths(round_index)["round_brief"].exists():
                raise HarnessError("--resume requires an existing prepared markdown round")
        else:
            cleaned = str(instructions or "").strip()
            if not cleaned:
                raise HarnessError("instructions are required unless --resume is used")
            self.prepare_round(
                round_index=round_index,
                instructions=cleaned,
                repo_files=repo_files,
                evidence_files=evidence_files,
            )
        self.run_step0(round_index, model=model, reasoning_effort=reasoning_effort, resume=resume)
        self.run_families(round_index, model=model, reasoning_effort=reasoning_effort, resume=resume)
        self.run_patches(round_index, model=model, reasoning_effort=reasoning_effort, resume=resume)
        return self.round_paths(round_index)["round_dir"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Markdown-only Copilot file-round harness under pg_enigma/file_round.")
    sub = parser.add_subparsers(dest="command", required=True)

    prepare = sub.add_parser("prepare-round", help="Create the markdown brief, prompts, and frozen snapshots for a round.")
    prepare.add_argument("--config", required=True)
    prepare.add_argument("--round", type=int, required=True)
    prepare_group = prepare.add_mutually_exclusive_group(required=True)
    prepare_group.add_argument("--instructions")
    prepare_group.add_argument("--instructions-file")
    prepare.add_argument("--repo-file", action="append", default=[], help="Repo file to emphasize. Repeatable.")
    prepare.add_argument("--evidence-file", action="append", default=[], help="Evidence file to snapshot. Repeatable.")

    step0 = sub.add_parser("run-step0", help="Run Copilot Step 0 and write STEP_0.md.")
    step0.add_argument("--config", required=True)
    step0.add_argument("--round", type=int, required=True)
    step0.add_argument("--model")
    step0.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"])
    step0.add_argument("--resume", action="store_true", help="Reuse an existing valid STEP_0.md if present.")

    families = sub.add_parser("run-families", help="Run Copilot to write the family slate and budget allocation.")
    families.add_argument("--config", required=True)
    families.add_argument("--round", type=int, required=True)
    families.add_argument("--model")
    families.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"])
    families.add_argument("--resume", action="store_true", help="Reuse an existing valid family slate if present.")

    patches = sub.add_parser(
        "run-patches",
        help="Run Copilot to write the six candidate files, two controls, and patch index from surviving families.",
    )
    patches.add_argument("--config", required=True)
    patches.add_argument("--round", type=int, required=True)
    patches.add_argument("--model")
    patches.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"])
    patches.add_argument("--resume", action="store_true", help="Reuse an existing valid patch slate if present.")

    run_round = sub.add_parser(
        "run-round",
        help="Prepare the markdown round, then run Step 0, the family slate, and the final patch slate.",
    )
    run_round.add_argument("--config", required=True)
    run_round.add_argument("--round", type=int, required=True)
    run_group = run_round.add_mutually_exclusive_group()
    run_group.add_argument("--instructions")
    run_group.add_argument("--instructions-file")
    run_round.add_argument("--repo-file", action="append", default=[], help="Repo file to emphasize. Repeatable.")
    run_round.add_argument("--evidence-file", action="append", default=[], help="Evidence file to snapshot. Repeatable.")
    run_round.add_argument("--model")
    run_round.add_argument("--reasoning-effort", choices=["low", "medium", "high", "xhigh"])
    run_round.add_argument("--resume", action="store_true", help="Resume from an existing prepared markdown round.")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    harness = FileRoundHarness(Path(args.config).resolve())

    if args.command == "prepare-round":
        harness.prepare_round(
            round_index=args.round,
            instructions=load_instructions_arg(args),
            repo_files=args.repo_file,
            evidence_files=args.evidence_file,
        )
        return

    if args.command == "run-step0":
        harness.run_step0(
            round_index=args.round,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            resume=args.resume,
        )
        return

    if args.command == "run-families":
        harness.run_families(
            round_index=args.round,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            resume=args.resume,
        )
        return

    if args.command == "run-patches":
        harness.run_patches(
            round_index=args.round,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            resume=args.resume,
        )
        return

    if args.command == "run-round":
        harness.run_round(
            round_index=args.round,
            instructions=load_optional_instructions_arg(args),
            repo_files=args.repo_file,
            evidence_files=args.evidence_file,
            model=args.model,
            reasoning_effort=args.reasoning_effort,
            resume=args.resume,
        )
        return

    raise SystemExit(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
