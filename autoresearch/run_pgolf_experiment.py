#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

RESULTS_HEADER = (
    "iteration\ttimestamp\tmodel\tpost_review_model\trun_id\tdecision\tval_bpb\tval_loss\t"
    "size_bytes\tcommit\tidea\tenv\tnotes\n"
)
REVIEWS_HEADER = "iteration\ttimestamp\tmodel\trun_id\tdecision\tcommit\tsummary\tfindings\n"
RUN_ID_PATTERN = re.compile(r"^(?P<tag>.+)_(?P<num>\d+)$")
BASELINE_IDEA = "baseline"
BASELINE_HYPOTHESIS = "establish initial reference metric and verify the harness"
BASELINE_EXPECTED_SIGNALS = "valid final metric line, valid size line, stable completion"
BASELINE_NOTES = "bootstrap baseline run from the reviewed base commit"


class ControllerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    proposer_model: str
    pre_review_model: str
    post_review_model: str
    execution_mode: str
    tag: str
    deadline: float | None
    max_pre_review_rounds: int
    repo_dir: Path
    data_path: str
    tokenizer_path: str
    vocab_size: int
    nproc_per_node: int
    max_wallclock_seconds: int
    val_loss_every: int
    iterations: int
    remote_host: str
    remote_port: int
    remote_repo_dir: str
    remote_branch: str
    push_remote: str
    remote_fetch_remote: str
    remote_torchrun: str
    remote_identity: str
    remote_force_tty: bool
    local_torchrun: str
    base_extra_env_text: str
    base_extra_env_pairs: list[tuple[str, str]]
    results_file: Path
    reviews_file: Path
    harness_log: Path
    proposer_protocol_file: Path
    pre_review_protocol_file: Path
    post_review_protocol_file: Path
    trace_root: Path
    history_dir: Path
    candidates_dir: Path
    runs_dir: Path
    prep_clones_dir: Path
    remote_log_dir: Path
    prep_queue_depth: int
    prep_poll_seconds: float
    codex_binary: str


@dataclass(frozen=True)
class CandidateSpec:
    idea: str
    hypothesis: str
    expected_signals: str
    notes: str
    extra_env_text: str
    extra_env_pairs: list[tuple[str, str]]


@dataclass(frozen=True)
class PreReviewDecision:
    decision: str
    summary: str
    findings: str
    feedback: str


@dataclass(frozen=True)
class PostReviewDecision:
    decision: str
    summary: str
    findings: str


@dataclass(frozen=True)
class BaselineReviewDecision:
    decision: str
    summary: str
    findings: str


@dataclass(frozen=True)
class PreparedCandidate:
    candidate_id: str
    base_commit: str
    patch_file: Path
    spec: CandidateSpec
    approved_round: int
    manifest_path: Path
    candidate_dir: Path


@dataclass(frozen=True)
class RunOutcome:
    val_bpb: str
    val_loss: str
    size_bytes: str
    remote_log: Path


class HarnessLogger:
    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)
        self._lock = threading.Lock()

    def close(self) -> None:
        with self._lock:
            self._fh.close()

    def log(self, message: str) -> None:
        line = f"{iso_now()} {message}"
        with self._lock:
            print(line, flush=True)
            self._fh.write(line + "\n")

    def stream_line(self, prefix: str, line: str) -> None:
        text = line.rstrip("\n")
        with self._lock:
            print(f"{prefix}{text}", flush=True)
            self._fh.write(f"{prefix}{text}\n")


def iso_now() -> str:
    return datetime.now(UTC).astimezone().isoformat(timespec="seconds")


def sanitize_tsv(value: str) -> str:
    return value.replace("\t", " ").replace("\r", " ").replace("\n", " ")


def run_cmd(
    cmd: list[str],
    *,
    cwd: Path,
    capture_output: bool = True,
    check: bool = True,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        cmd,
        cwd=cwd,
        env=env,
        text=True,
        capture_output=capture_output,
        check=check,
    )


def git_output(repo_dir: Path, *args: str) -> str:
    result = run_cmd(["git", *args], cwd=repo_dir)
    return result.stdout.strip()


def ensure_clean_git(repo_dir: Path) -> None:
    diff_ok = subprocess.run(
        ["git", "diff", "--quiet"],
        cwd=repo_dir,
        check=False,
    ).returncode == 0
    cached_ok = (
        subprocess.run(
            ["git", "diff", "--cached", "--quiet"],
            cwd=repo_dir,
            check=False,
        ).returncode
        == 0
    )
    if not diff_ok or not cached_ok:
        raise ControllerError(
            "git worktree is dirty; commit or stash before starting the controller"
        )


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise ControllerError(f"missing {label}: {path}")


def ensure_file_with_header(path: Path, header: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header, encoding="utf-8")


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def append_jsonl(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(payload, sort_keys=True) + "\n")


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    return path.read_text(encoding="utf-8").splitlines()


def latest_kept_bpb(results_file: Path) -> str:
    values: list[float] = []
    for line in read_lines(results_file)[1:]:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 7 or parts[5] != "keep" or not parts[6]:
            continue
        try:
            values.append(float(parts[6]))
        except ValueError:
            continue
    if not values:
        return ""
    return f"{min(values):.8f}"


def has_completed_result(results_file: Path) -> bool:
    for line in read_lines(results_file)[1:]:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 8 or parts[5] not in {"keep", "revert"}:
            continue
        if parts[6] and parts[7]:
            return True
    return False


def detect_next_iteration(results_file: Path) -> int:
    maximum = 0
    for line in read_lines(results_file)[1:]:
        parts = line.split("\t", 1)
        if parts and parts[0].isdigit():
            maximum = max(maximum, int(parts[0]))
    return maximum + 1


def detect_next_run_number(results_file: Path, tag: str) -> int:
    maximum = 0
    for line in read_lines(results_file)[1:]:
        parts = line.rstrip("\n").split("\t")
        if len(parts) < 5:
            continue
        match = RUN_ID_PATTERN.match(parts[4])
        if match and match.group("tag") == tag:
            maximum = max(maximum, int(match.group("num")))
    return maximum + 1


def detect_next_candidate_number(candidates_dir: Path) -> int:
    maximum = 0
    if not candidates_dir.exists():
        return 1
    for path in candidates_dir.iterdir():
        if not path.is_dir():
            continue
        match = re.fullmatch(r"candidate_(\d+)", path.name)
        if match:
            maximum = max(maximum, int(match.group(1)))
    return maximum + 1


def parse_shell_assignments(spec_file: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw_line in spec_file.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
        if not match:
            raise ControllerError(f"invalid assignment in {spec_file}: {raw_line}")
        key, rhs = match.group(1), match.group(2)
        tokens = shlex.split(f"{key}={rhs}", posix=True)
        if len(tokens) != 1 or "=" not in tokens[0]:
            raise ControllerError(f"invalid shell assignment in {spec_file}: {raw_line}")
        parsed_key, parsed_value = tokens[0].split("=", 1)
        values[parsed_key] = parsed_value
    return values


def parse_extra_env(extra_env_text: str) -> list[tuple[str, str]]:
    if not extra_env_text.strip():
        return []
    pairs: list[tuple[str, str]] = []
    for token in shlex.split(extra_env_text):
        if "=" not in token:
            raise ControllerError(f"invalid EXTRA_ENV token: {token}")
        key, value = token.split("=", 1)
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", key):
            raise ControllerError(f"invalid EXTRA_ENV key: {key}")
        pairs.append((key, value))
    return pairs


def env_flag(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in {"1", "true", "yes", "on"}:
        return True
    if value in {"0", "false", "no", "off"}:
        return False
    raise ControllerError(f"invalid boolean value for {name}: {raw}")


def shell_assignments(pairs: list[tuple[str, str]]) -> str:
    return " ".join(f"{key}={shlex.quote(value)}" for key, value in pairs)


def grep_last(pattern: str, path: Path) -> str:
    last = ""
    with path.open("r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            if pattern in line:
                last = line.rstrip("\n")
    return last


def resolve_repo_path(repo_dir: Path, raw_value: str) -> Path:
    path = Path(raw_value)
    if path.is_absolute():
        return path
    return repo_dir / path


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def git_changed_files(repo_dir: Path, start: str, end: str) -> list[str]:
    output = git_output(repo_dir, "diff", "--name-only", f"{start}..{end}")
    return [line for line in output.splitlines() if line]


def load_candidate_spec(spec_file: Path) -> CandidateSpec:
    values = parse_shell_assignments(spec_file)
    required = ("IDEA", "HYPOTHESIS", "EXPECTED_SIGNALS", "NOTES", "EXTRA_ENV")
    missing = [key for key in required if key not in values]
    if missing:
        raise ControllerError(f"missing fields in {spec_file}: {', '.join(missing)}")
    return CandidateSpec(
        idea=values["IDEA"],
        hypothesis=values["HYPOTHESIS"],
        expected_signals=values["EXPECTED_SIGNALS"],
        notes=values["NOTES"],
        extra_env_text=values["EXTRA_ENV"],
        extra_env_pairs=parse_extra_env(values["EXTRA_ENV"]),
    )


def load_pre_review_decision(path: Path) -> PreReviewDecision:
    values = parse_shell_assignments(path)
    decision = values.get("DECISION", "")
    if decision not in {"approve", "revise"}:
        raise ControllerError(f"invalid pre-review decision in {path}: {decision}")
    return PreReviewDecision(
        decision=decision,
        summary=values.get("SUMMARY", ""),
        findings=values.get("FINDINGS", ""),
        feedback=values.get("FEEDBACK", ""),
    )


def load_post_review_decision(path: Path) -> PostReviewDecision:
    values = parse_shell_assignments(path)
    decision = values.get("DECISION", "")
    if decision not in {"keep", "revert"}:
        raise ControllerError(f"invalid post-review decision in {path}: {decision}")
    return PostReviewDecision(
        decision=decision,
        summary=values.get("SUMMARY", ""),
        findings=values.get("FINDINGS", ""),
    )


def load_baseline_review_decision(path: Path) -> BaselineReviewDecision:
    values = parse_shell_assignments(path)
    decision = values.get("DECISION", "")
    if decision not in {"keep", "invalid_baseline"}:
        raise ControllerError(f"invalid baseline review decision in {path}: {decision}")
    return BaselineReviewDecision(
        decision=decision,
        summary=values.get("SUMMARY", ""),
        findings=values.get("FINDINGS", ""),
    )


class PgolfController:
    def __init__(self, config: Config):
        self.config = config
        self.logger = HarnessLogger(config.harness_log)
        self.next_iteration = detect_next_iteration(config.results_file)
        self.next_run_number = detect_next_run_number(config.results_file, config.tag)
        self.next_candidate_number = detect_next_candidate_number(config.candidates_dir)
        self.reviewed_base_lock = threading.Lock()
        self.reviewed_base_commit = git_output(config.repo_dir, "rev-parse", "HEAD")
        self.stop_event = threading.Event()
        self.ready_queue: queue.Queue[PreparedCandidate] = queue.Queue(
            maxsize=config.prep_queue_depth
        )
        self.prep_thread = threading.Thread(
            target=self._prep_worker,
            name="pgolf-prep",
            daemon=True,
        )
        self.history_ledger = self.config.history_dir / "ledger.jsonl"
        self.history_summary = self.config.history_dir / "summary.md"
        for path in (
            self.config.trace_root,
            self.config.history_dir,
            self.config.candidates_dir,
            self.config.runs_dir,
            self.config.prep_clones_dir,
            self.config.remote_log_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)

    def close(self) -> None:
        self.stop_event.set()
        if self.prep_thread.is_alive():
            self.prep_thread.join()
        self.logger.close()

    def run(self) -> None:
        self.logger.log(
            "controller_start "
            f"proposer_model={self.config.proposer_model} "
            f"pre_review_model={self.config.pre_review_model} "
            f"post_review_model={self.config.post_review_model} "
            f"execution_mode={self.config.execution_mode} "
            f"tag={self.config.tag} "
            f"deadline={self.config.deadline if self.config.deadline is not None else 'forever'} "
            f"prep_queue_depth={self.config.prep_queue_depth} "
            f"max_pre_review_rounds={self.config.max_pre_review_rounds}"
        )
        self.prep_thread.start()
        try:
            while not self._deadline_reached():
                if self._needs_bootstrap_baseline():
                    self._run_bootstrap_baseline()
                    continue
                candidate = self._wait_for_candidate()
                if candidate is None:
                    break
                iteration = self.next_iteration
                run_number = self.next_run_number
                run_id = f"{self.config.tag}_{run_number:04d}"
                run_dir = self.config.runs_dir / run_id
                run_dir.mkdir(parents=True, exist_ok=True)
                copy_file(candidate.patch_file, run_dir / "candidate.patch")
                write_json(
                    run_dir / "candidate_ref.json",
                    {
                        "candidate_id": candidate.candidate_id,
                        "candidate_manifest": str(candidate.manifest_path),
                        "approved_round": candidate.approved_round,
                    },
                )
                if not self._apply_candidate(candidate, run_id, run_dir):
                    continue
                experiment_commit = git_output(self.config.repo_dir, "rev-parse", "HEAD")
                try:
                    outcome = self._run_experiment(
                        candidate=candidate,
                        iteration=iteration,
                        run_id=run_id,
                        run_dir=run_dir,
                        experiment_commit=experiment_commit,
                    )
                except ControllerError as exc:
                    self._record_run_error(
                        candidate=candidate,
                        iteration=iteration,
                        run_id=run_id,
                        run_dir=run_dir,
                        experiment_commit=experiment_commit,
                        stage="experiment",
                        error=str(exc),
                    )
                    ensure_clean_git(self.config.repo_dir)
                    self.next_iteration += 1
                    self.next_run_number += 1
                    continue
                try:
                    decision = self._run_post_review(
                        candidate=candidate,
                        iteration=iteration,
                        run_id=run_id,
                        run_dir=run_dir,
                        experiment_commit=experiment_commit,
                        outcome=outcome,
                    )
                except ControllerError as exc:
                    self._record_run_error(
                        candidate=candidate,
                        iteration=iteration,
                        run_id=run_id,
                        run_dir=run_dir,
                        experiment_commit=experiment_commit,
                        stage="post_review",
                        error=str(exc),
                        outcome=outcome,
                    )
                    ensure_clean_git(self.config.repo_dir)
                    self.next_iteration += 1
                    self.next_run_number += 1
                    continue
                self._finalize_decision(
                    candidate=candidate,
                    iteration=iteration,
                    run_id=run_id,
                    run_dir=run_dir,
                    experiment_commit=experiment_commit,
                    outcome=outcome,
                    decision=decision,
                )
                ensure_clean_git(self.config.repo_dir)
                self.logger.log(f"iteration_complete iteration={iteration} run_id={run_id}")
                self.next_iteration += 1
                self.next_run_number += 1
        finally:
            self.stop_event.set()
            self.prep_thread.join()
            self._cleanup_unused_candidates()
            self.logger.log("controller_finished")

    def _deadline_reached(self) -> bool:
        return self.config.deadline is not None and time.time() >= self.config.deadline

    def _needs_bootstrap_baseline(self) -> bool:
        return not has_completed_result(self.config.results_file)

    def _run_bootstrap_baseline(self) -> None:
        iteration = self.next_iteration
        run_number = self.next_run_number
        run_id = f"{self.config.tag}_{run_number:04d}"
        run_dir = self.config.runs_dir / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        with self.reviewed_base_lock:
            base_commit = self.reviewed_base_commit

        write_json(
            run_dir / "baseline_ref.json",
            {
                "run_type": "baseline",
                "base_commit": base_commit,
                "hypothesis": BASELINE_HYPOTHESIS,
                "expected_signals": BASELINE_EXPECTED_SIGNALS,
            },
        )
        self.logger.log(
            f"baseline_start iteration={iteration} run_id={run_id} base_commit={base_commit}"
        )
        try:
            outcome = self._run_baseline_experiment(
                iteration=iteration,
                run_id=run_id,
                run_dir=run_dir,
                experiment_commit=base_commit,
            )
        except ControllerError as exc:
            self._record_baseline_error(
                iteration=iteration,
                run_id=run_id,
                run_dir=run_dir,
                experiment_commit=base_commit,
                stage="experiment",
                error=str(exc),
            )
            ensure_clean_git(self.config.repo_dir)
            self.next_iteration += 1
            self.next_run_number += 1
            return
        try:
            decision = self._run_baseline_post_review(
                iteration=iteration,
                run_id=run_id,
                run_dir=run_dir,
                experiment_commit=base_commit,
                outcome=outcome,
            )
        except ControllerError as exc:
            self._record_baseline_error(
                iteration=iteration,
                run_id=run_id,
                run_dir=run_dir,
                experiment_commit=base_commit,
                stage="post_review",
                error=str(exc),
                outcome=outcome,
            )
            ensure_clean_git(self.config.repo_dir)
            self.next_iteration += 1
            self.next_run_number += 1
            return
        self._finalize_baseline(
            iteration=iteration,
            run_id=run_id,
            run_dir=run_dir,
            experiment_commit=base_commit,
            outcome=outcome,
            decision=decision,
        )
        ensure_clean_git(self.config.repo_dir)
        self.logger.log(
            f"baseline_complete iteration={iteration} run_id={run_id} decision={decision.decision}"
        )
        self.next_iteration += 1
        self.next_run_number += 1

    def _prep_worker(self) -> None:
        while not self.stop_event.is_set():
            if self._deadline_reached():
                return
            if self.ready_queue.full():
                time.sleep(self.config.prep_poll_seconds)
                continue
            candidate = self._prepare_candidate()
            if candidate is None:
                time.sleep(self.config.prep_poll_seconds)
                continue
            self._update_candidate_status(candidate.manifest_path, "queued")
            try:
                self.ready_queue.put(candidate, timeout=self.config.prep_poll_seconds)
                self._append_history(
                    {
                        "event": "candidate_queued",
                        "candidate_id": candidate.candidate_id,
                        "manifest_path": str(candidate.manifest_path),
                        "timestamp": iso_now(),
                    }
                )
            except queue.Full:
                self._update_candidate_status(candidate.manifest_path, "approved")

    def _prepare_candidate(self) -> PreparedCandidate | None:
        candidate_id = f"candidate_{self.next_candidate_number:04d}"
        self.next_candidate_number += 1
        candidate_dir = self.config.candidates_dir / candidate_id
        candidate_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = candidate_dir / "manifest.json"
        with self.reviewed_base_lock:
            base_commit = self.reviewed_base_commit
        manifest: dict[str, Any] = {
            "candidate_id": candidate_id,
            "base_commit": base_commit,
            "created_at": iso_now(),
            "status": "drafting",
            "proposer_model": self.config.proposer_model,
            "pre_review_model": self.config.pre_review_model,
            "rounds": [],
        }
        self._write_candidate_manifest(manifest_path, manifest)
        self.logger.log(f"candidate_start candidate_id={candidate_id} base_commit={base_commit}")

        prior_feedback = ""
        for round_number in range(1, self.config.max_pre_review_rounds + 1):
            round_dir = candidate_dir / f"round_{round_number:02d}"
            round_dir.mkdir(parents=True, exist_ok=True)
            clone_dir = self.config.prep_clones_dir / f"{candidate_id}_round_{round_number:02d}"
            if clone_dir.exists():
                shutil.rmtree(clone_dir)
            proposer_log = round_dir / "proposer.log"
            proposer_prompt_file = round_dir / "proposer_prompt.txt"
            pre_review_log = round_dir / "pre_review.log"
            pre_review_prompt_file = round_dir / "pre_review_prompt.txt"
            patch_file = round_dir / "candidate.patch"
            spec_file = round_dir / "candidate.env"
            review_decision_file = round_dir / "pre_review.env"

            self._refresh_history_summary()
            try:
                run_cmd(
                    ["git", "clone", "--quiet", str(self.config.repo_dir), str(clone_dir)],
                    cwd=self.config.repo_dir,
                )
                run_cmd(
                    ["git", "checkout", "--quiet", "-b", candidate_id, base_commit],
                    cwd=clone_dir,
                )
                proposer_prompt = self._build_proposer_prompt(
                    candidate_id=candidate_id,
                    round_number=round_number,
                    clone_dir=clone_dir,
                    prior_feedback=prior_feedback,
                )
                proposer_prompt_file.write_text(proposer_prompt, encoding="utf-8")
                self.logger.log(
                    f"proposer_start candidate_id={candidate_id} round={round_number} "
                    f"base_commit={base_commit}"
                )
                exit_code = self._stream_subprocess(
                    [
                        self.config.codex_binary,
                        "exec",
                        "-m",
                        self.config.proposer_model,
                        "--dangerously-bypass-approvals-and-sandbox",
                        proposer_prompt,
                    ],
                    cwd=clone_dir,
                    prefix=f"proposer[{candidate_id}:r{round_number}] ",
                    raw_log_path=proposer_log,
                )
                if exit_code != 0:
                    raise ControllerError(f"proposer exited with code {exit_code}")
                generated_spec = clone_dir / "controller_state" / "current_candidate.env"
                if not generated_spec.exists():
                    raise ControllerError(
                        "proposer did not write controller_state/current_candidate.env"
                    )
                commit_count = int(
                    git_output(clone_dir, "rev-list", "--count", f"{base_commit}..HEAD")
                )
                if commit_count != 1:
                    raise ControllerError(
                        f"expected exactly one candidate commit, found {commit_count}"
                    )
                head_commit = self._sanitize_candidate_commit(
                    clone_dir=clone_dir,
                    base_commit=base_commit,
                    spec_path=generated_spec,
                )
                ensure_clean_git(clone_dir)
                patch_text = run_cmd(
                    ["git", "format-patch", "--quiet", "--stdout", f"{base_commit}..HEAD"],
                    cwd=clone_dir,
                ).stdout
                patch_file.write_text(patch_text, encoding="utf-8")
                copy_file(generated_spec, spec_file)
                spec = load_candidate_spec(spec_file)

                pre_review_prompt = self._build_pre_review_prompt(
                    candidate_id=candidate_id,
                    round_number=round_number,
                    patch_file=patch_file,
                    spec_file=spec_file,
                    output_file=review_decision_file,
                )
                pre_review_prompt_file.write_text(pre_review_prompt, encoding="utf-8")
                self.logger.log(
                    f"pre_review_start candidate_id={candidate_id} round={round_number}"
                )
                review_exit = self._stream_subprocess(
                    [
                        self.config.codex_binary,
                        "exec",
                        "-m",
                        self.config.pre_review_model,
                        "--dangerously-bypass-approvals-and-sandbox",
                        pre_review_prompt,
                    ],
                    cwd=self.config.repo_dir,
                    prefix=f"pre-review[{candidate_id}:r{round_number}] ",
                    raw_log_path=pre_review_log,
                )
                if review_exit != 0:
                    raise ControllerError(f"pre-review exited with code {review_exit}")
                decision = load_pre_review_decision(review_decision_file)
                round_manifest = {
                    "round": round_number,
                    "proposer_log": str(proposer_log),
                    "proposer_prompt": str(proposer_prompt_file),
                    "patch_file": str(patch_file),
                    "spec_file": str(spec_file),
                    "pre_review_log": str(pre_review_log),
                    "pre_review_prompt": str(pre_review_prompt_file),
                    "pre_review_decision_file": str(review_decision_file),
                    "idea": spec.idea,
                    "hypothesis": spec.hypothesis,
                    "expected_signals": spec.expected_signals,
                    "notes": spec.notes,
                    "extra_env": spec.extra_env_text,
                    "commit": head_commit,
                    "pre_review_decision": decision.decision,
                    "pre_review_summary": decision.summary,
                    "pre_review_findings": decision.findings,
                    "pre_review_feedback": decision.feedback,
                }
                manifest["rounds"].append(round_manifest)
                self._write_candidate_manifest(manifest_path, manifest)
                self._append_history(
                    {
                        "event": "candidate_round",
                        "candidate_id": candidate_id,
                        "round": round_number,
                        "decision": decision.decision,
                        "idea": spec.idea,
                        "hypothesis": spec.hypothesis,
                        "manifest_path": str(manifest_path),
                        "timestamp": iso_now(),
                    }
                )
                if decision.decision == "approve":
                    approved_patch = candidate_dir / "approved.patch"
                    approved_spec = candidate_dir / "approved.env"
                    copy_file(patch_file, approved_patch)
                    copy_file(spec_file, approved_spec)
                    manifest["status"] = "approved"
                    manifest["approved_at"] = iso_now()
                    manifest["approved_round"] = round_number
                    manifest["approved_patch"] = str(approved_patch)
                    manifest["approved_spec"] = str(approved_spec)
                    manifest["idea"] = spec.idea
                    manifest["hypothesis"] = spec.hypothesis
                    manifest["expected_signals"] = spec.expected_signals
                    manifest["notes"] = spec.notes
                    manifest["extra_env"] = spec.extra_env_text
                    self._write_candidate_manifest(manifest_path, manifest)
                    self.logger.log(
                        f"candidate_approved candidate_id={candidate_id} "
                        f"round={round_number} idea={sanitize_tsv(spec.idea)}"
                    )
                    return PreparedCandidate(
                        candidate_id=candidate_id,
                        base_commit=base_commit,
                        patch_file=approved_patch,
                        spec=spec,
                        approved_round=round_number,
                        manifest_path=manifest_path,
                        candidate_dir=candidate_dir,
                    )
                prior_feedback = decision.feedback
                self.logger.log(
                    f"candidate_revise candidate_id={candidate_id} round={round_number} "
                    f"feedback={sanitize_tsv(decision.feedback)}"
                )
            except (ControllerError, subprocess.CalledProcessError) as exc:
                manifest["status"] = "failed"
                manifest["failed_at"] = iso_now()
                manifest["failure"] = str(exc)
                self._write_candidate_manifest(manifest_path, manifest)
                self._append_history(
                    {
                        "event": "candidate_failed",
                        "candidate_id": candidate_id,
                        "manifest_path": str(manifest_path),
                        "error": str(exc),
                        "timestamp": iso_now(),
                    }
                )
                self.logger.log(
                    f"candidate_failed candidate_id={candidate_id} error={sanitize_tsv(str(exc))}"
                )
                return None
            finally:
                shutil.rmtree(clone_dir, ignore_errors=True)

        manifest["status"] = "rejected_pre_review"
        manifest["rejected_at"] = iso_now()
        manifest["rejection_reason"] = "max_pre_review_rounds_exhausted"
        self._write_candidate_manifest(manifest_path, manifest)
        self._append_history(
            {
                "event": "candidate_rejected_pre_review",
                "candidate_id": candidate_id,
                "manifest_path": str(manifest_path),
                "timestamp": iso_now(),
            }
        )
        self.logger.log(
            f"candidate_rejected candidate_id={candidate_id} reason=max_pre_review_rounds"
        )
        return None

    def _wait_for_candidate(self) -> PreparedCandidate | None:
        while not self.stop_event.is_set():
            if self._deadline_reached() and self.ready_queue.empty():
                return None
            try:
                candidate = self.ready_queue.get(timeout=self.config.prep_poll_seconds)
                self._update_candidate_status(candidate.manifest_path, "dequeued")
                self.logger.log(f"candidate_dequeued candidate_id={candidate.candidate_id}")
                return candidate
            except queue.Empty:
                continue
        return None

    def _sanitize_candidate_commit(
        self,
        *,
        clone_dir: Path,
        base_commit: str,
        spec_path: Path,
    ) -> str:
        head_commit = git_output(clone_dir, "rev-parse", "HEAD")
        changed_files = git_changed_files(clone_dir, base_commit, head_commit)
        allowed_files = {"train_gpt.py", "controller_state/current_candidate.env"}
        unexpected_files = [path for path in changed_files if path not in allowed_files]
        if unexpected_files:
            raise ControllerError(
                "candidate commit must only touch train_gpt.py; changed files were "
                + ", ".join(changed_files)
            )
        if "controller_state/current_candidate.env" not in changed_files:
            return head_commit

        spec_text = spec_path.read_text(encoding="utf-8")
        run_cmd(
            ["git", "rm", "--cached", "--quiet", "--", "controller_state/current_candidate.env"],
            cwd=clone_dir,
        )
        run_cmd(["git", "commit", "--amend", "--no-edit"], cwd=clone_dir)
        spec_path.parent.mkdir(parents=True, exist_ok=True)
        spec_path.write_text(spec_text, encoding="utf-8")
        head_commit = git_output(clone_dir, "rev-parse", "HEAD")
        changed_files = git_changed_files(clone_dir, base_commit, head_commit)
        if changed_files != ["train_gpt.py"]:
            raise ControllerError(
                (
                    "candidate commit must only touch train_gpt.py after controller "
                    "sanitization; changed files were "
                )
                + ", ".join(changed_files)
            )
        return head_commit

    def _apply_candidate(self, candidate: PreparedCandidate, run_id: str, run_dir: Path) -> bool:
        self.logger.log(
            f"apply_start candidate_id={candidate.candidate_id} run_id={run_id} "
            f"base_commit={candidate.base_commit}"
        )
        try:
            run_cmd(["git", "am", "--3way", str(candidate.patch_file)], cwd=self.config.repo_dir)
        except subprocess.CalledProcessError as exc:
            subprocess.run(["git", "am", "--abort"], cwd=self.config.repo_dir, check=False)
            self._update_candidate_status(candidate.manifest_path, "apply_failed")
            write_json(
                run_dir / "apply_failure.json",
                {
                    "candidate_id": candidate.candidate_id,
                    "run_id": run_id,
                    "error": exc.stderr or exc.stdout or "git am failed",
                },
            )
            self._append_history(
                {
                    "event": "candidate_apply_failed",
                    "candidate_id": candidate.candidate_id,
                    "run_id": run_id,
                    "error": exc.stderr or exc.stdout or "git am failed",
                    "timestamp": iso_now(),
                }
            )
            ensure_clean_git(self.config.repo_dir)
            self.logger.log(
                f"apply_failed candidate_id={candidate.candidate_id} run_id={run_id} "
                f"reason={sanitize_tsv(exc.stderr or exc.stdout or 'git am failed')}"
            )
            return False
        self._update_candidate_status(candidate.manifest_path, "running")
        self.logger.log(f"apply_ready candidate_id={candidate.candidate_id} run_id={run_id}")
        return True

    def _run_experiment(
        self,
        *,
        candidate: PreparedCandidate,
        iteration: int,
        run_id: str,
        run_dir: Path,
        experiment_commit: str,
    ) -> RunOutcome:
        if self.config.execution_mode == "local":
            return self._run_local_experiment(
                iteration=iteration,
                run_id=run_id,
                run_dir=run_dir,
                experiment_commit=experiment_commit,
                extra_env_pairs=candidate.spec.extra_env_pairs,
            )
        return self._run_remote_experiment(
            iteration=iteration,
            run_id=run_id,
            run_dir=run_dir,
            experiment_commit=experiment_commit,
            extra_env_pairs=candidate.spec.extra_env_pairs,
        )

    def _run_baseline_experiment(
        self,
        *,
        iteration: int,
        run_id: str,
        run_dir: Path,
        experiment_commit: str,
    ) -> RunOutcome:
        if self.config.execution_mode == "local":
            return self._run_local_experiment(
                iteration=iteration,
                run_id=run_id,
                run_dir=run_dir,
                experiment_commit=experiment_commit,
                extra_env_pairs=[],
            )
        return self._run_remote_experiment(
            iteration=iteration,
            run_id=run_id,
            run_dir=run_dir,
            experiment_commit=experiment_commit,
            extra_env_pairs=[],
        )

    def _run_remote_experiment(
        self,
        *,
        iteration: int,
        run_id: str,
        run_dir: Path,
        experiment_commit: str,
        extra_env_pairs: list[tuple[str, str]],
    ) -> RunOutcome:
        branch_name = git_output(self.config.repo_dir, "branch", "--show-current")
        self.logger.log(
            f"push_start iteration={iteration} run_id={run_id} "
            f"commit={experiment_commit} branch={branch_name} push_remote={self.config.push_remote}"
        )
        run_cmd(
            [
                "git",
                "push",
                self.config.push_remote,
                f"HEAD:refs/heads/{self.config.remote_branch}",
            ],
            cwd=self.config.repo_dir,
        )
        remote_log = self.config.remote_log_dir / f"{run_id}.log"
        remote_command = self._build_remote_command(run_id, extra_env_pairs)
        stdin_text: str | None = None
        ssh_cmd = ["ssh", *self._ssh_options(), self.config.remote_host]
        if self.config.remote_force_tty:
            stdin_text = remote_command + "exit\n"
        else:
            ssh_cmd.append(remote_command)
        self.logger.log(
            f"remote_start iteration={iteration} run_id={run_id} "
            f"remote_branch={self.config.remote_branch}"
        )
        exit_code = self._stream_subprocess(
            ssh_cmd,
            cwd=self.config.repo_dir,
            prefix=f"remote[{run_id}] ",
            raw_log_path=remote_log,
            stdin_text=stdin_text,
        )
        if exit_code != 0:
            raise ControllerError(
                f"remote training failed for run_id={run_id} with exit code {exit_code}"
            )
        copy_file(remote_log, run_dir / "remote.log")
        metrics_line = grep_last("final_int8_zlib_roundtrip_exact", remote_log)
        size_line = grep_last("Total submission size int8+zlib:", remote_log)
        if not metrics_line:
            raise ControllerError(f"missing final metric line in {remote_log}")
        val_loss_match = re.search(r"val_loss:([0-9.]+)", metrics_line)
        val_bpb_match = re.search(r"val_bpb:([0-9.]+)", metrics_line)
        size_match = re.search(r"Total submission size int8\+zlib: ([0-9]+) bytes", size_line)
        if not val_loss_match or not val_bpb_match:
            raise ControllerError(f"unable to parse metrics from {remote_log}")
        outcome = RunOutcome(
            val_bpb=val_bpb_match.group(1),
            val_loss=val_loss_match.group(1),
            size_bytes=size_match.group(1) if size_match else "",
            remote_log=remote_log,
        )
        write_json(
            run_dir / "metrics.json",
            {
                "val_bpb": outcome.val_bpb,
                "val_loss": outcome.val_loss,
                "size_bytes": outcome.size_bytes,
            },
        )
        return outcome

    def _run_local_experiment(
        self,
        *,
        iteration: int,
        run_id: str,
        run_dir: Path,
        experiment_commit: str,
        extra_env_pairs: list[tuple[str, str]],
    ) -> RunOutcome:
        local_log = self.config.remote_log_dir / f"{run_id}.log"
        env = os.environ.copy()
        env_pairs = [
            ("RUN_ID", run_id),
            ("DATA_PATH", self.config.data_path),
            ("TOKENIZER_PATH", self.config.tokenizer_path),
            ("VOCAB_SIZE", str(self.config.vocab_size)),
            ("VAL_LOSS_EVERY", str(self.config.val_loss_every)),
            ("ITERATIONS", str(self.config.iterations)),
            ("MAX_WALLCLOCK_SECONDS", str(self.config.max_wallclock_seconds)),
            *self.config.base_extra_env_pairs,
            *extra_env_pairs,
        ]
        for key, value in env_pairs:
            env[key] = value
        cmd = [
            self.config.local_torchrun,
            "--standalone",
            f"--nproc_per_node={self.config.nproc_per_node}",
            "train_gpt.py",
        ]
        self.logger.log(
            f"local_start iteration={iteration} run_id={run_id} commit={experiment_commit} "
            f"base_env={sanitize_tsv(self.config.base_extra_env_text)}"
        )
        exit_code = self._stream_subprocess(
            cmd,
            cwd=self.config.repo_dir,
            prefix=f"local[{run_id}] ",
            raw_log_path=local_log,
            env=env,
        )
        if exit_code != 0:
            raise ControllerError(
                f"local training failed for run_id={run_id} with exit code {exit_code}"
            )
        copy_file(local_log, run_dir / "remote.log")
        metrics_line = grep_last("final_int8_zlib_roundtrip_exact", local_log)
        size_line = grep_last("Total submission size int8+zlib:", local_log)
        if not metrics_line:
            raise ControllerError(f"missing final metric line in {local_log}")
        val_loss_match = re.search(r"val_loss:([0-9.]+)", metrics_line)
        val_bpb_match = re.search(r"val_bpb:([0-9.]+)", metrics_line)
        size_match = re.search(r"Total submission size int8\+zlib: ([0-9]+) bytes", size_line)
        if not val_loss_match or not val_bpb_match:
            raise ControllerError(f"unable to parse metrics from {local_log}")
        outcome = RunOutcome(
            val_bpb=val_bpb_match.group(1),
            val_loss=val_loss_match.group(1),
            size_bytes=size_match.group(1) if size_match else "",
            remote_log=local_log,
        )
        write_json(
            run_dir / "metrics.json",
            {
                "val_bpb": outcome.val_bpb,
                "val_loss": outcome.val_loss,
                "size_bytes": outcome.size_bytes,
            },
        )
        return outcome

    def _run_post_review(
        self,
        *,
        candidate: PreparedCandidate,
        iteration: int,
        run_id: str,
        run_dir: Path,
        experiment_commit: str,
        outcome: RunOutcome,
    ) -> PostReviewDecision:
        best_prior_bpb = latest_kept_bpb(self.config.results_file) or "none"
        output_file = run_dir / "post_review.env"
        prompt = self._build_post_review_prompt(
            candidate=candidate,
            iteration=iteration,
            run_id=run_id,
            run_dir=run_dir,
            experiment_commit=experiment_commit,
            outcome=outcome,
            best_prior_bpb=best_prior_bpb,
            output_file=output_file,
        )
        prompt_file = run_dir / "post_review_prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        self.logger.log(
            f"post_review_start iteration={iteration} "
            f"reviewer={self.config.post_review_model} run_id={run_id}"
        )
        exit_code = self._stream_subprocess(
            [
                self.config.codex_binary,
                "exec",
                "-m",
                self.config.post_review_model,
                "--dangerously-bypass-approvals-and-sandbox",
                prompt,
            ],
            cwd=self.config.repo_dir,
            prefix=f"post-review[{run_id}] ",
            raw_log_path=run_dir / "post_review.log",
        )
        if exit_code != 0:
            raise ControllerError(
                f"post-review failed for run_id={run_id} with exit code {exit_code}"
            )
        decision = load_post_review_decision(output_file)
        write_json(
            run_dir / "post_review.json",
            {
                "decision": decision.decision,
                "summary": decision.summary,
                "findings": decision.findings,
                "model": self.config.post_review_model,
            },
        )
        return decision

    def _run_baseline_post_review(
        self,
        *,
        iteration: int,
        run_id: str,
        run_dir: Path,
        experiment_commit: str,
        outcome: RunOutcome,
    ) -> BaselineReviewDecision:
        best_prior_bpb = latest_kept_bpb(self.config.results_file) or "none"
        output_file = run_dir / "post_review.env"
        prompt = self._build_baseline_post_review_prompt(
            iteration=iteration,
            run_id=run_id,
            run_dir=run_dir,
            experiment_commit=experiment_commit,
            outcome=outcome,
            best_prior_bpb=best_prior_bpb,
            output_file=output_file,
        )
        prompt_file = run_dir / "post_review_prompt.txt"
        prompt_file.write_text(prompt, encoding="utf-8")
        self.logger.log(
            f"baseline_post_review_start iteration={iteration} "
            f"reviewer={self.config.post_review_model} run_id={run_id}"
        )
        exit_code = self._stream_subprocess(
            [
                self.config.codex_binary,
                "exec",
                "-m",
                self.config.post_review_model,
                "--dangerously-bypass-approvals-and-sandbox",
                prompt,
            ],
            cwd=self.config.repo_dir,
            prefix=f"post-review[{run_id}] ",
            raw_log_path=run_dir / "post_review.log",
        )
        if exit_code != 0:
            raise ControllerError(
                f"baseline post-review failed for run_id={run_id} with exit code {exit_code}"
            )
        decision = load_baseline_review_decision(output_file)
        write_json(
            run_dir / "post_review.json",
            {
                "decision": decision.decision,
                "summary": decision.summary,
                "findings": decision.findings,
                "model": self.config.post_review_model,
                "run_type": "baseline",
            },
        )
        return decision

    def _record_run_error(
        self,
        *,
        candidate: PreparedCandidate,
        iteration: int,
        run_id: str,
        run_dir: Path,
        experiment_commit: str,
        stage: str,
        error: str,
        outcome: RunOutcome | None = None,
    ) -> None:
        self.logger.log(f"run_error run_id={run_id} stage={stage} error={sanitize_tsv(error)}")
        active_commit = git_output(self.config.repo_dir, "rev-parse", "HEAD")
        if active_commit == experiment_commit:
            self.logger.log(f"run_error_revert_start run_id={run_id} commit={experiment_commit}")
            run_cmd(["git", "revert", "--no-edit", experiment_commit], cwd=self.config.repo_dir)

        remote_log = self.config.remote_log_dir / f"{run_id}.log"
        if remote_log.exists():
            copy_file(remote_log, run_dir / "remote.log")

        timestamp = iso_now()
        note_parts = [
            f"stage={stage}",
            f"error={error}",
            f"hypothesis={candidate.spec.hypothesis}",
            f"expected_signals={candidate.spec.expected_signals}",
            f"pre_review_round={candidate.approved_round}",
        ]
        if candidate.spec.notes:
            note_parts.insert(0, candidate.spec.notes)
        results_row = (
            f"{iteration}\t{timestamp}\t{self.config.proposer_model}\t{self.config.post_review_model}\t{run_id}\terror\t"
            f"{outcome.val_bpb if outcome else ''}\t{outcome.val_loss if outcome else ''}\t"
            f"{outcome.size_bytes if outcome else ''}\t{experiment_commit}\t"
            f"{sanitize_tsv(candidate.spec.idea)}\t{sanitize_tsv(candidate.spec.extra_env_text)}\t"
            f"{sanitize_tsv(' | '.join(note_parts))}\n"
        )
        with self.config.results_file.open("a", encoding="utf-8") as fh:
            fh.write(results_row)
        reviews_row = (
            f"{iteration}\t{timestamp}\t{self.config.post_review_model}\t{run_id}\terror\t{experiment_commit}\t"
            f"{sanitize_tsv(stage + ' failed')}\t{sanitize_tsv(error)}\n"
        )
        with self.config.reviews_file.open("a", encoding="utf-8") as fh:
            fh.write(reviews_row)

        failure_manifest = {
            "run_id": run_id,
            "candidate_id": candidate.candidate_id,
            "iteration": iteration,
            "experiment_commit": experiment_commit,
            "candidate_manifest": str(candidate.manifest_path),
            "decision": "error",
            "failure_stage": stage,
            "error": error,
            "idea": candidate.spec.idea,
            "hypothesis": candidate.spec.hypothesis,
            "expected_signals": candidate.spec.expected_signals,
            "notes": candidate.spec.notes,
            "extra_env": candidate.spec.extra_env_text,
            "metrics": (
                {
                    "val_bpb": outcome.val_bpb,
                    "val_loss": outcome.val_loss,
                    "size_bytes": outcome.size_bytes,
                }
                if outcome
                else None
            ),
            "remote_log": str(remote_log) if remote_log.exists() else "",
            "timestamp": timestamp,
        }
        write_json(run_dir / "failure.json", failure_manifest)
        self._append_history({"event": "run_failed", **failure_manifest})
        self._update_candidate_status(candidate.manifest_path, "error")
        self._commit_ledger_updates(run_id=run_id, decision="error")
        with self.reviewed_base_lock:
            self.reviewed_base_commit = git_output(self.config.repo_dir, "rev-parse", "HEAD")

    def _record_baseline_error(
        self,
        *,
        iteration: int,
        run_id: str,
        run_dir: Path,
        experiment_commit: str,
        stage: str,
        error: str,
        outcome: RunOutcome | None = None,
    ) -> None:
        self.logger.log(
            f"baseline_error run_id={run_id} stage={stage} error={sanitize_tsv(error)}"
        )
        remote_log = self.config.remote_log_dir / f"{run_id}.log"
        if remote_log.exists():
            copy_file(remote_log, run_dir / "remote.log")

        timestamp = iso_now()
        note_parts = [
            BASELINE_NOTES,
            f"stage={stage}",
            f"error={error}",
            f"hypothesis={BASELINE_HYPOTHESIS}",
            f"expected_signals={BASELINE_EXPECTED_SIGNALS}",
        ]
        results_row = (
            f"{iteration}\t{timestamp}\tbaseline\t{self.config.post_review_model}\t{run_id}\terror\t"
            f"{outcome.val_bpb if outcome else ''}\t{outcome.val_loss if outcome else ''}\t"
            f"{outcome.size_bytes if outcome else ''}\t{experiment_commit}\t"
            f"{BASELINE_IDEA}\t{sanitize_tsv(self.config.base_extra_env_text)}\t"
            f"{sanitize_tsv(' | '.join(note_parts))}\n"
        )
        with self.config.results_file.open("a", encoding="utf-8") as fh:
            fh.write(results_row)
        reviews_row = (
            f"{iteration}\t{timestamp}\t{self.config.post_review_model}\t{run_id}\terror\t"
            f"{experiment_commit}\t{sanitize_tsv(stage + ' failed')}\t{sanitize_tsv(error)}\n"
        )
        with self.config.reviews_file.open("a", encoding="utf-8") as fh:
            fh.write(reviews_row)

        failure_manifest = {
            "run_type": "baseline",
            "run_id": run_id,
            "iteration": iteration,
            "experiment_commit": experiment_commit,
            "decision": "error",
            "failure_stage": stage,
            "error": error,
            "idea": BASELINE_IDEA,
            "hypothesis": BASELINE_HYPOTHESIS,
            "expected_signals": BASELINE_EXPECTED_SIGNALS,
            "notes": BASELINE_NOTES,
            "extra_env": self.config.base_extra_env_text,
            "metrics": (
                {
                    "val_bpb": outcome.val_bpb,
                    "val_loss": outcome.val_loss,
                    "size_bytes": outcome.size_bytes,
                }
                if outcome
                else None
            ),
            "remote_log": str(remote_log) if remote_log.exists() else "",
            "timestamp": timestamp,
        }
        write_json(run_dir / "failure.json", failure_manifest)
        self._append_history({"event": "baseline_failed", **failure_manifest})
        self._commit_ledger_updates(run_id=run_id, decision="error")
        with self.reviewed_base_lock:
            self.reviewed_base_commit = git_output(self.config.repo_dir, "rev-parse", "HEAD")

    def _finalize_decision(
        self,
        *,
        candidate: PreparedCandidate,
        iteration: int,
        run_id: str,
        run_dir: Path,
        experiment_commit: str,
        outcome: RunOutcome,
        decision: PostReviewDecision,
    ) -> None:
        if decision.decision == "revert":
            self.logger.log(f"revert_start run_id={run_id} commit={experiment_commit}")
            run_cmd(["git", "revert", "--no-edit", experiment_commit], cwd=self.config.repo_dir)

        timestamp = iso_now()
        note_parts = [
            f"hypothesis={candidate.spec.hypothesis}",
            f"expected_signals={candidate.spec.expected_signals}",
            f"pre_review_round={candidate.approved_round}",
            f"post_review={decision.summary}",
        ]
        if candidate.spec.notes:
            note_parts.insert(0, candidate.spec.notes)
        results_row = (
            f"{iteration}\t{timestamp}\t{self.config.proposer_model}\t{self.config.post_review_model}\t{run_id}\t"
            f"{decision.decision}\t{outcome.val_bpb}\t{outcome.val_loss}\t{outcome.size_bytes}\t"
            f"{experiment_commit}\t{sanitize_tsv(candidate.spec.idea)}\t{sanitize_tsv(candidate.spec.extra_env_text)}\t"
            f"{sanitize_tsv(' | '.join(note_parts))}\n"
        )
        with self.config.results_file.open("a", encoding="utf-8") as fh:
            fh.write(results_row)
        reviews_row = (
            f"{iteration}\t{timestamp}\t{self.config.post_review_model}\t{run_id}\t{decision.decision}\t"
            f"{experiment_commit}\t{sanitize_tsv(decision.summary)}\t{sanitize_tsv(decision.findings)}\n"
        )
        with self.config.reviews_file.open("a", encoding="utf-8") as fh:
            fh.write(reviews_row)

        run_manifest = {
            "run_id": run_id,
            "candidate_id": candidate.candidate_id,
            "iteration": iteration,
            "experiment_commit": experiment_commit,
            "candidate_manifest": str(candidate.manifest_path),
            "post_review_model": self.config.post_review_model,
            "decision": decision.decision,
            "decision_summary": decision.summary,
            "decision_findings": decision.findings,
            "idea": candidate.spec.idea,
            "hypothesis": candidate.spec.hypothesis,
            "expected_signals": candidate.spec.expected_signals,
            "notes": candidate.spec.notes,
            "extra_env": candidate.spec.extra_env_text,
            "metrics": {
                "val_bpb": outcome.val_bpb,
                "val_loss": outcome.val_loss,
                "size_bytes": outcome.size_bytes,
            },
            "remote_log": str(outcome.remote_log),
            "timestamp": timestamp,
        }
        write_json(run_dir / "manifest.json", run_manifest)
        self._append_history({"event": "run_finalized", **run_manifest})
        self._update_candidate_status(candidate.manifest_path, decision.decision)
        self._commit_ledger_updates(run_id=run_id, decision=decision.decision)
        with self.reviewed_base_lock:
            self.reviewed_base_commit = git_output(self.config.repo_dir, "rev-parse", "HEAD")

    def _finalize_baseline(
        self,
        *,
        iteration: int,
        run_id: str,
        run_dir: Path,
        experiment_commit: str,
        outcome: RunOutcome,
        decision: BaselineReviewDecision,
    ) -> None:
        timestamp = iso_now()
        note_parts = [
            BASELINE_NOTES,
            f"hypothesis={BASELINE_HYPOTHESIS}",
            f"expected_signals={BASELINE_EXPECTED_SIGNALS}",
            f"post_review={decision.summary}",
        ]
        results_row = (
            f"{iteration}\t{timestamp}\tbaseline\t{self.config.post_review_model}\t{run_id}\t"
            f"{decision.decision}\t{outcome.val_bpb}\t{outcome.val_loss}\t{outcome.size_bytes}\t"
            f"{experiment_commit}\t{BASELINE_IDEA}\t{sanitize_tsv(self.config.base_extra_env_text)}\t"
            f"{sanitize_tsv(' | '.join(note_parts))}\n"
        )
        with self.config.results_file.open("a", encoding="utf-8") as fh:
            fh.write(results_row)
        reviews_row = (
            f"{iteration}\t{timestamp}\t{self.config.post_review_model}\t{run_id}\t"
            f"{decision.decision}\t{experiment_commit}\t{sanitize_tsv(decision.summary)}\t"
            f"{sanitize_tsv(decision.findings)}\n"
        )
        with self.config.reviews_file.open("a", encoding="utf-8") as fh:
            fh.write(reviews_row)

        run_manifest = {
            "run_type": "baseline",
            "run_id": run_id,
            "iteration": iteration,
            "experiment_commit": experiment_commit,
            "post_review_model": self.config.post_review_model,
            "decision": decision.decision,
            "decision_summary": decision.summary,
            "decision_findings": decision.findings,
            "idea": BASELINE_IDEA,
            "hypothesis": BASELINE_HYPOTHESIS,
            "expected_signals": BASELINE_EXPECTED_SIGNALS,
            "notes": BASELINE_NOTES,
            "extra_env": self.config.base_extra_env_text,
            "metrics": {
                "val_bpb": outcome.val_bpb,
                "val_loss": outcome.val_loss,
                "size_bytes": outcome.size_bytes,
            },
            "remote_log": str(outcome.remote_log),
            "timestamp": timestamp,
        }
        write_json(run_dir / "manifest.json", run_manifest)
        self._append_history({"event": "baseline_finalized", **run_manifest})
        self._commit_ledger_updates(run_id=run_id, decision=decision.decision)
        with self.reviewed_base_lock:
            self.reviewed_base_commit = git_output(self.config.repo_dir, "rev-parse", "HEAD")

    def _commit_ledger_updates(self, *, run_id: str, decision: str) -> None:
        run_cmd(
            ["git", "add", str(self.config.results_file), str(self.config.reviews_file)],
            cwd=self.config.repo_dir,
        )
        message = f"chore(autoresearch): record {decision} for {run_id}"
        run_cmd(["git", "commit", "-m", message], cwd=self.config.repo_dir)

    def _cleanup_unused_candidates(self) -> None:
        while True:
            try:
                candidate = self.ready_queue.get_nowait()
            except queue.Empty:
                return
            self._update_candidate_status(candidate.manifest_path, "approved")

    def _update_candidate_status(self, manifest_path: Path, status: str) -> None:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        payload["status"] = status
        payload["updated_at"] = iso_now()
        write_json(manifest_path, payload)

    def _write_candidate_manifest(self, manifest_path: Path, payload: dict[str, Any]) -> None:
        write_json(manifest_path, payload)

    def _append_history(self, payload: dict[str, Any]) -> None:
        append_jsonl(self.history_ledger, payload)

    def _refresh_history_summary(self) -> None:
        result_lines = read_lines(self.config.results_file)
        lines = [
            "# Autoresearch Summary",
            "",
            f"Generated: {iso_now()}",
            "",
            "## Best kept results",
        ]
        kept_rows = []
        for line in result_lines[1:]:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 13 or parts[5] != "keep":
                continue
            kept_rows.append(parts)
        kept_rows.sort(key=lambda row: float(row[6]))
        if kept_rows:
            for row in kept_rows[:5]:
                lines.append(
                    f"- {row[4]} val_bpb={row[6]} idea={row[10]} env={row[11]} notes={row[12]}"
                )
        else:
            lines.append("- none yet")
        lines.extend(["", "## Recent results"])
        recent_rows = result_lines[max(1, len(result_lines) - 10) :]
        if not recent_rows:
            lines.append("- none yet")
        else:
            for row_line in recent_rows:
                row = row_line.rstrip("\n").split("\t")
                if len(row) < 13:
                    continue
                lines.append(
                    f"- {row[4]} decision={row[5]} val_bpb={row[6]} idea={row[10]} notes={row[12]}"
                )
        lines.extend(["", "## Recent history events"])
        history_lines = read_lines(self.history_ledger)[-15:]
        if history_lines:
            for raw in history_lines:
                event = json.loads(raw)
                event_name = event.get("event", "unknown")
                timestamp = event.get("timestamp", "?")
                candidate_id = event.get("candidate_id", "")
                run_id = event.get("run_id", "")
                idea = event.get("idea", "")
                lines.append(
                    f"- {timestamp} event={event_name} candidate={candidate_id} "
                    f"run={run_id} idea={idea}"
                )
        else:
            lines.append("- none yet")
        self.history_summary.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def _ssh_options(self) -> list[str]:
        options = ["-p", str(self.config.remote_port)]
        if self.config.remote_identity:
            options.extend(["-i", self.config.remote_identity])
        if self.config.remote_force_tty:
            options.append("-tt")
        options.extend(["-o", "StrictHostKeyChecking=accept-new"])
        return options

    def _build_remote_command(self, run_id: str, extra_env_pairs: list[tuple[str, str]]) -> str:
        env_pairs = [
            ("RUN_ID", run_id),
            ("DATA_PATH", self.config.data_path),
            ("TOKENIZER_PATH", self.config.tokenizer_path),
            ("VOCAB_SIZE", str(self.config.vocab_size)),
            ("VAL_LOSS_EVERY", str(self.config.val_loss_every)),
            ("ITERATIONS", str(self.config.iterations)),
            ("MAX_WALLCLOCK_SECONDS", str(self.config.max_wallclock_seconds)),
            *self.config.base_extra_env_pairs,
            *extra_env_pairs,
        ]
        env_prefix = shell_assignments(env_pairs)
        return (
            "set -euo pipefail\n"
            f"cd {shlex.quote(self.config.remote_repo_dir)}\n"
            f"git fetch {shlex.quote(self.config.remote_fetch_remote)} "
            f"{shlex.quote(self.config.remote_branch)}\n"
            f"git checkout -B {shlex.quote(self.config.remote_branch)} FETCH_HEAD\n"
            "git reset --hard FETCH_HEAD\n"
            "mkdir -p logs\n"
            f"env {env_prefix} {shlex.quote(self.config.remote_torchrun)} "
            f"--standalone --nproc_per_node={self.config.nproc_per_node} train_gpt.py "
            f"2>&1 | tee logs/{shlex.quote(run_id)}.log\n"
        )

    def _build_proposer_prompt(
        self,
        *,
        candidate_id: str,
        round_number: int,
        clone_dir: Path,
        prior_feedback: str,
    ) -> str:
        spec_file = clone_dir / "controller_state" / "current_candidate.env"
        protocol_path = clone_dir / "autoresearch" / self.config.proposer_protocol_file.name
        history_summary = self.history_summary
        extra_env_prefix = (
            f"{self.config.base_extra_env_text} " if self.config.base_extra_env_text else ""
        )
        torchrun_bin = (
            self.config.remote_torchrun
            if self.config.execution_mode == "remote"
            else self.config.local_torchrun
        )
        feedback_section = (
            "Pre-review feedback from the previous round:\n"
            f"{prior_feedback}\n"
            if prior_feedback
            else "There is no previous pre-review feedback for this candidate.\n"
        )
        return "\n".join(
            [
                f"This is Parameter Golf proposer candidate {candidate_id}, round {round_number}.",
                "",
                "Working repository clone:",
                str(clone_dir),
                "",
                "Base training command for this candidate:",
                (
                    "RUN_ID=<assigned-by-controller> "
                    f"DATA_PATH={self.config.data_path} "
                    f"TOKENIZER_PATH={self.config.tokenizer_path} "
                    f"VOCAB_SIZE={self.config.vocab_size} "
                    f"VAL_LOSS_EVERY={self.config.val_loss_every} "
                    f"ITERATIONS={self.config.iterations} "
                    f"MAX_WALLCLOCK_SECONDS={self.config.max_wallclock_seconds} "
                    f"{extra_env_prefix}"
                    f"{torchrun_bin} "
                    f"--standalone --nproc_per_node={self.config.nproc_per_node} train_gpt.py"
                ),
                "",
                "Read these repository files before you act:",
                f"- results: {self.config.results_file}",
                f"- reviews: {self.config.reviews_file}",
                f"- autoresearch summary: {history_summary}",
                f"- autoresearch history ledger: {self.history_ledger}",
                f"- proposer protocol: {protocol_path}",
                "",
                feedback_section,
                "Important:",
                "- Propose exactly one bounded change to train_gpt.py.",
                "- Do not edit any tracked file other than train_gpt.py.",
                "- Do not run training yourself.",
                "- Make exactly one git commit for the candidate.",
                f"- Write the rationale and run spec to {spec_file}.",
                (
                    "- Your rationale must include the hypothesis and the exact signals "
                    "that would support or falsify it."
                ),
                (
                    "- The next Codex instance will pre-review your patch for correctness "
                    "and trustworthiness before it is queued."
                ),
            ]
        )

    def _build_pre_review_prompt(
        self,
        *,
        candidate_id: str,
        round_number: int,
        patch_file: Path,
        spec_file: Path,
        output_file: Path,
    ) -> str:
        return "\n".join(
            [
                f"This is the pre-review step for candidate {candidate_id}, round {round_number}.",
                "",
                "Review these candidate artifacts:",
                f"- patch: {patch_file}",
                f"- rationale and env spec: {spec_file}",
                f"- protocol: {self.config.pre_review_protocol_file}",
                "",
                "Important:",
                (
                    "- Focus on code quality, correctness, trustworthiness, and whether "
                    "the claimed ablation is actually valid."
                ),
                "- Assume the controller will only queue the patch if you approve it.",
                "- Be strict. Reject speculative or weakly justified changes.",
                f"- Write your decision to {output_file}.",
                "- Do not edit train_gpt.py yourself.",
                "- Do not run training.",
            ]
        )

    def _build_post_review_prompt(
        self,
        *,
        candidate: PreparedCandidate,
        iteration: int,
        run_id: str,
        run_dir: Path,
        experiment_commit: str,
        outcome: RunOutcome,
        best_prior_bpb: str,
        output_file: Path,
    ) -> str:
        return "\n".join(
            [
                f"This is the post-review step for Parameter Golf iteration {iteration}.",
                "",
                "Review these artifacts:",
                f"- results: {self.config.results_file}",
                f"- reviews: {self.config.reviews_file}",
                f"- run manifest directory: {run_dir}",
                f"- candidate manifest: {candidate.manifest_path}",
                f"- remote log: {outcome.remote_log}",
                f"- post-review protocol: {self.config.post_review_protocol_file}",
                "",
                "Run metadata:",
                f"- run_id: {run_id}",
                f"- experiment_commit: {experiment_commit}",
                f"- best prior kept val_bpb: {best_prior_bpb}",
                f"- current val_bpb: {outcome.val_bpb}",
                f"- current val_loss: {outcome.val_loss}",
                f"- current size_bytes: {outcome.size_bytes}",
                "",
                "Important:",
                (
                    "- Focus mainly on metric quality and trustworthiness. Pre-review "
                    "already handled most code-quality filtering."
                ),
                (
                    "- Keep the change only if the result looks like a real improvement "
                    "or a clearly worthwhile retained change."
                ),
                "- Revert if the result regresses or if the evidence is not trustworthy.",
                f"- Write your decision to {output_file}.",
                "- Do not edit the repository yourself.",
                "- Do not run training.",
            ]
        )

    def _build_baseline_post_review_prompt(
        self,
        *,
        iteration: int,
        run_id: str,
        run_dir: Path,
        experiment_commit: str,
        outcome: RunOutcome,
        best_prior_bpb: str,
        output_file: Path,
    ) -> str:
        return "\n".join(
            [
                f"This is the bootstrap baseline post-review step for iteration {iteration}.",
                "",
                "Review these artifacts:",
                f"- results: {self.config.results_file}",
                f"- reviews: {self.config.reviews_file}",
                f"- run manifest directory: {run_dir}",
                f"- remote log: {outcome.remote_log}",
                f"- post-review protocol: {self.config.post_review_protocol_file}",
                "",
                "Run metadata:",
                f"- run_id: {run_id}",
                f"- experiment_commit: {experiment_commit}",
                f"- best prior kept val_bpb: {best_prior_bpb}",
                f"- current val_bpb: {outcome.val_bpb}",
                f"- current val_loss: {outcome.val_loss}",
                f"- current size_bytes: {outcome.size_bytes}",
                "",
                "Important:",
                "- No candidate patch was applied in this run.",
                (
                    "- This is a bootstrap baseline run meant to establish the initial "
                    "reference metric and validate the harness."
                ),
                (
                    "- Focus on trustworthiness of the run and whether the result is a "
                    "usable baseline reference."
                ),
                f"- Write your decision to {output_file}.",
                "- Use DECISION=keep if the baseline is trustworthy and usable.",
                (
                    "- Use DECISION=invalid_baseline if the baseline should be discarded "
                    "and retried."
                ),
                "- Do not edit the repository yourself.",
                "- Do not run training.",
            ]
        )

    def _stream_subprocess(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        prefix: str,
        raw_log_path: Path,
        env: dict[str, str] | None = None,
        stdin_text: str | None = None,
    ) -> int:
        raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            env=env,
            stdin=subprocess.PIPE if stdin_text is not None else None,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        if stdin_text is not None:
            assert process.stdin is not None
            process.stdin.write(stdin_text)
            process.stdin.close()
        assert process.stdout is not None
        with raw_log_path.open("w", encoding="utf-8") as raw_fh:
            for line in process.stdout:
                raw_fh.write(line)
                self.logger.stream_line(prefix, line)
        return process.wait()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parameter Golf autoresearch controller with a proposer, bounded pre-review loop, "
            "queued reviewed patches, remote execution, and post-review decisions."
        )
    )
    parser.add_argument("--repo-dir", default=os.environ.get("REPO_DIR"))
    parser.add_argument("--proposer-model", default=os.environ.get("PROPOSER_MODEL", "gpt-5.4"))
    parser.add_argument("--pre-review-model", default=os.environ.get("PRE_REVIEW_MODEL"))
    parser.add_argument("--post-review-model", default=os.environ.get("POST_REVIEW_MODEL"))
    parser.add_argument(
        "--executor",
        choices=("remote", "local"),
        default=os.environ.get("EXECUTION_MODE", "remote"),
    )
    parser.add_argument("--tag", default=os.environ.get("TAG", "pgolf"))
    time_group = parser.add_mutually_exclusive_group()
    time_group.add_argument("--hours", type=float, default=None)
    time_group.add_argument("--forever", action="store_true")
    parser.add_argument(
        "--max-pre-review-rounds",
        type=int,
        default=int(os.environ.get("MAX_PRE_REVIEW_ROUNDS", "3")),
    )
    parser.add_argument(
        "--prep-queue-depth",
        type=int,
        default=int(os.environ.get("PREP_QUEUE_DEPTH", "1")),
    )
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    repo_dir = Path(args.repo_dir).expanduser() if args.repo_dir else Path.cwd()
    if shutil.which("codex") is None:
        raise ControllerError("codex CLI not found in PATH")
    if (
        subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            cwd=repo_dir,
            check=False,
        ).returncode
        != 0
    ):
        raise ControllerError("run this program from inside the parameter-golf git repo")
    repo_dir = Path(git_output(repo_dir, "rev-parse", "--show-toplevel"))
    ensure_clean_git(repo_dir)

    def env_path(name: str, default: str) -> Path:
        return resolve_repo_path(repo_dir, os.environ.get(name, default))

    proposer_model = args.proposer_model
    pre_review_model = args.pre_review_model or os.environ.get("REVIEW_MODEL", proposer_model)
    post_review_model = args.post_review_model or os.environ.get("REVIEW_MODEL", proposer_model)
    base_extra_env_text = os.environ.get("BASE_EXTRA_ENV", "")
    default_hours = float(os.environ.get("HOURS", "8"))
    if args.forever:
        deadline = None
    else:
        hours = args.hours if args.hours is not None else default_hours
        deadline = time.time() + hours * 3600

    trace_root = env_path("TRACE_ROOT", "controller_state/autoresearch")
    history_dir = trace_root / "history"
    candidates_dir = trace_root / "candidates"
    runs_dir = trace_root / "runs"
    prep_clones_dir = trace_root / "prep_clones"

    config = Config(
        proposer_model=proposer_model,
        pre_review_model=pre_review_model,
        post_review_model=post_review_model,
        execution_mode=args.executor,
        tag=args.tag,
        deadline=deadline,
        max_pre_review_rounds=max(1, args.max_pre_review_rounds),
        repo_dir=repo_dir,
        data_path=os.environ.get("DATA_PATH", str(repo_dir / "data/datasets/fineweb10B_sp1024")),
        tokenizer_path=os.environ.get(
            "TOKENIZER_PATH", str(repo_dir / "data/tokenizers/fineweb_1024_bpe.model")
        ),
        vocab_size=int(os.environ.get("VOCAB_SIZE", "1024")),
        nproc_per_node=int(os.environ.get("NPROC_PER_NODE", "1")),
        max_wallclock_seconds=int(os.environ.get("MAX_WALLCLOCK_SECONDS", "600")),
        val_loss_every=int(os.environ.get("VAL_LOSS_EVERY", "0")),
        iterations=int(os.environ.get("ITERATIONS", "20000")),
        remote_host=os.environ.get("REMOTE_HOST", ""),
        remote_port=int(os.environ.get("REMOTE_PORT", "22")),
        remote_repo_dir=os.environ.get("REMOTE_REPO_DIR", "/workspace/parameter-golf"),
        remote_branch=os.environ.get("REMOTE_BRANCH", "runpod-autoresearch"),
        push_remote=os.environ.get("PUSH_REMOTE", "origin"),
        remote_fetch_remote=os.environ.get("REMOTE_FETCH_REMOTE", "origin"),
        remote_torchrun=os.environ.get("REMOTE_TORCHRUN", "torchrun"),
        remote_identity=os.environ.get("REMOTE_IDENTITY", ""),
        remote_force_tty=env_flag(
            "REMOTE_SSH_FORCE_TTY",
            "runpod.io" in os.environ.get("REMOTE_HOST", ""),
        ),
        local_torchrun=os.environ.get("LOCAL_TORCHRUN", str(repo_dir / ".venv/bin/torchrun")),
        base_extra_env_text=base_extra_env_text,
        base_extra_env_pairs=parse_extra_env(base_extra_env_text),
        results_file=env_path("RESULTS_FILE", "results.tsv"),
        reviews_file=env_path("REVIEWS_FILE", "reviews.tsv"),
        harness_log=env_path("HARNESS_LOG", f"logs/autoresearch_{args.tag}.log"),
        proposer_protocol_file=env_path(
            "PROPOSER_PROTOCOL_FILE", "autoresearch/pgolf_autoresearch_prompt.md"
        ),
        pre_review_protocol_file=env_path(
            "PRE_REVIEW_PROTOCOL_FILE", "autoresearch/pgolf_pre_review_prompt.md"
        ),
        post_review_protocol_file=env_path(
            "POST_REVIEW_PROTOCOL_FILE", "autoresearch/pgolf_review_prompt.md"
        ),
        trace_root=trace_root,
        history_dir=history_dir,
        candidates_dir=candidates_dir,
        runs_dir=runs_dir,
        prep_clones_dir=prep_clones_dir,
        remote_log_dir=env_path("REMOTE_LOG_DIR", "remote_logs"),
        prep_queue_depth=max(1, args.prep_queue_depth),
        prep_poll_seconds=max(1.0, float(os.environ.get("PREP_POLL_SECONDS", "5"))),
        codex_binary=os.environ.get("CODEX_BIN", "codex"),
    )
    if config.execution_mode == "remote" and not config.remote_host:
        raise ControllerError("set REMOTE_HOST=user@host for the GPU box")
    ensure_exists(config.proposer_protocol_file, "proposer protocol file")
    ensure_exists(config.pre_review_protocol_file, "pre-review protocol file")
    ensure_exists(config.post_review_protocol_file, "post-review protocol file")
    ensure_file_with_header(config.results_file, RESULTS_HEADER)
    ensure_file_with_header(config.reviews_file, REVIEWS_HEADER)
    return config


def main(argv: list[str]) -> int:
    try:
        config = build_config(parse_args(argv))
        controller = PgolfController(config)
        try:
            controller.run()
        finally:
            controller.close()
        return 0
    except ControllerError as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
