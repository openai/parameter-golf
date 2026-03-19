#!/usr/bin/env python3
from __future__ import annotations

import argparse
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
from datetime import datetime, timezone
from pathlib import Path


RESULTS_HEADER = (
    "iteration\ttimestamp\tmodel\treview_model\trun_id\tdecision\tval_bpb\tval_loss\t"
    "size_bytes\tcommit\tidea\tenv\tnotes\n"
)
REVIEWS_HEADER = "iteration\ttimestamp\tmodel\trun_id\tdecision\tcommit\tsummary\tfindings\n"
RUN_ID_PATTERN = re.compile(r"^(?P<tag>.+)_(?P<num>\d+)$")


class ControllerError(RuntimeError):
    pass


@dataclass(frozen=True)
class Config:
    model: str
    tag: str
    hours: float
    review_model: str
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
    remote_python: str
    remote_torchrun: str
    remote_identity: str
    results_file: Path
    harness_log: Path
    program_file: Path
    review_program_file: Path
    reviews_file: Path
    state_dir: Path
    remote_log_dir: Path
    prep_queue_depth: int
    prep_poll_seconds: float
    codex_binary: str


@dataclass(frozen=True)
class PreparedExperiment:
    prep_id: str
    base_commit: str
    patch_file: Path
    spec_file: Path
    prep_log_file: Path
    idea: str
    notes: str
    extra_env_pairs: list[tuple[str, str]]
    extra_env_text: str


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
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


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
    diff_ok = subprocess.run(["git", "diff", "--quiet"], cwd=repo_dir, check=False).returncode == 0
    cached_ok = subprocess.run(["git", "diff", "--cached", "--quiet"], cwd=repo_dir, check=False).returncode == 0
    if not diff_ok or not cached_ok:
        raise ControllerError("git worktree is dirty; commit or stash before starting the controller")


def ensure_exists(path: Path, label: str) -> None:
    if not path.exists():
        raise ControllerError(f"missing {label}: {path}")


def ensure_file_with_header(path: Path, header: str) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(header, encoding="utf-8")


def latest_kept_bpb(results_file: Path) -> str:
    if not results_file.exists():
        return ""
    values: list[float] = []
    with results_file.open("r", encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
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


def detect_next_iteration(results_file: Path) -> int:
    if not results_file.exists():
        return 1
    maximum = 0
    with results_file.open("r", encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            parts = line.split("\t", 1)
            if not parts or not parts[0].isdigit():
                continue
            maximum = max(maximum, int(parts[0]))
    return maximum + 1


def detect_next_run_number(results_file: Path, tag: str) -> int:
    if not results_file.exists():
        return 1
    maximum = 0
    with results_file.open("r", encoding="utf-8") as fh:
        next(fh, None)
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 5:
                continue
            match = RUN_ID_PATTERN.match(parts[4])
            if match and match.group("tag") == tag:
                maximum = max(maximum, int(match.group("num")))
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


def shell_assignments(pairs: list[tuple[str, str]]) -> str:
    tokens = [f"{key}={shlex.quote(value)}" for key, value in pairs]
    return " ".join(tokens)


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


class PgolfController:
    def __init__(self, config: Config):
        self.config = config
        self.logger = HarnessLogger(config.harness_log)
        self.deadline = time.time() + config.hours * 3600
        self.next_iteration = detect_next_iteration(config.results_file)
        self.next_run_number = detect_next_run_number(config.results_file, config.tag)
        self.next_prep_number = 1
        self.stop_event = threading.Event()
        self.ready_queue: queue.Queue[PreparedExperiment] = queue.Queue(maxsize=config.prep_queue_depth)
        self.prep_thread = threading.Thread(target=self._prep_worker, name="pgolf-prep", daemon=True)
        self.active_prep_lock = threading.Lock()
        self.active_prep_count = 0
        self.prepared_dir = config.state_dir / "prepared"
        self.prep_logs_dir = config.state_dir / "prep_logs"
        self.prep_specs_dir = config.state_dir / "prep_specs"
        self.prep_clones_dir = config.state_dir / "prep_clones"
        for path in (
            self.prepared_dir,
            self.prep_logs_dir,
            self.prep_specs_dir,
            self.prep_clones_dir,
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
            f"controller_start model={self.config.model} review_model={self.config.review_model} "
            f"tag={self.config.tag} hours={self.config.hours} prep_queue_depth={self.config.prep_queue_depth}"
        )
        self.prep_thread.start()
        try:
            while time.time() < self.deadline:
                prepared = self._wait_for_candidate()
                if prepared is None:
                    break
                iteration = self.next_iteration
                run_number = self.next_run_number
                run_id = f"{self.config.tag}_{run_number:04d}"
                if not self._apply_patch(prepared, run_id):
                    continue
                experiment_commit = git_output(self.config.repo_dir, "rev-parse", "HEAD")
                outcome = self._run_remote_experiment(prepared, iteration, run_id, experiment_commit)
                self._append_pending_result(iteration, run_id, experiment_commit, prepared, outcome)
                self._run_review(iteration, run_id, experiment_commit, outcome.remote_log)
                ensure_clean_git(self.config.repo_dir)
                self.logger.log(f"iteration_complete iteration={iteration} run_id={run_id}")
                self.next_iteration += 1
                self.next_run_number += 1
        finally:
            self.stop_event.set()
            self.prep_thread.join()
            self._cleanup_unused_candidates()
            self.logger.log("controller_finished")

    def _prep_worker(self) -> None:
        while not self.stop_event.is_set():
            if time.time() >= self.deadline:
                return
            if self.ready_queue.full():
                time.sleep(self.config.prep_poll_seconds)
                continue
            prepared = self._prepare_candidate()
            if prepared is None:
                time.sleep(self.config.prep_poll_seconds)
                continue
            try:
                self.ready_queue.put(prepared, timeout=self.config.prep_poll_seconds)
            except queue.Full:
                self._delete_candidate_files(prepared)

    def _prepare_candidate(self) -> PreparedExperiment | None:
        prep_number = self.next_prep_number
        self.next_prep_number += 1
        prep_id = f"prep_{prep_number:04d}"
        clone_dir = self.prep_clones_dir / prep_id
        prep_log_file = self.prep_logs_dir / f"{prep_id}.log"
        spec_copy_path = self.prep_specs_dir / f"{prep_id}.env"
        patch_file = self.prepared_dir / f"{prep_id}.patch"
        base_commit = git_output(self.config.repo_dir, "rev-parse", "HEAD")
        clone_dir.parent.mkdir(parents=True, exist_ok=True)
        if clone_dir.exists():
            shutil.rmtree(clone_dir)
        self.logger.log(f"prep_start prep_id={prep_id} base_commit={base_commit}")
        with self.active_prep_lock:
            self.active_prep_count += 1
        try:
            run_cmd(["git", "clone", "--quiet", str(self.config.repo_dir), str(clone_dir)], cwd=self.config.repo_dir)
            run_cmd(["git", "checkout", "--quiet", "-b", prep_id, base_commit], cwd=clone_dir)
            prompt = self._build_prep_prompt(prep_id=prep_id, clone_dir=clone_dir)
            exit_code = self._stream_subprocess(
                [self.config.codex_binary, "exec", "-m", self.config.model, "--dangerously-bypass-approvals-and-sandbox", prompt],
                cwd=clone_dir,
                prefix=f"prep[{prep_id}] ",
                raw_log_path=prep_log_file,
            )
            if exit_code != 0:
                self.logger.log(f"prep_failed prep_id={prep_id} reason=codex_exit_{exit_code}")
                return None
            spec_file = clone_dir / "controller_state" / "current_run.env"
            if not spec_file.exists():
                self.logger.log(f"prep_failed prep_id={prep_id} reason=missing_spec")
                return None
            ensure_clean_git(clone_dir)
            commit_count = int(git_output(clone_dir, "rev-list", "--count", f"{base_commit}..HEAD"))
            if commit_count != 1:
                self.logger.log(f"prep_failed prep_id={prep_id} reason=expected_one_commit actual_commits={commit_count}")
                return None
            patch_text = run_cmd(["git", "format-patch", "--quiet", "--stdout", f"{base_commit}..HEAD"], cwd=clone_dir).stdout
            patch_file.write_text(patch_text, encoding="utf-8")
            spec_values = parse_shell_assignments(spec_file)
            idea = spec_values.get("IDEA", "unspecified")
            notes = spec_values.get("NOTES", "")
            extra_env_text = spec_values.get("EXTRA_ENV", "")
            extra_env_pairs = parse_extra_env(extra_env_text)
            shutil.copyfile(spec_file, spec_copy_path)
            prepared = PreparedExperiment(
                prep_id=prep_id,
                base_commit=base_commit,
                patch_file=patch_file,
                spec_file=spec_copy_path,
                prep_log_file=prep_log_file,
                idea=idea,
                notes=notes,
                extra_env_pairs=extra_env_pairs,
                extra_env_text=extra_env_text,
            )
            self.logger.log(f"prep_ready prep_id={prep_id} idea={sanitize_tsv(idea)}")
            return prepared
        except (ControllerError, subprocess.CalledProcessError) as exc:
            self.logger.log(f"prep_failed prep_id={prep_id} reason={sanitize_tsv(str(exc))}")
            return None
        finally:
            with self.active_prep_lock:
                self.active_prep_count -= 1
            shutil.rmtree(clone_dir, ignore_errors=True)

    def _wait_for_candidate(self) -> PreparedExperiment | None:
        while not self.stop_event.is_set():
            if time.time() >= self.deadline and self.ready_queue.empty():
                return None
            try:
                prepared = self.ready_queue.get(timeout=self.config.prep_poll_seconds)
                self.logger.log(f"prep_dequeued prep_id={prepared.prep_id}")
                return prepared
            except queue.Empty:
                continue
        return None

    def _apply_patch(self, prepared: PreparedExperiment, run_id: str) -> bool:
        self.logger.log(
            f"apply_start prep_id={prepared.prep_id} run_id={run_id} base_commit={prepared.base_commit}"
        )
        try:
            run_cmd(["git", "am", "--3way", str(prepared.patch_file)], cwd=self.config.repo_dir)
        except subprocess.CalledProcessError as exc:
            subprocess.run(["git", "am", "--abort"], cwd=self.config.repo_dir, check=False)
            self.logger.log(
                f"apply_failed prep_id={prepared.prep_id} run_id={run_id} "
                f"reason={sanitize_tsv(exc.stderr or exc.stdout or 'git am failed')}"
            )
            self._delete_candidate_files(prepared)
            ensure_clean_git(self.config.repo_dir)
            return False
        self.logger.log(f"apply_ready prep_id={prepared.prep_id} run_id={run_id}")
        self._delete_candidate_files(prepared)
        return True

    def _run_remote_experiment(
        self,
        prepared: PreparedExperiment,
        iteration: int,
        run_id: str,
        experiment_commit: str,
    ) -> RunOutcome:
        branch_name = git_output(self.config.repo_dir, "branch", "--show-current")
        self.logger.log(
            f"push_start iteration={iteration} run_id={run_id} commit={experiment_commit} branch={branch_name}"
        )
        run_cmd(
            ["git", "push", "origin", f"HEAD:refs/heads/{self.config.remote_branch}"],
            cwd=self.config.repo_dir,
        )
        remote_log = self.config.remote_log_dir / f"{run_id}.log"
        remote_command = self._build_remote_command(run_id, prepared.extra_env_pairs)
        ssh_cmd = ["ssh", *self._ssh_options(), self.config.remote_host, remote_command]
        self.logger.log(
            f"remote_start iteration={iteration} run_id={run_id} remote_branch={self.config.remote_branch}"
        )
        exit_code = self._stream_subprocess(
            ssh_cmd,
            cwd=self.config.repo_dir,
            prefix=f"remote[{run_id}] ",
            raw_log_path=remote_log,
        )
        if exit_code != 0:
            raise ControllerError(f"remote training failed for run_id={run_id} with exit code {exit_code}")
        metrics_line = grep_last("final_int8_zlib_roundtrip_exact", remote_log)
        size_line = grep_last("Total submission size int8+zlib:", remote_log)
        if not metrics_line:
            raise ControllerError(f"missing final metric line in {remote_log}")
        val_loss_match = re.search(r"val_loss:([0-9.]+)", metrics_line)
        val_bpb_match = re.search(r"val_bpb:([0-9.]+)", metrics_line)
        size_match = re.search(r"Total submission size int8\+zlib: ([0-9]+) bytes", size_line)
        if not val_loss_match or not val_bpb_match:
            raise ControllerError(f"unable to parse metrics from {remote_log}")
        return RunOutcome(
            val_bpb=val_bpb_match.group(1),
            val_loss=val_loss_match.group(1),
            size_bytes=size_match.group(1) if size_match else "",
            remote_log=remote_log,
        )

    def _append_pending_result(
        self,
        iteration: int,
        run_id: str,
        experiment_commit: str,
        prepared: PreparedExperiment,
        outcome: RunOutcome,
    ) -> None:
        timestamp = iso_now()
        row = (
            f"{iteration}\t{timestamp}\t{self.config.model}\t{self.config.review_model}\t{run_id}\t"
            f"pending_review\t{outcome.val_bpb}\t{outcome.val_loss}\t{outcome.size_bytes}\t"
            f"{experiment_commit}\t{sanitize_tsv(prepared.idea)}\t{sanitize_tsv(prepared.extra_env_text)}\t"
            f"{sanitize_tsv(prepared.notes)}\n"
        )
        with self.config.results_file.open("a", encoding="utf-8") as fh:
            fh.write(row)
        self.logger.log(f"result_recorded iteration={iteration} run_id={run_id} val_bpb={outcome.val_bpb}")

    def _run_review(self, iteration: int, run_id: str, experiment_commit: str, remote_log: Path) -> None:
        best_prior_bpb = latest_kept_bpb(self.config.results_file) or "none"
        prompt = self._build_review_prompt(
            iteration=iteration,
            run_id=run_id,
            experiment_commit=experiment_commit,
            remote_log=remote_log,
            best_prior_bpb=best_prior_bpb,
        )
        self.logger.log(f"review_start iteration={iteration} reviewer={self.config.review_model} run_id={run_id}")
        exit_code = self._stream_subprocess(
            [self.config.codex_binary, "exec", "-m", self.config.review_model, "--dangerously-bypass-approvals-and-sandbox", prompt],
            cwd=self.config.repo_dir,
            prefix=f"review[{run_id}] ",
            raw_log_path=self.config.state_dir / f"review_{run_id}.log",
        )
        if exit_code != 0:
            raise ControllerError(f"review failed for run_id={run_id} with exit code {exit_code}")

    def _cleanup_unused_candidates(self) -> None:
        while True:
            try:
                prepared = self.ready_queue.get_nowait()
            except queue.Empty:
                return
            self._delete_candidate_files(prepared)

    def _delete_candidate_files(self, prepared: PreparedExperiment) -> None:
        for path in (prepared.patch_file, prepared.spec_file):
            if path.exists():
                path.unlink()

    def _ssh_options(self) -> list[str]:
        opts = ["-p", str(self.config.remote_port)]
        if self.config.remote_identity:
            opts.extend(["-i", self.config.remote_identity])
        opts.extend(["-o", "StrictHostKeyChecking=accept-new"])
        return opts

    def _build_remote_command(self, run_id: str, extra_env_pairs: list[tuple[str, str]]) -> str:
        env_pairs = [
            ("RUN_ID", run_id),
            ("DATA_PATH", self.config.data_path),
            ("TOKENIZER_PATH", self.config.tokenizer_path),
            ("VOCAB_SIZE", str(self.config.vocab_size)),
            ("VAL_LOSS_EVERY", str(self.config.val_loss_every)),
            ("ITERATIONS", str(self.config.iterations)),
            ("MAX_WALLCLOCK_SECONDS", str(self.config.max_wallclock_seconds)),
            *extra_env_pairs,
        ]
        env_prefix = shell_assignments(env_pairs)
        quoted_torchrun = shlex.quote(self.config.remote_torchrun)
        quoted_repo_dir = shlex.quote(self.config.remote_repo_dir)
        quoted_branch = shlex.quote(self.config.remote_branch)
        quoted_nproc = shlex.quote(str(self.config.nproc_per_node))
        return (
            "set -euo pipefail\n"
            f"cd {quoted_repo_dir}\n"
            f"git fetch origin {quoted_branch}\n"
            f"git checkout -B {quoted_branch} FETCH_HEAD\n"
            "git reset --hard FETCH_HEAD\n"
            "mkdir -p logs\n"
            f"env {env_prefix} {quoted_torchrun} --standalone --nproc_per_node={quoted_nproc} train_gpt.py "
            f"2>&1 | tee logs/{shlex.quote(run_id)}.log\n"
        )

    def _build_prep_prompt(self, *, prep_id: str, clone_dir: Path) -> str:
        spec_file = clone_dir / "controller_state" / "current_run.env"
        protocol_path = self._clone_visible_path(clone_dir, self.config.program_file)
        return "\n".join(
            [
                f"This is Parameter Golf autoresearch experiment-preparation candidate {prep_id}.",
                "",
                "Working repository clone:",
                str(clone_dir),
                "",
                f"Model tag:\n{self.config.model}",
                "",
                "Base training command for this candidate:",
                (
                    "RUN_ID=<assigned-by-controller> "
                    f"DATA_PATH={self.config.data_path} TOKENIZER_PATH={self.config.tokenizer_path} "
                    f"VOCAB_SIZE={self.config.vocab_size} VAL_LOSS_EVERY={self.config.val_loss_every} "
                    f"ITERATIONS={self.config.iterations} MAX_WALLCLOCK_SECONDS={self.config.max_wallclock_seconds} "
                    f"{self.config.remote_torchrun} --standalone --nproc_per_node={self.config.nproc_per_node} train_gpt.py"
                ),
                "",
                "Use this results file:",
                str(self.config.results_file),
                "",
                "Follow this protocol file exactly:",
                str(protocol_path),
                "",
                "Important:",
                "- Lower final roundtrip val_bpb is better.",
                "- Only prepare one candidate, then stop.",
                f"- Write the run spec to {spec_file}.",
                "- Do not run training yourself.",
                "- You may add or change experiment-specific env vars like TRAIN_SEQ_LEN, EVAL_SEQ_LEN, NUM_KV_HEADS, TIE_EMBEDDINGS, MODEL_DIM, NUM_LAYERS, or learning rates for this candidate.",
                "- Keep the dataset path, tokenizer path, entrypoint, and wallclock cap unless the experiment is explicitly about one of those.",
                "- Use exactly one git commit for the candidate change.",
                "- Keep the change self-contained and easy to cherry-pick onto a slightly newer controller state.",
            ]
        )

    def _clone_visible_path(self, clone_dir: Path, path: Path) -> Path:
        try:
            return clone_dir / path.relative_to(self.config.repo_dir)
        except ValueError:
            return path

    def _build_review_prompt(
        self,
        *,
        iteration: int,
        run_id: str,
        experiment_commit: str,
        remote_log: Path,
        best_prior_bpb: str,
    ) -> str:
        return "\n".join(
            [
                f"This is the review half of Parameter Golf autoresearch iteration {iteration}.",
                "",
                "Repository:",
                str(self.config.repo_dir),
                "",
                f"Experiment model tag:\n{self.config.model}",
                "",
                f"Review model tag:\n{self.config.review_model}",
                "",
                "The experiment run_id for this review is:",
                run_id,
                "",
                "The experiment commit for this review is:",
                experiment_commit,
                "",
                "The local fetched remote log for this review is:",
                str(remote_log),
                "",
                "The best prior kept val_bpb before this run was:",
                best_prior_bpb,
                "",
                "Use these files:",
                f"- results: {self.config.results_file}",
                f"- reviews: {self.config.reviews_file}",
                f"- review protocol: {self.config.review_program_file}",
                "",
                "Important:",
                "- Do not run training.",
                f"- Review only the latest experiment commit and the log for run_id {run_id}.",
                "- Decide whether to keep or revert the latest experiment commit.",
                "- Leave the repository clean when you are done.",
            ]
        )

    def _stream_subprocess(
        self,
        cmd: list[str],
        *,
        cwd: Path,
        prefix: str,
        raw_log_path: Path,
    ) -> int:
        raw_log_path.parent.mkdir(parents=True, exist_ok=True)
        process = subprocess.Popen(
            cmd,
            cwd=cwd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        with raw_log_path.open("w", encoding="utf-8") as raw_fh:
            for line in process.stdout:
                raw_fh.write(line)
                self.logger.stream_line(prefix, line)
        return process.wait()


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Parameter Golf autoresearch controller with a background prep queue so the "
            "next candidate can be drafted while the current remote training job is running."
        )
    )
    parser.add_argument("model", nargs="?", default=os.environ.get("MODEL", "gpt-5.4"))
    parser.add_argument("tag", nargs="?", default=os.environ.get("TAG", "pgolf"))
    parser.add_argument("hours", nargs="?", type=float, default=float(os.environ.get("HOURS", "8")))
    return parser.parse_args(argv)


def build_config(args: argparse.Namespace) -> Config:
    repo_dir = Path.cwd()
    if shutil.which("codex") is None:
        raise ControllerError("codex CLI not found in PATH")
    if subprocess.run(["git", "rev-parse", "--is-inside-work-tree"], cwd=repo_dir, check=False).returncode != 0:
        raise ControllerError("run this program from inside the parameter-golf git repo")
    repo_dir = Path(git_output(repo_dir, "rev-parse", "--show-toplevel"))
    ensure_clean_git(repo_dir)

    def env_path(name: str, default: str) -> Path:
        return resolve_repo_path(repo_dir, os.environ.get(name, default))

    config = Config(
        model=args.model,
        tag=args.tag,
        hours=args.hours,
        review_model=os.environ.get("REVIEW_MODEL", args.model),
        repo_dir=repo_dir,
        data_path=os.environ.get("DATA_PATH", str(repo_dir / "data/datasets/fineweb10B_sp1024")),
        tokenizer_path=os.environ.get("TOKENIZER_PATH", str(repo_dir / "data/tokenizers/fineweb_1024_bpe.model")),
        vocab_size=int(os.environ.get("VOCAB_SIZE", "1024")),
        nproc_per_node=int(os.environ.get("NPROC_PER_NODE", "1")),
        max_wallclock_seconds=int(os.environ.get("MAX_WALLCLOCK_SECONDS", "600")),
        val_loss_every=int(os.environ.get("VAL_LOSS_EVERY", "0")),
        iterations=int(os.environ.get("ITERATIONS", "20000")),
        remote_host=os.environ.get("REMOTE_HOST", ""),
        remote_port=int(os.environ.get("REMOTE_PORT", "22")),
        remote_repo_dir=os.environ.get("REMOTE_REPO_DIR", "/workspace/parameter-golf"),
        remote_branch=os.environ.get("REMOTE_BRANCH", "runpod-autoresearch"),
        remote_python=os.environ.get("REMOTE_PYTHON", "python3"),
        remote_torchrun=os.environ.get("REMOTE_TORCHRUN", "torchrun"),
        remote_identity=os.environ.get("REMOTE_IDENTITY", ""),
        results_file=env_path("RESULTS_FILE", "results.tsv"),
        harness_log=env_path("HARNESS_LOG", f"logs/autoresearch_{args.tag}.log"),
        program_file=env_path("PROGRAM_FILE", "autoresearch/pgolf_autoresearch_prompt.md"),
        review_program_file=env_path("REVIEW_PROGRAM_FILE", "autoresearch/pgolf_review_prompt.md"),
        reviews_file=env_path("REVIEWS_FILE", "reviews.tsv"),
        state_dir=env_path("STATE_DIR", "controller_state"),
        remote_log_dir=env_path("REMOTE_LOG_DIR", "remote_logs"),
        prep_queue_depth=max(1, int(os.environ.get("PREP_QUEUE_DEPTH", "1"))),
        prep_poll_seconds=max(1.0, float(os.environ.get("PREP_POLL_SECONDS", "5"))),
        codex_binary=os.environ.get("CODEX_BIN", "codex"),
    )
    if not config.remote_host:
        raise ControllerError("set REMOTE_HOST=user@host for the Runpod box")
    ensure_exists(config.program_file, "prompt file")
    ensure_exists(config.review_program_file, "review prompt file")
    config.state_dir.mkdir(parents=True, exist_ok=True)
    config.remote_log_dir.mkdir(parents=True, exist_ok=True)
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
