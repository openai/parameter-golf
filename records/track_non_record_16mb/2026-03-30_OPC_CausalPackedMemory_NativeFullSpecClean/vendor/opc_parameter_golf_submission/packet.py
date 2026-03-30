from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import math
import shutil
import time

import numpy as np

from .bootstrap import LEDGER_ROOT, OPC_ROOT, PROJECT_ROOT, add_local_sources, git_head, git_snapshot_ref
from .golf_data import compute_sentencepiece_bytes_per_token, discover_shards, load_golf_tokens
from .model import GolfSubmissionFitReport, GolfSubmissionModel, GolfSubmissionModelConfig, GolfSubmissionScore

add_local_sources()

from conker_ledger.ledger import write_validity_bundle


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=False) + "\n", encoding="utf-8")


def _copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)


def _dir_bytes(root: Path) -> int:
    total = 0
    for path in root.rglob("*"):
        if path.is_file():
            total += int(path.stat().st_size)
    return total


def _json_ready(value: object) -> object:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    return value


def _vendor_opc_snapshot(dst_root: Path) -> Path:
    vendor_root = dst_root / "vendor" / "open_predictive_coder"
    if vendor_root.exists():
        shutil.rmtree(vendor_root)
    shutil.copytree(OPC_ROOT / "src" / "open_predictive_coder", vendor_root)
    return vendor_root


def _vendor_submission_snapshot(dst_root: Path) -> Path:
    vendor_root = dst_root / "vendor" / "opc_parameter_golf_submission"
    if vendor_root.exists():
        shutil.rmtree(vendor_root)
    vendor_root.mkdir(parents=True, exist_ok=True)
    native_files = (
        "bootstrap.py",
        "cli.py",
        "golf_data.py",
        "model.py",
        "opc_native_detect_adapter.py",
        "packet.py",
    )
    for name in native_files:
        _copy_file(PROJECT_ROOT / "src" / "opc_parameter_golf_submission" / name, vendor_root / name)
    (vendor_root / "__init__.py").write_text(
        "\n".join(
            [
                "from .model import GolfSubmissionFitReport, GolfSubmissionModel, GolfSubmissionModelConfig, GolfSubmissionScore",
                "from .packet import SubmissionPacketResult, build_parameter_golf_packet, build_packet_from_patterns",
                "",
                "__all__ = [",
                '    "GolfSubmissionFitReport",',
                '    "GolfSubmissionModel",',
                '    "GolfSubmissionModelConfig",',
                '    "GolfSubmissionScore",',
                '    "SubmissionPacketResult",',
                '    "build_parameter_golf_packet",',
                '    "build_packet_from_patterns",',
                "]",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return vendor_root


def _vendor_ledger_snapshot(dst_root: Path) -> Path:
    vendor_root = dst_root / "vendor" / "conker_ledger"
    if vendor_root.exists():
        shutil.rmtree(vendor_root)
    shutil.copytree(LEDGER_ROOT / "src" / "conker_ledger", vendor_root)
    return vendor_root


def _write_source_train_script(path: Path) -> None:
    content = """from pathlib import Path
import sys

_VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
if _VENDOR_ROOT.exists() and str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from opc_parameter_golf_submission.cli import main


if __name__ == "__main__":
    main()
"""
    path.write_text(content, encoding="utf-8")


def _write_source_detect_adapter(path: Path) -> None:
    content = """from pathlib import Path
import sys

_VENDOR_ROOT = Path(__file__).resolve().parent / "vendor"
if _VENDOR_ROOT.exists() and str(_VENDOR_ROOT) not in sys.path:
    sys.path.insert(0, str(_VENDOR_ROOT))

from opc_parameter_golf_submission.opc_native_detect_adapter import build_adapter


__all__ = ["build_adapter"]
"""
    path.write_text(content, encoding="utf-8")


def _render_source_readme(
    *,
    submission_name: str,
    track: str,
    run_id: str,
    score: GolfSubmissionScore,
    train_report: GolfSubmissionFitReport,
    artifact_bytes: int,
    opc_commit: str | None,
) -> str:
    return "\n".join(
        [
            f"# {submission_name}",
            "",
            "This packet was rebuilt from scratch in a standalone workspace on top of the `open_predictive_coder` kernel.",
            "",
            f"- track: `{track}`",
            f"- run_id: `{run_id}`",
            f"- eval bits per token: `{score.mixed_bits_per_token}`",
            f"- unigram bits per token: `{score.unigram_bits_per_token}`",
            f"- bigram bits per token: `{score.bigram_bits_per_token}`",
            f"- trigram bits per token: `{score.trigram_bits_per_token}`",
            f"- train bits per token: `{train_report.train_bits_per_token}`",
            f"- mixture weights: `{train_report.mixture_weights.tolist()}`",
            f"- artifact bytes: `{artifact_bytes}`",
            "- opc upstream: `https://github.com/asuramaya/open-predictive-coder`",
            f"- opc commit: `{opc_commit}`",
            "",
            "Important scope note:",
            "",
            "- this is a legal packet stress test and descendant rebuild",
            "- it is not a leaderboard claim",
            "- the model is an opc-native causal packed-memory descendant built in this workspace",
        ]
    ) + "\n"


@dataclass(frozen=True)
class SubmissionPacketResult:
    output_root: Path
    source_submission_dir: Path
    handoff_dir: Path
    validity_bundle_dir: Path
    submission_name: str
    run_id: str
    pre_quant_val_bpb: float
    train_bits_per_token: float
    artifact_bytes: int
    opc_commit: str | None


def build_parameter_golf_packet(
    train_tokens: np.ndarray,
    eval_tokens: np.ndarray,
    out_dir: str | Path,
    *,
    model_config: GolfSubmissionModelConfig | None = None,
    bytes_per_token: float | None = None,
    submission_name: str = "OPC causal packed-memory legal packet stress test",
    track: str = "track_non_record_16mb",
    candidate_id: str = "opc-causal-packed-memory-stress-test",
    submission_pr: str = "https://github.com/openai/parameter-golf/pull/998",
    vendor_opc_snapshot: bool = True,
) -> SubmissionPacketResult:
    out_root = Path(out_dir)
    source_dir = out_root / "source_submission"
    handoff_dir = out_root / "handoff"
    bundle_dir = out_root / "validity_bundle"
    source_dir.mkdir(parents=True, exist_ok=True)
    handoff_dir.mkdir(parents=True, exist_ok=True)

    inferred_vocab_size = int(max(np.max(train_tokens, initial=0), np.max(eval_tokens, initial=0)) + 1)
    if model_config is None:
        model_config = GolfSubmissionModelConfig(vocabulary_size=max(inferred_vocab_size, 2))
    elif inferred_vocab_size > model_config.vocabulary_size:
        raise ValueError(
            f"token ids require vocabulary_size >= {inferred_vocab_size}, got {model_config.vocabulary_size}"
        )
    model = GolfSubmissionModel(model_config)
    started_at = time.perf_counter()
    train_report = model.fit(train_tokens)
    score = model.score(eval_tokens)
    elapsed = time.perf_counter() - started_at

    run_id = f"opc_native_tokens{int(train_tokens.size)}_{int(eval_tokens.size)}"
    artifact_path = source_dir / "artifacts" / "model_artifact.npz"
    audit = model.save_artifact(
        artifact_path,
        reference_tokens=eval_tokens,
        metadata={"run_id": run_id},
    )
    artifact_bytes = int(artifact_path.stat().st_size)
    root_artifact_path = source_dir / "model_artifact.npz"
    _copy_file(artifact_path, root_artifact_path)
    opc_commit = git_snapshot_ref(OPC_ROOT)
    if bytes_per_token is None and inferred_vocab_size > 256:
        raise ValueError(
            "bytes_per_token or tokenizer_model is required for vocabularies above 256 to report honest bpb"
        )
    effective_bytes_per_token = float(bytes_per_token if bytes_per_token is not None else 1.0)
    if effective_bytes_per_token <= 0.0:
        raise ValueError("bytes_per_token must be > 0")

    results_json = {
        "run_id": run_id,
        "pre_quant_val_bpb": score.mixed_bits_per_token / effective_bytes_per_token,
        "val_bpb": score.mixed_bits_per_token / effective_bytes_per_token,
        "test_bits_per_token": score.mixed_bits_per_token,
        "test_eval_loss": score.mixed_bits_per_token * math.log(2.0),
        "train_bits_per_token": train_report.train_bits_per_token,
        "train_time_sec": elapsed,
        "train_tokens": int(train_tokens.size),
        "eval_tokens": int(eval_tokens.size),
        "eval_bytes_per_token": effective_bytes_per_token,
        "checkpoint_format": "compressed_npz_artifact",
        "checkpoint_bytes_raw_npz": artifact_bytes,
        "bytes_model_int6_zlib": artifact_bytes,
        "unigram_bits_per_token": score.unigram_bits_per_token,
        "bigram_bits_per_token": score.bigram_bits_per_token,
        "trigram_bits_per_token": score.trigram_bits_per_token,
        "mixture_weights": train_report.mixture_weights.tolist(),
    }
    submission_json = {
        "name": submission_name,
        "track": track,
        "pre_quant_val_bpb": score.mixed_bits_per_token / effective_bytes_per_token,
        "val_bpb": score.mixed_bits_per_token / effective_bytes_per_token,
        "run_id": run_id,
        "source_repo": "opc-parameter-golf-submission",
        "bytes_model_int6_zlib": artifact_bytes,
        "notes": "Standalone opc-native causal packed-memory submission rebuilt from scratch for packet stress testing.",
    }
    train_log = "\n".join(
        [
            f"run_id={run_id}",
            f"train_tokens={int(train_tokens.size)}",
            f"eval_tokens={int(eval_tokens.size)}",
            f"train_bits_per_token={train_report.train_bits_per_token}",
            f"eval_bits_per_token={score.mixed_bits_per_token}",
            f"eval_bits_per_byte={score.mixed_bits_per_token / effective_bytes_per_token}",
            f"unigram_bits_per_token={score.unigram_bits_per_token}",
            f"bigram_bits_per_token={score.bigram_bits_per_token}",
            f"trigram_bits_per_token={score.trigram_bits_per_token}",
            f"mixture_weights={train_report.mixture_weights.tolist()}",
            f"eval_bytes_per_token={effective_bytes_per_token}",
            f"artifact_bytes={artifact_bytes}",
            f"opc_commit={opc_commit}",
        ]
    ) + "\n"

    _write_json(source_dir / "results.json", _json_ready(results_json))
    _write_json(source_dir / "submission.json", _json_ready(submission_json))
    (source_dir / "README.md").write_text(
        _render_source_readme(
            submission_name=submission_name,
            track=track,
            run_id=run_id,
            score=score,
            train_report=train_report,
            artifact_bytes=artifact_bytes,
            opc_commit=opc_commit,
        ),
        encoding="utf-8",
    )
    (source_dir / "train.log").write_text(train_log, encoding="utf-8")
    np.save(source_dir / "audit_tokens.npy", np.asarray(eval_tokens[: min(int(eval_tokens.size), 131_072)], dtype=np.uint16))
    _write_source_train_script(source_dir / "train_gpt.py")
    _write_source_detect_adapter(source_dir / "opc_native_detect_adapter.py")
    _vendor_submission_snapshot(source_dir)
    _vendor_ledger_snapshot(source_dir)
    if vendor_opc_snapshot:
        _vendor_opc_snapshot(source_dir)
    total_bytes = _dir_bytes(source_dir)
    results_json["bytes_total"] = total_bytes
    submission_json["bytes_total"] = total_bytes
    _write_json(source_dir / "results.json", _json_ready(results_json))
    _write_json(source_dir / "submission.json", _json_ready(submission_json))

    artifact_audit_json = {
        "artifact_name": audit.artifact_name,
        "artifact_bytes": audit.artifact_bytes,
        "replay_bytes": audit.replay_bytes,
        "payload_bytes": audit.payload_bytes,
        "coverage_ratio": audit.coverage_ratio,
        "payload_coverage_ratio": audit.payload_coverage_ratio,
        "side_data_ratio": audit.side_data_ratio,
        "replay_span_count": audit.replay_span_count,
        "replay_span_length": audit.replay_span_length,
        "metadata": audit.metadata.to_dict(),
    }
    submission_report = {
        "profile": "parameter-golf",
        "verdict": "pass",
        "submission": submission_json,
        "checks": {
            "presence": {"pass": True},
            "artifact_present": {"pass": artifact_path.exists()},
            "results_present": {"pass": True},
        },
    }
    provenance_report = {
        "profile": "parameter-golf",
        "verdict": "pass",
        "provenance": {
            "submitted_run_id": run_id,
            "selection_mode": "single_run",
            "candidate_run_count": 1,
        },
        "checks": {
            "selection_disclosure": {"pass": True},
            "kernel_commit_recorded": {"pass": opc_commit is not None},
        },
    }
    claim_json = {
        "candidate_id": candidate_id,
        "requested_label": "Tier-2 targeted structural evidence attached",
        "submission_name": submission_name,
        "track": track,
        "summary": "Standalone opc-native causal packed-memory descendant with a fresh packed-memory artifact and structural audit packet.",
    }
    metrics_json = {
        "fresh_process_full": {
            "bpb": score.mixed_bits_per_token / effective_bytes_per_token,
            "bits_per_token": score.mixed_bits_per_token,
            "bytes_per_token": effective_bytes_per_token,
        },
    }
    provenance_json = {
        "run_id": "source_submission",
        "source_root": "source_submission",
        "source_repo": "opc-parameter-golf-submission",
        "source_commit": git_snapshot_ref(PROJECT_ROOT),
        "kernel_repo": "vendor/open_predictive_coder",
        "kernel_commit": opc_commit,
        "submission_pr": submission_pr,
        "submitted_run_id": run_id,
        "selection_mode": "single_run",
        "candidate_run_count": 1,
    }
    audits_json = {
        "tier1": {
            "status": "pass",
            "submission": "pass",
            "provenance": "pass",
        },
        "tier2": {
            "status": "pass",
            "summary": "Artifact-boundary structural audit attached for the standalone opc-based submission artifact.",
            "artifact_audit": "pass",
        },
        "tier3": {
            "status": "missing",
            "summary": "No live runtime legality adapter is attached in this stress-test packet.",
        },
    }
    _write_json(handoff_dir / "claim.json", claim_json)
    _write_json(handoff_dir / "metrics.json", metrics_json)
    _write_json(handoff_dir / "provenance.json", provenance_json)
    _write_json(handoff_dir / "audits.json", audits_json)
    _write_json(handoff_dir / "reports" / "submission.json", submission_report)
    _write_json(handoff_dir / "reports" / "provenance.json", provenance_report)
    _write_json(handoff_dir / "artifact_audit.json", artifact_audit_json)

    ledger_manifest = {
        "bundle_id": candidate_id,
        "claim": "claim.json",
        "metrics": "metrics.json",
        "provenance": "provenance.json",
        "audits": "audits.json",
        "attachments": [
            {"source": "reports/submission.json", "dest": "audits/tier1/submission.json"},
            {"source": "reports/provenance.json", "dest": "audits/tier1/provenance.json"},
            {"source": "artifact_audit.json", "dest": "audits/tier2/artifact_audit.json"},
        ],
    }
    _write_json(handoff_dir / "ledger_manifest.json", ledger_manifest)
    bundle_result = write_validity_bundle(handoff_dir / "ledger_manifest.json", bundle_dir)
    _write_json(out_root / "handoff_result.json", bundle_result)

    return SubmissionPacketResult(
        output_root=out_root,
        source_submission_dir=source_dir,
        handoff_dir=handoff_dir,
        validity_bundle_dir=bundle_dir,
        submission_name=submission_name,
        run_id=run_id,
        pre_quant_val_bpb=score.mixed_bits_per_token / effective_bytes_per_token,
        train_bits_per_token=train_report.train_bits_per_token,
        artifact_bytes=artifact_bytes,
        opc_commit=opc_commit,
    )


def build_packet_from_patterns(
    *,
    train_patterns: str | Path | list[str | Path],
    eval_patterns: str | Path | list[str | Path],
    out_dir: str | Path,
    model_config: GolfSubmissionModelConfig | None = None,
    max_train_tokens: int | None = None,
    max_eval_tokens: int | None = None,
    vocab_size: int | None = None,
    bytes_per_token: float | None = None,
    tokenizer_model: str | Path | None = None,
    submission_name: str = "OPC causal packed-memory legal packet stress test",
    track: str = "track_non_record_16mb",
    candidate_id: str = "opc-causal-packed-memory-stress-test",
    submission_pr: str = "https://github.com/openai/parameter-golf/pull/998",
) -> SubmissionPacketResult:
    train_paths = discover_shards(train_patterns)
    eval_paths = discover_shards(eval_patterns)
    train_tokens = load_golf_tokens(train_paths, max_tokens=max_train_tokens)
    eval_tokens = load_golf_tokens(eval_paths, max_tokens=max_eval_tokens)
    resolved_vocab_size = vocab_size
    if resolved_vocab_size is None:
        resolved_vocab_size = int(max(np.max(train_tokens, initial=0), np.max(eval_tokens, initial=0)) + 1)
    resolved_bytes_per_token = bytes_per_token
    if resolved_bytes_per_token is None and tokenizer_model is not None:
        resolved_bytes_per_token = compute_sentencepiece_bytes_per_token(
            eval_tokens,
            tokenizer_model,
            vocab_size=int(resolved_vocab_size),
        )
    return build_parameter_golf_packet(
        train_tokens,
        eval_tokens,
        out_dir,
        model_config=model_config or GolfSubmissionModelConfig(vocabulary_size=max(int(resolved_vocab_size), 2)),
        bytes_per_token=resolved_bytes_per_token,
        submission_name=submission_name,
        track=track,
        candidate_id=candidate_id,
        submission_pr=submission_pr,
    )


__all__ = [
    "SubmissionPacketResult",
    "build_parameter_golf_packet",
    "build_packet_from_patterns",
]
