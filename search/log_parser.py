from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re


ARTIFACT_LIMIT_BYTES = 16_000_000

STEP_RE = re.compile(
    r"^step:(?P<step>\d+)/(?P<total>\d+) val_loss:(?P<loss>[-+0-9.eEnNaA]+) "
    r"val_bpb:(?P<bpb>[-+0-9.eEnNaA]+) train_time:(?P<train_time>\d+)ms",
    re.MULTILINE,
)
ROUNDTRIP_RE = re.compile(
    r"^(?P<label>(?:roundtrip|final)_int(?P<bits>6|8)\S*) val_loss:(?P<loss>[-+0-9.eEnNaA]+) "
    r"val_bpb:(?P<bpb>[-+0-9.eEnNaA]+) eval_time:(?P<eval_time>\d+)ms",
    re.MULTILINE,
)
SLIDING_WINDOW_RE = re.compile(
    r"^sliding_window val_loss:(?P<loss>[-+0-9.eEnNaA]+) "
    r"val_bpb:(?P<bpb>[-+0-9.eEnNaA]+) "
    r"seq_len:(?P<seq_len>\d+) stride:(?P<stride>\d+) "
    r"eval_time:(?P<eval_time>\d+)ms$",
    re.MULTILINE,
)
QUANT_SUMMARY_RE = re.compile(
    r"^quant_summary int8_bpb:(?P<int8_bpb>[-+0-9.eEnNaA]+) "
    r"int6_bpb:(?P<int6_bpb>[-+0-9.eEnNaA]+) int8_sz:(?P<int8_sz>\d+) int6_sz:(?P<int6_sz>\d+)$",
    re.MULTILINE,
)
NAN_RE = re.compile(r":(?:nan|NaN|NAN)\b")
STOPPING_RE = re.compile(r"^stopping_early:", re.MULTILINE)


def _to_float(raw: str) -> float:
    return float(raw)


@dataclass(frozen=True)
class StepValidation:
    step: int
    total_steps: int
    val_loss: float
    val_bpb: float
    train_time_ms: int


@dataclass(frozen=True)
class QuantizedMetric:
    label: str
    bits: int
    val_loss: float
    val_bpb: float
    eval_time_ms: int


@dataclass(frozen=True)
class SlidingWindowMetric:
    label: str
    val_loss: float
    val_bpb: float
    seq_len: int
    stride: int
    eval_time_ms: int


@dataclass(frozen=True)
class ParsedRunLog:
    log_path: str
    step_validations: tuple[StepValidation, ...]
    terminal_validation: StepValidation | None
    roundtrip_int6: QuantizedMetric | None
    roundtrip_int8: QuantizedMetric | None
    sliding_window_int6: SlidingWindowMetric | None
    int6_artifact_bytes: int | None
    int8_artifact_bytes: int | None
    has_nan: bool
    stopped_early: bool
    oversize_int6: bool
    status: str
    failure_reason: str | None

    def to_dict(self) -> dict[str, object]:
        return {
            "log_path": self.log_path,
            "step_validations": [asdict(e) for e in self.step_validations],
            "terminal_validation": asdict(self.terminal_validation) if self.terminal_validation else None,
            "roundtrip_int6": asdict(self.roundtrip_int6) if self.roundtrip_int6 else None,
            "roundtrip_int8": asdict(self.roundtrip_int8) if self.roundtrip_int8 else None,
            "sliding_window_int6": asdict(self.sliding_window_int6) if self.sliding_window_int6 else None,
            "int6_artifact_bytes": self.int6_artifact_bytes,
            "int8_artifact_bytes": self.int8_artifact_bytes,
            "has_nan": self.has_nan,
            "stopped_early": self.stopped_early,
            "oversize_int6": self.oversize_int6,
            "status": self.status,
            "failure_reason": self.failure_reason,
        }

    def objective_source(self) -> str | None:
        if self.sliding_window_int6 is not None:
            return "sliding_window_int6"
        if self.roundtrip_int6 is not None:
            return "roundtrip_int6"
        return None

    def objective_bpb(self) -> float | None:
        if self.sliding_window_int6 is not None:
            return self.sliding_window_int6.val_bpb
        if self.roundtrip_int6 is not None:
            return self.roundtrip_int6.val_bpb
        return None


def parse_run_log(path: str | Path) -> ParsedRunLog:
    log_path = Path(path)
    text = log_path.read_text(encoding="utf-8")

    validations = tuple(
        StepValidation(
            step=int(match.group("step")),
            total_steps=int(match.group("total")),
            val_loss=_to_float(match.group("loss")),
            val_bpb=_to_float(match.group("bpb")),
            train_time_ms=int(match.group("train_time")),
        )
        for match in STEP_RE.finditer(text)
    )
    terminal_validation = validations[-1] if validations else None

    roundtrip_int6 = None
    roundtrip_int8 = None
    for match in ROUNDTRIP_RE.finditer(text):
        metric = QuantizedMetric(
            label=match.group("label"),
            bits=int(match.group("bits")),
            val_loss=_to_float(match.group("loss")),
            val_bpb=_to_float(match.group("bpb")),
            eval_time_ms=int(match.group("eval_time")),
        )
        if metric.bits == 6:
            roundtrip_int6 = metric
        elif metric.bits == 8:
            roundtrip_int8 = metric

    sliding_window_int6 = None
    sliding_window_match = SLIDING_WINDOW_RE.search(text)
    if sliding_window_match:
        sliding_window_int6 = SlidingWindowMetric(
            label="sliding_window_int6",
            val_loss=_to_float(sliding_window_match.group("loss")),
            val_bpb=_to_float(sliding_window_match.group("bpb")),
            seq_len=int(sliding_window_match.group("seq_len")),
            stride=int(sliding_window_match.group("stride")),
            eval_time_ms=int(sliding_window_match.group("eval_time")),
        )

    quant_summary = QUANT_SUMMARY_RE.search(text)
    int8_artifact_bytes = int(quant_summary.group("int8_sz")) if quant_summary else None
    int6_artifact_bytes = int(quant_summary.group("int6_sz")) if quant_summary else None
    if quant_summary and roundtrip_int6 is None:
        roundtrip_int6 = QuantizedMetric(
            label="quant_summary_int6",
            bits=6,
            val_loss=terminal_validation.val_loss if terminal_validation else float("nan"),
            val_bpb=_to_float(quant_summary.group("int6_bpb")),
            eval_time_ms=0,
        )
    if quant_summary and roundtrip_int8 is None:
        roundtrip_int8 = QuantizedMetric(
            label="quant_summary_int8",
            bits=8,
            val_loss=terminal_validation.val_loss if terminal_validation else float("nan"),
            val_bpb=_to_float(quant_summary.group("int8_bpb")),
            eval_time_ms=0,
        )

    has_nan = bool(NAN_RE.search(text))
    stopped_early = bool(STOPPING_RE.search(text))
    oversize_int6 = int6_artifact_bytes is not None and int6_artifact_bytes >= ARTIFACT_LIMIT_BYTES

    failure_reason = None
    status = "completed"
    if oversize_int6:
        status = "oversize"
        failure_reason = f"int6 artifact {int6_artifact_bytes} >= {ARTIFACT_LIMIT_BYTES}"
    elif roundtrip_int6 is None:
        status = "failed"
        failure_reason = "missing final int6 metric"
    elif has_nan:
        status = "failed"
        failure_reason = "nan detected in log"
    elif terminal_validation is None:
        status = "failed"
        failure_reason = "missing validation metrics"

    return ParsedRunLog(
        log_path=str(log_path),
        step_validations=validations,
        terminal_validation=terminal_validation,
        roundtrip_int6=roundtrip_int6,
        roundtrip_int8=roundtrip_int8,
        sliding_window_int6=sliding_window_int6,
        int6_artifact_bytes=int6_artifact_bytes,
        int8_artifact_bytes=int8_artifact_bytes,
        has_nan=has_nan,
        stopped_early=stopped_early,
        oversize_int6=oversize_int6,
        status=status,
        failure_reason=failure_reason,
    )
