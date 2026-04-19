from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

from experiment_env import guard_experiment_env

guard_experiment_env(script_name="build_three_layer_prototype_data", require_torch=True)

import torch
from prototype.model import RoutedTinyLM
from run_understanding_gate_probe_eval import load_sentencepiece, make_config


DEFAULT_INPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_prototype_seed_samples.jsonl")
DEFAULT_OUTPUT = Path("/Users/seryn/Documents/Parameter_Golf/repo/three_layer_prototype_builder_run1")

ErrorMode = Literal["none", "premature_close", "relation_drift", "fit_mismatch", "order_break"]
UncertaintyBand = Literal["low", "mid", "high"]
TopGapBand = Literal["wide", "narrow"]
InterventionEffect = Literal["none", "short_redirect", "stable_redirect", "still_drifting"]
LocalSite = Literal["break", "join", "hinge", "close", "fit", "order", "unknown"]
MondayMove = Literal["no", "too_fast", "where", "break", "hinge", "go_back", "leave_open", "not_enough"]
SerynAudit = Literal[
    "too_early",
    "too_weak",
    "too_much",
    "held",
    "missed",
    "right_place",
    "wrong_place",
    "still_drifting",
]


@dataclass(frozen=True)
class WorldSample:
    sample_id: str
    prefix: str
    gold_continuation: str


@dataclass(frozen=True)
class CaelTrace:
    error_mode: ErrorMode
    error_persist_steps: int
    uncertainty_band: UncertaintyBand
    top_gap_band: TopGapBand
    intervention_effect: InterventionEffect
    local_site: LocalSite


@dataclass(frozen=True)
class CaelRecord:
    sample_id: str
    prefix: str
    gold_continuation: str
    cael_continuation: str
    cael_trace: CaelTrace


@dataclass(frozen=True)
class MondayRecord:
    sample_id: str
    prefix: str
    cael_continuation: str
    cael_trace: CaelTrace
    target_move: MondayMove


@dataclass(frozen=True)
class SerynRecord:
    sample_id: str
    prefix: str
    cael_trace: CaelTrace
    monday_move: MondayMove
    post_monday_continuation: str
    target_audit: SerynAudit


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build a very small three-layer prototype dataset.")
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--cael-source", choices=("mock_v0", "real_cael"), default="mock_v0")
    parser.add_argument("--cael-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--tokenizer-model",
        type=Path,
        default=Path("/Users/seryn/Documents/Parameter_Golf/repo/data/tokenizers/fineweb_1024_bpe.model"),
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--d-model", type=int, default=256)
    parser.add_argument("--num-heads", type=int, default=2)
    parser.add_argument("--head-dim", type=int, default=32)
    parser.add_argument("--shared-repeats", type=int, default=2)
    parser.add_argument("--router-temperature", type=float, default=1.5)
    parser.add_argument("--op3-hidden-dim", type=int, default=96)
    parser.add_argument("--binary-forward", action="store_true", default=True)
    return parser


def load_world_samples(path: Path) -> list[WorldSample]:
    rows: list[WorldSample] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            payload = json.loads(line)
            rows.append(
                WorldSample(
                    sample_id=str(payload["sample_id"]),
                    prefix=str(payload["prefix"]).strip(),
                    gold_continuation=str(payload["gold_continuation"]).strip(),
                )
            )
    if not rows:
        raise ValueError(f"No world samples found in {path}")
    return rows


def infer_expected_error_mode(sample: WorldSample) -> ErrorMode:
    gold = sample.gold_continuation.lower()
    prefix = sample.prefix.lower()
    joined = f"{prefix} {gold}"
    if any(token in joined for token in ("1, 2, 4", "3, 6, 9", "pattern", "order")):
        return "order_break"
    if "fit" in joined:
        return "fit_mismatch"
    if "and—" in joined or "and-" in joined or "open" in joined or "scene" in joined:
        return "premature_close"
    if any(token in joined for token in ("relation", "hinge", "join", "missing part")):
        return "relation_drift"
    return "none"


def infer_error_mode(sample: WorldSample, cael_continuation: str) -> ErrorMode:
    gold = sample.gold_continuation.lower().strip()
    cael = cael_continuation.lower().strip()
    if gold == cael:
        return "none"
    return infer_expected_error_mode(sample)


def infer_local_site(sample: WorldSample, error_mode: ErrorMode) -> LocalSite:
    gold = sample.gold_continuation.lower()
    prefix = sample.prefix.lower()
    joined = f"{prefix} {gold}"
    if "hinge" in joined:
        return "hinge"
    if "join" in joined:
        return "join"
    if error_mode == "premature_close":
        return "close"
    if error_mode == "fit_mismatch":
        return "fit"
    if error_mode == "order_break":
        return "order"
    if "break" in joined:
        return "break"
    return "unknown"


def derive_uncertainty_band(error_mode: ErrorMode) -> UncertaintyBand:
    if error_mode in {"premature_close", "relation_drift"}:
        return "high"
    if error_mode in {"fit_mismatch", "order_break"}:
        return "mid"
    return "low"


def derive_top_gap_band(error_mode: ErrorMode) -> TopGapBand:
    if error_mode in {"premature_close", "relation_drift"}:
        return "narrow"
    return "wide"


def derive_error_persist_steps(error_mode: ErrorMode) -> int:
    if error_mode == "none":
        return 0
    if error_mode in {"fit_mismatch", "order_break"}:
        return 2
    return 3


def mock_cael_continuation(sample: WorldSample, error_mode: ErrorMode) -> str:
    gold = sample.gold_continuation.strip()
    if error_mode == "premature_close":
        return "It closes here."
    if error_mode == "relation_drift":
        return "The words still work, so it holds."
    if error_mode == "fit_mismatch":
        return "It fits."
    if error_mode == "order_break":
        return "It still holds."
    return gold


def choose_device(device_name: str) -> torch.device:
    if device_name == "mps" and torch.backends.mps.is_available():
        return torch.device("mps")
    if device_name == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def entropy_band(value: float) -> UncertaintyBand:
    if value >= 5.0:
        return "high"
    if value >= 2.5:
        return "mid"
    return "low"


def gap_band(value: float) -> TopGapBand:
    return "wide" if value >= 2.0 else "narrow"


def load_real_cael_model(args: argparse.Namespace, device: torch.device) -> RoutedTinyLM:
    if args.cael_checkpoint is None:
        raise ValueError("--cael-source real_cael requires --cael-checkpoint")
    if not args.cael_checkpoint.exists():
        raise FileNotFoundError(f"Cael checkpoint not found: {args.cael_checkpoint}")
    config = make_config(args)
    model = RoutedTinyLM(config)
    state = torch.load(args.cael_checkpoint, map_location="cpu")
    if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
        state = state["state_dict"]
    if not isinstance(state, dict):
        raise TypeError(f"Unsupported checkpoint payload type: {type(state)!r}")
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def real_cael_continuation(
    sample: WorldSample,
    *,
    model: RoutedTinyLM,
    sp,
    device: torch.device,
    max_new_tokens: int,
    binary_forward: bool,
) -> tuple[str, UncertaintyBand, TopGapBand]:
    token_ids = [int(sp.bos_id()), *sp.encode(sample.prefix, out_type=int)]
    input_ids = torch.tensor([token_ids], dtype=torch.long, device=device)
    entropies: list[float] = []
    gaps: list[float] = []
    with torch.inference_mode():
        for _ in range(max_new_tokens):
            context = input_ids[:, -256:]
            logits, _stats = model(context, target_ids=None, binary_forward=binary_forward)
            next_logits = logits[:, -1, :].float()
            probs = torch.softmax(next_logits, dim=-1)
            entropy = -(probs * probs.clamp_min(1e-9).log()).sum(dim=-1)
            top2 = torch.topk(next_logits, k=2, dim=-1).values
            entropies.append(float(entropy.item()))
            gaps.append(float((top2[:, 0] - top2[:, 1]).item()))
            next_id = next_logits.argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
    full_text = sp.decode(input_ids[0].tolist())
    continuation = full_text[len(sample.prefix) :].strip() if full_text.startswith(sample.prefix) else full_text.strip()
    continuation = continuation or sample.gold_continuation.strip()
    mean_entropy = sum(entropies) / max(len(entropies), 1)
    mean_gap = sum(gaps) / max(len(gaps), 1)
    return continuation, entropy_band(mean_entropy), gap_band(mean_gap)


def derive_monday_move(trace: CaelTrace) -> MondayMove:
    if trace.error_mode == "premature_close":
        return "leave_open" if trace.error_persist_steps <= 2 else "too_fast"
    if trace.error_mode == "relation_drift":
        if trace.local_site in {"hinge", "join"}:
            return "hinge"
        return "where"
    if trace.error_mode == "fit_mismatch":
        return "break" if trace.top_gap_band == "wide" else "where"
    if trace.error_mode == "order_break":
        return "go_back" if trace.error_persist_steps >= 2 else "break"
    return "no"


def mock_post_monday_continuation(sample: WorldSample, move: MondayMove) -> str:
    gold = sample.gold_continuation.strip()
    if move in {"go_back", "leave_open", "hinge", "break"}:
        return gold
    if move == "where":
        return gold if "where" not in gold.lower() else gold
    if move == "too_fast":
        return "It does not close yet."
    if move == "not_enough":
        return "It changes, but does not hold yet."
    return gold


def mismatch_score(text: str, gold: str) -> int:
    a = {token.strip(".,;:!?").lower() for token in text.split() if token.strip()}
    b = {token.strip(".,;:!?").lower() for token in gold.split() if token.strip()}
    return len(a.symmetric_difference(b))


def derive_intervention_effect(
    *,
    cael_continuation: str,
    post_monday_continuation: str,
    gold_continuation: str,
) -> InterventionEffect:
    before = mismatch_score(cael_continuation, gold_continuation)
    after = mismatch_score(post_monday_continuation, gold_continuation)
    if after >= before and post_monday_continuation.strip() == cael_continuation.strip():
        return "none"
    if after == 0:
        return "stable_redirect"
    if after < before:
        return "short_redirect"
    return "still_drifting"


def derive_seryn_audit(trace: CaelTrace, monday_move: MondayMove) -> SerynAudit:
    if trace.error_mode == "none" and monday_move == "no":
        return "held"
    if trace.error_mode != "none" and monday_move == "no":
        return "missed"
    if trace.intervention_effect == "stable_redirect":
        if monday_move in {"hinge", "break", "go_back", "leave_open"}:
            return "right_place"
        return "held"
    if trace.intervention_effect == "short_redirect":
        return "too_weak"
    if trace.intervention_effect == "still_drifting":
        if monday_move in {"where", "not_enough"}:
            return "still_drifting"
        return "too_much"
    if monday_move in {"hinge", "break", "go_back", "leave_open", "where"}:
        return "missed"
    return "wrong_place"


def build_records(
    sample: WorldSample,
    *,
    cael_source: str,
    real_model: RoutedTinyLM | None,
    sp,
    device: torch.device,
    max_new_tokens: int,
    binary_forward: bool,
) -> tuple[CaelRecord, MondayRecord, SerynRecord]:
    if cael_source == "real_cael":
        if real_model is None or sp is None:
            raise RuntimeError("real_cael source requires loaded model and tokenizer")
        cael_continuation, uncertainty_band, top_gap_band = real_cael_continuation(
            sample,
            model=real_model,
            sp=sp,
            device=device,
            max_new_tokens=max_new_tokens,
            binary_forward=binary_forward,
        )
        error_mode = infer_error_mode(sample, cael_continuation)
    else:
        provisional_mode = infer_expected_error_mode(sample)
        cael_continuation = mock_cael_continuation(sample, provisional_mode)
        error_mode = infer_error_mode(sample, cael_continuation)
        uncertainty_band = derive_uncertainty_band(error_mode)
        top_gap_band = derive_top_gap_band(error_mode)
    local_site = infer_local_site(sample, error_mode)
    trace = CaelTrace(
        error_mode=error_mode,
        error_persist_steps=derive_error_persist_steps(error_mode),
        uncertainty_band=uncertainty_band,
        top_gap_band=top_gap_band,
        intervention_effect="none",
        local_site=local_site,
    )
    monday_move = derive_monday_move(trace)
    post_monday_continuation = mock_post_monday_continuation(sample, monday_move)
    filled_trace = CaelTrace(
        error_mode=trace.error_mode,
        error_persist_steps=trace.error_persist_steps,
        uncertainty_band=trace.uncertainty_band,
        top_gap_band=trace.top_gap_band,
        intervention_effect=derive_intervention_effect(
            cael_continuation=cael_continuation,
            post_monday_continuation=post_monday_continuation,
            gold_continuation=sample.gold_continuation,
        ),
        local_site=trace.local_site,
    )
    cael_record = CaelRecord(
        sample_id=sample.sample_id,
        prefix=sample.prefix,
        gold_continuation=sample.gold_continuation,
        cael_continuation=cael_continuation,
        cael_trace=filled_trace,
    )
    monday_record = MondayRecord(
        sample_id=sample.sample_id,
        prefix=sample.prefix,
        cael_continuation=cael_continuation,
        cael_trace=filled_trace,
        target_move=monday_move,
    )
    seryn_record = SerynRecord(
        sample_id=sample.sample_id,
        prefix=sample.prefix,
        cael_trace=filled_trace,
        monday_move=monday_move,
        post_monday_continuation=post_monday_continuation,
        target_audit=derive_seryn_audit(filled_trace, monday_move),
    )
    return cael_record, monday_record, seryn_record


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_inspection(
    path: Path,
    *,
    cael_source: str,
    world_samples: list[WorldSample],
    cael_records: list[CaelRecord],
    monday_records: list[MondayRecord],
    seryn_records: list[SerynRecord],
) -> None:
    lines = [
        "# Three-layer prototype builder inspection",
        "",
        f"- sample_count: {len(world_samples)}",
        f"- cael_source: {cael_source}",
        "- trace_mode: rule-derived schema v0",
        "- monday_mode: policy(error trace)",
        "- seryn_mode: audit(governance effect)",
        "",
    ]
    for sample, cael, monday, seryn in zip(world_samples, cael_records, monday_records, seryn_records):
        lines.extend(
            [
                f"## {sample.sample_id}",
                "",
                f"- prefix: `{sample.prefix}`",
                f"- gold: `{sample.gold_continuation}`",
                f"- cael_continuation: `{cael.cael_continuation}`",
                f"- cael_trace: `{json.dumps(asdict(cael.cael_trace), ensure_ascii=False)}`",
                f"- monday_target_move: `{monday.target_move}`",
                f"- post_monday_continuation: `{seryn.post_monday_continuation}`",
                f"- seryn_target_audit: `{seryn.target_audit}`",
                "",
            ]
        )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = build_parser().parse_args()
    world_samples = load_world_samples(args.input)
    device = choose_device(args.device)
    sp = None
    real_model = None
    if args.cael_source == "real_cael":
        sp = load_sentencepiece(args.tokenizer_model)
        real_model = load_real_cael_model(args, device)
    cael_records: list[CaelRecord] = []
    monday_records: list[MondayRecord] = []
    seryn_records: list[SerynRecord] = []
    for sample in world_samples:
        cael, monday, seryn = build_records(
            sample,
            cael_source=args.cael_source,
            real_model=real_model,
            sp=sp,
            device=device,
            max_new_tokens=args.max_new_tokens,
            binary_forward=args.binary_forward,
        )
        cael_records.append(cael)
        monday_records.append(monday)
        seryn_records.append(seryn)

    output_root = args.output_root
    output_root.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_root / "world_samples.jsonl", [asdict(row) for row in world_samples])
    write_jsonl(output_root / "cael_records.jsonl", [asdict(row) for row in cael_records])
    write_jsonl(output_root / "monday_records.jsonl", [asdict(row) for row in monday_records])
    write_jsonl(output_root / "seryn_records.jsonl", [asdict(row) for row in seryn_records])
    write_inspection(
        output_root / "inspection.md",
        cael_source=args.cael_source,
        world_samples=world_samples,
        cael_records=cael_records,
        monday_records=monday_records,
        seryn_records=seryn_records,
    )
    manifest = {
        "input_path": str(args.input),
        "output_root": str(output_root),
        "sample_count": len(world_samples),
        "cael_source": args.cael_source,
        "cael_checkpoint": str(args.cael_checkpoint) if args.cael_checkpoint is not None else None,
        "device": str(device),
        "trace_fields": [
            "error_mode",
            "error_persist_steps",
            "uncertainty_band",
            "top_gap_band",
            "intervention_effect",
            "local_site",
        ],
        "monday_vocab": ["no", "too_fast", "where", "break", "hinge", "go_back", "leave_open", "not_enough"],
        "seryn_vocab": [
            "too_early",
            "too_weak",
            "too_much",
            "held",
            "missed",
            "right_place",
            "wrong_place",
            "still_drifting",
        ],
    }
    (output_root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
