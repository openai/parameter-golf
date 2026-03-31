from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from controller_dsl import ControllerSpec, GateSpec, PulseSpec, SnapshotSpec


@dataclass(frozen=True)
class ControllerCandidate:
    key: str
    name: str
    family: str
    spec: ControllerSpec
    broken_invariant: str
    mechanism: str
    why: str
    dominant_metric: str
    expected_impact: dict[str, Any]
    expected_horizon: str
    early_signal: str
    failure_mode: str
    kill_rule: str
    validates: str
    falsifies: str
    notes: tuple[str, ...] = ()
    code_burden: str = "patchable"
    role: str = "candidate"
    parent_slot: str = "R0A"
    compare_to: str = "R0A"

    def to_slot(self, slot_id: str) -> dict[str, Any]:
        return {
            "slot": slot_id,
            "name": self.name,
            "role": self.role,
            "implementation_state": "ready",
            "parent": self.parent_slot,
            "compare_to": self.compare_to,
            "patches": ["state_controller"],
            "family": self.family,
            "lane": "controller",
            "broken_invariant": self.broken_invariant,
            "mechanism": self.mechanism,
            "why": self.why,
            "dominant_metric": self.dominant_metric,
            "expected_impact": self.expected_impact,
            "expected_horizon": self.expected_horizon,
            "early_signal": self.early_signal,
            "failure_mode": self.failure_mode,
            "kill_rule": self.kill_rule,
            "code_burden": self.code_burden,
            "validates": self.validates,
            "falsifies": self.falsifies,
            "env": {
                "CTRL_ENABLE": "1",
                "CTRL_SPEC_JSON": self.spec.to_env_json(),
            },
            "notes": list(self.notes),
        }

    def to_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "name": self.name,
            "family": self.family,
            "spec": self.spec.to_dict(),
            "broken_invariant": self.broken_invariant,
            "mechanism": self.mechanism,
            "why": self.why,
            "dominant_metric": self.dominant_metric,
            "expected_impact": self.expected_impact,
            "expected_horizon": self.expected_horizon,
            "early_signal": self.early_signal,
            "failure_mode": self.failure_mode,
            "kill_rule": self.kill_rule,
            "validates": self.validates,
            "falsifies": self.falsifies,
            "notes": list(self.notes),
            "code_burden": self.code_burden,
            "role": self.role,
            "parent_slot": self.parent_slot,
            "compare_to": self.compare_to,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ControllerCandidate":
        return cls(
            key=str(data["key"]),
            name=str(data["name"]),
            family=str(data["family"]),
            spec=ControllerSpec.from_dict(data["spec"]),
            broken_invariant=str(data["broken_invariant"]),
            mechanism=str(data["mechanism"]),
            why=str(data["why"]),
            dominant_metric=str(data["dominant_metric"]),
            expected_impact=dict(data["expected_impact"]),
            expected_horizon=str(data["expected_horizon"]),
            early_signal=str(data["early_signal"]),
            failure_mode=str(data["failure_mode"]),
            kill_rule=str(data["kill_rule"]),
            validates=str(data["validates"]),
            falsifies=str(data["falsifies"]),
            notes=tuple(data.get("notes", [])),
            code_burden=str(data.get("code_burden", "patchable")),
            role=str(data.get("role", "candidate")),
            parent_slot=str(data.get("parent_slot", "R0A")),
            compare_to=str(data.get("compare_to", "R0A")),
        )


def seed_candidates() -> list[ControllerCandidate]:
    return [
        ControllerCandidate(
            key="H201",
            name="late_deploy_gate",
            family="LATE_DEPLOY_GATE",
            spec=ControllerSpec(
                phase_boundaries=(0.58, 0.82),
                phase_defaults={
                    "early": {"ema_decay": 0.997},
                    "mid": {"ema_decay": 0.9975, "checkpoint_capture_rate": 200},
                    "late": {
                        "ema_decay": 0.999,
                        "checkpoint_capture_rate": 100,
                        "checkpoint_selection_mode": "ema",
                        "export_surrogate_weight": 0.0005,
                        "qat_alpha": 0.35,
                    },
                },
                snapshot=SnapshotSpec(every=100, start_frac=0.76, last_k=6, score="deployed", mode="ema"),
            ),
            broken_invariant="One deploy-alignment law should apply for the whole run.",
            mechanism="Keep early and mid training mostly static, then activate late export-surrogate pressure plus a soft QAT ramp and denser late snapshot capture.",
            why="Deploy-facing pressure is likely harmful early and valuable late. The controller should internalize deploy alignment only after the representation is mostly formed.",
            dominant_metric="post_quant_bpb",
            expected_impact={"magnitude": "medium", "bpb_range": [0.004, 0.012]},
            expected_horizon="screen_to_decision",
            early_signal="Late-phase logs should show nonzero qat_alpha and snapshot capture without a large step-time collapse.",
            failure_mode="Late deploy pressure still arrives too early or is too weak to matter.",
            kill_rule="No post_quant_bpb improvement vs control by screen, or step_avg_ms regresses by >12% without a score win.",
            validates="Late gated deploy alignment beats the static default.",
            falsifies="Static late-QAT behavior is already sufficient and controller timing does not matter.",
            notes=("Lead family H201.",),
        ),
        ControllerCandidate(
            key="H202",
            name="best_state_deployed",
            family="BEST_STATE_CONTROLLER",
            spec=ControllerSpec(
                phase_boundaries=(0.55, 0.8),
                phase_defaults={
                    "early": {},
                    "mid": {"checkpoint_capture_rate": 150},
                    "late": {
                        "ema_decay": 0.9985,
                        "checkpoint_capture_rate": 75,
                        "checkpoint_selection_mode": "best_deployed_last_k",
                    },
                },
                snapshot=SnapshotSpec(every=75, start_frac=0.7, last_k=8, score="deployed", mode="best_deployed_last_k"),
            ),
            broken_invariant="The last checkpoint is the correct export target.",
            mechanism="Capture late snapshots and choose the best deployed-state candidate from the last-k states instead of always exporting the last state.",
            why="The best compressed artifact often does not coincide with the latest raw checkpoint. This controller attacks train-to-deploy mismatch directly.",
            dominant_metric="post_quant_bpb",
            expected_impact={"magnitude": "medium", "bpb_range": [0.003, 0.01]},
            expected_horizon="decision",
            early_signal="Snapshot scoring logs should show a chosen checkpoint different from raw_final or ema_final.",
            failure_mode="The late-state window is too narrow, or the best deployed state still coincides with EMA or the final step.",
            kill_rule="By decision, chosen snapshot is always final/EMA and post_quant_bpb does not beat control.",
            validates="Export-state selection is a real first-order mechanism.",
            falsifies="The static end state is already the best deployable state.",
            notes=("Lead family H202.",),
        ),
        ControllerCandidate(
            key="H202B",
            name="best_state_raw",
            family="BEST_STATE_CONTROLLER_RAW",
            spec=ControllerSpec(
                phase_boundaries=(0.55, 0.8),
                phase_defaults={
                    "early": {},
                    "mid": {"checkpoint_capture_rate": 150},
                    "late": {
                        "ema_decay": 0.9985,
                        "checkpoint_capture_rate": 75,
                        "checkpoint_selection_mode": "best_raw_last_k",
                    },
                },
                snapshot=SnapshotSpec(every=75, start_frac=0.7, last_k=8, score="raw", mode="best_raw_last_k"),
            ),
            broken_invariant="If checkpoint selection matters, the deployed criterion and raw criterion should pick the same late state.",
            mechanism="Capture the same late snapshots as H202 but select by raw validation instead of deployed validation.",
            why="This is a matched falsification of H202. If raw and deployed selection tie, then checkpoint choice matters less than the objective used to score it.",
            dominant_metric="post_quant_bpb",
            expected_impact={"magnitude": "small_to_medium", "bpb_range": [0.001, 0.006]},
            expected_horizon="decision",
            early_signal="Snapshot selection logs should diverge from H202 if the deploy criterion really matters.",
            failure_mode="Raw and deployed ranking of the late snapshots are effectively identical.",
            kill_rule="No difference from H202 in chosen snapshot or final post_quant_bpb.",
            validates="Deploy-scored checkpoint selection is meaningfully different from raw-scored selection.",
            falsifies="The selection criterion is noise; only having EMA snapshots matters.",
            notes=("Support variant for H202.",),
        ),
        ControllerCandidate(
            key="H204",
            name="family_split_warmdown",
            family="FAMILY_SPLIT_WARMDOWN",
            spec=ControllerSpec(
                phase_boundaries=(0.62, 0.84),
                phase_defaults={
                    "early": {},
                    "mid": {},
                    "late": {
                        "ema_decay": 0.9985,
                        "token_lr_mult": 0.0,
                        "head_lr_mult": 0.45,
                        "scalar_lr_mult": 0.6,
                        "matrix_lr_mult": 1.1,
                        "checkpoint_capture_rate": 100,
                        "checkpoint_selection_mode": "ema",
                    },
                },
                snapshot=SnapshotSpec(every=100, start_frac=0.78, last_k=6, score="deployed", mode="ema"),
            ),
            broken_invariant="All parameter families should follow the same late adaptation law.",
            mechanism="During warmdown, freeze token updates, damp head/scalar movement, and let matrix trunk updates continue slightly more aggressively.",
            why="Embeddings, head, scalars, and trunk matrices likely do not want the same late dynamics. This controller changes the late basin shape instead of changing the whole run.",
            dominant_metric="post_quant_bpb",
            expected_impact={"magnitude": "small_to_medium", "bpb_range": [0.002, 0.008]},
            expected_horizon="decision",
            early_signal="Step cost should stay neutral while late deployed score improves. Raw loss may move little or even worsen slightly.",
            failure_mode="Selective freezing hurts fit more than it helps deploy robustness.",
            kill_rule="No post_quant gain by decision, or clear raw-collapse in screen with no deploy compensation.",
            validates="Parameter-family splitting matters in the late phase.",
            falsifies="Uniform late adaptation is already near-optimal for this stack.",
            notes=("Lead family H204.",),
        ),
        ControllerCandidate(
            key="H205",
            name="alternating_objective",
            family="ALTERNATING_OBJECTIVE",
            spec=ControllerSpec(
                phase_boundaries=(0.6, 0.8),
                phase_defaults={
                    "early": {},
                    "mid": {},
                    "late": {"ema_decay": 0.9985, "checkpoint_capture_rate": 100},
                },
                snapshot=SnapshotSpec(every=100, start_frac=0.74, last_k=6, score="deployed", mode="ema"),
                pulse=PulseSpec(every=8, late_start=0.72, mode="export_surrogate", weight=0.0007),
            ),
            broken_invariant="One blended objective should be applied every step.",
            mechanism="Keep the main loss clean on most steps, then inject sparse late export-surrogate pulses instead of a globally blended deploy term.",
            why="If deploy alignment is helpful late but poisonous when applied continuously, sparse pulses should recover the benefit at lower throughput cost.",
            dominant_metric="post_quant_bpb",
            expected_impact={"magnitude": "medium", "bpb_range": [0.004, 0.015]},
            expected_horizon="screen_to_decision",
            early_signal="Pulse logs should appear late, step cost should remain close to control, and post_quant should improve more than pre_quant.",
            failure_mode="Pulses are either too weak to matter or too expensive relative to the step budget.",
            kill_rule="No post_quant win by screen, or step_avg_ms regresses by >10% without deployed gain.",
            validates="Alternating late objective pressure beats static always-on pressure.",
            falsifies="Sparse pulses are too weak; only continuous objective shaping works.",
            notes=("Lead family H205.",),
        ),
        ControllerCandidate(
            key="H206",
            name="systems_aware_late_deploy",
            family="SYSTEMS_AWARE_CONTROLLER",
            spec=ControllerSpec(
                phase_boundaries=(0.58, 0.82),
                phase_defaults={
                    "early": {"ema_decay": 0.997},
                    "mid": {"ema_decay": 0.9975, "checkpoint_capture_rate": 200},
                    "late": {
                        "ema_decay": 0.999,
                        "checkpoint_capture_rate": 100,
                        "checkpoint_selection_mode": "ema",
                        "export_surrogate_weight": 0.0005,
                        "qat_alpha": 0.35,
                    },
                },
                gates=(
                    GateSpec("step_avg_ms", ">", 650.0, "export_surrogate_weight", 0.0),
                    GateSpec("step_avg_ms", ">", 650.0, "qat_alpha", 0.15),
                ),
                snapshot=SnapshotSpec(every=100, start_frac=0.76, last_k=6, score="deployed", mode="ema"),
            ),
            broken_invariant="Heavy late controls should remain enabled regardless of wallclock cost.",
            mechanism="Use the same late deploy gate as H201, but automatically disable late export pressure if step_avg_ms drifts too high.",
            why="Some controllers fail only because they cost too much step time. This tests whether throughput-aware throttling preserves the win.",
            dominant_metric="post_quant_bpb",
            expected_impact={"magnitude": "small", "bpb_range": [0.0, 0.004]},
            expected_horizon="screen",
            early_signal="Gate-trigger logs should show step-cost-based throttling. Step cost should stay closer to control than H201.",
            failure_mode="The throughput guard disables the very pressure that made H201 useful.",
            kill_rule="It is slower than H201 and no better on post_quant_bpb, or it collapses to control behavior entirely.",
            validates="Systems-aware gating preserves late deploy wins under the wallclock cap.",
            falsifies="The best late deploy policy is not throughput-limited in practice.",
            notes=("Support family H206.",),
        ),
    ]


def candidate_by_key(key: str) -> ControllerCandidate:
    for candidate in seed_candidates():
        if candidate.key == key:
            return candidate
    raise KeyError(key)


def child_candidate(
    parent: ControllerCandidate,
    child_key: str,
    child_name: str,
    spec: ControllerSpec,
    mutation_note: str,
) -> ControllerCandidate:
    return replace(
        parent,
        key=child_key,
        name=child_name,
        family=f"{parent.family}_CHILD",
        spec=spec,
        mechanism=f"{parent.mechanism} Child optimization: {mutation_note}.",
        why=f"Optimizes {parent.key}. {mutation_note} {parent.why}",
        expected_horizon="screen_to_decision",
        early_signal=f"Child should beat parent or control on deployed score if {mutation_note.lower()} is a real improvement.",
        failure_mode=f"Mutation does not materially improve {parent.key}, or it preserves the mechanism while adding noise.",
        kill_rule="No deployed-score improvement vs the parent family’s control or no meaningful behavioral difference in logs.",
        notes=parent.notes + (f"child_of:{parent.key}", mutation_note),
    )
