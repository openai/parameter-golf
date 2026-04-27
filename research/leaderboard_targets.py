from __future__ import annotations

from dataclasses import asdict, dataclass


LEADERBOARD_SNAPSHOT_DATE = "2026-04-11"


@dataclass(frozen=True)
class LeaderboardTarget:
    key: str
    name: str
    scope: str
    metric_bpb: float
    pr_number: int
    recipe: str
    as_of: str = LEADERBOARD_SNAPSHOT_DATE


LEADERBOARD_TARGETS: dict[str, LeaderboardTarget] = {
    "merged_safe": LeaderboardTarget(
        key="merged_safe",
        name="Merged Safe Bar",
        scope="merged",
        metric_bpb=1.0810,
        pr_number=1493,
        recipe="SP8192 + 3-layer recurrence + parallel residuals + QK-Gain 5.25 + legal TTT",
    ),
    "live_safe": LeaderboardTarget(
        key="live_safe",
        name="Live Safe Bar",
        scope="open_or_merged",
        metric_bpb=1.0803,
        pr_number=1532,
        recipe="SP8192 + 3-layer recurrence + parallel residuals + QK-Gain 5.25 + legal TTT + async loader",
    ),
    "live_overall": LeaderboardTarget(
        key="live_overall",
        name="Live Overall Bar",
        scope="open_or_merged",
        metric_bpb=1.077741,
        pr_number=1518,
        recipe="Wider loop + per-pass embeddings + Tap-In V6 + legal TTT",
    ),
    "live_prequant": LeaderboardTarget(
        key="live_prequant",
        name="Live Pre-Quant TTT Bar",
        scope="open_or_merged",
        metric_bpb=1.0600,
        pr_number=1487,
        recipe="SP8192 + full stack + tuned pre-quant TTT",
    ),
}


def leaderboard_snapshot_record() -> dict[str, object]:
    return {
        "as_of": LEADERBOARD_SNAPSHOT_DATE,
        "targets": {key: asdict(value) for key, value in LEADERBOARD_TARGETS.items()},
    }


def leaderboard_deltas(final_bpb: float | None) -> dict[str, float | None]:
    if final_bpb is None:
        return {
            "delta_vs_merged_bar": None,
            "delta_vs_live_safe_bar": None,
            "delta_vs_live_overall_bar": None,
            "delta_vs_live_prequant_bar": None,
        }
    return {
        "delta_vs_merged_bar": round(final_bpb - LEADERBOARD_TARGETS["merged_safe"].metric_bpb, 6),
        "delta_vs_live_safe_bar": round(final_bpb - LEADERBOARD_TARGETS["live_safe"].metric_bpb, 6),
        "delta_vs_live_overall_bar": round(final_bpb - LEADERBOARD_TARGETS["live_overall"].metric_bpb, 6),
        "delta_vs_live_prequant_bar": round(final_bpb - LEADERBOARD_TARGETS["live_prequant"].metric_bpb, 6),
    }

