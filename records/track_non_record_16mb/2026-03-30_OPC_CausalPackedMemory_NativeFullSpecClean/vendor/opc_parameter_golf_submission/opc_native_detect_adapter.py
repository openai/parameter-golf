from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .bootstrap import add_local_sources
from .model import GolfSubmissionModel

add_local_sources()


class OpcNativeDetectAdapter:
    def __init__(self, artifact_path: str):
        self.artifact_path = str(artifact_path)
        self.model = GolfSubmissionModel.load_artifact(artifact_path)
        self.vocab_size = int(self.model.config.vocabulary_size)

    def fork(self) -> "OpcNativeDetectAdapter":
        return self

    def describe(self) -> dict[str, Any]:
        return {
            "adapter": "OpcNativeDetectAdapter",
            "artifact_path": self.artifact_path,
            "vocab_size": self.vocab_size,
            "notes": "opc-native packed-memory replay adapter for legality and replay scans.",
        }

    def score_chunk(self, tokens: np.ndarray, sample_positions: np.ndarray | None = None) -> dict[str, Any]:
        seq = np.asarray(tokens, dtype=np.int64).reshape(-1)
        if seq.size == 0 or sample_positions is None:
            return {}
        idx = np.asarray(sample_positions, dtype=np.int64).reshape(-1)
        sampled = np.zeros((idx.shape[0], self.vocab_size), dtype=np.float64)
        gold = np.zeros((idx.shape[0],), dtype=np.float64)
        for row, pos in enumerate(idx.tolist()):
            if pos < 0 or pos >= seq.size:
                raise ValueError("sample position is out of bounds")
            distribution = self.model.predictive_distribution(seq[:pos])
            sampled[row] = distribution
            gold[row] = float(np.log(max(float(distribution[int(seq[pos])]), 1e-300)))
        return {
            "sample_predictions": sampled,
            "sample_gold_logprobs": gold,
            "sample_trace": {
                "gold_logprobs": gold,
                "loss_nats": -gold,
                "weights": np.ones((idx.shape[0],), dtype=np.float64),
                "counted": np.ones((idx.shape[0],), dtype=bool),
                "path_ids": np.asarray([f"opc_native:{int(pos)}" for pos in idx.tolist()], dtype=object),
                "state_hash_before": np.asarray([f"pre:{int(pos)}" for pos in idx.tolist()], dtype=object),
                "state_hash_after": np.asarray([f"post:{int(pos)}" for pos in idx.tolist()], dtype=object),
            },
        }

    def adapt_chunk(self, tokens: np.ndarray) -> None:
        _ = tokens
        return None


def build_adapter(config: dict[str, Any]) -> OpcNativeDetectAdapter:
    artifact_path = Path(str(config["artifact_path"]))
    if not artifact_path.is_absolute():
        artifact_path = Path(__file__).resolve().parents[2] / artifact_path
    return OpcNativeDetectAdapter(str(artifact_path))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit("This module is meant to be loaded by conker-detect.")
