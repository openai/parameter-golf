from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .config import HierarchicalSubstrateConfig
from .hierarchical import HierarchicalStateSlices


@dataclass(frozen=True)
class HierarchicalSummary:
    fast_mean: np.ndarray
    mid_mean: np.ndarray
    slow_mean: np.ndarray
    fast_energy: float
    mid_energy: float
    slow_energy: float


class HierarchicalFeatureView:
    def __init__(self, config: HierarchicalSubstrateConfig):
        self.config = config
        self._fast_slice = slice(0, config.fast_size)
        self._mid_slice = slice(config.fast_size, config.fast_size + config.mid_size)
        self._slow_slice = slice(
            config.fast_size + config.mid_size,
            config.fast_size + config.mid_size + config.slow_size,
        )

    @property
    def state_dim(self) -> int:
        return self.config.state_dim

    @property
    def predictive_dim(self) -> int:
        return (
            self.config.state_dim
            + min(self.config.fast_size, self.config.mid_size)
            + min(self.config.mid_size, self.config.slow_size)
            + 3
            + 6
        )

    @property
    def feature_dim(self) -> int:
        return 3 + 3 + self.predictive_dim

    @property
    def bank_slices(self) -> HierarchicalStateSlices:
        return HierarchicalStateSlices(fast=self._fast_slice, mid=self._mid_slice, slow=self._slow_slice)

    def split(self, state: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        state = np.asarray(state, dtype=np.float64)
        if state.ndim != 1:
            raise ValueError("state must be rank-1")
        if state.shape[0] != self.state_dim:
            raise ValueError("state does not match configured state_dim")
        return state[self._fast_slice], state[self._mid_slice], state[self._slow_slice]

    def pooled_summary(self, state: np.ndarray) -> HierarchicalSummary:
        fast, mid, slow = self.split(state)
        return HierarchicalSummary(
            fast_mean=np.array([float(np.mean(fast))], dtype=np.float64),
            mid_mean=np.array([float(np.mean(mid))], dtype=np.float64),
            slow_mean=np.array([float(np.mean(slow))], dtype=np.float64),
            fast_energy=float(np.mean(np.square(fast))),
            mid_energy=float(np.mean(np.square(mid))),
            slow_energy=float(np.mean(np.square(slow))),
        )

    @staticmethod
    def _aligned_delta(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        width = min(left.shape[0], right.shape[0])
        if width == 0:
            return np.zeros((0,), dtype=np.float64)
        return left[:width] - right[:width]

    def predictive_features(
        self,
        state: np.ndarray,
        previous_state: np.ndarray | None = None,
    ) -> np.ndarray:
        fast, mid, slow = self.split(state)
        summary = self.pooled_summary(state)

        if previous_state is None:
            prev_fast = np.zeros_like(fast)
            prev_mid = np.zeros_like(mid)
            prev_slow = np.zeros_like(slow)
        else:
            prev_fast, prev_mid, prev_slow = self.split(previous_state)

        fast_delta = fast - prev_fast
        mid_delta = mid - prev_mid
        slow_delta = slow - prev_slow

        fast_mid_surprise = self._aligned_delta(fast, self.config.fast_leak * np.tanh(mid))
        mid_slow_surprise = self._aligned_delta(mid, self.config.mid_leak * np.tanh(slow))

        return np.concatenate(
            [
                fast_delta,
                mid_delta,
                slow_delta,
                fast_mid_surprise,
                mid_slow_surprise,
                summary.fast_mean,
                summary.mid_mean,
                summary.slow_mean,
                np.array(
                    [
                        summary.fast_energy,
                        summary.mid_energy,
                        summary.slow_energy,
                        float(np.mean(np.abs(fast_delta))),
                        float(np.mean(np.abs(mid_delta))),
                        float(np.mean(np.abs(slow_delta))),
                    ],
                    dtype=np.float64,
                ),
            ]
        )

    def encode(
        self,
        state: np.ndarray,
        previous_state: np.ndarray | None = None,
    ) -> np.ndarray:
        summary = self.pooled_summary(state)
        predictive = self.predictive_features(state, previous_state=previous_state)
        return np.concatenate(
            [
                summary.fast_mean,
                summary.mid_mean,
                summary.slow_mean,
                np.array(
                    [
                        summary.fast_energy,
                        summary.mid_energy,
                        summary.slow_energy,
                    ],
                    dtype=np.float64,
                ),
                predictive,
            ]
        )


__all__ = [
    "HierarchicalFeatureView",
    "HierarchicalSummary",
]
