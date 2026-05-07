from __future__ import annotations

import numpy as np

from .linear_memory import LinearMemorySubstrate


class LinearMemoryFeatureView:
    def __init__(self, substrate: LinearMemorySubstrate):
        self.substrate = substrate
        self.feature_dim = substrate.state_dim + (3 * substrate.bank_count)

    def encode(self, state: np.ndarray, previous_state: np.ndarray | None = None) -> np.ndarray:
        banks = self.substrate.state_view(state)
        means = np.mean(banks, axis=1)
        energies = np.mean(np.square(banks), axis=1)
        if previous_state is None:
            drift = np.zeros(self.substrate.bank_count, dtype=np.float64)
        else:
            previous_banks = self.substrate.state_view(previous_state)
            drift = np.mean(np.abs(banks - previous_banks), axis=1)
        return np.concatenate([state, means, energies, drift])


__all__ = [
    "LinearMemoryFeatureView",
]
