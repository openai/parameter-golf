from __future__ import annotations

from typing import Protocol

import numpy as np

from .reservoir import EchoStateReservoir, build_recurrent_matrix, spectral_radius


class TokenSubstrate(Protocol):
    @property
    def state_dim(self) -> int: ...

    def initial_state(self) -> np.ndarray: ...

    def step(self, state: np.ndarray, token: int) -> np.ndarray: ...


EchoStateSubstrate = EchoStateReservoir

from .delay import DelayLineSubstrate
from .hierarchical import HierarchicalSubstrate
from .linear_memory import LinearMemorySubstrate
from .mixed_memory import MixedMemorySubstrate
from .oscillatory_memory import OscillatoryMemorySubstrate

__all__ = [
    "DelayLineSubstrate",
    "EchoStateSubstrate",
    "HierarchicalSubstrate",
    "LinearMemorySubstrate",
    "MixedMemorySubstrate",
    "OscillatoryMemorySubstrate",
    "TokenSubstrate",
    "build_recurrent_matrix",
    "spectral_radius",
]
