from __future__ import annotations

import json
import os
import random
from pathlib import Path

import numpy as np
import torch


def atomic_json_dump(obj: dict[str, object], path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2, sort_keys=True)
        f.write("\n")
    os.replace(tmp_path, path)


def atomic_torch_save(obj: object, path: Path) -> None:
    tmp_path = path.with_name(path.name + ".tmp")
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def capture_rng_state() -> dict[str, object]:
    return {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state_all": torch.cuda.get_rng_state_all(),
    }


def restore_rng_state(state: dict[str, object]) -> None:
    random.setstate(state["python_random_state"])
    np.random.set_state(state["numpy_random_state"])
    torch.set_rng_state(state["torch_rng_state"])
    torch.cuda.set_rng_state_all(state["cuda_rng_state_all"])
