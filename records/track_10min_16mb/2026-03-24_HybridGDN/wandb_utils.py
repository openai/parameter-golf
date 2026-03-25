from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

from dotenv import load_dotenv


def _is_truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}

def wandb_enabled() -> bool:
    load_dotenv(override=False)
    enabled = os.environ.get("WANDB_ENABLED")
    if enabled is not None:
        return _is_truthy(enabled)
    return bool(os.environ.get("WANDB_API_KEY"))


def hyperparameters_to_config(args: Any) -> dict[str, Any]:
    config: dict[str, Any] = {}
    for key, value in vars(type(args)).items():
        if key.startswith("_") or isinstance(value, property) or callable(value):
            continue
        if isinstance(value, (str, int, float, bool)) or value is None:
            config[key] = value
    return config


def init_wandb(
    *,
    run_id: str,
    backend: str,
    config: Mapping[str, Any],
    extra_config: Mapping[str, Any] | None = None,
):
    if not wandb_enabled():
        return None

    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            "WANDB_ENABLED=1 but wandb is not installed. Install it with `pip install wandb`."
        ) from exc

    init_config = dict(config)
    if extra_config is not None:
        init_config.update(dict(extra_config))

    return wandb.init(
        project=os.environ.get("WANDB_PROJECT", "parameter-golf"),
        entity=os.environ.get("WANDB_ENTITY") or None,
        group=os.environ.get("WANDB_GROUP") or None,
        job_type=os.environ.get("WANDB_JOB_TYPE", "train"),
        mode=os.environ.get("WANDB_MODE") or None,
        dir=os.environ.get("WANDB_DIR") or None,
        id=os.environ.get("WANDB_RUN_ID") or run_id,
        name=os.environ.get("WANDB_NAME") or run_id,
        resume=os.environ.get("WANDB_RESUME", "allow"),
        tags=[backend],
        config=init_config,
    )


def log_wandb(run: Any, metrics: Mapping[str, Any], *, step: int | None = None) -> None:
    if run is None:
        return
    if step is None:
        run.log(dict(metrics))
        return
    run.log(dict(metrics), step=step)


def update_summary(run: Any, metrics: Mapping[str, Any]) -> None:
    if run is None:
        return
    for key, value in metrics.items():
        run.summary[key] = value


def finish_wandb(run: Any) -> None:
    if run is None:
        return
    run.finish()
