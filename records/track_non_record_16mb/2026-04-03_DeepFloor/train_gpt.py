#!/usr/bin/env python3
from __future__ import annotations

import json
import os

from deepfloor_snapshot import V3Config, train_and_evaluate


def env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    return default if value is None else int(value)


def env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    return default if value is None else float(value)


def env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off"}


def build_config_from_env() -> V3Config:
    enwik8_path = os.environ.get("ENWIK8_PATH")
    if not enwik8_path:
        raise SystemExit("ENWIK8_PATH is required")
    return V3Config(
        enwik8_path=enwik8_path,
        output_json=os.environ.get("OUTPUT_JSON"),
        device=os.environ.get("DEVICE", V3Config.device),
        seed=env_int("SEED", V3Config.seed),
        seq_len=env_int("SEQ_LEN", V3Config.seq_len),
        stride=env_int("STRIDE", V3Config.stride),
        batch_size=env_int("BATCH_SIZE", V3Config.batch_size),
        train_steps=env_int("TRAIN_STEPS", V3Config.train_steps),
        eval_batches=env_int("EVAL_BATCHES", V3Config.eval_batches),
        report_every=env_int("REPORT_EVERY", V3Config.report_every),
        cache_dataset_on_device=env_bool("CACHE_DATASET_ON_DEVICE", V3Config.cache_dataset_on_device),
        recurrent_dim=env_int("RECURRENT_DIM", V3Config.recurrent_dim),
        recurrent_heads=env_int("RECURRENT_HEADS", V3Config.recurrent_heads),
        num_distinct_blocks=env_int("NUM_DISTINCT_BLOCKS", V3Config.num_distinct_blocks),
        view_count=env_int("VIEW_COUNT", V3Config.view_count),
        view_combination=os.environ.get("VIEW_COMBINATION", V3Config.view_combination),
        cross_token_mode=os.environ.get("CROSS_TOKEN_MODE", V3Config.cross_token_mode),
        block_has_residual=env_bool("BLOCK_HAS_RESIDUAL", V3Config.block_has_residual),
        block_nonlinearity=os.environ.get("BLOCK_NONLINEARITY", V3Config.block_nonlinearity),
        recurrence_step_size=env_float("RECURRENCE_STEP_SIZE", V3Config.recurrence_step_size),
        state_decay=env_float("STATE_DECAY", V3Config.state_decay),
        contraction_target=env_float("CONTRACTION_TARGET", V3Config.contraction_target),
        train_recurrence_steps=env_int("TRAIN_RECURRENCE_STEPS", V3Config.train_recurrence_steps),
        eval_recurrence_steps=env_int("EVAL_RECURRENCE_STEPS", V3Config.eval_recurrence_steps),
        tbptt_chunk=env_int("TBPTT_CHUNK", V3Config.tbptt_chunk),
        norm_interval_k=env_int("NORM_INTERVAL_K", V3Config.norm_interval_k),
        train_floor_interval=env_int("TRAIN_FLOOR_INTERVAL", V3Config.train_floor_interval),
        floor_min_interval=env_int("FLOOR_MIN_INTERVAL", V3Config.floor_min_interval),
        floor_max_interval=env_int("FLOOR_MAX_INTERVAL", V3Config.floor_max_interval),
        floor_threshold=env_float("FLOOR_THRESHOLD", V3Config.floor_threshold),
        kernel_feature_map=os.environ.get("KERNEL_FEATURE_MAP", V3Config.kernel_feature_map),
        accumulator_decay=env_float("ACCUMULATOR_DECAY", V3Config.accumulator_decay),
        state_core=os.environ.get("STATE_CORE", V3Config.state_core),
        hippo_delta_scale=env_float("HIPPO_DELTA_SCALE", V3Config.hippo_delta_scale),
        hippo_rank=env_int("HIPPO_RANK", V3Config.hippo_rank),
        quantization=os.environ.get("QUANTIZATION", V3Config.quantization),
        jacobian_lambda=env_float("JACOBIAN_LAMBDA", V3Config.jacobian_lambda),
        stochastic_round_p=env_float("STOCHASTIC_ROUND_P", V3Config.stochastic_round_p),
        base_lr=env_float("BASE_LR", V3Config.base_lr),
        weight_decay=env_float("WEIGHT_DECAY", V3Config.weight_decay),
        grad_clip_norm=env_float("GRAD_CLIP_NORM", V3Config.grad_clip_norm),
        warmup_steps=env_int("WARMUP_STEPS", V3Config.warmup_steps),
        min_lr_scale=env_float("MIN_LR_SCALE", V3Config.min_lr_scale),
    )


def main() -> None:
    result = train_and_evaluate(build_config_from_env())
    artifact = result["artifact"]
    val = result["val"]
    test = result["test"]
    print(
        "deepfloor_final "
        f"val_bpb:{float(val['bpb']):.8f} "
        f"test_bpb:{float(test['bpb']):.8f} "
        f"artifact_bytes:{int(artifact['estimated_bytes'])}"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
