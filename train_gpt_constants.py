import importlib.util
from dataclasses import dataclass
from pathlib import Path

try:
    import chz
except ModuleNotFoundError:
    class _ChzCompat:
        @staticmethod
        def chz(cls):
            return dataclass(frozen=True)(cls)

    chz = _ChzCompat()


@chz.chz
class TrainConstants:
    target_total_gpus: int
    model_num_heads: int
    model_dim: int
    stage_batch_sizes: tuple[int, int, int, int]
    val_batch_size: int
    model_num_layers: int = 11
    optimizer_lr_scale: float = 1.0
    optimizer_wd_scale: float = 1.0
    drop_zero_length_seqlens: bool = False
    compile_model: bool = True
    empty_cache_every_steps: int = 0


def load_train_constants(config_path: str) -> TrainConstants:
    resolved = Path(config_path).resolve()
    assert resolved.is_file(), f"Missing config file: {resolved}"

    spec = importlib.util.spec_from_file_location("train_gpt_config_module", resolved)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    config = getattr(module, "CONFIG")
    assert isinstance(config, TrainConstants), "CONFIG must be a TrainConstants instance"
    return config
