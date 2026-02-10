from train_gpt_constants import TrainConstants

CONFIG = TrainConstants(
    target_total_gpus=1,
    model_num_layers=6,
    model_num_heads=4,
    model_dim=512,
    stage_batch_sizes=(
        3072,
        3072,
        3072,
        3072,
    ),
    val_batch_size=4 * 1024,
    optimizer_lr_scale=0.005,
    optimizer_wd_scale=0.0,
    drop_zero_length_seqlens=True,
    compile_model=False,
    empty_cache_every_steps=1,
)
