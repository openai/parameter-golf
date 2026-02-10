from train_gpt_constants import TrainConstants


CONFIG = TrainConstants(
    target_total_gpus=8,
    model_num_layers=11,
    model_num_heads=6,
    model_dim=768,
    stage_batch_sizes=(
        8 * 2048 * 8,
        16 * 2048 * 8,
        24 * 2048 * 8,
        24 * 2048 * 8,
    ),
    val_batch_size=4 * 64 * 1024 * 8,
)
