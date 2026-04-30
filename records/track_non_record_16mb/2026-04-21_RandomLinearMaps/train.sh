ITERATIONS=2000 TRAIN_LOG_EVERY=20 WARMUP_STEPS=0 MAX_WALLCLOCK_SECONDS=0 torchrun --standalone --nproc_per_node=4 train_gpt.py
