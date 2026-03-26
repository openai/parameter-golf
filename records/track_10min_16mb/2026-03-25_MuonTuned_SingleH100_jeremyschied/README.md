# Muon Optimizer Tuning — Single H100
MATRIX_LR=0.05 MUON_BACKEND_STEPS=6 MUON_MOMENTUM_WARMUP_STEPS=300 GRAD_CLIP_NORM=1.0 WARMDOWN_ITERS=900 torchrun --standalone --nproc_per_node=1 train_gpt.py

