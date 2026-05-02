 MoE MLP + EMA + LeakyReLU + Sliding Window Eval

What changed over baseline:
replaced the standard MLP with a shared expert + 2 specialized experts with a learned router. Each token is processed by the shared expert (full hidden dim) plus a weighted combination of specialized experts (hidden//4 each). This gives more effective capacity without proportionally more parameters.
EMA weight averaging — exponential moving average (decay=0.997) maintained every training step, applied before quantization for smoother final weights.
LeakyReLU(0.5)² — activation change in both shared and specialized experts.
Sliding window eval — stride=64 for better context utilization during evaluation.
Zstd compression — level 22
Results (8xH100 SXM, RunPod)
seed | steps | val_bpb | artifact size
1337 | 9274 | 1.2283 | 15.6MB

Config:
MODEL_DIM=448 EVAL_STRIDE=64
torchrun --standalone --nproc_per_node=8 train_gpt.py
