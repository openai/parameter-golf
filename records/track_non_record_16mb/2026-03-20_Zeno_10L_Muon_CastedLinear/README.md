# Zeno 10L Muon CastedLinear (Unfinished)

This is a non-record submission detailing a high-capacity architecture approach. 

## Approach
The goal of this submission is to push the Pareto frontier of the 10-minute, 16MB Parameter Golf challenge by aggressively maximizing model capacity while staying within the strict artifact limits. The core of my approach abandons the naive low-rank adapter baseline structures. Instead, I replaced the `SeededLoRALinear` layers in both the CausalSelfAttention and MLP components with full-capacity `CastedLinear` matrices. This architectural shift increases the parameter count from ~4.7M to over 18.5M, vastly expanding the model's expressibility.

To fit these ~18.5M parameters into a 16MB artifact, the weight matrices are quantized down to `int8` post-training and then heavily compressed using `zlib` level 9. 

To train effectively within the 10-minute 8xH100 constraint, I replaced AdamW with the Muon optimizer specifically for the 2D matrix parameters, utilizing a decoupled weight decay to improve generalization and quantization robustness. Extending the transformer to 10 layers, I applied an "Overtone" spectrum initialization for embeddings alongside a Sigmoid-scheduled phase-transition residual mixing strategy. Finally, at evaluation, the model leverages Batched Test-Time Training (TTT) via dynamically updated LoRA adapters on the validation stream.

## Status
The provided `train_gpt.py` script successfully compiles and launches with optimal hyperparameters (`warmdown_iters=1200`, `matrix_lr`/`scalar_lr` tuning, etc.). However, because I ran out of compute credits on RunPod exactly 4-6 minutes into training, I was unable to complete a full 10-minute 8xH100 run to collect the required 3 random seeds for a record submission. 

I am submitting this to the non-record track to share the architectural and optimal hyperparameter implementations.
