
# Implementation notes and Log

Going from recent to first
---

### SWA (Stochastic Weight Averaging) - experiment
  - Averages model checkpoints from the last 40% of training (when LR is warming down)
  - Collects a snapshot every 50 steps once `scale < swa_start_frac (0.4)`
  - After training, averages all collected checkpoints and loads as final weights
  - Proven ~0.0006 bpb improvement in SOTA submission
  - Costs nothing during training — just CPU copies every 50 steps
  - Config: SWA_ENABLED=1, SWA_START_FRAC=0.4, SWA_EVERY=50

### Depth Recurrence — FAILED
  - Tested NUM_LAYERS=8 RECURRENCE=2 (16 effective layers) on 1xH100
  - BPB 1.3976 vs 1.3477 for 11L unique — worse
  - Slower per step (678ms vs 579ms) and used more memory (19.7GB vs 13.9GB)
  - Running same weights twice doubles compute graph, doesn't help quality
  - Unique layers are simply more expressive per byte at this scale
  - Don't revisit this approach

### SmearGate
  - One parameter per dimension (512 floats), initialized to zero (sigmoid(0) = 0.5 = equal mix)
  - Applied after RMS norm, before transformer blocks
  - `x = (1 - gate) * current_token + gate * previous_token`
  - Gives the model cheap 1-token context before attention kicks in
  - Gate param added to scalar optimizer, listed in control tensor patterns so it stays fp32

### BigramHash                                                                                                
  - 10240 buckets, 128-dim embeddings projected to 512 (model_dim)              
  - XOR hash of consecutive token pairs
  - Learned scale starting at 0.05                                              
  - Weights initialized to zero so it gradually learns                          
  - Embedding goes to token optimizer (Adam), projection to Muon with the other matrix params                                                                 
  - Quantized as int8 in the export  

### Quicker wins

Bump to 10-11 layers       
3x MLP expansion. The SOTA uses 3x, lets follow suite     
Weight decay on Muon (0.04) as the baseline has none                
Gradient clipping (0.3) to stabalise training                
Longer warmdown (3000 vs 1200) hopefully for convergence                
Higher Muon momentum (0.99 vs 0.95) copy from SOTA                     
Magnitude pruning before quantization, compression ratio

### Int4 instead of int5
This aim is to reduce the model size so we can slam more things in there. needs to be QAT

QAT hyperparameters (lines ~89-90) — QAT_ENABLED=1 and QAT_CLIP_RANGE=7  (int4)                                                          
                                                                                
CastedLinear now has STE fake-quantization — during training, MLP weights see int4 quantization noise but gradients flow through unmodified           
                                                                                
Mixed post-training quantization — replaces uniform int8 with:             
  - Int4 (clip_range=7) for MLP weights — half the bits of int8                 
  - Int8 for attention weights — precision-sensitive                            
  - FP16 for embeddings — small, needs full precision                           
                                                                                
QAT activation — after model creation, all block.mlp.fc and block.mlp.proj layers get QAT enabled          