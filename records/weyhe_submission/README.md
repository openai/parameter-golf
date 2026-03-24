# Parameter Golf Submission: 1.0941 BPB (Document-Level LoRA TTT)

Submitted by: David Weyh (@dweyh)  
*Based on the outstanding architectural foundation "Loqui Auris" by Eli Pancamo.*

## Result
**val_bpb:** `1.0941`  
**Artifact Size:** `14.99 MB` (Fits under 16MB)  
**Total Eval Time:** `167.8s` (Fits under 10min)  
**Training Time (Wallclock):** `600s`

## Architecture Highlights
- 10 transformer layers, dim=512, 8 heads / 4 KV heads (GQA)
- MLP 3x expansion (1536 hidden)
- SmearGate + BigramHash + OrthoInit
- U-Net skip connections, tied embeddings
- ~22.3M parameters

## Quantization Method
- **INT8 Zstandard (zlib) Compression:** Instead of forcing INT6 on 11 layers, this submission leverages a highly efficient 10-layer model compressed to exactly 14.99 MB using INT8, preserving crucial weight fidelity.

## The Secret: Causal Document-Level LoRA
The extreme score drop to `1.0941` is achieved via **Test-Time Training (TTT)** using LoRA injected into the `c_proj` and `mlp_proj` layers (`TTT_MODE=lora`). 
Key logic:
- The validation dataset is grouped by documents.
- Shorter documents (< 512 tokens) are evaluated zero-shot.
- Longer documents are processed chronologically in chunks. A LoRA adapter (Rank 8) is trained using Adam (`lr=0.01`) over 2 epochs on the past chunks, constantly adapting the model to the specific document context.
- After each document, the LoRA state is cleanly flushed.

This perfectly exploits the chronological autoregressive rule structure of the leaderboard!
