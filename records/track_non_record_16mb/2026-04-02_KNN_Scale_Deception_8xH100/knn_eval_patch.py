"""
KNN Hidden State Retrieval — Drop-in eval augmentation for train_gpt.py
=========================================================================

This module adds vectorized KNN scoring to the competition eval pipeline.
Import it in train_gpt.py and call eval_knn() instead of eval_standard().

INTEGRATION:
  1. Add this file alongside train_gpt.py
  2. In train_gpt.py main(), after model training:
     - Import: from knn_eval_patch import eval_with_knn
     - Replace: result = eval_standard(config, model, ...)
     - With:    result = eval_with_knn(config, model, ...)

ALTERNATIVELY: paste eval_with_knn() directly into train_gpt.py.

The function signature matches eval_standard() for drop-in compatibility.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import time
from dataclasses import dataclass


@dataclass
class EvalResult:
    """Matches the competition's EvalResult format."""
    val_loss: float
    val_bpb: float
    eval_ms: float
    eval_seq_len: int


class VectorizedKNN:
    """GPU-accelerated KNN hidden state retrieval.

    Stores hidden states from scored tokens and retrieves K nearest
    neighbors for each new token. Fully vectorized via torch.cdist.

    Protocol: score-first, causal, chunk-based.
    """

    def __init__(self, dim: int, max_stored: int, k: int = 8,
                 lam: float = 0.12, device: str = "cuda",
                 dtype: torch.dtype = torch.float16,
                 subsample_rate: int = 1):
        """
        Args:
            dim: hidden state dimension (e.g., 512)
            max_stored: maximum number of hidden states to store
            k: number of nearest neighbors
            lam: mixing weight (0.12 = 12% KNN)
            device: cuda device
            dtype: float16 saves memory (32GB vs 64GB for 62M vectors)
            subsample_rate: store every Nth token (1=all, 4=25%)
        """
        self.dim = dim
        self.k = k
        self.lam = lam
        self.device = device
        self.dtype = dtype
        self.subsample_rate = subsample_rate

        # Pre-allocate store
        self.stored_h = torch.zeros(max_stored, dim, device=device, dtype=dtype)
        self.stored_tok = torch.zeros(max_stored, device=device, dtype=torch.long)
        self.n_stored = 0
        self.n_seen = 0  # total tokens seen (before subsampling)

    def store_chunk(self, hidden_states: torch.Tensor, tokens: torch.Tensor):
        """Store hidden states from a scored chunk. CALL AFTER SCORING.

        Args:
            hidden_states: (chunk_len, dim) — hidden states from model
            tokens: (chunk_len,) — the scored tokens (targets)
        """
        if self.subsample_rate > 1:
            # Subsample: keep every Nth token
            indices = torch.arange(0, len(tokens), self.subsample_rate, device=tokens.device)
            hidden_states = hidden_states[indices]
            tokens = tokens[indices]

        n_new = len(tokens)
        if self.n_stored + n_new > self.stored_h.shape[0]:
            n_new = self.stored_h.shape[0] - self.n_stored
            if n_new <= 0:
                return
            hidden_states = hidden_states[:n_new]
            tokens = tokens[:n_new]

        self.stored_h[self.n_stored:self.n_stored + n_new] = hidden_states.to(self.dtype)
        self.stored_tok[self.n_stored:self.n_stored + n_new] = tokens
        self.n_stored += n_new
        self.n_seen += len(tokens) * self.subsample_rate

    def get_knn_distribution(self, queries: torch.Tensor,
                              vocab_size: int = 1024) -> torch.Tensor:
        """Batch KNN query. CALL BEFORE SCORING (uses only previously stored states).

        Args:
            queries: (chunk_len, dim) — hidden states for current chunk

        Returns:
            knn_probs: (chunk_len, vocab_size) — KNN probability distribution
                       Returns uniform if not enough stored states.
        """
        chunk_len = queries.shape[0]

        if self.n_stored < self.k:
            return torch.ones(chunk_len, vocab_size, device=self.device) / vocab_size

        # Compute squared L2 distances via cdist
        # queries: (C, dim), stored: (N, dim)
        q = queries.to(self.dtype)
        dists = torch.cdist(q, self.stored_h[:self.n_stored], p=2).pow(2)  # (C, N)

        # Top-K
        actual_k = min(self.k, self.n_stored)
        topk_dists, topk_idx = dists.topk(actual_k, dim=1, largest=False)  # (C, K)

        # Get tokens of neighbors
        topk_toks = self.stored_tok[topk_idx]  # (C, K)

        # Distance-weighted softmax
        weights = torch.exp(-topk_dists.float() / self.dim)  # (C, K)
        weights = weights / (weights.sum(dim=1, keepdim=True) + 1e-30)

        # Build distribution via scatter
        knn_dist = torch.zeros(chunk_len, vocab_size, device=self.device)
        knn_dist.scatter_add_(1, topk_toks, weights)

        # Smooth
        knn_dist = 0.99 * knn_dist + 0.01 / vocab_size
        knn_dist = knn_dist / knn_dist.sum(dim=1, keepdim=True)

        return knn_dist

    def reset(self):
        self.n_stored = 0
        self.n_seen = 0


def eval_with_knn(
    model: nn.Module,
    val_tokens: torch.Tensor,
    vocab_size: int = 1024,
    seq_len: int = 512,
    eval_batch_seqs: int = 64,
    knn_k: int = 8,
    knn_lam: float = 0.12,
    knn_chunk: int = 1024,
    knn_subsample: int = 4,
    knn_max_stored: int = 16_000_000,
    device: str = "cuda",
    luts: tuple = None,
) -> EvalResult:
    """Evaluate model with KNN hidden state augmentation.

    Drop-in replacement for eval_standard() with KNN mixing.

    The eval processes sequences in chunks:
      1. Forward pass → logits + hidden states
      2. Softmax → neural probabilities
      3. Query KNN → KNN probabilities
      4. Mix: (1-lam)*neural + lam*KNN
      5. Score with mixed distribution
      6. Store hidden states AFTER scoring

    Args:
        model: ParameterGolfModel (must have _hidden() method)
        val_tokens: 1D tensor of validation tokens
        vocab_size: vocabulary size (1024)
        seq_len: evaluation sequence length
        eval_batch_seqs: sequences per forward pass batch
        knn_k: number of nearest neighbors
        knn_lam: KNN mixing weight
        knn_chunk: tokens per KNN scoring chunk
        knn_subsample: store every Nth token (reduces memory)
        knn_max_stored: maximum stored hidden states
        device: cuda device
        luts: byte scoring lookup tables (for BPB computation)

    Returns:
        EvalResult with val_loss and val_bpb
    """
    start_time = time.perf_counter()
    model.eval()

    # Determine model dimension
    dim = model.config.model_dim if hasattr(model, 'config') else model.tok_emb.weight.shape[1]

    # Initialize KNN store
    knn = VectorizedKNN(
        dim=dim,
        max_stored=knn_max_stored,
        k=knn_k,
        lam=knn_lam,
        device=device,
        dtype=torch.float16,
        subsample_rate=knn_subsample,
    )

    # Prepare sequences
    total_tokens = val_tokens.numel() - 1
    n_seqs = total_tokens // seq_len
    batch_tokens = eval_batch_seqs * seq_len

    total_loss_sum = 0.0
    total_token_count = 0
    total_byte_count = 0.0

    # Process sequences in batches
    with torch.inference_mode():
        for batch_start in range(0, n_seqs, eval_batch_seqs):
            batch_end = min(batch_start + eval_batch_seqs, n_seqs)
            actual_batch = batch_end - batch_start

            # Get batch tokens
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.long)

            x = local[:-1].reshape(-1, seq_len)  # (B, seq_len) — inputs
            y = local[1:].reshape(-1, seq_len)    # (B, seq_len) — targets

            # Forward pass: get hidden states AND logits
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                hidden = model._hidden(x)  # (B, seq_len, dim)

            # Compute logits from hidden
            hidden_f32 = hidden.float()
            if hasattr(model, 'tie_embeddings') and model.tie_embeddings:
                logits = F.linear(hidden_f32, model.tok_emb.weight.float())
            elif hasattr(model, 'lm_head') and model.lm_head is not None:
                logits = model.lm_head(hidden_f32)
            else:
                logits = F.linear(hidden_f32, model.tok_emb.weight.float())

            # Flatten for chunk-based KNN scoring
            flat_logits = logits.reshape(-1, vocab_size)     # (B*seq_len, V)
            flat_hidden = hidden_f32.reshape(-1, dim)        # (B*seq_len, dim)
            flat_targets = y.reshape(-1)                     # (B*seq_len,)
            n_tokens = flat_logits.shape[0]

            # Neural probabilities
            neural_probs = F.softmax(flat_logits, dim=-1)    # (B*seq_len, V)

            # Score in KNN chunks
            for chunk_start in range(0, n_tokens, knn_chunk):
                chunk_end = min(chunk_start + knn_chunk, n_tokens)
                chunk_len = chunk_end - chunk_start

                c_neural = neural_probs[chunk_start:chunk_end]   # (C, V)
                c_hidden = flat_hidden[chunk_start:chunk_end]     # (C, dim)
                c_targets = flat_targets[chunk_start:chunk_end]   # (C,)

                # Get KNN distribution (from previously stored states)
                knn_dist = knn.get_knn_distribution(c_hidden, vocab_size)

                # Mix
                if knn.n_stored >= knn_k:
                    mixed = (1.0 - knn_lam) * c_neural + knn_lam * knn_dist
                else:
                    mixed = c_neural

                mixed = mixed / mixed.sum(dim=1, keepdim=True)

                # Score: cross-entropy
                target_probs = mixed.gather(1, c_targets.unsqueeze(1)).squeeze(1)
                log_probs = torch.log(target_probs.clamp(min=1e-30))
                chunk_loss = -log_probs.sum()

                total_loss_sum += chunk_loss.item()
                total_token_count += chunk_len

                # Byte counting for BPB (if luts provided)
                if luts is not None:
                    x_chunk = local[:-1].reshape(-1)[chunk_start:chunk_end]
                    y_chunk = flat_targets[chunk_start:chunk_end]
                    # Use competition's byte scoring
                    try:
                        from train_gpt import _score_token_bytes
                        token_bytes = _score_token_bytes(x_chunk, y_chunk, luts)
                        total_byte_count += token_bytes.float().sum().item()
                    except ImportError:
                        total_byte_count += chunk_len  # fallback: 1 byte per token

                # Store AFTER scoring (causal)
                knn.store_chunk(c_hidden.detach(), c_targets.detach())

            if (batch_start // eval_batch_seqs) % 10 == 0:
                elapsed = time.perf_counter() - start_time
                bpc = total_loss_sum / max(total_token_count, 1) / math.log(2)
                print(f"  KNN eval [{batch_end}/{n_seqs}] "
                      f"loss={total_loss_sum/max(total_token_count,1):.4f} "
                      f"BPC={bpc:.4f} "
                      f"stored={knn.n_stored:,} "
                      f"({elapsed:.0f}s)", flush=True)

    # Compute final metrics
    val_loss = total_loss_sum / max(total_token_count, 1)
    bits_per_token = val_loss / math.log(2.0)
    if total_byte_count > 0:
        tokens_per_byte = total_token_count / total_byte_count
    else:
        tokens_per_byte = 1.0  # fallback
    val_bpb = bits_per_token * tokens_per_byte

    elapsed_ms = 1000.0 * (time.perf_counter() - start_time)

    print(f"\n  KNN eval complete: val_loss={val_loss:.4f} val_bpb={val_bpb:.4f} "
          f"stored={knn.n_stored:,} time={elapsed_ms/1000:.0f}s", flush=True)

    return EvalResult(
        val_loss=val_loss,
        val_bpb=val_bpb,
        eval_ms=elapsed_ms,
        eval_seq_len=seq_len,
    )


# ============================================================
# Integration helper: patch into existing eval pipeline
# ============================================================
def patch_eval_with_knn(original_eval_fn, knn_lam=0.12, knn_k=8,
                         knn_subsample=4):
    """Decorator to add KNN to any eval function.

    Usage in train_gpt.py:
        original_result = eval_standard(config, model, ...)
        knn_result = eval_with_knn(model, val_tokens, ...)
        # Use knn_result instead
    """
    pass  # Not needed if we directly call eval_with_knn


# ============================================================
# Quick test
# ============================================================
if __name__ == "__main__":
    print("KNN Eval Patch — ready for integration")
    print("Import eval_with_knn() and call it with your trained model.")
    print()
    print("Example:")
    print("  from knn_eval_patch import eval_with_knn")
    print("  result = eval_with_knn(model, val_tokens, vocab_size=1024)")
    print("  print(f'BPB: {result.val_bpb}')")
