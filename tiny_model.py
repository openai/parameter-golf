"""
Tiny Transformer Language Model
================================
A minimal, efficient transformer for constrained environments.
Target: <16MB model size, <10min training time.
"""

import torch
import torch.nn as nn


# ── Configuration ─────────────────────────────────────────────────────────────

VOCAB_SIZE   = 10_000   # Token vocabulary size
D_MODEL      = 64       # Embedding / model dimension
N_HEADS      = 2        # Attention heads
D_FF         = 128      # Feed-forward hidden dimension
N_LAYERS     = 2        # Transformer encoder layers
SEQ_LEN      = 128      # Maximum sequence length


# ── Model ─────────────────────────────────────────────────────────────────────

class TinyTransformer(nn.Module):
    """
    Minimal transformer language model.

    Key efficiency choices:
      - Weight tying: input embedding matrix is reused as the output
        projection, halving the largest parameter block.
      - Small d_model (64) and d_ff (128) keep every layer thin.
      - Only 2 layers — enough capacity for a baseline, minimal overhead.
      - No dropout, layer-norm epsilon left at PyTorch default.
    """

    def __init__(
        self,
        vocab_size: int = VOCAB_SIZE,
        d_model:    int = D_MODEL,
        n_heads:    int = N_HEADS,
        d_ff:       int = D_FF,
        n_layers:   int = N_LAYERS,
        seq_len:    int = SEQ_LEN,
    ):
        super().__init__()

        # ── Token + positional embeddings ──────────────────────────────────
        # Both live in d_model space; positional is learnable (not sinusoidal)
        # to keep the implementation simple and allow the model to adapt them.
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb   = nn.Embedding(seq_len, d_model)

        # ── Transformer encoder stack ──────────────────────────────────────
        # PyTorch's built-in encoder layer: multi-head self-attention + FF block.
        # batch_first=True → input shape (B, T, C) throughout.
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=0.0,          # no dropout at baseline stage
            activation="relu",
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)

        # ── Language-model head ────────────────────────────────────────────
        # Linear(d_model → vocab_size).  Weight is tied to token_emb below,
        # so this adds only a bias vector (vocab_size,) of extra parameters.
        self.lm_head = nn.Linear(d_model, vocab_size, bias=True)

        # ── Weight tying (CRITICAL optimisation) ──────────────────────────
        # Sharing the embedding matrix with the output projection removes
        # vocab_size × d_model ≈ 640 000 parameters from the count.
        self.lm_head.weight = self.token_emb.weight

        # ── Parameter initialisation ───────────────────────────────────────
        self._init_weights()

    # ── Initialisation ────────────────────────────────────────────────────────

    def _init_weights(self):
        """Small normal init — stable for a tiny model."""
        nn.init.normal_(self.token_emb.weight, std=0.02)
        nn.init.normal_(self.pos_emb.weight,   std=0.02)

    # ── Forward pass ──────────────────────────────────────────────────────────

    def forward(self, idx: torch.Tensor, targets: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            idx: LongTensor of token ids, shape (B, T)
            targets: Optional LongTensor of target token ids for computing loss

        Returns:
            loss or logits depending on whether targets provided
        """
        B, T = idx.shape

        # Build position indices [0, 1, …, T-1] for every item in the batch
        positions = torch.arange(T, device=idx.device).unsqueeze(0)  # (1, T)

        # Combine token and positional embeddings
        x = self.token_emb(idx) + self.pos_emb(positions)            # (B, T, d_model)

        # Pass through transformer encoder layers
        x = self.transformer(x)                                       # (B, T, d_model)

        # Project to vocabulary (weight-tied)
        logits = self.lm_head(x)                                      # (B, T, vocab_size)

        if targets is None:
            return logits
        else:
            # Compute cross-entropy loss
            loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1)
            )
            return loss


# ── Parameter analysis ────────────────────────────────────────────────────────

def count_parameters(model: nn.Module) -> None:
    """Print a breakdown of parameter counts per component."""
    total = 0
    print(f"\n{'Component':<35} {'Parameters':>12}")
    print("─" * 49)
    for name, param in model.named_parameters():
        n = param.numel()
        print(f"  {name:<33} {n:>12,}")
        total += n

    size_mb = total * 4 / 1024 / 1024   # float32 → bytes → MB
    print("─" * 49)
    print(f"  {'TOTAL (unique)':<33} {total:>12,}")
    print(f"  {'Estimated size (fp32)':<33} {size_mb:>11.2f} MB\n")


# ── Quick smoke-test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    model = TinyTransformer()
    model.eval()

    # Random token IDs in [0, VOCAB_SIZE)
    dummy_input = torch.randint(0, VOCAB_SIZE, (1, SEQ_LEN))   # shape: (1, 128)

    with torch.no_grad():
        output = model(dummy_input)

    print("=" * 49)
    print("  TinyTransformer — forward-pass smoke test")
    print("=" * 49)
    print(f"  Input  shape : {tuple(dummy_input.shape)}")
    print(f"  Output shape : {tuple(output.shape)}")

    count_parameters(model)
