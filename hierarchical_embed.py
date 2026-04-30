import torch, torch.nn as nn, torch.nn.functional as F

class HierarchicalQuantizedEmbedding(nn.Module):
    def __init__(self, vocab_size, dim, tier_boundaries=None, tier_bits=None):
        super().__init__()
        if tier_boundaries is None:
            if vocab_size <= 1024:
                tier_boundaries = [0, 32, 128, 512, vocab_size]
            else:
                tier_boundaries = [0, 256, 2048, 16384, vocab_size]
        if tier_bits is None:
            tier_bits = [16, 8, 6, 4]

        self.vocab_size = vocab_size
        self.dim = dim
        self.tier_boundaries = tier_boundaries
        self.tier_bits = tier_bits
        self.num_tiers = len(tier_bits)

        self.tier_weights = nn.ParameterList([
            nn.Parameter(torch.randn(
                tier_boundaries[i+1] - tier_boundaries[i], dim) * 0.02)
            for i in range(self.num_tiers)
        ])
        self.tier_scales = nn.ParameterList([
            nn.Parameter(torch.ones(1, dim) * 0.1)
            for _ in range(self.num_tiers)
        ])
        # tie_embeddings用のダミーleaf weight（lm_headから参照される）
        # 実際の学習はtier_weightsで行う
        self.weight = nn.Parameter(
            torch.zeros(vocab_size, dim), requires_grad=False
        )

    def quantize_ste(self, x, bits):
        if bits >= 16:
            return x
        max_val = 2 ** (bits - 1) - 1
        scale = x.abs().max().clamp(min=1e-8) / max_val
        x_q = torch.round(x / scale).clamp(-max_val, max_val) * scale
        return x + (x_q - x).detach()

    def get_quantized_weight(self):
        return torch.cat([
            self.quantize_ste(self.tier_weights[i] * self.tier_scales[i],
                              self.tier_bits[i])
            for i in range(self.num_tiers)
        ], dim=0)

    def forward(self, input_ids):
        return F.embedding(input_ids, self.get_quantized_weight())
