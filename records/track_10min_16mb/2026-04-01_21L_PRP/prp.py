import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ParametrizedRandomProjection(nn.Module):
    _layer_counter = 0
    def __init__(
        self,
        in_features,
        out_features,
        projection_type="random",
        lora_rank: int = 32,
        lora_alpha: float = 1.0,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.projection_type = projection_type
        self._layer_id = ParametrizedRandomProjection._layer_counter
        ParametrizedRandomProjection._layer_counter += 1
        self.seed = hash((in_features, out_features, projection_type, self._layer_id)) % (2 ** 32)

        self.alpha = nn.Parameter(torch.empty(in_features))
        self.weight = nn.Parameter(torch.empty(out_features))
        self.bias = nn.Parameter(torch.empty(out_features))

        std_in = 1 / math.sqrt(in_features)
        nn.init.normal_(self.alpha, mean=1.0, std=std_in)
        nn.init.normal_(self.weight, mean=1.0, std=std_in)
        nn.init.zeros_(self.bias)

        # ========== Fixed random projection (non-learnable) ==========
        g = torch.Generator(device="cpu")
        g.manual_seed(self.seed)
        std_dev = 1 / math.sqrt(self.in_features)
        if self.projection_type == "random":
            proj = torch.randn(self.in_features, self.out_features, generator=g) * std_dev
        elif self.projection_type == "sparse":
            val_scale = math.sqrt(3.0 / self.in_features)
            u = torch.rand(self.in_features, self.out_features, generator=g)
            proj = torch.zeros(self.in_features, self.out_features)
            proj[u < 1 / 6] = -val_scale
            proj[u >= 5 / 6] = val_scale
        elif self.projection_type == "orthogonal":
            proj = torch.randn(self.in_features, self.out_features, generator=g) * std_dev
            nn.init.orthogonal_(proj)
        elif self.projection_type == "uniform":
            proj = (torch.rand(self.in_features, self.out_features, generator=g) * 2 - 1) * std_dev * math.sqrt(3)
        else:
            raise ValueError("Unsupported projection_type")

        # keep both names for compatibility
        self.register_buffer("proj", proj, persistent=False)
        self.register_buffer("fixed_proj", proj, persistent=False)

        # ========== LoRA low-rank adaptation ==========
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_A = nn.Parameter(torch.empty(self.in_features, lora_rank))
        self.lora_B = nn.Parameter(torch.empty(lora_rank, self.out_features))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)
        self.lora_scale = float(self.lora_alpha) / max(1, self.lora_rank)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Move proj to correct device/dtype
        proj = self.proj.to(dtype=x.dtype, device=x.device)

        # Element-wise scaling (no per-input bias)
        x_scaled = x * self.alpha.unsqueeze(0)

        # Fixed projection path
        out_fixed = x_scaled @ proj
        out_fixed = out_fixed * self.weight.unsqueeze(0) + self.bias.unsqueeze(0)

        # LoRA adaptation as separate path
        lora_out = (x @ self.lora_A) @ self.lora_B
        lora_out = lora_out * self.lora_scale

        return out_fixed + lora_out

    def count_trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}("
            f"in_features={self.in_features}, "
            f"out_features={self.out_features}, "
            f"projection_type='{self.projection_type}', "
            f"lora_rank={self.lora_rank}, "
            f"lora_alpha={self.lora_alpha}, "
            f"trainable_params={self.count_trainable_params():,})"
        )

