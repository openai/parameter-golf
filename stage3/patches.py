"""
Composable code patches for train_gpt.py.

Each patch function takes the full source text and returns modified source text.
Patches target non-overlapping regions so they can be composed freely for
Pack 2 composite candidates.

Usage:
    from patches import apply_patches
    patched = apply_patches(original_source, ["normuon_wd", "qat", "orthoinit"])
"""
from __future__ import annotations

import os


def _replace_unique(source: str, old: str, new: str, patch_name: str) -> str:
    count = source.count(old)
    if count == 0:
        raise ValueError(f"Patch '{patch_name}': target string not found in source")
    if count > 1:
        raise ValueError(
            f"Patch '{patch_name}': target string found {count} times (must be unique)"
        )
    return source.replace(old, new)


# ---------------------------------------------------------------------------
# Patch: normuon_wd
# Target regions: Muon class, optimizer construction in main()
# ---------------------------------------------------------------------------

def patch_normuon_wd(source: str) -> str:
    """Add NorMuon (per-row second-moment normalization) and weight decay to Muon."""
    name = "normuon_wd"

    # 1. Add weight_decay and beta2 to Muon constructor
    source = _replace_unique(
        source,
        "    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):\n"
        "        super().__init__(\n"
        "            params,\n"
        "            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),\n"
        "        )",
        "    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0, beta2: float = 0.95):\n"
        "        super().__init__(\n"
        "            params,\n"
        "            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay, beta2=beta2),\n"
        "        )",
        name,
    )

    # 2. Add second_momentum state buffer
    source = _replace_unique(
        source,
        "                    if \"momentum_buffer\" not in state:\n"
        "                        state[\"momentum_buffer\"] = torch.zeros_like(g)\n"
        "                    buf = state[\"momentum_buffer\"]",
        "                    if \"momentum_buffer\" not in state:\n"
        "                        state[\"momentum_buffer\"] = torch.zeros_like(g)\n"
        "                    if \"second_momentum\" not in state:\n"
        "                        state[\"second_momentum\"] = torch.zeros(g.shape[0], 1, device=g.device, dtype=g.dtype)\n"
        "                    buf = state[\"momentum_buffer\"]",
        name,
    )

    # 3. Add NorMuon normalization after Newton-Schulz
    source = _replace_unique(
        source,
        "                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)\n"
        "                    # Scale correction from Muon reference implementations.\n"
        "                    g *= max(1, g.size(0) / g.size(1)) ** 0.5",
        "                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)\n"
        "                    # NorMuon: per-row second-moment normalization\n"
        "                    _sm = state[\"second_momentum\"]\n"
        "                    _vnorm = g.norm()\n"
        "                    _v_mean = (g * g).mean(dim=-1, keepdim=True)\n"
        "                    _sm.lerp_(_v_mean, 1 - group.get(\"beta2\", 0.95))\n"
        "                    g.div_(_sm.sqrt().add_(1e-10))\n"
        "                    g.mul_(_vnorm / g.norm().add_(1e-10))\n"
        "                    # Scale correction from Muon reference implementations.\n"
        "                    g *= max(1, g.size(0) / g.size(1)) ** 0.5",
        name,
    )

    # 4. Add weight decay to the update step
    source = _replace_unique(
        source,
        "            curr = 0\n"
        "            for p in params:\n"
        "                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)\n"
        "                p.add_(g, alpha=-lr)\n"
        "                curr += p.numel()",
        "            _wd = group.get(\"weight_decay\", 0.0)\n"
        "            curr = 0\n"
        "            for p in params:\n"
        "                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)\n"
        "                if _wd > 0:\n"
        "                    p.data.mul_(1.0 - lr * _wd)\n"
        "                p.add_(g, alpha=-lr)\n"
        "                curr += p.numel()",
        name,
    )

    # 5. Pass weight_decay to Muon constructor in main()
    source = _replace_unique(
        source,
        "    optimizer_muon = Muon(\n"
        "        matrix_params,\n"
        "        lr=args.matrix_lr,\n"
        "        momentum=args.muon_momentum,\n"
        "        backend_steps=args.muon_backend_steps,\n"
        "    )",
        "    optimizer_muon = Muon(\n"
        "        matrix_params,\n"
        "        lr=args.matrix_lr,\n"
        "        momentum=args.muon_momentum,\n"
        "        backend_steps=args.muon_backend_steps,\n"
        "        weight_decay=float(os.environ.get(\"MUON_WD\", \"0.0\")),\n"
        "    )",
        name,
    )

    return source


# ---------------------------------------------------------------------------
# Patch: qat
# Target region: CastedLinear class
# ---------------------------------------------------------------------------

def patch_qat(source: str) -> str:
    """Add STE int6 fake quantization to CastedLinear forward pass."""
    name = "qat"

    source = _replace_unique(
        source,
        "class CastedLinear(nn.Linear):\n"
        "    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.\n"
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        bias = self.bias.to(x.dtype) if self.bias is not None else None\n"
        "        return F.linear(x, self.weight.to(x.dtype), bias)",

        "class _FakeQuantInt6STE(torch.autograd.Function):\n"
        "    @staticmethod\n"
        "    def forward(ctx, w: Tensor) -> Tensor:\n"
        "        w32 = w.float()\n"
        "        abs_max = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)\n"
        "        scale = abs_max / 31.0\n"
        "        q = torch.clamp(torch.round(w32 / scale), -32, 31)\n"
        "        return (q * scale).to(w.dtype)\n"
        "    @staticmethod\n"
        "    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:\n"
        "        return (grad_output,)\n"
        "\n"
        "_fake_quant_int6 = _FakeQuantInt6STE.apply\n"
        "_QAT_ENABLED = os.environ.get(\"QAT\", \"0\") == \"1\"\n"
        "\n"
        "\n"
        "class CastedLinear(nn.Linear):\n"
        "    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.\n"
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        w = self.weight.to(x.dtype)\n"
        "        if _QAT_ENABLED and self.training and w.ndim == 2:\n"
        "            w = _fake_quant_int6(w)\n"
        "        bias = self.bias.to(x.dtype) if self.bias is not None else None\n"
        "        return F.linear(x, w, bias)",
        name,
    )

    return source


# ---------------------------------------------------------------------------
# Patch: smeargate
# Target regions: before Rotary class, GPT.__init__, GPT.forward, optimizer setup
# Composable with bigram_hash in any order.
# ---------------------------------------------------------------------------

def patch_smeargate(source: str) -> str:
    """Add SmearGate module: learned sigmoid gate blending each token with its predecessor."""
    name = "smeargate"

    # 1. Insert SmearGate class before Rotary
    source = _replace_unique(
        source,
        "class Rotary(nn.Module):\n"
        "    # Caches cos/sin tables per sequence length on the current device.",

        "class SmearGate(nn.Module):\n"
        "    def __init__(self, dim: int):\n"
        "        super().__init__()\n"
        "        self.gate = nn.Parameter(torch.zeros(dim, dtype=torch.float32))\n"
        "\n"
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        g = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]\n"
        "        x_prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)\n"
        "        return (1 - g) * x + g * x_prev\n"
        "\n"
        "\n"
        "class Rotary(nn.Module):\n"
        "    # Caches cos/sin tables per sequence length on the current device.",
        name,
    )

    # 2. Add smear module to GPT.__init__ (inserts before final_norm)
    source = _replace_unique(
        source,
        "        self.final_norm = RMSNorm()",
        "        self.smear = SmearGate(model_dim) if os.environ.get(\"SMEARGATE\", \"0\") == \"1\" else None\n"
        "        self.final_norm = RMSNorm()",
        name,
    )

    # 3. Modify GPT.forward: apply smear after rms_norm, before x0
    source = _replace_unique(
        source,
        "        x = F.rms_norm(x, (x.size(-1),))\n"
        "        x0 = x",
        "        x = F.rms_norm(x, (x.size(-1),))\n"
        "        if getattr(self, 'smear', None) is not None:\n"
        "            x = self.smear(x)\n"
        "        x0 = x",
        name,
    )

    # 4. Add smear params to optimizer
    source = _replace_unique(
        source,
        "    if base_model.skip_weights.numel() > 0:\n"
        "        scalar_params.append(base_model.skip_weights)",
        "    if base_model.skip_weights.numel() > 0:\n"
        "        scalar_params.append(base_model.skip_weights)\n"
        "    if getattr(base_model, 'smear', None) is not None:\n"
        "        scalar_params.extend(base_model.smear.parameters())",
        name,
    )

    return source


# ---------------------------------------------------------------------------
# Patch: bigram_hash
# Target regions: before Rotary class, GPT.__init__, GPT.forward, optimizer setup
# Composable with smeargate in any order.
# ---------------------------------------------------------------------------

def patch_bigram_hash(source: str) -> str:
    """Add BigramHash embedding: hashed token-pair lookup table projected to model dim."""
    name = "bigram_hash"

    # 1. Insert BigramHashEmbedding class before Rotary
    source = _replace_unique(
        source,
        "class Rotary(nn.Module):\n"
        "    # Caches cos/sin tables per sequence length on the current device.",

        "class BigramHashEmbedding(nn.Module):\n"
        "    def __init__(self, bigram_vocab_size: int, bigram_dim: int, model_dim: int):\n"
        "        super().__init__()\n"
        "        self.bigram_vocab_size = bigram_vocab_size\n"
        "        self.embed = nn.Embedding(bigram_vocab_size, bigram_dim)\n"
        "        nn.init.zeros_(self.embed.weight)\n"
        "        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None\n"
        "        if self.proj is not None:\n"
        "            nn.init.zeros_(self.proj.weight)\n"
        "        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))\n"
        "\n"
        "    def bigram_hash(self, tokens: Tensor) -> Tensor:\n"
        "        t = tokens.to(torch.int32)\n"
        "        mod = self.bigram_vocab_size - 1\n"
        "        out = torch.empty_like(t)\n"
        "        out[..., 0] = mod\n"
        "        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % mod\n"
        "        return out.long()\n"
        "\n"
        "    def forward(self, token_ids: Tensor) -> Tensor:\n"
        "        h = self.embed(self.bigram_hash(token_ids))\n"
        "        if self.proj is not None:\n"
        "            h = self.proj(h)\n"
        "        return h * self.scale.to(dtype=h.dtype)\n"
        "\n"
        "\n"
        "class Rotary(nn.Module):\n"
        "    # Caches cos/sin tables per sequence length on the current device.",
        name,
    )

    # 2. Add bigram module to GPT.__init__ (inserts before final_norm)
    source = _replace_unique(
        source,
        "        self.final_norm = RMSNorm()",
        "        self.bigram = BigramHashEmbedding(\n"
        "            bigram_vocab_size=int(os.environ.get(\"BIGRAM_BUCKETS\", \"4096\")),\n"
        "            bigram_dim=int(os.environ.get(\"BIGRAM_DIM\", \"128\")),\n"
        "            model_dim=model_dim,\n"
        "        ) if os.environ.get(\"BIGRAM_HASH\", \"0\") == \"1\" else None\n"
        "        self.final_norm = RMSNorm()",
        name,
    )

    # 3. Modify GPT.forward: add bigram after tok_emb, before rms_norm
    source = _replace_unique(
        source,
        "        x = self.tok_emb(input_ids)\n"
        "        x = F.rms_norm(x, (x.size(-1),))",
        "        x = self.tok_emb(input_ids)\n"
        "        if getattr(self, 'bigram', None) is not None:\n"
        "            x = x + self.bigram(input_ids)\n"
        "        x = F.rms_norm(x, (x.size(-1),))",
        name,
    )

    # 4. Add bigram params to optimizer
    source = _replace_unique(
        source,
        "    if base_model.skip_weights.numel() > 0:\n"
        "        scalar_params.append(base_model.skip_weights)",
        "    if base_model.skip_weights.numel() > 0:\n"
        "        scalar_params.append(base_model.skip_weights)\n"
        "    if getattr(base_model, 'bigram', None) is not None:\n"
        "        scalar_params.extend(base_model.bigram.parameters())",
        name,
    )

    return source


# ---------------------------------------------------------------------------
# Patch: orthoinit
# Target region: GPT._init_weights
# ---------------------------------------------------------------------------

def patch_orthoinit(source: str) -> str:
    """Replace default Kaiming init with orthogonal init for large linear layers."""
    name = "orthoinit"

    source = _replace_unique(
        source,
        "    def _init_weights(self) -> None:\n"
        "        if self.tie_embeddings:\n"
        "            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)\n"
        "        for module in self.modules():\n"
        "            if isinstance(module, nn.Linear) and getattr(module, \"_zero_init\", False):\n"
        "                nn.init.zeros_(module.weight)",
        "    def _init_weights(self) -> None:\n"
        "        if self.tie_embeddings:\n"
        "            nn.init.normal_(self.tok_emb.weight, mean=0.0, std=self.tied_embed_init_std)\n"
        "        for module in self.modules():\n"
        "            if isinstance(module, nn.Linear):\n"
        "                if getattr(module, \"_zero_init\", False):\n"
        "                    nn.init.zeros_(module.weight)\n"
        "                elif module.weight.ndim == 2 and min(module.weight.shape) >= 64:\n"
        "                    nn.init.orthogonal_(module.weight, gain=1.0)",
        name,
    )

    return source


# ---------------------------------------------------------------------------
# Patch: label_smoothing
# Target region: cross_entropy call in GPT.forward
# ---------------------------------------------------------------------------

def patch_label_smoothing(source: str) -> str:
    """Add label smoothing to the training cross-entropy loss."""
    name = "label_smoothing"

    source = _replace_unique(
        source,
        "        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction=\"mean\")",
        "        _ls = float(os.environ.get(\"LABEL_SMOOTHING\", \"0.0\"))\n"
        "        return F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), target_ids.reshape(-1), reduction=\"mean\", label_smoothing=_ls)",
        name,
    )

    return source


# ---------------------------------------------------------------------------
# Patch: compile_autotune
# Target region: torch.compile call in main()
# ---------------------------------------------------------------------------

def patch_compile_autotune(source: str) -> str:
    """Make torch.compile mode configurable via COMPILE_MODE env var."""
    name = "compile_autotune"

    source = _replace_unique(
        source,
        "    compiled_model = torch.compile(base_model, dynamic=False, fullgraph=True)",
        "    _compile_mode = os.environ.get(\"COMPILE_MODE\")\n"
        "    _compile_kwargs = dict(dynamic=False, fullgraph=True)\n"
        "    if _compile_mode:\n"
        "        _compile_kwargs[\"mode\"] = _compile_mode\n"
        "    compiled_model = torch.compile(base_model, **_compile_kwargs)",
        name,
    )

    return source


# ---------------------------------------------------------------------------
# Registry and apply function
# ---------------------------------------------------------------------------

PATCH_REGISTRY: dict[str, callable] = {
    "normuon_wd": patch_normuon_wd,
    "qat": patch_qat,
    "smeargate": patch_smeargate,
    "bigram_hash": patch_bigram_hash,
    "orthoinit": patch_orthoinit,
    "label_smoothing": patch_label_smoothing,
    "compile_autotune": patch_compile_autotune,
}


def apply_patches(source: str, patch_names: list[str]) -> str:
    """Apply a list of named patches to the source code, in order."""
    for patch_name in patch_names:
        if patch_name not in PATCH_REGISTRY:
            raise ValueError(
                f"Unknown patch: '{patch_name}'. Available: {sorted(PATCH_REGISTRY)}"
            )
        source = PATCH_REGISTRY[patch_name](source)
    return source


def list_patches() -> list[str]:
    """Return sorted list of available patch names."""
    return sorted(PATCH_REGISTRY)
