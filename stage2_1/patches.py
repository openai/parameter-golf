"""
Runtime code patches for stage2_1 train_gpt.py experiments.

Each patch function takes the full root train_gpt.py source and returns a modified
copy. The orchestrator writes that copy into the slot run directory so experiments
can run in parallel without mutating the shared source file.
"""
from __future__ import annotations


def _replace_unique(source: str, old: str, new: str, patch_name: str) -> str:
    count = source.count(old)
    if count == 0:
        raise ValueError(f"Patch '{patch_name}': target string not found")
    if count > 1:
        raise ValueError(f"Patch '{patch_name}': target string matched {count} times")
    return source.replace(old, new)


def patch_muon_weight_decay(source: str) -> str:
    name = "muon_weight_decay"
    source = _replace_unique(
        source,
        "class Muon(torch.optim.Optimizer):\n"
        "    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):\n"
        "        super().__init__(\n"
        "            params,\n"
        "            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),\n"
        "        )",
        "class Muon(torch.optim.Optimizer):\n"
        "    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, weight_decay: float = 0.0):\n"
        "        super().__init__(\n"
        "            params,\n"
        "            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, weight_decay=weight_decay),\n"
        "        )",
        name,
    )
    source = _replace_unique(
        source,
        "            curr = 0\n"
        "            for p in params:\n"
        "                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)\n"
        "                p.add_(g, alpha=-lr)\n"
        "                curr += p.numel()",
        "            weight_decay = group.get(\"weight_decay\", 0.0)\n"
        "            curr = 0\n"
        "            for p in params:\n"
        "                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)\n"
        "                if weight_decay > 0:\n"
        "                    p.mul_(1.0 - lr * weight_decay)\n"
        "                p.add_(g, alpha=-lr)\n"
        "                curr += p.numel()",
        name,
    )
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
        "        weight_decay=float(os.environ.get(\"MUON_WEIGHT_DECAY\", \"0.0\")),\n"
        "    )",
        name,
    )
    return source


def patch_normuon(source: str) -> str:
    name = "normuon"
    source = _replace_unique(
        source,
        "class Muon(torch.optim.Optimizer):\n"
        "    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True):\n"
        "        super().__init__(\n"
        "            params,\n"
        "            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov),\n"
        "        )",
        "class Muon(torch.optim.Optimizer):\n"
        "    def __init__(self, params, lr: float, momentum: float, backend_steps: int, nesterov: bool = True, beta2: float = 0.95):\n"
        "        super().__init__(\n"
        "            params,\n"
        "            dict(lr=lr, momentum=momentum, backend_steps=backend_steps, nesterov=nesterov, beta2=beta2),\n"
        "        )",
        name,
    )
    source = _replace_unique(
        source,
        "                    if \"momentum_buffer\" not in state:\n"
        "                        state[\"momentum_buffer\"] = torch.zeros_like(g)\n"
        "                    buf = state[\"momentum_buffer\"]",
        "                    if \"momentum_buffer\" not in state:\n"
        "                        state[\"momentum_buffer\"] = torch.zeros_like(g)\n"
        "                    if \"second_moment\" not in state:\n"
        "                        state[\"second_moment\"] = torch.zeros(g.shape[0], 1, device=g.device, dtype=g.dtype)\n"
        "                    buf = state[\"momentum_buffer\"]",
        name,
    )
    source = _replace_unique(
        source,
        "                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)\n"
        "                    # Scale correction from Muon reference implementations.\n"
        "                    g *= max(1, g.size(0) / g.size(1)) ** 0.5",
        "                    g = zeropower_via_newtonschulz5(g, steps=backend_steps)\n"
        "                    second_moment = state[\"second_moment\"]\n"
        "                    second_moment.lerp_((g * g).mean(dim=1, keepdim=True), 1.0 - group.get(\"beta2\", 0.95))\n"
        "                    g_norm = g.norm()\n"
        "                    g = g / second_moment.sqrt().add_(1e-10)\n"
        "                    g = g * (g_norm / g.norm().add_(1e-10))\n"
        "                    # Scale correction from Muon reference implementations.\n"
        "                    g *= max(1, g.size(0) / g.size(1)) ** 0.5",
        name,
    )
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
        "        beta2=float(os.environ.get(\"BETA2\", \"0.95\")),\n"
        "    )",
        name,
    )
    return source


def patch_ste_qat(source: str) -> str:
    name = "ste_qat"
    source = _replace_unique(
        source,
        "class CastedLinear(nn.Linear):\n"
        "    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.\n"
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        bias = self.bias.to(x.dtype) if self.bias is not None else None\n"
        "        return F.linear(x, self.weight.to(x.dtype), bias)",
        "class _FakeQuantSTE(torch.autograd.Function):\n"
        "    @staticmethod\n"
        "    def forward(ctx, w: Tensor) -> Tensor:\n"
        "        bits = max(int(os.environ.get(\"STE_QAT_BITS\", \"6\")), 2)\n"
        "        qmax = (1 << (bits - 1)) - 1\n"
        "        qmin = -(1 << (bits - 1))\n"
        "        w32 = w.float()\n"
        "        abs_max = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)\n"
        "        scale = abs_max / float(qmax)\n"
        "        q = torch.clamp(torch.round(w32 / scale), qmin, qmax)\n"
        "        return (q * scale).to(w.dtype)\n"
        "\n"
        "    @staticmethod\n"
        "    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:\n"
        "        return (grad_output,)\n"
        "\n"
        "\n"
        "_fake_quant_ste = _FakeQuantSTE.apply\n"
        "_STE_QAT_ENABLED = os.environ.get(\"STE_QAT_ENABLE\", \"0\") == \"1\"\n"
        "\n"
        "\n"
        "class CastedLinear(nn.Linear):\n"
        "    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.\n"
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        weight = self.weight.to(x.dtype)\n"
        "        if _STE_QAT_ENABLED and self.training and weight.ndim == 2:\n"
        "            weight = _fake_quant_ste(weight)\n"
        "        bias = self.bias.to(x.dtype) if self.bias is not None else None\n"
        "        return F.linear(x, weight, bias)",
        name,
    )
    return source


def patch_orthoinit_mup(source: str) -> str:
    name = "orthoinit_mup"
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
    source = _replace_unique(
        source,
        "        return self.proj(y)",
        "        return self.proj(y) * float(os.environ.get(\"MUP_OUTPUT_SCALE\", \"1.0\"))",
        name,
    )
    source = _replace_unique(
        source,
        "        return self.proj(x.square())",
        "        return self.proj(x.square()) * float(os.environ.get(\"MUP_OUTPUT_SCALE\", \"1.0\"))",
        name,
    )
    return source


def patch_smeargate(source: str) -> str:
    name = "smeargate"
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
        "        gate = torch.sigmoid(self.gate.to(dtype=x.dtype))[None, None, :]\n"
        "        prev = torch.cat([torch.zeros_like(x[:, :1]), x[:, :-1]], dim=1)\n"
        "        return (1.0 - gate) * x + gate * prev\n"
        "\n"
        "\n"
        "class Rotary(nn.Module):\n"
        "    # Caches cos/sin tables per sequence length on the current device.",
        name,
    )
    source = _replace_unique(
        source,
        "        self.final_norm = RMSNorm()\n"
        "        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)",
        "        self.smear = SmearGate(model_dim)\n"
        "        self.final_norm = RMSNorm()\n"
        "        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)",
        name,
    )
    source = _replace_unique(
        source,
        "        x = self.tok_emb(input_ids)\n"
        "        x = F.rms_norm(x, (x.size(-1),))\n"
        "        x0 = x",
        "        x = self.tok_emb(input_ids)\n"
        "        x = F.rms_norm(x, (x.size(-1),))\n"
        "        x = self.smear(x)\n"
        "        x0 = x",
        name,
    )
    source = _replace_unique(
        source,
        "    if base_model.skip_weights.numel() > 0:\n"
        "        scalar_params.append(base_model.skip_weights)\n"
        "    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr",
        "    if base_model.skip_weights.numel() > 0:\n"
        "        scalar_params.append(base_model.skip_weights)\n"
        "    scalar_params.extend(base_model.smear.parameters())\n"
        "    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr",
        name,
    )
    return source


def patch_bigramhash(source: str) -> str:
    name = "bigramhash"
    source = _replace_unique(
        source,
        "class Rotary(nn.Module):\n"
        "    # Caches cos/sin tables per sequence length on the current device.",
        "class BigramHashEmbedding(nn.Module):\n"
        "    def __init__(self, bucket_count: int, bigram_dim: int, model_dim: int):\n"
        "        super().__init__()\n"
        "        self.bucket_count = bucket_count\n"
        "        self.embed = nn.Embedding(bucket_count, bigram_dim)\n"
        "        self.proj = CastedLinear(bigram_dim, model_dim, bias=False) if bigram_dim != model_dim else None\n"
        "        self.scale = nn.Parameter(torch.tensor(0.05, dtype=torch.float32))\n"
        "        nn.init.zeros_(self.embed.weight)\n"
        "        if self.proj is not None:\n"
        "            nn.init.zeros_(self.proj.weight)\n"
        "\n"
        "    def _hash(self, token_ids: Tensor) -> Tensor:\n"
        "        t = token_ids.to(torch.int32)\n"
        "        out = torch.empty_like(t)\n"
        "        out[..., 0] = 0\n"
        "        out[..., 1:] = torch.bitwise_xor(36313 * t[..., 1:], 27191 * t[..., :-1]) % self.bucket_count\n"
        "        return out.long()\n"
        "\n"
        "    def forward(self, token_ids: Tensor) -> Tensor:\n"
        "        h = self.embed(self._hash(token_ids))\n"
        "        if self.proj is not None:\n"
        "            h = self.proj(h)\n"
        "        return h * self.scale.to(dtype=h.dtype)\n"
        "\n"
        "\n"
        "class Rotary(nn.Module):\n"
        "    # Caches cos/sin tables per sequence length on the current device.",
        name,
    )
    source = _replace_unique(
        source,
        "        self.final_norm = RMSNorm()\n"
        "        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)",
        "        self.bigram = BigramHashEmbedding(\n"
        "            bucket_count=int(os.environ.get(\"BIGRAM_HASH_BUCKETS\", \"4096\")),\n"
        "            bigram_dim=int(os.environ.get(\"BIGRAM_HASH_DIM\", \"128\")),\n"
        "            model_dim=model_dim,\n"
        "        )\n"
        "        self.final_norm = RMSNorm()\n"
        "        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)",
        name,
    )
    source = _replace_unique(
        source,
        "        x = self.tok_emb(input_ids)\n"
        "        x = F.rms_norm(x, (x.size(-1),))\n"
        "        x0 = x",
        "        x = self.tok_emb(input_ids)\n"
        "        x = x + self.bigram(input_ids)\n"
        "        x = F.rms_norm(x, (x.size(-1),))\n"
        "        x0 = x",
        name,
    )
    source = _replace_unique(
        source,
        "    if base_model.skip_weights.numel() > 0:\n"
        "        scalar_params.append(base_model.skip_weights)\n"
        "    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr",
        "    if base_model.skip_weights.numel() > 0:\n"
        "        scalar_params.append(base_model.skip_weights)\n"
        "    scalar_params.extend(base_model.bigram.parameters())\n"
        "    token_lr = args.tied_embed_lr if args.tie_embeddings else args.embed_lr",
        name,
    )
    return source


PATCH_REGISTRY = {
    "muon_weight_decay": patch_muon_weight_decay,
    "normuon": patch_normuon,
    "ste_qat": patch_ste_qat,
    "orthoinit_mup": patch_orthoinit_mup,
    "smeargate": patch_smeargate,
    "bigramhash": patch_bigramhash,
}


def apply_patches(source: str, patch_names: list[str]) -> str:
    for patch_name in patch_names:
        try:
            patch_fn = PATCH_REGISTRY[patch_name]
        except KeyError as exc:
            raise ValueError(f"Unknown patch '{patch_name}'. Available: {sorted(PATCH_REGISTRY)}") from exc
        source = patch_fn(source)
    return source


def list_patches() -> list[str]:
    return sorted(PATCH_REGISTRY)
