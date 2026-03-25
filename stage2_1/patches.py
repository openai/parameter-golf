"""
Runtime code patches for stage2_1 train_gpt.py experiments.

Each patch function takes the current root train_gpt.py source and returns a
modified copy. The orchestrator writes that copy into the slot run directory so
experiments can run in parallel without mutating the shared source file.
"""
from __future__ import annotations


def _replace_unique(source: str, old: str, new: str, patch_name: str) -> str:
    count = source.count(old)
    if count == 0:
        raise ValueError(f"Patch '{patch_name}': target string not found")
    if count > 1:
        raise ValueError(f"Patch '{patch_name}': target string matched {count} times")
    return source.replace(old, new)


def _replace_any(source: str, olds: list[str], new: str, patch_name: str) -> str:
    for old in olds:
        count = source.count(old)
        if count == 1:
            return source.replace(old, new)
        if count > 1:
            raise ValueError(f"Patch '{patch_name}': target string matched {count} times")
    raise ValueError(f"Patch '{patch_name}': target string not found")


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
        "                if weight_decay > 0.0:\n"
        "                    p.data.mul_(1.0 - lr * weight_decay)\n"
        "                g = updates_flat[curr : curr + p.numel()].view_as(p).to(dtype=p.dtype)\n"
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


def patch_leaky_relu_sq(source: str) -> str:
    name = "leaky_relu_sq"
    return _replace_unique(
        source,
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        x = torch.relu(self.fc(x))\n"
        "        return self.proj(x.square())",
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        x = F.leaky_relu(self.fc(x), negative_slope=float(os.environ.get(\"LEAKY_RELU_SLOPE\", \"0.5\")))\n"
        "        return self.proj(x.square())",
        name,
    )


def patch_ema(source: str) -> str:
    name = "ema"
    source = _replace_unique(
        source,
        "    if base_model.lm_head is not None:\n"
        "        optimizer_head = torch.optim.Adam(\n"
        "            [{\"params\": [base_model.lm_head.weight], \"lr\": args.head_lr, \"base_lr\": args.head_lr}],\n"
        "            betas=(args.beta1, args.beta2),\n"
        "            eps=args.adam_eps,\n"
        "            fused=True,\n"
        "        )\n"
        "        optimizers.insert(1, optimizer_head)",
        "    if base_model.lm_head is not None:\n"
        "        optimizer_head = torch.optim.Adam(\n"
        "            [{\"params\": [base_model.lm_head.weight], \"lr\": args.head_lr, \"base_lr\": args.head_lr}],\n"
        "            betas=(args.beta1, args.beta2),\n"
        "            eps=args.adam_eps,\n"
        "            fused=True,\n"
        "        )\n"
        "        optimizers.insert(1, optimizer_head)\n"
        "    ema_state: dict[str, Tensor] | None = None\n"
        "    ema_decay = float(os.environ.get(\"EMA_DECAY\", \"0.997\"))\n"
        "    ema_export_only = os.environ.get(\"EMA_EXPORT_ONLY\", \"1\") == \"1\"\n"
        "    if os.environ.get(\"EMA_ENABLE\", \"0\") == \"1\":\n"
        "        ema_state = {name: t.detach().float().clone() for name, t in base_model.state_dict().items()}",
        name,
    )
    source = _replace_unique(
        source,
        "        for opt in optimizers:\n"
        "            opt.step()\n"
        "        zero_grad_all()",
        "        for opt in optimizers:\n"
        "            opt.step()\n"
        "        if ema_state is not None:\n"
        "            for state_name, tensor in base_model.state_dict().items():\n"
        "                ema_state[state_name].mul_(ema_decay).add_(tensor.detach().float(), alpha=1.0 - ema_decay)\n"
        "        zero_grad_all()",
        name,
    )
    source = _replace_unique(
        source,
        "    # -----------------------------\n"
        "    # SERIALIZATION + ROUNDTRIP VALIDATION\n",
        "    if ema_state is not None and ema_export_only:\n"
        "        log0(\"ema:applying EMA weights\")\n"
        "        current_state = base_model.state_dict()\n"
        "        avg_state = {state_name: tensor.to(dtype=current_state[state_name].dtype) for state_name, tensor in ema_state.items()}\n"
        "        base_model.load_state_dict(avg_state, strict=True)\n"
        "\n"
        "    # -----------------------------\n"
        "    # SERIALIZATION + ROUNDTRIP VALIDATION\n",
        name,
    )
    return source


def patch_xsa4(source: str) -> str:
    name = "xsa4"
    source = _replace_any(
        source,
        [
            "        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))\n"
            "        self.rotary = Rotary(self.head_dim, base=rope_base)\n"
            "\n"
            "    def forward(self, x: Tensor) -> Tensor:",
            "        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))\n"
            "        self.rope_partial_dim = int(os.environ.get(\"ROPE_PARTIAL_DIM\", \"0\"))\n"
            "        self.rotary = Rotary(self.head_dim, base=rope_base, rope_dims=self.rope_partial_dim)\n"
            "\n"
            "    def forward(self, x: Tensor) -> Tensor:",
        ],
        "        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))\n"
        "        self.use_xsa = False\n"
        "        if os.environ.get(\"ROPE_PARTIAL_DIM\") is not None:\n"
        "            self.rope_partial_dim = int(os.environ.get(\"ROPE_PARTIAL_DIM\", \"0\"))\n"
        "            self.rotary = Rotary(self.head_dim, base=rope_base, rope_dims=self.rope_partial_dim)\n"
        "        else:\n"
        "            self.rotary = Rotary(self.head_dim, base=rope_base)\n"
        "\n"
        "    def _xsa_efficient(self, y: Tensor, v: Tensor) -> Tensor:\n"
        "        bsz, heads, seqlen, head_dim = y.shape\n"
        "        kv_heads = v.size(1)\n"
        "        group = heads // kv_heads\n"
        "        y_grouped = y.reshape(bsz, kv_heads, group, seqlen, head_dim)\n"
        "        v_norm = F.normalize(v, dim=-1).unsqueeze(2)\n"
        "        proj = (y_grouped * v_norm).sum(dim=-1, keepdim=True) * v_norm\n"
        "        return (y_grouped - proj).reshape(bsz, heads, seqlen, head_dim)\n"
        "\n"
        "\n"
        "    def forward(self, x: Tensor) -> Tensor:",
        name,
    )
    source = _replace_unique(
        source,
        "        y = F.scaled_dot_product_attention(\n"
        "            q,\n"
        "            k,\n"
        "            v,\n"
        "            attn_mask=None,\n"
        "            is_causal=True,\n"
        "            enable_gqa=(self.num_kv_heads != self.num_heads),\n"
        "        )\n"
        "        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)\n"
        "        return self.proj(y)",
        "        y = F.scaled_dot_product_attention(\n"
        "            q,\n"
        "            k,\n"
        "            v,\n"
        "            attn_mask=None,\n"
        "            is_causal=True,\n"
        "            enable_gqa=(self.num_kv_heads != self.num_heads),\n"
        "        )\n"
        "        if self.use_xsa:\n"
        "            y = self._xsa_efficient(y, v)\n"
        "        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)\n"
        "        return self.proj(y)",
        name,
    )
    source = _replace_unique(
        source,
        "        self.blocks = nn.ModuleList(\n"
        "            [\n"
        "                Block(\n"
        "                    model_dim,\n"
        "                    num_heads,\n"
        "                    num_kv_heads,\n"
        "                    mlp_mult,\n"
        "                    rope_base,\n"
        "                    qk_gain_init,\n"
        "                )\n"
        "                for i in range(num_layers)\n"
        "            ]\n"
        "        )\n"
        "        self.final_norm = RMSNorm()",
        "        self.blocks = nn.ModuleList(\n"
        "            [\n"
        "                Block(\n"
        "                    model_dim,\n"
        "                    num_heads,\n"
        "                    num_kv_heads,\n"
        "                    mlp_mult,\n"
        "                    rope_base,\n"
        "                    qk_gain_init,\n"
        "                )\n"
        "                for i in range(num_layers)\n"
        "            ]\n"
        "        )\n"
        "        xsa_layers_raw = os.environ.get(\"XSA_LAYERS\", \"4\").strip() if os.environ.get(\"XSA_ENABLE\", \"0\") == \"1\" else \"\"\n"
        "        if xsa_layers_raw:\n"
        "            if xsa_layers_raw.lower() == \"all\":\n"
        "                xsa_indices = range(num_layers)\n"
        "            elif \",\" in xsa_layers_raw:\n"
        "                xsa_indices = [int(tok.strip()) for tok in xsa_layers_raw.split(\",\") if tok.strip()]\n"
        "            else:\n"
        "                xsa_last_n = int(xsa_layers_raw)\n"
        "                xsa_indices = range(max(0, num_layers - xsa_last_n), num_layers)\n"
        "            for i in xsa_indices:\n"
        "                if 0 <= i < num_layers:\n"
        "                    self.blocks[i].attn.use_xsa = True\n"
        "        self.final_norm = RMSNorm()",
        name,
    )
    return source


def patch_xsa_all(source: str) -> str:
    return patch_xsa4(source)


def patch_partial_rope(source: str) -> str:
    name = "partial_rope"
    source = _replace_unique(
        source,
        "class Rotary(nn.Module):\n"
        "    # Caches cos/sin tables per sequence length on the current device.\n"
        "    def __init__(self, dim: int, base: float = 10000.0):\n"
        "        super().__init__()\n"
        "        inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))\n"
        "        self.register_buffer(\"inv_freq\", inv_freq, persistent=False)\n"
        "        self._seq_len_cached = 0\n"
        "        self._cos_cached: Tensor | None = None\n"
        "        self._sin_cached: Tensor | None = None\n"
        "\n"
        "    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:\n"
        "        if (\n"
        "            self._cos_cached is None\n"
        "            or self._sin_cached is None\n"
        "            or self._seq_len_cached != seq_len\n"
        "            or self._cos_cached.device != device\n"
        "        ):\n"
        "            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)\n"
        "            freqs = torch.outer(t, self.inv_freq.to(device))\n"
        "            self._cos_cached = freqs.cos()[None, None, :, :]\n"
        "            self._sin_cached = freqs.sin()[None, None, :, :]\n"
        "            self._seq_len_cached = seq_len\n"
        "        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)\n"
        "\n"
        "\n"
        "def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor) -> Tensor:\n"
        "    half = x.size(-1) // 2\n"
        "    x1, x2 = x[..., :half], x[..., half:]\n"
        "    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)",
        "class Rotary(nn.Module):\n"
        "    # Caches cos/sin tables per sequence length on the current device.\n"
        "    def __init__(self, dim: int, base: float = 10000.0, rope_dims: int = 0):\n"
        "        super().__init__()\n"
        "        self.rope_dims = rope_dims if rope_dims > 0 else dim\n"
        "        inv_freq = 1.0 / (base ** (torch.arange(0, self.rope_dims, 2, dtype=torch.float32) / self.rope_dims))\n"
        "        self.register_buffer(\"inv_freq\", inv_freq, persistent=False)\n"
        "        self._seq_len_cached = 0\n"
        "        self._cos_cached: Tensor | None = None\n"
        "        self._sin_cached: Tensor | None = None\n"
        "\n"
        "    def forward(self, seq_len: int, device: torch.device, dtype: torch.dtype) -> tuple[Tensor, Tensor]:\n"
        "        if (\n"
        "            self._cos_cached is None\n"
        "            or self._sin_cached is None\n"
        "            or self._seq_len_cached != seq_len\n"
        "            or self._cos_cached.device != device\n"
        "        ):\n"
        "            t = torch.arange(seq_len, device=device, dtype=self.inv_freq.dtype)\n"
        "            freqs = torch.outer(t, self.inv_freq.to(device))\n"
        "            self._cos_cached = freqs.cos()[None, None, :, :]\n"
        "            self._sin_cached = freqs.sin()[None, None, :, :]\n"
        "            self._seq_len_cached = seq_len\n"
        "        return self._cos_cached.to(dtype=dtype), self._sin_cached.to(dtype=dtype)\n"
        "\n"
        "\n"
        "def apply_rotary_emb(x: Tensor, cos: Tensor, sin: Tensor, rope_dims: int = 0) -> Tensor:\n"
        "    if rope_dims > 0 and rope_dims < x.size(-1):\n"
        "        x_rope, x_pass = x[..., :rope_dims], x[..., rope_dims:]\n"
        "        half = rope_dims // 2\n"
        "        x1, x2 = x_rope[..., :half], x_rope[..., half:]\n"
        "        x_rot = torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)\n"
        "        return torch.cat((x_rot, x_pass), dim=-1)\n"
        "    half = x.size(-1) // 2\n"
        "    x1, x2 = x[..., :half], x[..., half:]\n"
        "    return torch.cat((x1 * cos + x2 * sin, x1 * (-sin) + x2 * cos), dim=-1)",
        name,
    )
    source = _replace_unique(
        source,
        "        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))\n"
        "        self.rotary = Rotary(self.head_dim, base=rope_base)",
        "        self.q_gain = nn.Parameter(torch.full((num_heads,), qk_gain_init, dtype=torch.float32))\n"
        "        self.rope_partial_dim = int(os.environ.get(\"ROPE_PARTIAL_DIM\", \"0\"))\n"
        "        self.rotary = Rotary(self.head_dim, base=rope_base, rope_dims=self.rope_partial_dim)",
        name,
    )
    if "        q = apply_rotary_emb(q, cos, sin)\n        k = apply_rotary_emb(k, cos, sin)" in source:
        source = _replace_unique(
            source,
            "        q = apply_rotary_emb(q, cos, sin)\n"
            "        k = apply_rotary_emb(k, cos, sin)",
            "        if hasattr(self, \"rope_partial_dim\"):\n"
            "            q = apply_rotary_emb(q, cos, sin, self.rope_partial_dim)\n"
            "            k = apply_rotary_emb(k, cos, sin, self.rope_partial_dim)\n"
            "        else:\n"
            "            q = apply_rotary_emb(q, cos, sin)\n"
            "            k = apply_rotary_emb(k, cos, sin)",
            name,
        )
    return source


def patch_ln_scale(source: str) -> str:
    name = "ln_scale"
    source = _replace_unique(
        source,
        "        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))\n"
        "        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))\n"
        "        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())",
        "        self.attn_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))\n"
        "        self.mlp_scale = nn.Parameter(torch.ones(dim, dtype=torch.float32))\n"
        "        self.resid_mix = nn.Parameter(torch.stack((torch.ones(dim), torch.zeros(dim))).float())\n"
        "        self.ln_scale_factor = 1.0",
        name,
    )
    source = _replace_unique(
        source,
        "        mix = self.resid_mix.to(dtype=x.dtype)\n"
        "        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0\n"
        "        attn_out = self.attn(self.attn_norm(x))\n"
        "        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out\n"
        "        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x))\n"
        "        return x",
        "        mix = self.resid_mix.to(dtype=x.dtype)\n"
        "        x = mix[0][None, None, :] * x + mix[1][None, None, :] * x0\n"
        "        scale = float(self.ln_scale_factor)\n"
        "        attn_out = self.attn(self.attn_norm(x) * scale)\n"
        "        x = x + self.attn_scale.to(dtype=x.dtype)[None, None, :] * attn_out\n"
        "        x = x + self.mlp_scale.to(dtype=x.dtype)[None, None, :] * self.mlp(self.mlp_norm(x) * scale)\n"
        "        return x",
        name,
    )
    source = _replace_unique(
        source,
        "        if self.lm_head is not None:\n"
        "            self.lm_head._zero_init = True\n"
        "        self._init_weights()",
        "        if self.lm_head is not None:\n"
        "            self.lm_head._zero_init = True\n"
        "        if os.environ.get(\"LN_SCALE_ENABLE\", \"0\") == \"1\":\n"
        "            for i, block in enumerate(self.blocks):\n"
        "                block.ln_scale_factor = 1.0 / math.sqrt(i + 1)\n"
        "        self._init_weights()",
        name,
    )
    return source


def patch_zstd_export(source: str) -> str:
    name = "zstd_export"
    source = _replace_unique(
        source,
        "import zlib\n"
        "from pathlib import Path",
        "import zlib\n"
        "from pathlib import Path\n"
        "try:\n"
        "    import zstandard\n"
        "except ImportError:\n"
        "    zstandard = None",
        name,
    )
    source = _replace_unique(
        source,
        "    quant_blob = zlib.compress(quant_raw, level=9)",
        "    use_zstd = os.environ.get(\"EXPORT_COMPRESSOR\", \"zlib\") == \"zstd\" and zstandard is not None\n"
        "    if use_zstd:\n"
        "        quant_blob = zstandard.ZstdCompressor(level=int(os.environ.get(\"ZSTD_LEVEL\", \"22\"))).compress(quant_raw)\n"
        "    else:\n"
        "        quant_blob = zlib.compress(quant_raw, level=9)",
        name,
    )
    source = _replace_unique(
        source,
        "    with open(\"final_model.int8.ptz\", \"rb\") as f:\n"
        "        quant_blob_disk = f.read()\n"
        "    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location=\"cpu\")",
        "    with open(\"final_model.int8.ptz\", \"rb\") as f:\n"
        "        quant_blob_disk = f.read()\n"
        "    use_zstd = os.environ.get(\"EXPORT_COMPRESSOR\", \"zlib\") == \"zstd\" and zstandard is not None\n"
        "    quant_payload = zstandard.ZstdDecompressor().decompress(quant_blob_disk) if use_zstd else zlib.decompress(quant_blob_disk)\n"
        "    quant_state = torch.load(io.BytesIO(quant_payload), map_location=\"cpu\")",
        name,
    )
    return source


def patch_fp16_embedding_export(source: str) -> str:
    name = "fp16_embedding_export"
    return _replace_unique(
        source,
        "def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:\n"
        "    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):\n"
        "        return t.float().contiguous()\n"
        "    if t.dtype in {torch.float32, torch.bfloat16}:\n"
        "        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix(\"torch.\")\n"
        "        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()\n"
        "    return t",
        "def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:\n"
        "    if name == \"tok_emb.weight\":\n"
        "        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix(\"torch.\")\n"
        "        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()\n"
        "    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):\n"
        "        return t.float().contiguous()\n"
        "    if t.dtype in {torch.float32, torch.bfloat16}:\n"
        "        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix(\"torch.\")\n"
        "        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()\n"
        "    return t",
        name,
    )


def patch_gptq_lite(source: str) -> str:
    name = "gptq_lite"
    return _replace_unique(
        source,
        "def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:\n"
        "    t32 = t.float()\n"
        "    if t32.ndim == 2:\n"
        "        # Matrices get one scale per row, which usually tracks output-channel\n"
        "        # ranges much better than a single tensor-wide scale.\n"
        "        clip_abs = (\n"
        "            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)\n"
        "            if t32.numel()\n"
        "            else torch.empty((t32.shape[0],), dtype=torch.float32)\n"
        "        )\n"
        "        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])\n"
        "        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)\n"
        "        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()\n"
        "        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()\n"
        "\n"
        "    # Vectors / scalars use a simpler per-tensor scale.\n"
        "    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0\n"
        "    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)\n"
        "    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()\n"
        "    return q, scale",
        "def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:\n"
        "    t32 = t.float()\n"
        "    use_clip_search = os.environ.get(\"GPTQ_LITE_ENABLE\", \"0\") == \"1\"\n"
        "    if use_clip_search:\n"
        "        raw_pcts = os.environ.get(\"GPTQ_LITE_PCTS\", \"99.9,99.95,99.99,99.99984,100.0\")\n"
        "        percentiles = []\n"
        "        for raw in raw_pcts.split(\",\"):\n"
        "            raw = raw.strip()\n"
        "            if not raw:\n"
        "                continue\n"
        "            pct = float(raw)\n"
        "            percentiles.append(pct / 100.0 if pct > 1.0 else pct)\n"
        "        percentiles = percentiles or [INT8_CLIP_Q]\n"
        "        if t32.ndim == 2:\n"
        "            best_err = None\n"
        "            best_q = None\n"
        "            best_scale = None\n"
        "            for pct in percentiles:\n"
        "                clip_abs = torch.quantile(t32.abs(), pct, dim=1) if t32.numel() else torch.empty((t32.shape[0],), dtype=torch.float32)\n"
        "                clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])\n"
        "                scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)\n"
        "                q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8)\n"
        "                recon = q.float() * scale[:, None]\n"
        "                err = (recon - t32).pow(2).mean(dim=1)\n"
        "                if best_err is None:\n"
        "                    best_err = err\n"
        "                    best_q = q\n"
        "                    best_scale = scale\n"
        "                else:\n"
        "                    mask = err < best_err\n"
        "                    best_err = torch.where(mask, err, best_err)\n"
        "                    best_q = torch.where(mask[:, None], q, best_q)\n"
        "                    best_scale = torch.where(mask, scale, best_scale)\n"
        "            return best_q.contiguous(), best_scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()\n"
        "        best_err = None\n"
        "        best_q = None\n"
        "        best_scale = None\n"
        "        for pct in percentiles:\n"
        "            clip_abs = float(torch.quantile(t32.abs().flatten(), pct).item()) if t32.numel() else 0.0\n"
        "            scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)\n"
        "            q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8)\n"
        "            recon = q.float() * scale\n"
        "            err = float((recon - t32).pow(2).mean().item()) if t32.numel() else 0.0\n"
        "            if best_err is None or err < best_err:\n"
        "                best_err = err\n"
        "                best_q = q\n"
        "                best_scale = scale\n"
        "        return best_q.contiguous(), best_scale\n"
        "    if t32.ndim == 2:\n"
        "        # Matrices get one scale per row, which usually tracks output-channel\n"
        "        # ranges much better than a single tensor-wide scale.\n"
        "        clip_abs = (\n"
        "            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)\n"
        "            if t32.numel()\n"
        "            else torch.empty((t32.shape[0],), dtype=torch.float32)\n"
        "        )\n"
        "        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])\n"
        "        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)\n"
        "        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()\n"
        "        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()\n"
        "\n"
        "    # Vectors / scalars use a simpler per-tensor scale.\n"
        "    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0\n"
        "    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)\n"
        "    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()\n"
        "    return q, scale",
        name,
    )


def patch_sliding_eval(source: str) -> str:
    name = "sliding_eval"
    source = _replace_unique(
        source,
        "    train_batch_tokens = int(os.environ.get(\"TRAIN_BATCH_TOKENS\", 524_288))\n"
        "    train_seq_len = int(os.environ.get(\"TRAIN_SEQ_LEN\", 1024))\n"
        "    max_wallclock_seconds = float(os.environ.get(\"MAX_WALLCLOCK_SECONDS\", 600.0))",
        "    train_batch_tokens = int(os.environ.get(\"TRAIN_BATCH_TOKENS\", 524_288))\n"
        "    train_seq_len = int(os.environ.get(\"TRAIN_SEQ_LEN\", 1024))\n"
        "    eval_seq_len = int(os.environ.get(\"EVAL_SEQ_LEN\", os.environ.get(\"TRAIN_SEQ_LEN\", 1024)))\n"
        "    eval_stride = int(os.environ.get(\"EVAL_STRIDE\", 256))\n"
        "    eval_batch_seqs = int(os.environ.get(\"EVAL_BATCH_SEQS\", 32))\n"
        "    max_wallclock_seconds = float(os.environ.get(\"MAX_WALLCLOCK_SECONDS\", 600.0))",
        name,
    )
    source = _replace_unique(
        source,
        "        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)\n"
        "        return F.cross_entropy(logits.float(), targets, reduction=\"mean\")",
        "        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)\n"
        "        return F.cross_entropy(logits.float(), targets, reduction=\"mean\")\n"
        "\n"
        "    def forward_logits(self, input_ids: Tensor) -> Tensor:\n"
        "        x = self.tok_emb(input_ids)\n"
        "        x = F.rms_norm(x, (x.size(-1),))\n"
        "        x0 = x\n"
        "        skips: list[Tensor] = []\n"
        "        for i in range(self.num_encoder_layers):\n"
        "            x = self.blocks[i](x, x0)\n"
        "            skips.append(x)\n"
        "        for i in range(self.num_decoder_layers):\n"
        "            if skips:\n"
        "                x = x + self.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()\n"
        "            x = self.blocks[self.num_encoder_layers + i](x, x0)\n"
        "        x = self.final_norm(x)\n"
        "        if self.tie_embeddings:\n"
        "            logits_proj = F.linear(x, self.tok_emb.weight)\n"
        "        else:\n"
        "            if self.lm_head is None:\n"
        "                raise RuntimeError(\"lm_head is required when tie_embeddings=False\")\n"
        "            logits_proj = self.lm_head(x)\n"
        "        return self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)",
        name,
    )
    source = _replace_unique(
        source,
        "    val_loss = val_loss_sum / val_token_count\n"
        "    bits_per_token = val_loss.item() / math.log(2.0)\n"
        "    tokens_per_byte = val_token_count.item() / val_byte_count.item()\n"
        "    model.train()\n"
        "    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)\n"
        "\n"
        "# -----------------------------\n"
        "# POST-TRAINING QUANTIZATION\n",
        "    val_loss = val_loss_sum / val_token_count\n"
        "    bits_per_token = val_loss.item() / math.log(2.0)\n"
        "    tokens_per_byte = val_token_count.item() / val_byte_count.item()\n"
        "    model.train()\n"
        "    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)\n"
        "\n"
        "\n"
        "def eval_val_sliding(\n"
        "    args: Hyperparameters,\n"
        "    base_model: nn.Module,\n"
        "    rank: int,\n"
        "    world_size: int,\n"
        "    device: torch.device,\n"
        "    val_tokens: Tensor,\n"
        "    base_bytes_lut: Tensor,\n"
        "    has_leading_space_lut: Tensor,\n"
        "    is_boundary_token_lut: Tensor,\n"
        "    stride: int,\n"
        "    batch_seqs: int = 32,\n"
        "    eval_seq_len: int | None = None,\n"
        ") -> tuple[float, float]:\n"
        "    seq_len = eval_seq_len or args.train_seq_len\n"
        "    total_tokens = val_tokens.numel() - 1\n"
        "    window_starts = [ws for ws in range(0, total_tokens, stride) if min(ws + seq_len, total_tokens) - ws >= 1]\n"
        "    total_windows = len(window_starts)\n"
        "    my_s = (total_windows * rank) // world_size\n"
        "    my_e = (total_windows * (rank + 1)) // world_size\n"
        "    my_windows = window_starts[my_s:my_e]\n"
        "    loss_sum = torch.zeros((), device=device, dtype=torch.float64)\n"
        "    token_count = torch.zeros((), device=device, dtype=torch.float64)\n"
        "    byte_count = torch.zeros((), device=device, dtype=torch.float64)\n"
        "    base_model.eval()\n"
        "    compiled_logits = torch.compile(base_model.forward_logits, dynamic=False, fullgraph=True)\n"
        "    with torch.inference_mode():\n"
        "        for bi in range(0, len(my_windows), batch_seqs):\n"
        "            batch_ws = my_windows[bi:bi + batch_seqs]\n"
        "            bsz = len(batch_ws)\n"
        "            x_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)\n"
        "            y_batch = torch.zeros(bsz, seq_len, dtype=torch.int64, device=device)\n"
        "            wlens: list[int] = []\n"
        "            for i, ws in enumerate(batch_ws):\n"
        "                end = min(ws + seq_len, total_tokens)\n"
        "                wlen = end - ws\n"
        "                wlens.append(wlen)\n"
        "                chunk = val_tokens[ws:end + 1].to(dtype=torch.int64, device=device)\n"
        "                x_batch[i, :wlen] = chunk[:-1]\n"
        "                y_batch[i, :wlen] = chunk[1:]\n"
        "            with torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16):\n"
        "                logits = compiled_logits(x_batch)\n"
        "            nll = F.cross_entropy(\n"
        "                logits.reshape(-1, logits.size(-1)).float(),\n"
        "                y_batch.reshape(-1),\n"
        "                reduction=\"none\",\n"
        "            ).reshape(bsz, seq_len)\n"
        "            for i, ws in enumerate(batch_ws):\n"
        "                wlen = wlens[i]\n"
        "                s = 0 if ws == 0 else max(wlen - stride, 0)\n"
        "                scored_nll = nll[i, s:wlen].to(torch.float64)\n"
        "                loss_sum += scored_nll.sum()\n"
        "                token_count += float(wlen - s)\n"
        "                tgt = y_batch[i, s:wlen]\n"
        "                prev = x_batch[i, s:wlen]\n"
        "                tb = base_bytes_lut[tgt].to(torch.float64)\n"
        "                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)\n"
        "                byte_count += tb.sum()\n"
        "    if dist.is_available() and dist.is_initialized():\n"
        "        dist.all_reduce(loss_sum, op=dist.ReduceOp.SUM)\n"
        "        dist.all_reduce(token_count, op=dist.ReduceOp.SUM)\n"
        "        dist.all_reduce(byte_count, op=dist.ReduceOp.SUM)\n"
        "    val_loss = loss_sum / token_count\n"
        "    bits_per_token = val_loss.item() / math.log(2.0)\n"
        "    tokens_per_byte = token_count.item() / byte_count.item()\n"
        "    base_model.train()\n"
        "    return float(val_loss.item()), float(bits_per_token * tokens_per_byte)\n"
        "\n"
        "# -----------------------------\n"
        "# POST-TRAINING QUANTIZATION\n",
        name,
    )
    source = _replace_unique(
        source,
        "    log0(f\"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}\")\n"
        "\n"
        "    if distributed:\n"
        "        dist.destroy_process_group()",
        "    log0(f\"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}\")\n"
        "    if args.eval_stride > 0:\n"
        "        torch.cuda.synchronize()\n"
        "        t_sw = time.perf_counter()\n"
        "        sw_val_loss, sw_val_bpb = eval_val_sliding(\n"
        "            args,\n"
        "            base_model,\n"
        "            rank,\n"
        "            world_size,\n"
        "            device,\n"
        "            val_tokens,\n"
        "            base_bytes_lut,\n"
        "            has_leading_space_lut,\n"
        "            is_boundary_token_lut,\n"
        "            stride=args.eval_stride,\n"
        "            batch_seqs=args.eval_batch_seqs,\n"
        "            eval_seq_len=args.eval_seq_len,\n"
        "        )\n"
        "        torch.cuda.synchronize()\n"
        "        log0(\n"
        "            f\"final_int8_zlib_roundtrip_sliding val_loss:{sw_val_loss:.4f} val_bpb:{sw_val_bpb:.4f} \"\n"
        "            f\"eval_time:{1000.0 * (time.perf_counter() - t_sw):.0f}ms\"\n"
        "        )\n"
        "        log0(f\"final_int8_zlib_roundtrip_sliding_exact val_loss:{sw_val_loss:.8f} val_bpb:{sw_val_bpb:.8f}\")\n"
        "\n"
        "    if distributed:\n"
        "        dist.destroy_process_group()",
        name,
    )
    return source


def patch_curriculum_shard_order(source: str) -> str:
    name = "curriculum_shard_order"
    source = _replace_unique(
        source,
        "def load_data_shard(file: Path) -> Tensor:\n"
        "    header_bytes = 256 * np.dtype(\"<i4\").itemsize\n"
        "    token_bytes = np.dtype(\"<u2\").itemsize\n"
        "    header = np.fromfile(file, dtype=\"<i4\", count=256)\n"
        "    # SHARD HEADER INTS & SHARD_MAGIC\n"
        "    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:\n"
        "        raise ValueError(f\"Unexpected shard header for {file}\")\n"
        "    num_tokens = int(header[2])\n"
        "    expected_size = header_bytes + num_tokens * token_bytes\n"
        "    if file.stat().st_size != expected_size:\n"
        "        raise ValueError(f\"Shard size mismatch for {file}: expected {expected_size} bytes\")\n"
        "    tokens_np = np.fromfile(file, dtype=\"<u2\", count=num_tokens, offset=header_bytes)\n"
        "    if tokens_np.size != num_tokens:\n"
        "        raise ValueError(f\"Short read for {file}\")\n"
        "    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))\n"
        "\n"
        "\n"
        "class TokenStream:\n",
        "def load_data_shard(file: Path) -> Tensor:\n"
        "    header_bytes = 256 * np.dtype(\"<i4\").itemsize\n"
        "    token_bytes = np.dtype(\"<u2\").itemsize\n"
        "    header = np.fromfile(file, dtype=\"<i4\", count=256)\n"
        "    # SHARD HEADER INTS & SHARD_MAGIC\n"
        "    if header.size != 256 or int(header[0]) != 20240520 or int(header[1]) != 1:\n"
        "        raise ValueError(f\"Unexpected shard header for {file}\")\n"
        "    num_tokens = int(header[2])\n"
        "    expected_size = header_bytes + num_tokens * token_bytes\n"
        "    if file.stat().st_size != expected_size:\n"
        "        raise ValueError(f\"Shard size mismatch for {file}: expected {expected_size} bytes\")\n"
        "    tokens_np = np.fromfile(file, dtype=\"<u2\", count=num_tokens, offset=header_bytes)\n"
        "    if tokens_np.size != num_tokens:\n"
        "        raise ValueError(f\"Short read for {file}\")\n"
        "    return torch.from_numpy(tokens_np.astype(np.uint16, copy=False))\n"
        "\n"
        "\n"
        "def order_shard_files(files: list[Path]) -> list[Path]:\n"
        "    mode = os.environ.get(\"SHARD_ORDER\", \"lexicographic\").strip()\n"
        "    source_path = os.environ.get(\"SHARD_ORDER_SOURCE\", \"\").strip()\n"
        "    if not files:\n"
        "        return files\n"
        "    if source_path:\n"
        "        order_path = Path(source_path)\n"
        "        if not order_path.exists():\n"
        "            raise FileNotFoundError(f\"SHARD_ORDER_SOURCE not found: {order_path}\")\n"
        "        ordered_names = [line.strip() for line in order_path.read_text(encoding=\"utf-8\").splitlines() if line.strip()]\n"
        "        rank_map = {name: idx for idx, name in enumerate(ordered_names)}\n"
        "        default_rank = len(rank_map)\n"
        "        return sorted(files, key=lambda path: (rank_map.get(path.name, default_rank), path.name))\n"
        "    if mode in {\"\", \"lexicographic\", \"sorted\"}:\n"
        "        return files\n"
        "    if mode == \"reverse\":\n"
        "        return list(reversed(files))\n"
        "    if mode == \"random\":\n"
        "        seed = int(os.environ.get(\"SHARD_ORDER_SEED\", os.environ.get(\"SEED\", \"1337\")))\n"
        "        shuffled = list(files)\n"
        "        random.Random(seed).shuffle(shuffled)\n"
        "        return shuffled\n"
        "    if mode == \"size_desc\":\n"
        "        return sorted(files, key=lambda path: (-path.stat().st_size, path.name))\n"
        "    if mode == \"size_asc\":\n"
        "        return sorted(files, key=lambda path: (path.stat().st_size, path.name))\n"
        "    order_path = Path(mode)\n"
        "    if order_path.exists():\n"
        "        ordered_names = [line.strip() for line in order_path.read_text(encoding=\"utf-8\").splitlines() if line.strip()]\n"
        "        rank_map = {name: idx for idx, name in enumerate(ordered_names)}\n"
        "        default_rank = len(rank_map)\n"
        "        return sorted(files, key=lambda path: (rank_map.get(path.name, default_rank), path.name))\n"
        "    raise ValueError(f\"Unsupported SHARD_ORDER={mode!r}\")\n"
        "\n"
        "\n"
        "class TokenStream:\n",
        name,
    )
    source = _replace_unique(
        source,
        "    def __init__(self, pattern: str):\n"
        "        self.files = [Path(p) for p in sorted(glob.glob(pattern))]\n"
        "        if not self.files:\n"
        "            raise FileNotFoundError(f\"No files found for pattern: {pattern}\")\n"
        "        self.file_idx = 0\n"
        "        self.tokens = load_data_shard(self.files[0])\n"
        "        self.pos = 0\n",
        "    def __init__(self, pattern: str):\n"
        "        self.files = order_shard_files([Path(p) for p in sorted(glob.glob(pattern))])\n"
        "        if not self.files:\n"
        "            raise FileNotFoundError(f\"No files found for pattern: {pattern}\")\n"
        "        self.file_idx = 0\n"
        "        self.tokens = load_data_shard(self.files[0])\n"
        "        self.pos = 0\n",
        name,
    )
    return source


def patch_weight_perturbation(source: str) -> str:
    name = "weight_perturbation"
    source = _replace_unique(
        source,
        "        step += 1\n"
        "        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)",
        "        step += 1\n"
        "\n"
        "        # Stochastic weight perturbation after the optimizer step.\n"
        "        _swp_scale = float(os.environ.get(\"WEIGHT_PERTURB_SCALE\", \"0.0\"))\n"
        "        if _swp_scale > 0:\n"
        "            _swp_start = float(os.environ.get(\"WEIGHT_PERTURB_START_FRAC\", \"0.0\"))\n"
        "            _swp_end = float(os.environ.get(\"WEIGHT_PERTURB_END_FRAC\", \"1.0\"))\n"
        "            _swp_frac = step / max(float(args.iterations), 1.0)\n"
        "            if _swp_start <= _swp_frac <= _swp_end:\n"
        "                with torch.no_grad():\n"
        "                    for p in matrix_params:\n"
        "                        p.add_(torch.randn_like(p) * (scale * _swp_scale))\n"
        "\n"
        "        approx_training_time_ms = training_time_ms + 1000.0 * (time.perf_counter() - t0)",
        name,
    )
    return source


def patch_grad_centralization(source: str) -> str:
    name = "grad_centralization"
    source = _replace_unique(
        source,
        "        if args.grad_clip_norm > 0:\n"
        "            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)\n"
        "        for opt in optimizers:\n"
        "            opt.step()",
        "        if args.grad_clip_norm > 0:\n"
        "            torch.nn.utils.clip_grad_norm_(base_model.parameters(), args.grad_clip_norm)\n"
        "        if os.environ.get(\"GRAD_CENTRALIZATION\", \"0\") == \"1\":\n"
        "            with torch.no_grad():\n"
        "                for p in matrix_params:\n"
        "                    if p.grad is not None and p.grad.ndim >= 2:\n"
        "                        p.grad.sub_(p.grad.mean(dim=tuple(range(1, p.grad.ndim)), keepdim=True))\n"
        "        for opt in optimizers:\n"
        "            opt.step()",
        name,
    )
    return source


def patch_fa3(source: str) -> str:
    name = "fa3"
    source = _replace_unique(
        source,
        "import torch.distributed as dist\n"
        "import torch.nn.functional as F\n"
        "from torch import Tensor, nn\n"
        "from torch.nn.parallel import DistributedDataParallel as DDP\n",
        "import torch.distributed as dist\n"
        "import torch.nn.functional as F\n"
        "from torch import Tensor, nn\n"
        "from torch.nn.parallel import DistributedDataParallel as DDP\n"
        "\n"
        "try:\n"
        "    from flash_attn_interface import flash_attn_func as flash_attn_3_func\n"
        "    _FLASH_ATTN_3_AVAILABLE = True\n"
        "except ImportError:\n"
        "    flash_attn_3_func = None\n"
        "    _FLASH_ATTN_3_AVAILABLE = False\n",
        name,
    )
    source = _replace_any(
        source,
        [
            "        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]\n"
            "        y = F.scaled_dot_product_attention(\n"
            "            q,\n"
            "            k,\n"
            "            v,\n"
            "            attn_mask=None,\n"
            "            is_causal=True,\n"
            "            enable_gqa=(self.num_kv_heads != self.num_heads),\n"
            "        )\n"
            "        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)\n"
            "        return self.proj(y)",
            "        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]\n"
            "        y = F.scaled_dot_product_attention(\n"
            "            q,\n"
            "            k,\n"
            "            v,\n"
            "            attn_mask=None,\n"
            "            is_causal=True,\n"
            "            enable_gqa=(self.num_kv_heads != self.num_heads),\n"
            "        )\n"
            "        if self.use_xsa:\n"
            "            y = self._xsa_efficient(y, v)\n"
            "        y = y.transpose(1, 2).contiguous().reshape(bsz, seqlen, dim)\n"
            "        return self.proj(y)",
        ],
        "        q = q * self.q_gain.to(dtype=q.dtype)[None, :, None, None]\n"
        "        if os.environ.get(\"FA3_ENABLE\", \"0\") == \"1\":\n"
        "            if not _FLASH_ATTN_3_AVAILABLE:\n"
        "                raise RuntimeError(\"FA3_ENABLE=1 but flash_attn_interface is unavailable\")\n"
        "            y = flash_attn_3_func(\n"
        "                q.transpose(1, 2).contiguous(),\n"
        "                k.transpose(1, 2).contiguous(),\n"
        "                v.transpose(1, 2).contiguous(),\n"
        "                causal=True,\n"
        "            )\n"
        "            if getattr(self, \"use_xsa\", False):\n"
        "                y = self._xsa_efficient(y.transpose(1, 2).contiguous(), v).transpose(1, 2).contiguous()\n"
        "        else:\n"
        "            y = F.scaled_dot_product_attention(\n"
        "                q,\n"
        "                k,\n"
        "                v,\n"
        "                attn_mask=None,\n"
        "                is_causal=True,\n"
        "                enable_gqa=(self.num_kv_heads != self.num_heads),\n"
        "            )\n"
        "            if getattr(self, \"use_xsa\", False):\n"
        "                y = self._xsa_efficient(y, v)\n"
        "            y = y.transpose(1, 2).contiguous()\n"
        "        y = y.reshape(bsz, seqlen, dim)\n"
        "        return self.proj(y)",
        name,
    )
    source = _replace_unique(
        source,
        "    log0(\"sdp_backends:cudnn=False flash=True mem_efficient=False math=False\")\n"
        "    log0(f\"attention_mode:gqa num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}\")\n",
        "    log0(f\"sdp_backends:cudnn=False flash=True mem_efficient=False math=False fa3:{os.environ.get('FA3_ENABLE', '0')}\")\n"
        "    log0(\n"
        "        f\"attention_mode:{'fa3_gqa' if os.environ.get('FA3_ENABLE', '0') == '1' else 'gqa'} \"\n"
        "        f\"num_heads:{args.num_heads} num_kv_heads:{args.num_kv_heads}\"\n"
        "    )\n",
        name,
    )
    return source


PATCH_REGISTRY = {
    "curriculum_shard_order": patch_curriculum_shard_order,
    "ema": patch_ema,
    "fa3": patch_fa3,
    "fp16_embedding_export": patch_fp16_embedding_export,
    "gptq_lite": patch_gptq_lite,
    "grad_centralization": patch_grad_centralization,
    "leaky_relu_sq": patch_leaky_relu_sq,
    "ln_scale": patch_ln_scale,
    "muon_weight_decay": patch_muon_weight_decay,
    "partial_rope": patch_partial_rope,
    "sliding_eval": patch_sliding_eval,
    "weight_perturbation": patch_weight_perturbation,
    "xsa4": patch_xsa4,
    "xsa_all": patch_xsa_all,
    "zstd_export": patch_zstd_export,
}

PATCH_ORDER = {
    "partial_rope": 10,
    "xsa4": 20,
    "xsa_all": 20,
    "ln_scale": 30,
}


def apply_patches(source: str, patch_names: list[str]) -> str:
    ordered_patch_names = sorted(
        enumerate(patch_names),
        key=lambda item: (PATCH_ORDER.get(item[1], 100), item[0]),
    )
    for _, patch_name in ordered_patch_names:
        try:
            patch_fn = PATCH_REGISTRY[patch_name]
        except KeyError as exc:
            raise ValueError(f"Unknown patch '{patch_name}'. Available: {sorted(PATCH_REGISTRY)}") from exc
        source = patch_fn(source)
    return source


def list_patches() -> list[str]:
    return sorted(PATCH_REGISTRY)
