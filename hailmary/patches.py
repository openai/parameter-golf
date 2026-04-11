"""
Moonshot runtime patches for hailmary experiments.

This file keeps the base train_gpt.py untouched. Each patch rewrites the source
for one run directory, like the stage2_1 and stage3 flows.

The first wave here focuses on distinct, tractable moonshot patches:

- exact_overlap_eval
- ema_export
- late_qat_active
- countinit_bigram

For reuse, some patches delegate to the already-validated stage2_1 patch set.
"""
from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_stage2_1_patches():
    path = Path(__file__).resolve().parents[1] / "stage2_1" / "patches.py"
    spec = importlib.util.spec_from_file_location("stage2_1_patches_for_hailmary", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load stage2_1 patches from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_S21 = _load_stage2_1_patches()
_replace_unique = _S21._replace_unique


def patch_exact_overlap_eval(source: str) -> str:
    # Reuse the validated sliding-window exact-tail implementation from stage2_1.
    return _S21.patch_sliding_eval(source)


def patch_ema_export(source: str) -> str:
    # Reuse the validated EMA export patch from stage2_1.
    return _S21.patch_ema(source)


def patch_xsa_all(source: str) -> str:
    # Reuse the generalized XSA patch and drive it with XSA_LAYERS=all.
    return _S21.patch_xsa_all(source)


def patch_curriculum_shard_order(source: str) -> str:
    # Reuse the shard-order patch from stage2_1 for moonshot curriculum screens.
    return _S21.patch_curriculum_shard_order(source)


def patch_full_gptq(source: str) -> str:
    name = "full_gptq"
    source = _replace_unique(
        source,
        "import io\n"
        "import math\n",
        "import io\n"
        "import lzma\n"
        "import math\n",
        name,
    )
    source = _replace_unique(
        source,
        "\n# -----------------------------\n# POST-TRAINING QUANTIZATION\n",
        "\n\ndef collect_hessians(\n"
        "    base_model: nn.Module,\n"
        "    train_loader: DistributedTokenLoader,\n"
        "    args: Hyperparameters,\n"
        "    device: torch.device,\n"
        "    grad_accum_steps: int,\n"
        "    num_batches: int = 256,\n"
        ") -> dict[str, Tensor]:\n"
        "    hessians: dict[str, Tensor] = {}\n"
        "    hooks: list[torch.utils.hooks.RemovableHandle] = []\n"
        "    was_training = base_model.training\n"
        "    for module_name, module in base_model.named_modules():\n"
        "        if isinstance(module, CastedLinear):\n"
        "            param_name = module_name + \".weight\"\n"
        "            cols = module.weight.shape[1]\n"
        "            hessians[param_name] = torch.zeros((cols, cols), dtype=torch.float32, device=\"cpu\")\n"
        "\n"
        "            def make_hook(pname: str):\n"
        "                def hook_fn(_module: nn.Module, module_input: tuple[Tensor, ...], _output: Tensor) -> None:\n"
        "                    x = module_input[0].detach().float()\n"
        "                    if x.ndim == 3:\n"
        "                        x = x.reshape(-1, x.shape[-1])\n"
        "                    hessians[pname].add_((x.T @ x).cpu())\n"
        "\n"
        "                return hook_fn\n"
        "\n"
        "            hooks.append(module.register_forward_hook(make_hook(param_name)))\n"
        "    base_model.eval()\n"
        "    try:\n"
        "        with torch.inference_mode(), torch.autocast(device_type=\"cuda\", dtype=torch.bfloat16, enabled=True):\n"
        "            for _ in range(num_batches):\n"
        "                x, y = train_loader.next_batch(args.train_batch_tokens, args.train_seq_len, grad_accum_steps)\n"
        "                base_model(x, y)\n"
        "    finally:\n"
        "        for hook in hooks:\n"
        "            hook.remove()\n"
        "        base_model.train(was_training)\n"
        "    norm = float(max(num_batches, 1))\n"
        "    for pname, hessian in hessians.items():\n"
        "        hessians[pname] = (hessian / norm).contiguous()\n"
        "    return hessians\n"
        "\n"
        "# -----------------------------\n"
        "# POST-TRAINING QUANTIZATION\n",
        name,
    )
    source = _replace_unique(
        source,
        "CONTROL_TENSOR_NAME_PATTERNS = tuple(\n"
        "    pattern\n"
        "    for pattern in os.environ.get(\n"
        "        \"CONTROL_TENSOR_NAME_PATTERNS\",\n"
        "        \"attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights\",\n"
        "    ).split(\",\")\n"
        "    if pattern\n"
        ")\n"
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(\n"
        "    pattern\n"
        "    for pattern in os.environ.get(\n"
        "        \"INT8_KEEP_FLOAT_FP32_NAME_PATTERNS\",\n"
        "        \",\".join(CONTROL_TENSOR_NAME_PATTERNS),\n"
        "    ).split(\",\")\n"
        "    if pattern\n"
        ")\n"
        "INT8_KEEP_FLOAT_MAX_NUMEL = 65_536\n"
        "INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16\n"
        "INT8_PER_ROW_SCALE_DTYPE = torch.float16\n"
        "INT8_CLIP_PERCENTILE = 99.99984\n"
        "INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0\n"
        "\n"
        "def tensor_nbytes(t: Tensor) -> int:\n"
        "    return int(t.numel()) * int(t.element_size())\n"
        "\n"
        "def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:\n"
        "    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):\n"
        "        return t.float().contiguous()\n"
        "    if t.dtype in {torch.float32, torch.bfloat16}:\n"
        "        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix(\"torch.\")\n"
        "        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()\n"
        "    return t\n"
        "\n"
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
        "    return q, scale\n"
        "\n"
        "def quantize_state_dict_int8(state_dict: dict[str, Tensor]):\n"
        "    # Single supported clean-script export format:\n"
        "    # - per-row int8 for 2D float tensors\n"
        "    # - per-tensor int8 for other float tensors\n"
        "    # - exact passthrough for non-floats\n"
        "    # - passthrough for small float tensors, stored as fp16 to save bytes\n"
        "    quantized: dict[str, Tensor] = {}\n"
        "    scales: dict[str, Tensor] = {}\n"
        "    dtypes: dict[str, str] = {}\n"
        "    passthrough: dict[str, Tensor] = {}\n"
        "    passthrough_orig_dtypes: dict[str, str] = {}\n"
        "    qmeta: dict[str, dict[str, object]] = {}\n"
        "    stats = dict.fromkeys(\n"
        "        (\"param_count\", \"num_tensors\", \"num_float_tensors\", \"num_nonfloat_tensors\", \"baseline_tensor_bytes\", \"int8_payload_bytes\"),\n"
        "        0,\n"
        "    )\n"
        "\n"
        "    for name, tensor in state_dict.items():\n"
        "        t = tensor.detach().to(\"cpu\").contiguous()\n"
        "        stats[\"param_count\"] += int(t.numel())\n"
        "        stats[\"num_tensors\"] += 1\n"
        "        stats[\"baseline_tensor_bytes\"] += tensor_nbytes(t)\n"
        "\n"
        "        if not t.is_floating_point():\n"
        "            stats[\"num_nonfloat_tensors\"] += 1\n"
        "            passthrough[name] = t\n"
        "            stats[\"int8_payload_bytes\"] += tensor_nbytes(t)\n"
        "            continue\n"
        "\n"
        "        # Small float tensors are cheap enough to keep directly. We still downcast\n"
        "        # fp32/bf16 passthrough tensors to fp16 so metadata does not dominate size.\n"
        "        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:\n"
        "            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)\n"
        "            passthrough[name] = kept\n"
        "            stats[\"int8_payload_bytes\"] += tensor_nbytes(kept)\n"
        "            continue\n"
        "\n"
        "        stats[\"num_float_tensors\"] += 1\n"
        "        q, s = quantize_float_tensor(t)\n"
        "        if s.ndim > 0:\n"
        "            qmeta[name] = {\"scheme\": \"per_row\", \"axis\": 0}\n"
        "        quantized[name] = q\n"
        "        scales[name] = s\n"
        "        dtypes[name] = str(t.dtype).removeprefix(\"torch.\")\n"
        "        stats[\"int8_payload_bytes\"] += tensor_nbytes(q) + tensor_nbytes(s)\n"
        "\n"
        "    obj: dict[str, object] = {\n"
        "        \"__quant_format__\": \"int8_clean_per_row_v1\",\n"
        "        \"quantized\": quantized,\n"
        "        \"scales\": scales,\n"
        "        \"dtypes\": dtypes,\n"
        "        \"passthrough\": passthrough,\n"
        "    }\n"
        "    if qmeta:\n"
        "        obj[\"qmeta\"] = qmeta\n"
        "    if passthrough_orig_dtypes:\n"
        "        obj[\"passthrough_orig_dtypes\"] = passthrough_orig_dtypes\n"
        "    return obj, stats\n"
        "\n"
        "def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:\n"
        "    out: dict[str, Tensor] = {}\n"
        "    qmeta = obj.get(\"qmeta\", {})\n"
        "    passthrough_orig_dtypes = obj.get(\"passthrough_orig_dtypes\", {})\n"
        "    for name, q in obj[\"quantized\"].items():\n"
        "        dtype = getattr(torch, obj[\"dtypes\"][name])\n"
        "        s = obj[\"scales\"][name]\n"
        "        if qmeta.get(name, {}).get(\"scheme\") == \"per_row\" or s.ndim > 0:\n"
        "            s = s.to(dtype=torch.float32)\n"
        "            # Broadcast the saved row scale back across trailing dimensions.\n"
        "            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()\n"
        "        else:\n"
        "            scale = float(s.item())\n"
        "            out[name] = (q.float() * scale).to(dtype=dtype).contiguous()\n"
        "    for name, t in obj[\"passthrough\"].items():\n"
        "        # Restore small tensors, undoing the temporary fp16 storage cast if needed.\n"
        "        out_t = t.detach().to(\"cpu\").contiguous()\n"
        "        orig_dtype = passthrough_orig_dtypes.get(name)\n"
        "        if isinstance(orig_dtype, str):\n"
        "            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()\n"
        "        out[name] = out_t\n"
        "    return out\n",
        "CONTROL_TENSOR_NAME_PATTERNS = tuple(\n"
        "    pattern\n"
        "    for pattern in os.environ.get(\n"
        "        \"CONTROL_TENSOR_NAME_PATTERNS\",\n"
        "        \"attn_scale,attn_scales,mlp_scale,mlp_scales,resid_mix,resid_mixes,q_gain,skip_weight,skip_weights\",\n"
        "    ).split(\",\")\n"
        "    if pattern\n"
        ")\n"
        "INT8_KEEP_FLOAT_FP32_NAME_PATTERNS = tuple(\n"
        "    pattern\n"
        "    for pattern in os.environ.get(\n"
        "        \"INT8_KEEP_FLOAT_FP32_NAME_PATTERNS\",\n"
        "        \",\".join(CONTROL_TENSOR_NAME_PATTERNS),\n"
        "    ).split(\",\")\n"
        "    if pattern\n"
        ")\n"
        "INT8_KEEP_FLOAT_MAX_NUMEL = 65_536\n"
        "INT8_KEEP_FLOAT_STORE_DTYPE = torch.float16\n"
        "INT8_PER_ROW_SCALE_DTYPE = torch.float16\n"
        "INT8_CLIP_PERCENTILE = 99.99984\n"
        "INT8_CLIP_Q = INT8_CLIP_PERCENTILE / 100.0\n"
        "\n"
        "\n"
        "def tensor_nbytes(t: Tensor) -> int:\n"
        "    return int(t.numel()) * int(t.element_size())\n"
        "\n"
        "\n"
        "def _classify_param(name: str) -> str:\n"
        "    if \"tok_emb\" in name or \"lm_head\" in name:\n"
        "        return \"embed\"\n"
        "    if \".mlp.\" in name:\n"
        "        return \"mlp\"\n"
        "    if \".attn.\" in name or (\".proj.\" in name and \".mlp.\" not in name):\n"
        "        return \"attn\"\n"
        "    if \"count_bigram\" in name:\n"
        "        return \"bigram\"\n"
        "    return \"other\"\n"
        "\n"
        "\n"
        "def keep_float_tensor(name: str, t: Tensor, passthrough_orig_dtypes: dict[str, str]) -> Tensor:\n"
        "    if any(pattern in name for pattern in INT8_KEEP_FLOAT_FP32_NAME_PATTERNS):\n"
        "        return t.float().contiguous()\n"
        "    if t.dtype in {torch.float32, torch.bfloat16}:\n"
        "        passthrough_orig_dtypes[name] = str(t.dtype).removeprefix(\"torch.\")\n"
        "        return t.to(dtype=INT8_KEEP_FLOAT_STORE_DTYPE).contiguous()\n"
        "    return t\n"
        "\n"
        "\n"
        "def _quantize_int6_percentile(t: Tensor, clip_range: int = 31) -> tuple[Tensor, Tensor]:\n"
        "    t32 = t.float()\n"
        "    if t32.ndim == 2:\n"
        "        best_q = None\n"
        "        best_s = None\n"
        "        best_err = float(\"inf\")\n"
        "        for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:\n"
        "            row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)\n"
        "            s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)\n"
        "            q = torch.clamp(torch.round(t32 / s.float()[:, None]), -clip_range, clip_range).to(torch.int8)\n"
        "            recon = q.float() * s.float()[:, None]\n"
        "            err = float((t32 - recon).pow(2).mean().item())\n"
        "            if err < best_err:\n"
        "                best_q = q\n"
        "                best_s = s\n"
        "                best_err = err\n"
        "        return best_q.contiguous(), best_s.contiguous()\n"
        "    amax = float(t32.abs().max().item()) if t32.numel() else 0.0\n"
        "    scale = torch.tensor(amax / clip_range if amax > 0 else 1.0, dtype=torch.float16)\n"
        "    q = torch.clamp(torch.round(t32 / scale.float()), -clip_range, clip_range).to(torch.int8).contiguous()\n"
        "    return q, scale\n"
        "\n"
        "\n"
        "def quantize_int6_gptq(\n"
        "    weight: Tensor,\n"
        "    hessian: Tensor | None = None,\n"
        "    clip_range: int = 31,\n"
        "    block_size: int = 128,\n"
        ") -> tuple[Tensor, Tensor]:\n"
        "    t32 = weight.float()\n"
        "    if t32.ndim != 2 or hessian is None:\n"
        "        return _quantize_int6_percentile(t32, clip_range)\n"
        "    rows, cols = t32.shape\n"
        "    H = hessian.float().clone()\n"
        "    dead = torch.diag(H) == 0\n"
        "    if dead.any():\n"
        "        H[dead, dead] = 1\n"
        "    damp = 0.01 * torch.mean(torch.diag(H)).clamp_min(1e-6)\n"
        "    H[torch.arange(cols), torch.arange(cols)] += damp\n"
        "    perm = torch.argsort(torch.diag(H), descending=True)\n"
        "    inv_perm = torch.argsort(perm)\n"
        "    W = t32[:, perm].clone()\n"
        "    if dead.any():\n"
        "        W[:, dead[perm]] = 0\n"
        "    H = H[perm][:, perm]\n"
        "    try:\n"
        "        Hinv = torch.linalg.cholesky(H)\n"
        "        Hinv = torch.cholesky_inverse(Hinv)\n"
        "        Hinv = torch.linalg.cholesky(Hinv, upper=True)\n"
        "    except torch.linalg.LinAlgError:\n"
        "        return _quantize_int6_percentile(t32, clip_range)\n"
        "    best_q = None\n"
        "    best_scale = None\n"
        "    best_err = float(\"inf\")\n"
        "    for pct in [0.9990, 0.9995, 0.9999, 0.99999, 1.0]:\n"
        "        row_clip = torch.quantile(t32.abs(), pct, dim=1) if pct < 1.0 else t32.abs().amax(dim=1)\n"
        "        s = (row_clip / clip_range).clamp_min(1.0 / clip_range).to(torch.float16)\n"
        "        sf = s.float()\n"
        "        Q = torch.zeros_like(W, dtype=torch.int8)\n"
        "        W_work = W.clone()\n"
        "        for i1 in range(0, cols, block_size):\n"
        "            i2 = min(i1 + block_size, cols)\n"
        "            width = i2 - i1\n"
        "            W1 = W_work[:, i1:i2].clone()\n"
        "            Q1 = torch.zeros((rows, width), dtype=torch.int8)\n"
        "            Err1 = torch.zeros((rows, width), dtype=torch.float32)\n"
        "            Hinv1 = Hinv[i1:i2, i1:i2]\n"
        "            for i in range(width):\n"
        "                w = W1[:, i]\n"
        "                d = Hinv1[i, i].clamp_min(1e-8)\n"
        "                q = torch.clamp(torch.round(w / sf), -clip_range, clip_range).to(torch.int8)\n"
        "                Q1[:, i] = q\n"
        "                err = (w - q.float() * sf) / d\n"
        "                W1[:, i:] -= err.unsqueeze(1) * Hinv1[i, i:].unsqueeze(0)\n"
        "                Err1[:, i] = err\n"
        "            Q[:, i1:i2] = Q1\n"
        "            if i2 < cols:\n"
        "                W_work[:, i2:] -= Err1 @ Hinv[i1:i2, i2:]\n"
        "        recon = Q.float() * sf[:, None]\n"
        "        mse = float((W - recon).pow(2).mean().item())\n"
        "        if mse < best_err:\n"
        "            best_q = Q[:, inv_perm]\n"
        "            best_scale = s\n"
        "            best_err = mse\n"
        "    return best_q.contiguous(), best_scale.contiguous()\n"
        "\n"
        "\n"
        "def quantize_float_tensor(t: Tensor) -> tuple[Tensor, Tensor]:\n"
        "    t32 = t.float()\n"
        "    if t32.ndim == 2:\n"
        "        clip_abs = (\n"
        "            torch.quantile(t32.abs(), INT8_CLIP_Q, dim=1)\n"
        "            if t32.numel()\n"
        "            else torch.empty((t32.shape[0],), dtype=torch.float32)\n"
        "        )\n"
        "        clipped = torch.maximum(torch.minimum(t32, clip_abs[:, None]), -clip_abs[:, None])\n"
        "        scale = (clip_abs / 127.0).clamp_min(1.0 / 127.0)\n"
        "        q = torch.clamp(torch.round(clipped / scale[:, None]), -127, 127).to(torch.int8).contiguous()\n"
        "        return q, scale.to(dtype=INT8_PER_ROW_SCALE_DTYPE).contiguous()\n"
        "    clip_abs = float(torch.quantile(t32.abs().flatten(), INT8_CLIP_Q).item()) if t32.numel() else 0.0\n"
        "    scale = torch.tensor(clip_abs / 127.0 if clip_abs > 0 else 1.0, dtype=torch.float32)\n"
        "    q = torch.clamp(torch.round(torch.clamp(t32, -clip_abs, clip_abs) / scale), -127, 127).to(torch.int8).contiguous()\n"
        "    return q, scale\n"
        "\n"
        "\n"
        "def quantize_state_dict_int8(\n"
        "    state_dict: dict[str, Tensor],\n"
        "    hessians: dict[str, Tensor] | None = None,\n"
        "):\n"
        "    quantized: dict[str, Tensor] = {}\n"
        "    scales: dict[str, Tensor] = {}\n"
        "    dtypes: dict[str, str] = {}\n"
        "    passthrough: dict[str, Tensor] = {}\n"
        "    passthrough_orig_dtypes: dict[str, str] = {}\n"
        "    qmeta: dict[str, dict[str, object]] = {}\n"
        "    stats = dict.fromkeys(\n"
        "        (\"param_count\", \"num_tensors\", \"num_float_tensors\", \"num_nonfloat_tensors\", \"baseline_tensor_bytes\", \"int8_payload_bytes\"),\n"
        "        0,\n"
        "    )\n"
        "    gptq_enabled = os.environ.get(\"GPTQ_ENABLE\", \"0\") == \"1\"\n"
        "    gptq_block_size = int(os.environ.get(\"GPTQ_BLOCK_SIZE\", \"128\"))\n"
        "    int6_categories = {token.strip() for token in os.environ.get(\"GPTQ_INT6_CATEGORIES\", \"mlp,attn,bigram\").split(\",\") if token.strip()}\n"
        "    for name, tensor in state_dict.items():\n"
        "        t = tensor.detach().to(\"cpu\").contiguous()\n"
        "        stats[\"param_count\"] += int(t.numel())\n"
        "        stats[\"num_tensors\"] += 1\n"
        "        stats[\"baseline_tensor_bytes\"] += tensor_nbytes(t)\n"
        "        if not t.is_floating_point():\n"
        "            stats[\"num_nonfloat_tensors\"] += 1\n"
        "            passthrough[name] = t\n"
        "            stats[\"int8_payload_bytes\"] += tensor_nbytes(t)\n"
        "            continue\n"
        "        if t.numel() <= INT8_KEEP_FLOAT_MAX_NUMEL:\n"
        "            kept = keep_float_tensor(name, t, passthrough_orig_dtypes)\n"
        "            passthrough[name] = kept\n"
        "            stats[\"int8_payload_bytes\"] += tensor_nbytes(kept)\n"
        "            continue\n"
        "        stats[\"num_float_tensors\"] += 1\n"
        "        cat = _classify_param(name)\n"
        "        use_int6 = gptq_enabled and cat in int6_categories and t.ndim == 2\n"
        "        if use_int6:\n"
        "            q, s = quantize_int6_gptq(t, hessian=hessians.get(name) if hessians else None, clip_range=31, block_size=gptq_block_size)\n"
        "            qmeta[name] = {\"scheme\": \"per_row\", \"axis\": 0, \"bits\": 6}\n"
        "        else:\n"
        "            q, s = quantize_float_tensor(t)\n"
        "            if s.ndim > 0:\n"
        "                qmeta[name] = {\"scheme\": \"per_row\", \"axis\": 0, \"bits\": 8}\n"
        "        quantized[name] = q\n"
        "        scales[name] = s\n"
        "        dtypes[name] = str(t.dtype).removeprefix(\"torch.\")\n"
        "        stats[\"int8_payload_bytes\"] += tensor_nbytes(q) + tensor_nbytes(s)\n"
        "    obj: dict[str, object] = {\n"
        "        \"__quant_format__\": \"gptq_int6_lzma_v1\" if gptq_enabled else \"int8_clean_per_row_v1\",\n"
        "        \"quantized\": quantized,\n"
        "        \"scales\": scales,\n"
        "        \"dtypes\": dtypes,\n"
        "        \"passthrough\": passthrough,\n"
        "    }\n"
        "    if qmeta:\n"
        "        obj[\"qmeta\"] = qmeta\n"
        "    if passthrough_orig_dtypes:\n"
        "        obj[\"passthrough_orig_dtypes\"] = passthrough_orig_dtypes\n"
        "    return obj, stats\n"
        "\n"
        "\n"
        "def dequantize_state_dict_int8(obj: dict[str, object]) -> dict[str, Tensor]:\n"
        "    out: dict[str, Tensor] = {}\n"
        "    qmeta = obj.get(\"qmeta\", {})\n"
        "    passthrough_orig_dtypes = obj.get(\"passthrough_orig_dtypes\", {})\n"
        "    for name, q in obj[\"quantized\"].items():\n"
        "        dtype = getattr(torch, obj[\"dtypes\"][name])\n"
        "        s = obj[\"scales\"][name]\n"
        "        if qmeta.get(name, {}).get(\"scheme\") == \"per_row\" or s.ndim > 0:\n"
        "            s = s.to(dtype=torch.float32)\n"
        "            out[name] = (q.float() * s.view(q.shape[0], *([1] * (q.ndim - 1)))).to(dtype=dtype).contiguous()\n"
        "        else:\n"
        "            out[name] = (q.float() * float(s.item())).to(dtype=dtype).contiguous()\n"
        "    for name, t in obj[\"passthrough\"].items():\n"
        "        out_t = t.detach().to(\"cpu\").contiguous()\n"
        "        orig_dtype = passthrough_orig_dtypes.get(name)\n"
        "        if isinstance(orig_dtype, str):\n"
        "            out_t = out_t.to(dtype=getattr(torch, orig_dtype)).contiguous()\n"
        "        out[name] = out_t\n"
        "    return out\n",
        name,
    )
    source = _replace_unique(
        source,
        "    quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())\n"
        "    quant_buf = io.BytesIO()\n"
        "    torch.save(quant_obj, quant_buf)\n"
        "    quant_raw = quant_buf.getvalue()\n"
        "    quant_blob = zlib.compress(quant_raw, level=9)\n"
        "    quant_raw_bytes = len(quant_raw)\n"
        "    if master_process:\n"
        "        with open(\"final_model.int8.ptz\", \"wb\") as f:\n"
        "            f.write(quant_blob)\n"
        "        quant_file_bytes = os.path.getsize(\"final_model.int8.ptz\")\n"
        "        code_bytes = len(code.encode(\"utf-8\"))\n"
        "        ratio = quant_stats[\"baseline_tensor_bytes\"] / max(quant_stats[\"int8_payload_bytes\"], 1)\n"
        "        log0(\n"
        "            f\"Serialized model int8+zlib: {quant_file_bytes} bytes \"\n"
        "            f\"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)\"\n"
        "        )\n"
        "        log0(f\"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes\")\n"
        "\n"
        "    if distributed:\n"
        "        dist.barrier()\n"
        "    with open(\"final_model.int8.ptz\", \"rb\") as f:\n"
        "        quant_blob_disk = f.read()\n"
        "    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location=\"cpu\")\n"
        "    base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)",
        "    gptq_enabled = os.environ.get(\"GPTQ_ENABLE\", \"0\") == \"1\"\n"
        "    if gptq_enabled:\n"
        "        gptq_calib_batches = int(os.environ.get(\"GPTQ_CALIB_BATCHES\", \"256\"))\n"
        "        log0(f\"gptq:calibrating with {gptq_calib_batches} batches...\")\n"
        "        calib_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)\n"
        "        hessians = collect_hessians(\n"
        "            base_model,\n"
        "            calib_loader,\n"
        "            args,\n"
        "            device,\n"
        "            grad_accum_steps,\n"
        "            num_batches=gptq_calib_batches,\n"
        "        )\n"
        "        log0(f\"gptq:collected hessians for {len(hessians)} layers\")\n"
        "        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict(), hessians=hessians)\n"
        "        quant_buf = io.BytesIO()\n"
        "        torch.save(quant_obj, quant_buf)\n"
        "        quant_raw = quant_buf.getvalue()\n"
        "        quant_blob = lzma.compress(quant_raw, preset=int(os.environ.get(\"GPTQ_LZMA_PRESET\", \"6\")))\n"
        "        quant_raw_bytes = len(quant_raw)\n"
        "        if master_process:\n"
        "            with open(\"final_model.int6.ptz\", \"wb\") as f:\n"
        "                f.write(quant_blob)\n"
        "            quant_file_bytes = os.path.getsize(\"final_model.int6.ptz\")\n"
        "            code_bytes = len(code.encode(\"utf-8\"))\n"
        "            ratio = quant_stats[\"baseline_tensor_bytes\"] / max(quant_stats[\"int8_payload_bytes\"], 1)\n"
        "            log0(\n"
        "                f\"Serialized model int6+lzma: {quant_file_bytes} bytes \"\n"
        "                f\"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)\"\n"
        "            )\n"
        "            log0(f\"Total submission size int6+lzma: {quant_file_bytes + code_bytes} bytes\")\n"
        "\n"
        "        if distributed:\n"
        "            dist.barrier()\n"
        "        with open(\"final_model.int6.ptz\", \"rb\") as f:\n"
        "            quant_blob_disk = f.read()\n"
        "        quant_state = torch.load(io.BytesIO(lzma.decompress(quant_blob_disk)), map_location=\"cpu\")\n"
        "        base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)\n"
        "    else:\n"
        "        quant_obj, quant_stats = quantize_state_dict_int8(base_model.state_dict())\n"
        "        quant_buf = io.BytesIO()\n"
        "        torch.save(quant_obj, quant_buf)\n"
        "        quant_raw = quant_buf.getvalue()\n"
        "        quant_blob = zlib.compress(quant_raw, level=9)\n"
        "        quant_raw_bytes = len(quant_raw)\n"
        "        if master_process:\n"
        "            with open(\"final_model.int8.ptz\", \"wb\") as f:\n"
        "                f.write(quant_blob)\n"
        "            quant_file_bytes = os.path.getsize(\"final_model.int8.ptz\")\n"
        "            code_bytes = len(code.encode(\"utf-8\"))\n"
        "            ratio = quant_stats[\"baseline_tensor_bytes\"] / max(quant_stats[\"int8_payload_bytes\"], 1)\n"
        "            log0(\n"
        "                f\"Serialized model int8+zlib: {quant_file_bytes} bytes \"\n"
        "                f\"(payload:{quant_stats['int8_payload_bytes']} raw_torch:{quant_raw_bytes} payload_ratio:{ratio:.2f}x)\"\n"
        "            )\n"
        "            log0(f\"Total submission size int8+zlib: {quant_file_bytes + code_bytes} bytes\")\n"
        "\n"
        "        if distributed:\n"
        "            dist.barrier()\n"
        "        with open(\"final_model.int8.ptz\", \"rb\") as f:\n"
        "            quant_blob_disk = f.read()\n"
        "        quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob_disk)), map_location=\"cpu\")\n"
        "        base_model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)",
        name,
    )
    source = _replace_unique(
        source,
        "    log0(\n"
        "        f\"final_int8_zlib_roundtrip val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} \"\n"
        "        f\"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms\"\n"
        "    )\n"
        "    log0(f\"final_int8_zlib_roundtrip_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}\")",
        "    quant_prefix = \"final_int6_lzma_roundtrip\" if os.environ.get(\"GPTQ_ENABLE\", \"0\") == \"1\" else \"final_int8_zlib_roundtrip\"\n"
        "    log0(\n"
        "        f\"{quant_prefix} val_loss:{q_val_loss:.4f} val_bpb:{q_val_bpb:.4f} \"\n"
        "        f\"eval_time:{1000.0 * (time.perf_counter() - t_qeval):.0f}ms\"\n"
        "    )\n"
        "    log0(f\"{quant_prefix}_exact val_loss:{q_val_loss:.8f} val_bpb:{q_val_bpb:.8f}\")",
        name,
    )
    return source


def patch_late_qat_active(source: str) -> str:
    name = "late_qat_active"
    source = _replace_unique(
        source,
        "class CastedLinear(nn.Linear):\n"
        "    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.\n"
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        bias = self.bias.to(x.dtype) if self.bias is not None else None\n"
        "        return F.linear(x, self.weight.to(x.dtype), bias)",
        "class _LateQATInt6STE(torch.autograd.Function):\n"
        "    @staticmethod\n"
        "    def forward(ctx, w: Tensor) -> Tensor:\n"
        "        w32 = w.float()\n"
        "        abs_max = w32.abs().amax(dim=1, keepdim=True).clamp_min(1e-12)\n"
        "        scale = abs_max / 31.0\n"
        "        q = torch.clamp(torch.round(w32 / scale), -32, 31)\n"
        "        return (q * scale).to(w.dtype)\n"
        "\n"
        "    @staticmethod\n"
        "    def backward(ctx, grad_output: Tensor) -> tuple[Tensor]:\n"
        "        return (grad_output,)\n"
        "\n"
        "\n"
        "_late_qat_int6 = _LateQATInt6STE.apply\n"
        "\n"
        "\n"
        "class CastedLinear(nn.Linear):\n"
        "    # Keep weights in fp32 for optimizer/state quality, cast at matmul time for bf16 compute.\n"
        "    _late_qat_enabled = False\n"
        "\n"
        "    def forward(self, x: Tensor) -> Tensor:\n"
        "        bias = self.bias.to(x.dtype) if self.bias is not None else None\n"
        "        w = self.weight.to(x.dtype)\n"
        "        if self.training and type(self)._late_qat_enabled and w.ndim == 2:\n"
        "            w = _late_qat_int6(w)\n"
        "        return F.linear(x, w, bias)",
        name,
    )
    source = _replace_unique(
        source,
        "        scale = lr_mul(step, elapsed_ms)\n"
        "        zero_grad_all()",
        "        scale = lr_mul(step, elapsed_ms)\n"
        "        if os.environ.get(\"LATE_QAT_ENABLE\", \"0\") == \"1\":\n"
        "            _threshold = float(os.environ.get(\"LATE_QAT_THRESHOLD\", \"0.15\"))\n"
        "            CastedLinear._late_qat_enabled = scale <= _threshold\n"
        "        else:\n"
        "            CastedLinear._late_qat_enabled = False\n"
        "        zero_grad_all()",
        name,
    )
    return source


def patch_countinit_bigram(source: str) -> str:
    name = "countinit_bigram"
    source = _replace_unique(
        source,
        "class Rotary(nn.Module):\n"
        "    # Caches cos/sin tables per sequence length on the current device.",
        "class CountInitBigramHead(nn.Module):\n"
        "    def __init__(self, vocab_size: int):\n"
        "        super().__init__()\n"
        "        self.table = nn.Parameter(torch.zeros(vocab_size, vocab_size, dtype=torch.float32))\n"
        "        self.scale = nn.Parameter(torch.tensor(1.0, dtype=torch.float32))\n"
        "\n"
        "    @torch.no_grad()\n"
        "    def initialize_from_tokens(self, tokens: Tensor, vocab_size: int, alpha: float = 0.1) -> None:\n"
        "        if tokens.numel() < 2:\n"
        "            return\n"
        "        prev_ids = tokens[:-1].to(dtype=torch.int64)\n"
        "        next_ids = tokens[1:].to(dtype=torch.int64)\n"
        "        flat = prev_ids * vocab_size + next_ids\n"
        "        counts = torch.bincount(flat, minlength=vocab_size * vocab_size).reshape(vocab_size, vocab_size).float()\n"
        "        row_sum = counts.sum(dim=1, keepdim=True)\n"
        "        col_sum = counts.sum(dim=0, keepdim=True)\n"
        "        total = counts.sum().clamp_min(1.0)\n"
        "        cond = (counts + alpha) / (row_sum + alpha * vocab_size)\n"
        "        marg = (col_sum + alpha) / (total + alpha * vocab_size)\n"
        "        self.table.copy_((cond.log() - marg.log()).float())\n"
        "\n"
        "    def forward(self, prev_ids: Tensor) -> Tensor:\n"
        "        return self.scale.to(dtype=self.table.dtype) * self.table[prev_ids]\n"
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
        "        self.final_norm = RMSNorm()\n"
        "        self.count_bigram = CountInitBigramHead(vocab_size) if os.environ.get(\"COUNTINIT_BIGRAM\", \"0\") == \"1\" else None\n"
        "        self.lm_head = None if tie_embeddings else CastedLinear(model_dim, vocab_size, bias=False)",
        name,
    )
    source = _replace_unique(
        source,
        "        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)\n"
        "        return F.cross_entropy(logits.float(), targets, reduction=\"mean\")",
        "        if self.count_bigram is not None:\n"
        "            prev_flat = input_ids.reshape(-1)\n"
        "            logits_proj = logits_proj + self.count_bigram(prev_flat).to(dtype=logits_proj.dtype)\n"
        "        logits = self.logit_softcap * torch.tanh(logits_proj / self.logit_softcap)\n"
        "        return F.cross_entropy(logits.float(), targets, reduction=\"mean\")",
        name,
    )
    source = _replace_unique(
        source,
        "    if base_model.skip_weights.numel() > 0:\n"
        "        scalar_params.append(base_model.skip_weights)",
        "    if base_model.skip_weights.numel() > 0:\n"
        "        scalar_params.append(base_model.skip_weights)\n"
        "    if getattr(base_model, 'count_bigram', None) is not None:\n"
        "        scalar_params.extend(base_model.count_bigram.parameters())",
        name,
    )
    source = _replace_unique(
        source,
        "    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)\n"
        "\n"
        "    def zero_grad_all() -> None:\n",
        "    train_loader = DistributedTokenLoader(args.train_files, rank, world_size, device)\n"
        "    if getattr(base_model, 'count_bigram', None) is not None:\n"
        "        _sample_tokens = int(os.environ.get(\"COUNTINIT_BIGRAM_TOKENS\", \"2000000\"))\n"
        "        _stream = TokenStream(args.train_files)\n"
        "        _tokens = _stream.take(_sample_tokens + 1).to(dtype=torch.int64)\n"
        "        base_model.count_bigram.initialize_from_tokens(_tokens.cpu(), args.vocab_size)\n"
        "        log0(f\"countinit_bigram:initialized sample_tokens:{_sample_tokens}\")\n"
        "\n"
        "    def zero_grad_all() -> None:\n",
        name,
    )
    return source


PATCH_REGISTRY = {
    "countinit_bigram": patch_countinit_bigram,
    "curriculum_shard_order": patch_curriculum_shard_order,
    "ema_export": patch_ema_export,
    "exact_overlap_eval": patch_exact_overlap_eval,
    "full_gptq": patch_full_gptq,
    "late_qat_active": patch_late_qat_active,
    "xsa_all": patch_xsa_all,
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
