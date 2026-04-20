"""
SpinQuant hotstart script — spec 009.

Loads the pre-GPTQ FP checkpoint from spec 008 (runs/008-1736-reproduction/
seed_42/final_model.pt, auto-saved by train_gpt.py's serialize() before
quantization), optionally applies rotations to weight tensors in-memory,
then runs #1736's existing pipeline: GPTQ quantization + serialization +
eval + phased TTT.

Modes (SPINQUANT_MODE env var):
  baseline       No rotation. Closes spec 008's missed post-TTT gate number.
                 Also the apples-to-apples reference for SpinQuant deltas.
  internal_only  Per-layer, per-KV-group attention internal rotation R_a
                 (V-output / O-input, d_head = 64). Strict float-invariance
                 by construction — softmax(QK^T)V is rotation-equivariant in
                 V's d_head axis. Does NOT include MLP internal rotation
                 (LeakyReLU(0.5)^2 breaks equivariance; see spec 009 notes).
  full           DEFERRED — residual-stream rotation with per-channel folds
                 + resid_mix handling. Fails with an explanatory error.
  port_1695      DEFERRED — needs design pass after reading #1695's diff.
                 Fails with an explanatory error.

Env contract:
  SPINQUANT_MODE           mode selector (see above)
  SPINQUANT_SEED           int seed for rotation generator; default 42
  HOTSTART_FP_CKPT         absolute path to spec 008's final_model.pt
  ARTIFACT_DIR             standard #1736 artifact dir; per-mode subdir
  everything else          standard #1736 env block (CASEOPS_ENABLED,
                           PHASED_TTT_*, GATED_ATTN_*, MLP_CLIP_SIGMAS,
                           ATTN_CLIP_SIGMAS, EMBED_BITS, etc.)

Pipeline:
  1. Build empty GPT(h), load state_dict from HOTSTART_FP_CKPT.
  2. Apply rotations in-place on banked weight tensors (mode-dependent).
  3. Call serialize(h, base_model, code) — GPTQ + compress, writes
     {artifact_dir}/final_model.int6.ptz.
  4. Call deserialize(h, device) — loads quantized model.
  5. eval_val(...) on quantized — "diagnostic quantized" bpb.
  6. eval_val_ttt_phased(...) — "quantized_ttt_phased" bpb (the gate number).
  7. Write final.json with all three bpb numbers + rotation seeds.
"""

from __future__ import annotations

import json
import math
import os
import time
import sys
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F

# Re-use #1736's code. This file lives in the same directory as train_gpt.py,
# so a plain import works both on the pod and in a local checkout.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from train_gpt import (  # type: ignore[import-not-found]
    BatchedTTTLoRA,
    GPT,
    Hyperparameters,
    ValidationData,
    deserialize,
    eval_val,
    eval_val_ttt_phased,
    log,
    restore_fp32_params,
    serialize,
    set_logging_hparams,
    timed_eval,
)

import train_gpt  # for BOS_ID global


MODE_BASELINE = "baseline"
MODE_INTERNAL_ONLY = "internal_only"
MODE_FULL = "full"
MODE_PORT_1695 = "port_1695"
_KNOWN_MODES = {MODE_BASELINE, MODE_INTERNAL_ONLY, MODE_FULL, MODE_PORT_1695}


# ---------- rotation utilities ----------


def _signed_hadamard(d: int, seed: int) -> torch.Tensor:
    """Signed Hadamard matrix of order d (d must be a power of 2).

    Returns an orthogonal d×d tensor: H_n ⊗ H_n ⊗ ... with a random ±1 sign
    flip per column. Scale 1/√d makes rows orthonormal.
    """
    assert d > 0 and (d & (d - 1)) == 0, f"d={d} is not a power of 2"
    H = torch.ones((1, 1), dtype=torch.float32)
    while H.shape[0] < d:
        H = torch.cat(
            [torch.cat([H, H], dim=1), torch.cat([H, -H], dim=1)],
            dim=0,
        )
    H = H / math.sqrt(d)
    g = torch.Generator().manual_seed(seed)
    signs = torch.randint(0, 2, (d,), generator=g, dtype=torch.float32) * 2 - 1
    return H * signs.unsqueeze(0)


def _random_orthogonal(d: int, seed: int) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    a = torch.randn((d, d), generator=g, dtype=torch.float32)
    q, _ = torch.linalg.qr(a)
    return q


def build_rotation(d: int, seed: int) -> torch.Tensor:
    """Prefer signed Hadamard when d is a power of 2, else random orthogonal."""
    if d > 0 and (d & (d - 1)) == 0:
        return _signed_hadamard(d, seed)
    return _random_orthogonal(d, seed)


# ---------- apply rotations to banked state_dict ----------


def apply_internal_attention_rotations(
    model: GPT,
    *,
    base_seed: int,
    num_layers: int,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> Dict[str, int]:
    """Apply per-layer, per-KV-group attention internal rotation R_a.

    For each layer i and each KV-group g (0 .. num_kv_heads-1):
      - R_a[i, g] is a fixed d_head × d_head orthogonal matrix.
      - V rotation (kv_bank[n+i][g*head_dim : (g+1)*head_dim, :]):
          W ← R_a @ W     (pre-multiply rows)
      - O counter-rotation — for each Q-head h in the KV-group (h // (num_heads//num_kv_heads) == g):
          qo_bank[n+i][:, h*head_dim : (h+1)*head_dim] ← W @ R_a.T
                                                         (post-multiply columns)

    Returns a dict of rotation seeds used (layer, kv_group) -> seed offset.
    """
    assert num_heads % num_kv_heads == 0, (
        f"num_heads={num_heads} must be divisible by num_kv_heads={num_kv_heads}"
    )
    group_size = num_heads // num_kv_heads  # Q-heads per KV-group

    # Ensure we modify the Parameter data in place.
    qo_bank = model.qo_bank.data   # [2*n, d_model, d_model]
    kv_bank = model.kv_bank.data   # [2*n, kv_dim,  d_model]
    n = num_layers

    seeds_used: Dict[str, int] = {}

    for i in range(n):
        for g in range(num_kv_heads):
            # Distinct seed per (layer, kv_group).
            seed = base_seed + i * 1000 + g
            R = build_rotation(head_dim, seed)       # [head_dim, head_dim], fp32
            R = R.to(dtype=qo_bank.dtype, device=qo_bank.device)
            R_T = R.T.contiguous()

            # V rotation on kv_bank[n + i]
            v_slice = kv_bank[n + i, g * head_dim : (g + 1) * head_dim, :]
            kv_bank[n + i, g * head_dim : (g + 1) * head_dim, :] = R @ v_slice

            # O counter-rotation on qo_bank[n + i] for every Q-head in this group
            for h_in_group in range(group_size):
                h = g * group_size + h_in_group
                o_slice = qo_bank[n + i, :, h * head_dim : (h + 1) * head_dim]
                qo_bank[n + i, :, h * head_dim : (h + 1) * head_dim] = o_slice @ R_T

            seeds_used[f"layer{i}_kvgroup{g}"] = seed

    return seeds_used


def apply_rotations(model: GPT, h: Hyperparameters, mode: str, base_seed: int) -> Dict:
    """Mode dispatcher. Returns a rotation-manifest dict for logging."""
    manifest: Dict = {
        "mode": mode,
        "base_seed": base_seed,
        "num_layers": h.num_layers,
        "num_heads": h.num_heads,
        "num_kv_heads": h.num_kv_heads,
        "head_dim": h.model_dim // h.num_heads,
    }

    if mode == MODE_BASELINE:
        manifest["rotations"] = {}
        return manifest

    if mode == MODE_INTERNAL_ONLY:
        head_dim = h.model_dim // h.num_heads
        seeds = apply_internal_attention_rotations(
            model,
            base_seed=base_seed,
            num_layers=h.num_layers,
            num_heads=h.num_heads,
            num_kv_heads=h.num_kv_heads,
            head_dim=head_dim,
        )
        manifest["rotations"] = {"R_a": seeds}
        return manifest

    if mode == MODE_FULL:
        raise NotImplementedError(
            "SPINQUANT_MODE=full is deferred. Needs design for residual R_0 "
            "with per-channel folds (attn_scale, mlp_scale, skip_weights) and "
            "resid_mix handling. See research/ideas/spinquant-integration-notes.md."
        )

    if mode == MODE_PORT_1695:
        raise NotImplementedError(
            "SPINQUANT_MODE=port_1695 is deferred until #1695's diff is read "
            "and the rotation scheme is ported. Script will be updated in a "
            "follow-up spec."
        )

    raise ValueError(f"Unknown SPINQUANT_MODE={mode!r}; expected one of {_KNOWN_MODES}")


# ---------- main ----------


def _parse_env() -> Tuple[str, int, str]:
    mode = os.environ.get("SPINQUANT_MODE", MODE_BASELINE).strip()
    if mode not in _KNOWN_MODES:
        raise ValueError(
            f"SPINQUANT_MODE={mode!r} is not in the known set {sorted(_KNOWN_MODES)}"
        )
    seed = int(os.environ.get("SPINQUANT_SEED", "42"))
    ckpt = os.environ.get("HOTSTART_FP_CKPT", "").strip()
    if not ckpt:
        raise RuntimeError(
            "HOTSTART_FP_CKPT is required (path to spec 008's final_model.pt)"
        )
    if not Path(ckpt).is_file():
        raise FileNotFoundError(f"HOTSTART_FP_CKPT={ckpt} does not exist")
    return mode, seed, ckpt


def _run_quantized_eval(h, val_data, device):
    """Post-serialize quantized eval (no TTT). Mirrors lines 2980-2996 of
    train_and_eval. Returns (val_loss, val_bpb, eval_time_ms)."""
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    compiled_model = torch.compile(eval_model, dynamic=False, fullgraph=True)
    compiled_forward_logits = torch.compile(
        eval_model.forward_logits, dynamic=False, fullgraph=True
    )
    t0 = time.perf_counter()
    val_loss, val_bpb = eval_val(
        h, device, val_data, compiled_model, compiled_forward_logits
    )
    elapsed_ms = int((time.perf_counter() - t0) * 1000)
    del eval_model, compiled_model, compiled_forward_logits
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return float(val_loss), float(val_bpb), elapsed_ms


def _run_ttt_eval(h, val_data, device):
    """Phased TTT eval. Mirrors lines 2997-3075 of train_and_eval. Returns
    (val_loss, val_bpb, eval_time_ms) for quantized_ttt_phased."""
    ttt_model = deserialize(h, device)
    if h.num_loops > 0:
        ttt_model.looping_active = True
    for p in ttt_model.parameters():
        p.requires_grad_(False)

    if h.rope_yarn:
        _yarn_seqlen = h.train_batch_tokens // h.grad_accum_steps
        for block in ttt_model.blocks:
            block.attn.rotary(_yarn_seqlen, device, torch.bfloat16)
    else:
        for block in ttt_model.blocks:
            block.attn.rotary._cos_cached = None
            block.attn.rotary._sin_cached = None
            block.attn.rotary._seq_len_cached = 0
            block.attn.rotary(h.ttt_eval_seq_len, device, torch.bfloat16)

    def _fwd_ttt_inner(input_ids, target_ids, lora):
        return ttt_model.forward_ttt(input_ids, target_ids, lora=lora)

    _fwd_ttt_compiled_inner = None

    def _fwd_ttt(input_ids, target_ids, lora):
        nonlocal _fwd_ttt_compiled_inner
        if _fwd_ttt_compiled_inner is None:
            _fwd_ttt_compiled_inner = torch.compile(_fwd_ttt_inner, dynamic=True)
        return _fwd_ttt_compiled_inner(input_ids, target_ids, lora=lora)

    fwd_ttt_compiled = _fwd_ttt
    log("ttt_lora:warming up compile (random tokens, no val data)")
    if train_gpt.BOS_ID is None:
        train_gpt.BOS_ID = 1
    t_warmup = time.perf_counter()
    warmup_bszes = [h.ttt_batch_size]
    for bsz in warmup_bszes:
        wl = BatchedTTTLoRA(
            bsz, ttt_model, h.ttt_lora_rank,
            k_lora=h.ttt_k_lora, mlp_lora=h.ttt_mlp_lora, o_lora=h.ttt_o_lora,
        ).to(device)
        wo = torch.optim.AdamW(
            wl.parameters(),
            lr=h.ttt_lora_lr,
            betas=(h.ttt_beta1, h.ttt_beta2),
            eps=1e-10,
            weight_decay=h.ttt_weight_decay,
            fused=True,
        )
        for ctx_len in (h.ttt_chunk_size, h.ttt_eval_seq_len):
            xw = torch.randint(
                0, h.vocab_size, (bsz, ctx_len), device=device, dtype=torch.int64
            )
            yw = torch.randint(
                0, h.vocab_size, (bsz, ctx_len), device=device, dtype=torch.int64
            )
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                ptl = fwd_ttt_compiled(xw, yw, lora=wl)
            ptl[:, : min(h.ttt_chunk_size, ctx_len)].mean(dim=-1).sum().backward()
            wo.step()
            wo.zero_grad(set_to_none=True)
        del wl, wo
    torch.cuda.empty_cache()
    compile_elapsed = time.perf_counter() - t_warmup
    log(f"ttt_lora:compile warmup done ({compile_elapsed:.1f}s)")
    log("\nbeginning TTT eval timer")
    torch.cuda.synchronize()
    t_ttt = time.perf_counter()
    ttt_val_loss, ttt_val_bpb = eval_val_ttt_phased(
        h, ttt_model, device, val_data, forward_ttt_train=fwd_ttt_compiled
    )
    torch.cuda.synchronize()
    ttt_eval_elapsed = time.perf_counter() - t_ttt
    log(
        "quantized_ttt_phased "
        f"val_loss:{ttt_val_loss:.8f} val_bpb:{ttt_val_bpb:.8f} "
        f"eval_time:{1e3 * ttt_eval_elapsed:.0f}ms"
    )
    log(f"total_eval_time:{ttt_eval_elapsed:.1f}s")
    del ttt_model
    torch._dynamo.reset()
    torch.cuda.empty_cache()
    return float(ttt_val_loss), float(ttt_val_bpb), int(ttt_eval_elapsed * 1000)


def main() -> int:
    mode, base_seed, ckpt_path = _parse_env()

    # Standard distributed init, same pattern as train_gpt.main().
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)

    h = Hyperparameters()
    # grad_accum_steps is computed in train_gpt.main() and expected by the TTT
    # path; replicate that derivation so we can run without train_model().
    h.grad_accum_steps = max(1, 8 // max(1, world_size))
    h.world_size = world_size
    h.local_rank = local_rank
    h.is_main_process = (not distributed) or dist.get_rank() == 0
    h.distributed = distributed

    set_logging_hparams(h)

    log(f"spinquant_hotstart: mode={mode} seed={base_seed} ckpt={ckpt_path}")

    # 1. Build empty GPT(h) and load the FP state_dict.
    base_model = GPT(h).to(device).bfloat16()
    restore_fp32_params(base_model)
    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Some state_dict saves are the raw dict; some wrap under 'model'. Tolerate both.
    if isinstance(sd, dict) and "state_dict" in sd and "qo_bank" not in sd:
        sd = sd["state_dict"]
    missing, unexpected = base_model.load_state_dict(sd, strict=False)
    if missing:
        log(f"spinquant_hotstart: state_dict missing keys: {missing[:10]}"
            + (f" ... +{len(missing) - 10} more" if len(missing) > 10 else ""))
    if unexpected:
        log(f"spinquant_hotstart: state_dict unexpected keys: {unexpected[:10]}"
            + (f" ... +{len(unexpected) - 10} more" if len(unexpected) > 10 else ""))

    val_data = ValidationData(h, device)
    log(f"val_tokens: {val_data.val_tokens.numel() - 1}")

    # 2. Apply rotations in-place (no-op for baseline).
    t_rot = time.perf_counter()
    rotation_manifest = apply_rotations(base_model, h, mode, base_seed)
    t_rot = time.perf_counter() - t_rot
    log(f"spinquant_hotstart: applied mode={mode} rotations in {t_rot:.2f}s")

    if h.is_main_process and h.artifact_dir:
        manifest_path = Path(h.artifact_dir) / "rotation_manifest.json"
        manifest_path.write_text(json.dumps(rotation_manifest, indent=2, default=str))
        log(f"spinquant_hotstart: wrote rotation manifest -> {manifest_path}")

    # Barrier so all ranks have the rotated weights before serialize reads rank-0.
    if distributed:
        dist.barrier()

    # 3. Optional pre-quant post-rotation eval. Mirrors the diagnostic at line 2969.
    pre_quant_loss, pre_quant_bpb, pre_quant_ms = None, None, None
    try:
        torch._dynamo.reset()
        compiled = torch.compile(base_model, dynamic=False, fullgraph=True)
        compiled_logits = torch.compile(
            base_model.forward_logits, dynamic=False, fullgraph=True
        )
        t0 = time.perf_counter()
        pre_quant_loss_t, pre_quant_bpb_t = eval_val(
            h, device, val_data, compiled, compiled_logits
        )
        pre_quant_loss = float(pre_quant_loss_t)
        pre_quant_bpb = float(pre_quant_bpb_t)
        pre_quant_ms = int((time.perf_counter() - t0) * 1000)
        log(
            "diagnostic_pre_quant_post_rotation "
            f"val_loss:{pre_quant_loss:.8f} val_bpb:{pre_quant_bpb:.8f}"
        )
        del compiled, compiled_logits
        torch._dynamo.reset()
        torch.cuda.empty_cache()
    except Exception as e:  # pragma: no cover — diagnostic only
        log(f"diagnostic_pre_quant_post_rotation SKIPPED due to error: {e!r}")

    # 4. Call serialize() — this runs GPTQ on (possibly rotated) weights,
    #    writes {artifact_dir}/final_model.int6.ptz.
    code_text = Path(__file__).read_text(encoding="utf-8")
    bytes_total, quant_bytes = serialize(h, base_model, code_text)
    log(f"serialize done: total={bytes_total} quant_file={quant_bytes}")
    if distributed:
        dist.barrier()

    # 5. Quantized eval (no TTT).
    q_loss, q_bpb, q_ms = _run_quantized_eval(h, val_data, device)

    # 6. Phased TTT eval.
    ttt_loss, ttt_bpb, ttt_ms = None, None, None
    if h.ttt_enabled:
        ttt_loss, ttt_bpb, ttt_ms = _run_ttt_eval(h, val_data, device)

    # 7. Write final.json.
    if h.is_main_process and h.artifact_dir:
        out = {
            "spec": "009-spinquant-hotstart",
            "mode": mode,
            "spinquant_seed": base_seed,
            "hotstart_ckpt": ckpt_path,
            "status": "ok",
            "diagnostic_pre_quant_post_rotation": (
                {
                    "val_loss": pre_quant_loss,
                    "val_bpb": pre_quant_bpb,
                    "eval_time_ms": pre_quant_ms,
                }
                if pre_quant_bpb is not None
                else None
            ),
            "diagnostic_quantized": {
                "val_loss": q_loss,
                "val_bpb": q_bpb,
                "eval_time_ms": q_ms,
            },
            "quantized_ttt_phased": (
                {
                    "val_loss": ttt_loss,
                    "val_bpb": ttt_bpb,
                    "eval_time_ms": ttt_ms,
                }
                if ttt_bpb is not None
                else None
            ),
            "artifact_bytes": bytes_total,
            "quant_file_bytes": quant_bytes,
        }
        out_path = Path(h.artifact_dir) / "final.json"
        out_path.write_text(json.dumps(out, indent=2))
        log(f"wrote {out_path}")

    if distributed:
        dist.barrier()
        dist.destroy_process_group()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
