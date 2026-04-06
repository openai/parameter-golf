from __future__ import annotations

import importlib.util
import json
import os
import sys
import types
from pathlib import Path

import torch
import torch.nn.functional as F

sys.dont_write_bytecode = True


def _ensure_optional_stubs() -> None:
    if "sentencepiece" not in sys.modules:
        spm_stub = types.ModuleType("sentencepiece")

        class _SentencePieceProcessor:  # pragma: no cover - import stub only
            pass

        spm_stub.SentencePieceProcessor = _SentencePieceProcessor
        sys.modules["sentencepiece"] = spm_stub

    if "flash_attn_interface" not in sys.modules:
        fa3_stub = types.ModuleType("flash_attn_interface")

        def _flash_attn_proxy(q, k, v, causal=True):
            qh = q.permute(0, 2, 1, 3).contiguous()
            kh = k.permute(0, 2, 1, 3).contiguous()
            vh = v.permute(0, 2, 1, 3).contiguous()
            y = F.scaled_dot_product_attention(
                qh,
                kh,
                vh,
                is_causal=causal,
                enable_gqa=(k.size(2) != q.size(2)),
            )
            return y.permute(0, 2, 1, 3).contiguous()

        fa3_stub.flash_attn_func = _flash_attn_proxy
        sys.modules["flash_attn_interface"] = fa3_stub


def _load_module(file_path: Path, module_name: str):
    _ensure_optional_stubs()
    spec = importlib.util.spec_from_file_location(module_name, str(file_path))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load module spec for {file_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _build_model(mod):
    hp = mod.Hyperparameters()
    model = mod.GPT(
        vocab_size=hp.vocab_size,
        num_layers=hp.num_layers,
        model_dim=hp.model_dim,
        num_heads=hp.num_heads,
        num_kv_heads=hp.num_kv_heads,
        mlp_mult=hp.mlp_mult,
        tie_embeddings=hp.tie_embeddings,
        tied_embed_init_std=hp.tied_embed_init_std,
        logit_softcap=hp.logit_softcap,
        rope_base=hp.rope_base,
        qk_gain_init=hp.qk_gain_init,
        mtp_num_heads=hp.mtp_num_heads,
        mtp_loss_weight=hp.mtp_loss_weight,
        bigram_vocab_size=hp.bigram_vocab_size,
        bigram_dim=hp.bigram_dim,
        xsa_last_n=hp.xsa_last_n,
        rope_dims=hp.rope_dims,
        ln_scale=hp.ln_scale,
        dtg=hp.dtg_enabled,
        ve_enabled=hp.ve_enabled,
        ve_dim=hp.ve_dim,
        ve_layers=hp.ve_layers,
        gated_attention=hp.gated_attention,
        value_residual=hp.value_residual,
    ).to("cuda").bfloat16()
    model.train()
    return model, hp


def _maybe_compile(model: torch.nn.Module, use_compile: bool) -> tuple[torch.nn.Module, bool, str | None]:
    if not use_compile:
        return model, False, None
    try:
        compiled = torch.compile(model, dynamic=False, fullgraph=True)
        return compiled, True, None
    except Exception as exc:  # pragma: no cover - environment dependent
        return model, False, str(exc)


def _bench_train_step(
    model: torch.nn.Module,
    vocab_size: int,
    batch_size: int,
    seq_len: int,
    iters: int,
    warmup: int,
) -> tuple[float, float, float]:
    x = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda", dtype=torch.long)
    y = torch.randint(0, vocab_size, (batch_size, seq_len), device="cuda", dtype=torch.long)

    last_loss = torch.zeros((), device="cuda", dtype=torch.float32)

    def _step() -> None:
        nonlocal last_loss
        model.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)
        last_loss = loss.detach().float()
        loss.backward()

    for _ in range(warmup):
        _step()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        _step()
    end.record()
    torch.cuda.synchronize()

    ms = float(start.elapsed_time(end) / max(iters, 1))
    toks_per_s = float(batch_size * seq_len * 1000.0 / max(ms, 1e-9))
    return ms, toks_per_s, float(last_loss.item())


def _default_top1_path(train_path: Path) -> Path:
    repo_root = train_path.resolve().parents[3]
    return repo_root / "records/track_10min_16mb/2026-03-25_ValCalib_GPTQ_XSA_BigramHash3072/train_gpt.py"


def run_checks(train_path: Path, out_path: Path) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for these simulation benchmarks")

    torch.manual_seed(1337)
    torch.cuda.manual_seed_all(1337)

    sim_seq_len = int(os.environ.get("SIM_SEQ_LEN", "512"))
    sim_batch = int(os.environ.get("SIM_BATCH", "2"))
    sim_iters = int(os.environ.get("SIM_ITERS", "8"))
    sim_warmup = int(os.environ.get("SIM_WARMUP", "3"))
    sim_use_compile = bool(int(os.environ.get("SIM_USE_COMPILE", "0")))
    sim_winner_speed_min = float(os.environ.get("SIM_WINNER_SPEED_MIN", "1.0"))

    top1_train_path = Path(os.environ.get("SIM_TOP1_TRAIN_PATH", str(_default_top1_path(train_path)))).resolve()
    if not top1_train_path.exists():
        raise FileNotFoundError(f"Top-1 train file not found: {top1_train_path}")

    current_mod = _load_module(train_path, "epsilon_flash_only_train")
    current_model, current_hp = _build_model(current_mod)
    current_model, current_compiled, current_compile_error = _maybe_compile(current_model, sim_use_compile)
    current_ms, current_toks_per_s, current_loss = _bench_train_step(
        current_model,
        current_hp.vocab_size,
        sim_batch,
        sim_seq_len,
        sim_iters,
        sim_warmup,
    )
    epsilon_sparsity = float(getattr(current_mod, "_EPSILON_last_sparsity", 0.0))

    thermo_vals = [
        current_mod.get_thermodynamic_scale(1.0, 3.0, 0.85, 1.15),
        current_mod.get_thermodynamic_scale(3.0, 3.0, 0.85, 1.15),
        current_mod.get_thermodynamic_scale(8.0, 3.0, 0.85, 1.15),
    ]
    w = torch.randn(4096, 64, device="cuda", dtype=torch.float32)
    current_mod.project_rows_to_unit_sphere_(w)
    sphere_max_dev = float((w.norm(dim=1) - 1.0).abs().max().item())

    del current_model
    torch.cuda.empty_cache()

    top1_mod = _load_module(top1_train_path, "leaderboard_top1_train")
    top1_model, top1_hp = _build_model(top1_mod)
    top1_model, top1_compiled, top1_compile_error = _maybe_compile(top1_model, sim_use_compile)
    top1_ms, top1_toks_per_s, top1_loss = _bench_train_step(
        top1_model,
        top1_hp.vocab_size,
        sim_batch,
        sim_seq_len,
        sim_iters,
        sim_warmup,
    )

    speed_ratio_vs_top1 = top1_ms / max(current_ms, 1e-9)

    results = {
        "device": torch.cuda.get_device_name(torch.device("cuda")),
        "sim_hparams": {
            "sim_batch": sim_batch,
            "sim_seq_len": sim_seq_len,
            "sim_iters": sim_iters,
            "sim_warmup": sim_warmup,
            "sim_use_compile": sim_use_compile,
            "sim_winner_speed_min": sim_winner_speed_min,
            "top1_train_path": str(top1_train_path),
        },
        "current_run": {
            "train_path": str(train_path.resolve()),
            "compiled": current_compiled,
            "compile_error": current_compile_error,
            "step_ms": current_ms,
            "tokens_per_s": current_toks_per_s,
            "train_loss": current_loss,
            "epsilon_sparsity_observed": epsilon_sparsity,
            "num_layers": int(current_hp.num_layers),
            "xsa_last_n": int(current_hp.xsa_last_n),
            "ve_enabled": bool(current_hp.ve_enabled),
            "bigram_vocab_size": int(current_hp.bigram_vocab_size),
            "bigram_dim": int(current_hp.bigram_dim),
            "mlp_mult": float(current_hp.mlp_mult),
        },
        "top1_run": {
            "compiled": top1_compiled,
            "compile_error": top1_compile_error,
            "step_ms": top1_ms,
            "tokens_per_s": top1_toks_per_s,
            "train_loss": top1_loss,
            "num_layers": int(top1_hp.num_layers),
            "xsa_last_n": int(top1_hp.xsa_last_n),
            "ve_enabled": bool(top1_hp.ve_enabled),
            "bigram_vocab_size": int(top1_hp.bigram_vocab_size),
            "bigram_dim": int(top1_hp.bigram_dim),
            "mlp_mult": float(top1_hp.mlp_mult),
        },
        "speed_ratio_vs_top1": speed_ratio_vs_top1,
        "thermo_scales": thermo_vals,
        "sphere_max_norm_deviation": sphere_max_dev,
    }

    checks = {
        "beats_top1_speed": speed_ratio_vs_top1 >= sim_winner_speed_min,
        "loss_finite": torch.isfinite(torch.tensor([current_loss, top1_loss], device="cpu")).all().item(),
        "epsilon_sparse_off": 0.0 <= epsilon_sparsity <= 0.05,
        "thermo_ok": thermo_vals[0] <= thermo_vals[1] <= thermo_vals[2],
        "sphere_ok": sphere_max_dev < 1e-4,
    }
    results["checks"] = checks
    results["all_checks_passed"] = bool(all(checks.values()))

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main() -> None:
    train_path = Path(__file__).with_name("train_gpt.py")
    out_path = Path(__file__).parent / "sim_checks_local.json"
    results = run_checks(train_path, out_path)
    print(json.dumps(results, indent=2))
    if not results["all_checks_passed"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
