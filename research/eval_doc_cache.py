#!/usr/bin/env python3
from __future__ import annotations

import argparse
import glob
import importlib.util
import inspect
import math
import sys
import time
import io
import zlib
from pathlib import Path
from types import ModuleType
from typing import Iterable

import numpy as np
import sentencepiece as spm
import torch
import torch.nn.functional as F
try:
    import zstandard
except ImportError:
    zstandard = None

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


BOS_ID = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate doc-local sliding and cache interpolation.")
    parser.add_argument(
        "--train-script",
        type=Path,
        default=Path("train_gpt.py"),
        help="Path to the train_gpt.py variant that defines the model/checkpoint schema.",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("local_runs/proxy500/final_model.pt"),
        help="Path to a saved fp checkpoint.",
    )
    parser.add_argument(
        "--quant-roundtrip",
        action="store_true",
        help="Apply the train script's quantize/dequantize path to the loaded checkpoint before evaluation.",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=Path("data/tokenizers/fineweb_1024_bpe.model"),
        help="SentencePiece tokenizer path.",
    )
    parser.add_argument(
        "--val-files",
        type=str,
        default="data/datasets/fineweb10B_sp1024/fineweb_val_*.bin",
        help="Glob for validation token shards.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--max-tokens", type=int, default=1_048_576)
    parser.add_argument(
        "--start-token",
        type=int,
        default=0,
        help="Approximate starting token offset within validation stream; subset is BOS-aligned.",
    )
    parser.add_argument("--seq-len", type=int, default=1024)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--batch-seqs", type=int, default=64)
    parser.add_argument(
        "--mode",
        choices=(
            "flat_sliding",
            "flat_cache_bigram_adaptive",
            "flat_cache_bigram_adaptive_entropy",
            "doc_sliding",
            "doc_cache_bigram",
            "doc_cache_bigram_adaptive",
            "doc_cache_bigram_adaptive_entropy",
            "doc_cache_trigram_backoff",
            "doc_cache_unigram",
        ),
        default="doc_cache_bigram",
    )
    parser.add_argument("--alpha", type=float, default=0.03)
    parser.add_argument(
        "--alphas",
        type=float,
        nargs="*",
        default=None,
        help="Optional alpha sweep; overrides --alpha if provided.",
    )
    parser.add_argument(
        "--tau",
        type=float,
        default=8.0,
        help="Adaptive bigram pseudo-count temperature for doc_cache_bigram_adaptive.",
    )
    parser.add_argument(
        "--trigram-alpha",
        type=float,
        default=0.02,
        help="Trigram component weight for trigram backoff mode.",
    )
    parser.add_argument(
        "--entropy-power",
        type=float,
        default=1.0,
        help="Exponent for entropy-normalized gating in doc_cache_bigram_adaptive_entropy.",
    )
    parser.add_argument(
        "--reset-cache-on-bos",
        action="store_true",
        help="For flat cache modes, reset cache counts whenever the previous token is BOS.",
    )
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Compile the forward logits function. Useful for larger runs.",
    )
    parser.add_argument(
        "--ttt-epochs",
        type=int,
        default=0,
        help="Optional number of test-time training epochs to run on the evaluation subset before scoring.",
    )
    parser.add_argument(
        "--ttt-lr",
        type=float,
        default=1e-4,
        help="Learning rate for optional test-time training.",
    )
    parser.add_argument(
        "--ttt-batch-seqs",
        type=int,
        default=4,
        help="Number of sequences per TTT optimizer step.",
    )
    return parser.parse_args()


def find_docs(all_tokens: torch.Tensor, include_next_bos: bool = True) -> list[tuple[int, int]]:
    bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0].cpu().numpy()
    docs: list[tuple[int, int]] = []
    for i in range(len(bos_positions)):
        start = int(bos_positions[i])
        end = int(bos_positions[i + 1]) if i + 1 < len(bos_positions) else int(all_tokens.numel())
        if include_next_bos and i + 1 < len(bos_positions):
            end += 1
        if end - start >= 2:
            docs.append((start, end - start))
    return docs


def load_train_module(train_script: Path) -> ModuleType:
    path = train_script.resolve()
    spec = importlib.util.spec_from_file_location(f"pg_train_{path.stem}", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def instantiate_model(train_mod: ModuleType, device: torch.device, seq_len: int) -> torch.nn.Module:
    args = train_mod.Hyperparameters()
    if hasattr(args, "train_seq_len"):
        setattr(args, "train_seq_len", seq_len)

    sig = inspect.signature(train_mod.GPT)
    kwargs = {}
    missing: list[str] = []
    for name, param in sig.parameters.items():
        if name == "self":
            continue
        if hasattr(args, name):
            kwargs[name] = getattr(args, name)
        elif param.default is inspect._empty:
            missing.append(name)
    if missing:
        raise TypeError(f"Could not build GPT from {train_mod.__file__}; missing args: {missing}")

    model = train_mod.GPT(**kwargs).to(device).bfloat16()
    casted_linear_cls = getattr(train_mod, "CastedLinear", None)
    if casted_linear_cls is not None:
        for module in model.modules():
            if isinstance(module, casted_linear_cls):
                module.float()
    if hasattr(train_mod, "restore_low_dim_params_to_fp32"):
        train_mod.restore_low_dim_params_to_fp32(model)
    return model


def roundtrip_state(train_mod: ModuleType, state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    sd_cpu = {k: v.detach().cpu() for k, v in state.items()}

    if hasattr(train_mod, "mixed_quantize") and hasattr(train_mod, "dequantize_mixed"):
        int6_cats = getattr(train_mod, "MIXED_QUANT_INT6_CATS", {"mlp", "attn"})
        q_state, q_meta = train_mod.mixed_quantize(sd_cpu, int6_cats)
        return train_mod.dequantize_mixed(q_state, q_meta, sd_cpu)

    if hasattr(train_mod, "mixed_quantize_int6") and hasattr(train_mod, "dequantize_mixed_int6"):
        q_state, q_meta = train_mod.mixed_quantize_int6(sd_cpu, {"mlp", "attn"})
        return train_mod.dequantize_mixed_int6(q_state, q_meta, sd_cpu)

    if hasattr(train_mod, "quantize_state_dict_int6") and hasattr(train_mod, "dequantize_state_dict"):
        q_obj, _ = train_mod.quantize_state_dict_int6(sd_cpu)
        return train_mod.dequantize_state_dict(q_obj)

    if hasattr(train_mod, "quantize_state_dict_int8") and hasattr(train_mod, "dequantize_state_dict_int8"):
        q_obj = train_mod.quantize_state_dict_int8(sd_cpu)
        if isinstance(q_obj, tuple):
            q_obj = q_obj[0]
        return train_mod.dequantize_state_dict_int8(q_obj)

    raise RuntimeError(f"Don't know how to roundtrip-quantize using {train_mod.__file__}")


def load_checkpoint_state(args: argparse.Namespace, train_mod: ModuleType) -> dict[str, torch.Tensor]:
    # Support direct evaluation of .ptz artifacts when the raw container matches recent record formats.
    if args.checkpoint.suffix == ".ptz":
        blob = args.checkpoint.read_bytes()
        last_err = None
        for decompress in (
            (lambda b: zstandard.ZstdDecompressor().decompress(b)) if zstandard is not None else None,
            lambda b: zlib.decompress(b),
        ):
            if decompress is None:
                continue
            try:
                payload = torch.load(io.BytesIO(decompress(blob)), map_location="cpu")
                if isinstance(payload, dict):
                    if "w" in payload and "m" in payload:
                        if hasattr(train_mod, "dequantize_mixed"):
                            raise RuntimeError("Dequantizing .ptz directly needs a fp reference state; pass --checkpoint final_model.pt --quant-roundtrip instead.")
                        if hasattr(train_mod, "dequantize_mixed_int6"):
                            raise RuntimeError("Dequantizing .ptz directly needs a fp reference state; pass --checkpoint final_model.pt --quant-roundtrip instead.")
                        if hasattr(train_mod, "dequantize_state_dict"):
                            return train_mod.dequantize_state_dict(payload)
                    return payload
            except Exception as exc:  # noqa: BLE001
                last_err = exc
        raise RuntimeError(f"Unable to load .ptz checkpoint {args.checkpoint}: {last_err}")

    state_obj = torch.load(args.checkpoint, map_location="cpu")
    if not isinstance(state_obj, dict):
        raise TypeError(f"Unsupported checkpoint object type: {type(state_obj)}")

    state = state_obj
    if args.quant_roundtrip:
        state = roundtrip_state(train_mod, state)
    return state


def forward_logits_fallback(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    x = model.tok_emb(input_ids)
    if hasattr(model, "bigram") and model.bigram is not None:
        x = x + model.bigram(input_ids)
    if hasattr(model, "smear") and model.smear is not None:
        x = model.smear(x)
    x = F.rms_norm(x, (x.size(-1),))
    x0 = x
    skips: list[torch.Tensor] = []
    for i in range(model.num_encoder_layers):
        x = model.blocks[i](x, x0)
        skips.append(x)
    for i in range(model.num_decoder_layers):
        if skips:
            x = x + model.skip_weights[i].to(dtype=x.dtype)[None, None, :] * skips.pop()
        x = model.blocks[model.num_encoder_layers + i](x, x0)
    x = model.final_norm(x)
    if model.tie_embeddings:
        logits_proj = F.linear(x, model.tok_emb.weight)
    else:
        if model.lm_head is None:
            raise RuntimeError("lm_head is required when tie_embeddings=False")
        logits_proj = model.lm_head(x)
    return model.logit_softcap * torch.tanh(logits_proj / model.logit_softcap)


def forward_logits(model: torch.nn.Module, input_ids: torch.Tensor) -> torch.Tensor:
    if hasattr(model, "forward_logits"):
        return model.forward_logits(input_ids)
    return forward_logits_fallback(model, input_ids)


def load_validation_subset(
    train_mod: ModuleType, pattern: str, seq_len: int, max_tokens: int | None, start_token: int
) -> torch.Tensor:
    files = [Path(p) for p in sorted(glob.glob(pattern))]
    if not files:
        raise FileNotFoundError(f"No files found for pattern: {pattern}")
    all_tokens = torch.cat([train_mod.load_data_shard(file) for file in files]).contiguous()
    total_tokens = int(all_tokens.numel()) - 1
    start = max(0, min(start_token, total_tokens))
    end = int(all_tokens.numel())
    if max_tokens is not None and max_tokens > 0:
        end = min(start + max_tokens + 1, int(all_tokens.numel()))
    if start > 0 or end < int(all_tokens.numel()):
        bos_positions = (all_tokens == BOS_ID).nonzero(as_tuple=True)[0]
        if start > 0:
            i = int(torch.searchsorted(bos_positions, torch.tensor(start), right=False).item())
            if i < bos_positions.numel():
                start = int(bos_positions[i].item())
            else:
                start = int(bos_positions[-1].item())
        if end < int(all_tokens.numel()):
            j = int(torch.searchsorted(bos_positions, torch.tensor(end), right=False).item())
            if j > 0:
                end = int(bos_positions[j - 1].item()) + 1
        if end <= start:
            raise ValueError(f"Invalid BOS-aligned slice for start_token={start_token}, max_tokens={max_tokens}")
        all_tokens = all_tokens[start:end].contiguous()
    usable = all_tokens.numel() - 1
    if usable <= 0:
        raise ValueError("Validation subset has no predictible tokens.")
    if usable < seq_len:
        return all_tokens
    return all_tokens


def build_metric_luts(
    train_mod: ModuleType, tokenizer_path: Path, vocab_size: int, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    sp = spm.SentencePieceProcessor(model_file=str(tokenizer_path))
    return train_mod.build_sentencepiece_luts(sp, vocab_size, device)


def run_ttt(
    model: torch.nn.Module,
    val_tokens: torch.Tensor,
    seq_len: int,
    epochs: int,
    lr: float,
    batch_seqs: int,
    device: torch.device,
) -> None:
    if epochs <= 0:
        return
    total_pred = int(val_tokens.numel()) - 1
    total_seqs = total_pred // seq_len
    if total_seqs <= 0:
        print("ttt skipped: subset shorter than one sequence")
        return

    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.SGD(params, lr=lr)
    tic = time.perf_counter()
    print(f"ttt start epochs={epochs} lr={lr} batch_seqs={batch_seqs} total_seqs={total_seqs}", flush=True)
    for ep in range(epochs):
        loss_sum = 0.0
        count = 0
        for si in range(0, total_seqs, batch_seqs):
            se = min(si + batch_seqs, total_seqs)
            bsz = se - si
            rs = si * seq_len
            re = se * seq_len + 1
            local = val_tokens[rs:re].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(bsz, seq_len)
            y = local[1:].reshape(bsz, seq_len)
            opt.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = model(x, y)
            loss.backward()
            opt.step()
            loss_sum += float(loss.item()) * bsz
            count += bsz
        print(f"ttt epoch={ep + 1}/{epochs} loss={loss_sum / max(count, 1):.6f}", flush=True)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    print(f"ttt_time_sec={time.perf_counter() - tic:.3f}", flush=True)
    model.eval()


def iter_unique_windows(pred_len: int, seq_len: int, stride: int) -> Iterable[tuple[int, int, int]]:
    if pred_len <= 0:
        return
    first_end = min(seq_len, pred_len)
    yield 0, first_end, 0
    scored_end = first_end
    while scored_end < pred_len:
        next_end = min(scored_end + stride, pred_len)
        ws = max(0, next_end - seq_len)
        wlen = next_end - ws
        score_from = scored_end - ws
        yield ws, wlen, score_from
        scored_end = next_end


def iter_flat_windows(total_tokens: int, seq_len: int, stride: int) -> Iterable[tuple[int, int, int]]:
    yield from iter_unique_windows(total_tokens, seq_len, stride)


def iter_doc_windows(doc_len: int, seq_len: int, stride: int) -> Iterable[tuple[int, int, int]]:
    yield from iter_unique_windows(doc_len - 1, seq_len, stride)


def gather_doc_scores(
    model: torch.nn.Module,
    val_tokens: torch.Tensor,
    docs: list[tuple[int, int]],
    seq_len: int,
    stride: int,
    batch_seqs: int,
    device: torch.device,
    compile_model: bool,
) -> list[dict[str, np.ndarray]]:
    run_forward = forward_logits
    if compile_model:
        run_forward = torch.compile(run_forward, dynamic=False, fullgraph=False)

    doc_windows: list[tuple[int, int, int, int, int]] = []
    for doc_id, (doc_start, doc_len) in enumerate(docs):
        for rel_ws, wlen, score_from in iter_doc_windows(doc_len, seq_len, stride):
            doc_windows.append((doc_id, doc_start, doc_len, rel_ws, score_from))

    doc_bases: list[list[torch.Tensor]] = [[] for _ in docs]
    doc_prevs: list[list[torch.Tensor]] = [[] for _ in docs]
    doc_tgts: list[list[torch.Tensor]] = [[] for _ in docs]
    doc_entropy: list[list[torch.Tensor]] = [[] for _ in docs]

    model.eval()
    tic = time.perf_counter()
    with torch.inference_mode():
        for bi in range(0, len(doc_windows), batch_seqs):
            batch = doc_windows[bi : bi + batch_seqs]
            max_wlen = 0
            for _, _, doc_len, rel_ws, _ in batch:
                pred_len = doc_len - 1
                max_wlen = max(max_wlen, min(seq_len, pred_len - rel_ws))
            x_batch = torch.zeros(len(batch), max_wlen, dtype=torch.int64, device=device)
            y_batch = torch.zeros(len(batch), max_wlen, dtype=torch.int64, device=device)
            wlens: list[int] = []
            for i, (_, doc_start, doc_len, rel_ws, _) in enumerate(batch):
                pred_len = doc_len - 1
                wlen = min(seq_len, pred_len - rel_ws)
                toks = val_tokens[doc_start + rel_ws : doc_start + rel_ws + wlen + 1]
                wlens.append(wlen)
                x_batch[i, :wlen] = toks[:-1].to(device=device, dtype=torch.int64)
                y_batch[i, :wlen] = toks[1:].to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = run_forward(model, x_batch)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            for i, (doc_id, _, _, _, score_from) in enumerate(batch):
                wlen = wlens[i]
                prev = x_batch[i, score_from:wlen].detach().cpu()
                tgt = y_batch[i, score_from:wlen].detach().cpu()
                lp = log_probs[i, score_from:wlen].gather(1, tgt[:, None].to(device=log_probs.device)).squeeze(1).detach().cpu()
                ent = entropy[i, score_from:wlen].detach().cpu()
                doc_prevs[doc_id].append(prev)
                doc_tgts[doc_id].append(tgt)
                doc_bases[doc_id].append(lp)
                doc_entropy[doc_id].append(ent)
            if (bi // batch_seqs) % 50 == 0:
                done = min(bi + batch_seqs, len(doc_windows))
                pct = 100.0 * done / max(len(doc_windows), 1)
                print(f"gather [{pct:5.1f}%] {done}/{len(doc_windows)} windows", flush=True)

    elapsed = time.perf_counter() - tic
    print(f"gather_time_sec={elapsed:.3f}")

    results: list[dict[str, np.ndarray]] = []
    for doc_id, (doc_start, doc_len) in enumerate(docs):
        prev = torch.cat(doc_prevs[doc_id]).numpy()
        tgt = torch.cat(doc_tgts[doc_id]).numpy()
        base = torch.cat(doc_bases[doc_id]).numpy()
        ent = torch.cat(doc_entropy[doc_id]).numpy()
        if prev.shape[0] != doc_len - 1:
            raise RuntimeError(f"Doc {doc_id} expected {doc_len - 1} tokens, got {prev.shape[0]}")
        results.append({"start": np.int64(doc_start), "prev": prev, "tgt": tgt, "base_logp": base, "entropy": ent})
    return results


def gather_flat_scores(
    model: torch.nn.Module,
    val_tokens: torch.Tensor,
    seq_len: int,
    stride: int,
    batch_seqs: int,
    device: torch.device,
    compile_model: bool,
) -> dict[str, np.ndarray]:
    run_forward = forward_logits
    if compile_model:
        run_forward = torch.compile(run_forward, dynamic=False, fullgraph=False)

    windows = list(iter_flat_windows(int(val_tokens.numel()) - 1, seq_len, stride))
    prev_parts: list[torch.Tensor] = []
    tgt_parts: list[torch.Tensor] = []
    base_parts: list[torch.Tensor] = []
    entropy_parts: list[torch.Tensor] = []

    model.eval()
    tic = time.perf_counter()
    with torch.inference_mode():
        for bi in range(0, len(windows), batch_seqs):
            batch = windows[bi : bi + batch_seqs]
            max_wlen = max(wlen for _, wlen, _ in batch)
            x_batch = torch.zeros(len(batch), max_wlen, dtype=torch.int64, device=device)
            y_batch = torch.zeros(len(batch), max_wlen, dtype=torch.int64, device=device)
            for i, (ws, wlen, _) in enumerate(batch):
                toks = val_tokens[ws : ws + wlen + 1]
                x_batch[i, :wlen] = toks[:-1].to(device=device, dtype=torch.int64)
                y_batch[i, :wlen] = toks[1:].to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = run_forward(model, x_batch)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
            for i, (_, wlen, score_from) in enumerate(batch):
                prev = x_batch[i, score_from:wlen].detach().cpu()
                tgt = y_batch[i, score_from:wlen].detach().cpu()
                lp = log_probs[i, score_from:wlen].gather(1, tgt[:, None].to(device=log_probs.device)).squeeze(1).detach().cpu()
                ent = entropy[i, score_from:wlen].detach().cpu()
                prev_parts.append(prev)
                tgt_parts.append(tgt)
                base_parts.append(lp)
                entropy_parts.append(ent)
            if (bi // batch_seqs) % 50 == 0:
                done = min(bi + batch_seqs, len(windows))
                pct = 100.0 * done / max(len(windows), 1)
                print(f"gather_flat [{pct:5.1f}%] {done}/{len(windows)} windows", flush=True)

    elapsed = time.perf_counter() - tic
    print(f"gather_flat_time_sec={elapsed:.3f}")
    prev = torch.cat(prev_parts).numpy()
    tgt = torch.cat(tgt_parts).numpy()
    base = torch.cat(base_parts).numpy()
    ent = torch.cat(entropy_parts).numpy()
    expected = int(val_tokens.numel()) - 1
    if prev.shape[0] != expected:
        raise RuntimeError(f"Flat gather expected {expected} tokens, got {prev.shape[0]}")
    return {"prev": prev, "tgt": tgt, "base_logp": base, "entropy": ent}


def eval_flat_sliding(
    model: torch.nn.Module,
    val_tokens: torch.Tensor,
    seq_len: int,
    stride: int,
    batch_seqs: int,
    device: torch.device,
    compile_model: bool,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
) -> tuple[float, float]:
    run_forward = forward_logits
    if compile_model:
        run_forward = torch.compile(run_forward, dynamic=False, fullgraph=False)

    windows = list(iter_flat_windows(int(val_tokens.numel()) - 1, seq_len, stride))
    loss_sum = 0.0
    token_count = 0
    byte_count = 0.0
    tic = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for bi in range(0, len(windows), batch_seqs):
            batch = windows[bi : bi + batch_seqs]
            max_wlen = max(wlen for _, wlen, _ in batch)
            x_batch = torch.zeros(len(batch), max_wlen, dtype=torch.int64, device=device)
            y_batch = torch.zeros(len(batch), max_wlen, dtype=torch.int64, device=device)
            for i, (ws, wlen, _) in enumerate(batch):
                toks = val_tokens[ws : ws + wlen + 1]
                x_batch[i, :wlen] = toks[:-1].to(device=device, dtype=torch.int64)
                y_batch[i, :wlen] = toks[1:].to(device=device, dtype=torch.int64)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = run_forward(model, x_batch)
            log_probs = F.log_softmax(logits.float(), dim=-1)
            for i, (_, wlen, score_from) in enumerate(batch):
                prev = x_batch[i, score_from:wlen]
                tgt = y_batch[i, score_from:wlen]
                lp = log_probs[i, score_from:wlen].gather(1, tgt[:, None]).squeeze(1)
                loss_sum += float((-lp).sum().item())
                token_count += int(tgt.numel())
                tb = base_bytes_lut[tgt].to(torch.float64)
                tb += (has_leading_space_lut[tgt] & ~is_boundary_token_lut[prev]).to(torch.float64)
                byte_count += float(tb.sum().item())
            if (bi // batch_seqs) % 50 == 0:
                done = min(bi + batch_seqs, len(windows))
                pct = 100.0 * done / max(len(windows), 1)
                running_bpb = (loss_sum / math.log(2.0)) / byte_count if byte_count else 0.0
                print(f"flat [{pct:5.1f}%] {done}/{len(windows)} running_bpb={running_bpb:.6f}", flush=True)
    elapsed = time.perf_counter() - tic
    print(f"flat_time_sec={elapsed:.3f}")
    val_loss = loss_sum / token_count
    val_bpb = (loss_sum / math.log(2.0)) / byte_count
    return val_loss, val_bpb


def score_flat_cache(
    flat_scores: dict[str, np.ndarray],
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    mode: str,
    alpha: float,
    tau: float,
    entropy_power: float,
    reset_on_bos: bool,
) -> tuple[float, float]:
    base_bytes_np = base_bytes_lut.cpu().numpy().astype(np.int64)
    has_space_np = has_leading_space_lut.cpu().numpy().astype(np.bool_)
    is_boundary_np = is_boundary_token_lut.cpu().numpy().astype(np.bool_)
    vocab_size = base_bytes_np.shape[0]

    prev = flat_scores["prev"].astype(np.int64, copy=False)
    tgt = flat_scores["tgt"].astype(np.int64, copy=False)
    base_logp = flat_scores["base_logp"].astype(np.float64, copy=False)
    entropy = flat_scores["entropy"].astype(np.float64, copy=False)
    base_prob = np.exp(base_logp)

    row_totals = np.zeros(vocab_size, dtype=np.int32)
    counts = np.zeros((vocab_size, vocab_size), dtype=np.uint16)
    touched_pairs: list[tuple[int, int]] = []
    touched_rows: list[int] = []

    total_loss = 0.0
    max_entropy = math.log(vocab_size)
    tic = time.perf_counter()
    for i in range(len(tgt)):
        p = int(prev[i])
        y = int(tgt[i])
        if reset_on_bos and p == BOS_ID:
            for rp, ry in touched_pairs:
                counts[rp, ry] = 0
            for rp in touched_rows:
                row_totals[rp] = 0
            touched_pairs.clear()
            touched_rows.clear()

        total = int(row_totals[p])
        seen = int(counts[p, y])
        cache_p = (seen / total) if total > 0 else 0.0

        eff_alpha = alpha
        if mode in {"flat_cache_bigram_adaptive", "flat_cache_bigram_adaptive_entropy"}:
            eff_alpha = alpha * (total / (total + tau)) if total > 0 else 0.0
        if mode == "flat_cache_bigram_adaptive_entropy":
            ent_scale = max(float(entropy[i]) / max_entropy, 0.0)
            eff_alpha *= ent_scale ** entropy_power

        mix_p = (1.0 - eff_alpha) * float(base_prob[i]) + eff_alpha * cache_p
        total_loss += -math.log(max(mix_p, 1e-30))

        if total == 0:
            touched_rows.append(p)
        if seen == 0:
            touched_pairs.append((p, y))
        counts[p, y] = min(seen + 1, np.iinfo(np.uint16).max)
        row_totals[p] = total + 1

        if i % 200000 == 0:
            token_bytes_so_far = base_bytes_np[tgt[: i + 1]]
            token_bytes_so_far = token_bytes_so_far + np.logical_and(
                has_space_np[tgt[: i + 1]], np.logical_not(is_boundary_np[prev[: i + 1]])
            ).astype(np.int64)
            running_bpb = (total_loss / math.log(2.0)) / float(token_bytes_so_far.sum())
            print(f"flat_cache tok={i}/{len(tgt)} running_bpb={running_bpb:.6f}", flush=True)

    elapsed = time.perf_counter() - tic
    print(f"flat_cache_time_sec={elapsed:.3f}")
    tb = base_bytes_np[tgt]
    tb = tb + np.logical_and(has_space_np[tgt], np.logical_not(is_boundary_np[prev])).astype(np.int64)
    total_bytes = float(tb.sum())
    total_tokens = int(len(tgt))
    val_loss = total_loss / total_tokens
    val_bpb = (total_loss / math.log(2.0)) / total_bytes
    return val_loss, val_bpb


def score_doc_cache(
    doc_scores: list[dict[str, np.ndarray]],
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    mode: str,
    alpha: float,
    tau: float,
    trigram_alpha: float,
    entropy_power: float,
) -> tuple[float, float]:
    base_bytes_np = base_bytes_lut.cpu().numpy().astype(np.int64)
    has_space_np = has_leading_space_lut.cpu().numpy().astype(np.bool_)
    is_boundary_np = is_boundary_token_lut.cpu().numpy().astype(np.bool_)
    vocab_size = base_bytes_np.shape[0]

    total_loss = 0.0
    total_bytes = 0.0
    total_tokens = 0

    tic = time.perf_counter()
    for doc_idx, doc in enumerate(doc_scores):
        prev = doc["prev"].astype(np.int64, copy=False)
        tgt = doc["tgt"].astype(np.int64, copy=False)
        base_logp = doc["base_logp"].astype(np.float64, copy=False)
        base_prob = np.exp(base_logp)
        entropy = doc["entropy"].astype(np.float64, copy=False)

        if mode == "doc_sliding":
            total_loss += float((-base_logp).sum())
        elif mode == "doc_cache_unigram":
            unigram_total = 0
            unigram = {}
            losses = []
            for i in range(len(tgt)):
                y = int(tgt[i])
                cache_p = unigram.get(y, 0) / unigram_total if unigram_total > 0 else 0.0
                mix_p = (1.0 - alpha) * float(base_prob[i]) + alpha * cache_p
                losses.append(-math.log(max(mix_p, 1e-30)))
                unigram[y] = unigram.get(y, 0) + 1
                unigram_total += 1
            total_loss += float(sum(losses))
        elif mode in {"doc_cache_bigram", "doc_cache_bigram_adaptive", "doc_cache_bigram_adaptive_entropy"}:
            row_totals = np.zeros(vocab_size, dtype=np.int32)
            counts = np.zeros((vocab_size, vocab_size), dtype=np.uint16)
            touched_pairs: list[tuple[int, int]] = []
            touched_rows: list[int] = []
            doc_loss = 0.0
            max_entropy = math.log(vocab_size)
            for i in range(len(tgt)):
                p = int(prev[i])
                y = int(tgt[i])
                total = int(row_totals[p])
                seen = int(counts[p, y])
                cache_p = (seen / total) if total > 0 else 0.0
                eff_alpha = alpha
                if mode in {"doc_cache_bigram_adaptive", "doc_cache_bigram_adaptive_entropy"}:
                    eff_alpha = alpha * (total / (total + tau)) if total > 0 else 0.0
                if mode == "doc_cache_bigram_adaptive_entropy":
                    ent_scale = max(float(entropy[i]) / max_entropy, 0.0)
                    eff_alpha *= ent_scale ** entropy_power
                mix_p = (1.0 - eff_alpha) * float(base_prob[i]) + eff_alpha * cache_p
                doc_loss += -math.log(max(mix_p, 1e-30))
                if total == 0:
                    touched_rows.append(p)
                if seen == 0:
                    touched_pairs.append((p, y))
                counts[p, y] = min(seen + 1, np.iinfo(np.uint16).max)
                row_totals[p] = total + 1
            for p, y in touched_pairs:
                counts[p, y] = 0
            for p in touched_rows:
                row_totals[p] = 0
            total_loss += doc_loss
        elif mode == "doc_cache_trigram_backoff":
            unigram_total = 0
            unigram: dict[int, int] = {}
            bigram_total: dict[int, int] = {}
            bigram: dict[tuple[int, int], int] = {}
            trigram_total: dict[tuple[int, int], int] = {}
            trigram: dict[tuple[int, int, int], int] = {}
            doc_loss = 0.0
            prev2 = BOS_ID
            for i in range(len(tgt)):
                p1 = int(prev[i])
                y = int(tgt[i])
                uni_p = unigram.get(y, 0) / unigram_total if unigram_total > 0 else 0.0
                bi_total = bigram_total.get(p1, 0)
                bi_p = bigram.get((p1, y), 0) / bi_total if bi_total > 0 else 0.0
                tri_total = trigram_total.get((prev2, p1), 0)
                tri_p = trigram.get((prev2, p1, y), 0) / tri_total if tri_total > 0 else 0.0
                backoff_p = 0.5 * bi_p + 0.5 * uni_p
                cache_p = trigram_alpha * tri_p + (1.0 - trigram_alpha) * backoff_p
                mix_p = (1.0 - alpha) * float(base_prob[i]) + alpha * cache_p
                doc_loss += -math.log(max(mix_p, 1e-30))
                unigram[y] = unigram.get(y, 0) + 1
                unigram_total += 1
                bigram[(p1, y)] = bigram.get((p1, y), 0) + 1
                bigram_total[p1] = bi_total + 1
                trigram[(prev2, p1, y)] = trigram.get((prev2, p1, y), 0) + 1
                trigram_total[(prev2, p1)] = tri_total + 1
                prev2 = p1
            total_loss += doc_loss
        else:
            raise ValueError(f"Unsupported mode: {mode}")

        tb = base_bytes_np[tgt]
        tb = tb + np.logical_and(has_space_np[tgt], np.logical_not(is_boundary_np[prev])).astype(np.int64)
        total_bytes += float(tb.sum())
        total_tokens += int(len(tgt))

        if doc_idx % 2000 == 0:
            running_bpb = (total_loss / math.log(2.0)) / total_bytes if total_bytes else 0.0
            print(f"cache doc={doc_idx}/{len(doc_scores)} running_bpb={running_bpb:.6f}", flush=True)

    elapsed = time.perf_counter() - tic
    print(f"cache_time_sec={elapsed:.3f}")
    val_loss = total_loss / total_tokens
    val_bpb = (total_loss / math.log(2.0)) / total_bytes
    return val_loss, val_bpb


def main() -> None:
    args = parse_args()
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)

    train_mod = load_train_module(args.train_script)
    val_tokens = load_validation_subset(train_mod, args.val_files, args.seq_len, args.max_tokens, args.start_token)
    docs = find_docs(val_tokens)
    print(f"val_tokens={val_tokens.numel()} docs={len(docs)} seq_len={args.seq_len} stride={args.stride}")

    model = instantiate_model(train_mod, device, args.seq_len)
    state = load_checkpoint_state(args, train_mod)
    model.load_state_dict(state, strict=True)
    model.eval()

    run_ttt(
        model,
        val_tokens,
        args.seq_len,
        args.ttt_epochs,
        args.ttt_lr,
        args.ttt_batch_seqs,
        device,
    )

    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_metric_luts(
        train_mod, args.tokenizer, model.tok_emb.num_embeddings, device
    )

    if args.mode == "flat_sliding":
        val_loss, val_bpb = eval_flat_sliding(
            model,
            val_tokens,
            args.seq_len,
            args.stride,
            args.batch_seqs,
            device,
            args.compile,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
        )
        print(f"mode={args.mode} val_loss={val_loss:.8f} val_bpb={val_bpb:.8f}")
        return

    if args.mode in {"flat_cache_bigram_adaptive", "flat_cache_bigram_adaptive_entropy"}:
        flat_scores = gather_flat_scores(
            model,
            val_tokens,
            args.seq_len,
            args.stride,
            args.batch_seqs,
            device,
            args.compile,
        )
        alphas = args.alphas if args.alphas else [args.alpha]
        for alpha in alphas:
            val_loss, val_bpb = score_flat_cache(
                flat_scores,
                base_bytes_lut,
                has_leading_space_lut,
                is_boundary_token_lut,
                args.mode,
                alpha,
                args.tau,
                args.entropy_power,
                reset_on_bos=args.reset_cache_on_bos,
            )
            print(
                f"mode={args.mode} alpha={alpha:.6f} tau={args.tau:.4f} trigram_alpha={args.trigram_alpha:.4f} entropy_power={args.entropy_power:.4f} reset_cache_on_bos={args.reset_cache_on_bos} "
                f"val_loss={val_loss:.8f} val_bpb={val_bpb:.8f}"
            )
        return

    doc_scores = gather_doc_scores(
        model,
        val_tokens,
        docs,
        args.seq_len,
        args.stride,
        args.batch_seqs,
        device,
        args.compile,
    )

    alphas = args.alphas if args.alphas else [args.alpha]
    for alpha in alphas:
        val_loss, val_bpb = score_doc_cache(
            doc_scores,
            base_bytes_lut,
            has_leading_space_lut,
            is_boundary_token_lut,
            args.mode,
            alpha,
            args.tau,
            args.trigram_alpha,
            args.entropy_power,
        )
        print(
            f"mode={args.mode} alpha={alpha:.6f} tau={args.tau:.4f} trigram_alpha={args.trigram_alpha:.4f} entropy_power={args.entropy_power:.4f} "
            f"val_loss={val_loss:.8f} val_bpb={val_bpb:.8f}"
        )


if __name__ == "__main__":
    main()
