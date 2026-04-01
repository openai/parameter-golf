#!/usr/bin/env python3
from __future__ import annotations

import argparse
import copy
import random
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import final.dominationv2_cache.train_gpt as ft
import research.eval_doc_cache as re


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Verify DominationV2 cache evaluator parity and sharding.")
    parser.add_argument("--device", type=str, default="cuda")
    return parser.parse_args()


def make_model(device: torch.device, seed: int, vocab_size: int) -> torch.nn.Module:
    torch.manual_seed(seed)
    model = ft.GPT(
        vocab_size=vocab_size,
        num_layers=4,
        model_dim=64,
        num_heads=4,
        num_kv_heads=2,
        mlp_mult=2,
        tie_embeddings=True,
        tied_embed_init_std=0.02,
        logit_softcap=30.0,
        rope_base=10000.0,
        qk_gain_init=1.5,
        bigram_vocab_size=64,
        bigram_dim=16,
        xsa_last_n=1,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, ft.CastedLinear):
            module.float()
    ft.restore_low_dim_params_to_fp32(model)
    model.eval()
    return model


def make_tokens(seed: int, vocab_size: int, num_docs: int) -> torch.Tensor:
    rng = random.Random(seed)
    toks = []
    for _ in range(num_docs):
        doc_len = rng.randint(5, 31)
        toks.append(ft.BOS_ID)
        for _ in range(doc_len - 1):
            tok = rng.randrange(vocab_size)
            while tok == ft.BOS_ID:
                tok = rng.randrange(vocab_size)
            toks.append(tok)
    toks.append(ft.BOS_ID)
    return torch.tensor(toks, dtype=torch.int64)


def make_metric_luts(vocab_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    return (
        torch.ones(vocab_size, dtype=torch.int16, device=device),
        torch.zeros(vocab_size, dtype=torch.bool, device=device),
        torch.zeros(vocab_size, dtype=torch.bool, device=device),
    )


def verify_single_rank(device: torch.device) -> None:
    for seed in range(3):
        vocab_size = 32
        model = make_model(device, seed, vocab_size)
        val_tokens = make_tokens(seed, vocab_size, num_docs=12)
        docs = ft.find_docs(val_tokens)
        args = SimpleNamespace(train_seq_len=16, cache_enabled=True, cache_alpha=0.2, cache_tau=8.0, cache_entropy_power=1.0)
        base_bytes_lut, has_space_lut, boundary_lut = make_metric_luts(vocab_size, device)

        cache_loss, cache_bpb = ft.eval_val_sliding_cache(
            args, model, 0, 1, device, val_tokens, docs, base_bytes_lut, has_space_lut, boundary_lut, stride=4, batch_seqs=8
        )
        flat_scores = re.gather_flat_scores(
            model, val_tokens, seq_len=16, stride=4, batch_seqs=8, device=device, compile_model=False
        )
        ref_cache_loss, ref_cache_bpb = re.score_flat_cache(
            flat_scores,
            base_bytes_lut,
            has_space_lut,
            boundary_lut,
            mode="flat_cache_bigram_adaptive_entropy",
            alpha=0.2,
            tau=8.0,
            entropy_power=1.0,
            reset_on_bos=True,
        )
        if abs(cache_loss - ref_cache_loss) >= 1e-7 or abs(cache_bpb - ref_cache_bpb) >= 1e-7:
            raise AssertionError(
                f"cache parity failed for seed={seed}: "
                f"{cache_loss=} {ref_cache_loss=} {cache_bpb=} {ref_cache_bpb=}"
            )

        base_loss, base_bpb = ft.eval_val_sliding(
            args, model, 0, 1, device, val_tokens, base_bytes_lut, has_space_lut, boundary_lut, stride=4, batch_seqs=8
        )
        ref_base_loss, ref_base_bpb = re.eval_flat_sliding(
            model,
            val_tokens,
            seq_len=16,
            stride=4,
            batch_seqs=8,
            device=device,
            compile_model=False,
            base_bytes_lut=base_bytes_lut,
            has_leading_space_lut=has_space_lut,
            is_boundary_token_lut=boundary_lut,
        )
        if abs(base_loss - ref_base_loss) >= 1e-7 or abs(base_bpb - ref_base_bpb) >= 1e-7:
            raise AssertionError(
                f"base parity failed for seed={seed}: "
                f"{base_loss=} {ref_base_loss=} {base_bpb=} {ref_base_bpb=}"
            )
    print("single-rank parity: OK")


def verify_multi_rank(device: torch.device) -> None:
    vocab_size = 40
    model = make_model(device, 0, vocab_size)
    val_tokens = make_tokens(7, vocab_size, num_docs=17)
    docs = ft.find_docs(val_tokens)
    args = SimpleNamespace(train_seq_len=16, cache_enabled=True, cache_alpha=0.2, cache_tau=8.0, cache_entropy_power=1.0)
    base_bytes_lut, has_space_lut, boundary_lut = make_metric_luts(vocab_size, device)
    single_loss, single_bpb = ft.eval_val_sliding_cache(
        args, model, 0, 1, device, val_tokens, docs, base_bytes_lut, has_space_lut, boundary_lut, stride=4, batch_seqs=8
    )
    total_tokens = sum(doc_len - 1 for _, doc_len in docs)

    for world_size in (2, 3, 8, 32):
        agg_loss = 0.0
        agg_bpb = 0.0
        seen_tokens = 0
        for rank in range(world_size):
            shard = ft.shard_docs_by_tokens(docs, rank, world_size)
            shard_tokens = sum(doc_len - 1 for _, doc_len in shard)
            if shard_tokens == 0:
                continue
            loss_i, bpb_i = ft.eval_val_sliding_cache(
                args, model, rank, world_size, device, val_tokens, docs, base_bytes_lut, has_space_lut, boundary_lut, stride=4, batch_seqs=8
            )
            agg_loss += loss_i * shard_tokens
            agg_bpb += bpb_i * shard_tokens
            seen_tokens += shard_tokens
        agg_loss /= seen_tokens
        agg_bpb /= seen_tokens
        if seen_tokens != total_tokens or abs(agg_loss - single_loss) >= 1e-7 or abs(agg_bpb - single_bpb) >= 1e-7:
            raise AssertionError(
                f"multi-rank aggregation failed for world_size={world_size}: "
                f"{seen_tokens=} {total_tokens=} {agg_loss=} {single_loss=} {agg_bpb=} {single_bpb=}"
            )
    print("multi-rank aggregation: OK")


def verify_ttt_equivalence(device: torch.device) -> None:
    def run_final_style_ttt(model: torch.nn.Module, val_tokens: torch.Tensor, seq_len: int, epochs: int, lr: float, batch_seqs: int) -> None:
        model.train()
        ttt_params = [p for p in model.parameters() if p.requires_grad]
        ttt_opt = torch.optim.SGD(ttt_params, lr=lr)
        total_val = val_tokens.numel() - 1
        total_seqs = total_val // seq_len
        for _ in range(epochs):
            for si in range(0, total_seqs, batch_seqs):
                se = min(si + batch_seqs, total_seqs)
                bsz = se - si
                rs = si * seq_len
                re_idx = se * seq_len + 1
                local = val_tokens[rs:re_idx].to(device=device, dtype=torch.int64)
                x = local[:-1].reshape(bsz, seq_len)
                y = local[1:].reshape(bsz, seq_len)
                ttt_opt.zero_grad()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    loss = model(x, y)
                loss.backward()
                ttt_opt.step()
        model.eval()

    vocab_size = 32
    base = make_model(device, 0, vocab_size)
    val_tokens = make_tokens(11, vocab_size, num_docs=10)
    model_a = copy.deepcopy(base)
    model_b = copy.deepcopy(base)
    re.run_ttt(model_a, val_tokens, seq_len=16, epochs=3, lr=1e-4, batch_seqs=4, device=device)
    run_final_style_ttt(model_b, val_tokens, seq_len=16, epochs=3, lr=1e-4, batch_seqs=4)
    max_diff = 0.0
    for (_, pa), (_, pb) in zip(model_a.state_dict().items(), model_b.state_dict().items(), strict=True):
        max_diff = max(max_diff, (pa.float() - pb.float()).abs().max().item())
    if max_diff != 0.0:
        raise AssertionError(f"TTT equivalence failed: max_diff={max_diff}")
    print("TTT loop equivalence: OK")


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    verify_single_rank(device)
    verify_multi_rank(device)
    verify_ttt_equivalence(device)
    print("all checks passed")


if __name__ == "__main__":
    main()
