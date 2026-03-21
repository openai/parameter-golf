from __future__ import annotations

import argparse
import importlib
import json
import os
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import train_gpt as tg
from core.artifact_core import deserialize_quant_artifact, serialize_quant_artifact


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a quantized candidate from a dense state_dict.")
    parser.add_argument("--state-dict-path", type=Path, default=Path("final_model.pt"))
    parser.add_argument("--keep-large-patterns", type=str, default=None)
    parser.add_argument("--no-default-large-keeps", action="store_true")
    parser.add_argument("--quant-artifact-format", type=str, default="packed_zlib")
    parser.add_argument("--packed-scale-codec", type=str, default="raw")
    parser.add_argument("--train-seq-len", type=int, default=256)
    parser.add_argument("--val-batch-size", type=int, default=4096)
    parser.add_argument("--vocab-size", type=int, default=1024)
    parser.add_argument("--num-layers", type=int, default=9)
    parser.add_argument("--model-dim", type=int, default=512)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-kv-heads", type=int, default=4)
    parser.add_argument("--mlp-mult", type=int, default=2)
    parser.add_argument("--tie-embeddings", type=int, default=1)
    parser.add_argument("--tokenizer-path", type=str, default="./data/tokenizers/fineweb_1024_bpe.model")
    parser.add_argument("--data-path", type=str, default="./data/datasets/fineweb10B_sp1024")
    return parser.parse_args()


def build_args(cli: argparse.Namespace) -> tg.Hyperparameters:
    args = tg.Hyperparameters()
    args.data_path = cli.data_path
    args.train_files = os.path.join(args.data_path, "fineweb_train_*.bin")
    args.val_files = os.path.join(args.data_path, "fineweb_val_*.bin")
    args.tokenizer_path = cli.tokenizer_path
    args.train_seq_len = cli.train_seq_len
    args.val_batch_size = cli.val_batch_size
    args.vocab_size = cli.vocab_size
    args.num_layers = cli.num_layers
    args.model_dim = cli.model_dim
    args.num_heads = cli.num_heads
    args.num_kv_heads = cli.num_kv_heads
    args.mlp_mult = cli.mlp_mult
    args.tie_embeddings = bool(cli.tie_embeddings)
    args.eval_mode = "flat"
    args.eval_seq_len = cli.train_seq_len
    args.eval_stride = cli.train_seq_len
    args.eval_batch_seqs = 32
    args.quant_artifact_format = cli.quant_artifact_format
    args.packed_scale_codec = cli.packed_scale_codec
    return args


def main() -> None:
    cli = parse_args()
    if cli.no_default_large_keeps:
        os.environ["INT8_KEEP_LARGE_FLOAT_NAME_PATTERNS"] = ""
    elif cli.keep_large_patterns is not None:
        os.environ["INT8_KEEP_LARGE_FLOAT_NAME_PATTERNS"] = cli.keep_large_patterns

    import core.quant_core as quant_core

    quant_core = importlib.reload(quant_core)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    tg.enable_cudnn_sdp(False)
    tg.enable_flash_sdp(True)
    tg.enable_mem_efficient_sdp(False)
    tg.enable_math_sdp(False)

    args = build_args(cli)
    sp = tg.spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    val_tokens = tg.load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = tg.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    model = tg.GPT(
        vocab_size=args.vocab_size,
        num_layers=args.num_layers,
        model_dim=args.model_dim,
        num_heads=args.num_heads,
        num_kv_heads=args.num_kv_heads,
        mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings,
        tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap,
        rope_base=args.rope_base,
        qk_gain_init=args.qk_gain_init,
    ).to(device)

    dense_state = torch.load(cli.state_dict_path, map_location="cpu")
    quant_obj, quant_stats = quant_core.quantize_state_dict_int8(dense_state)
    artifact_blob, raw_len = serialize_quant_artifact(
        quant_obj,
        cli.quant_artifact_format,
        compression_level=9,
        scale_codec=cli.packed_scale_codec,
    )
    restored = deserialize_quant_artifact(artifact_blob, cli.quant_artifact_format)
    model.load_state_dict(quant_core.dequantize_state_dict_int8(restored), strict=True)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    result = tg.eval_val(
        args,
        model,
        rank=0,
        world_size=1,
        device=device,
        grad_accum_steps=8,
        val_tokens=val_tokens,
        base_bytes_lut=base_bytes_lut,
        has_leading_space_lut=has_leading_space_lut,
        is_boundary_token_lut=is_boundary_token_lut,
    )
    torch.cuda.synchronize()

    print(
        json.dumps(
            {
                "keep_large_patterns": cli.keep_large_patterns,
                "quant_artifact_format": cli.quant_artifact_format,
                "packed_scale_codec": cli.packed_scale_codec,
                "artifact_raw_serialized_bytes": raw_len,
                "artifact_compressed_bytes": len(artifact_blob),
                "int8_payload_bytes": quant_stats["int8_payload_bytes"],
                "large_float_passthrough_bytes": quant_stats["large_float_passthrough_bytes"],
                "num_large_float_passthrough_tensors": quant_stats["num_large_float_passthrough_tensors"],
                "val_loss": result.val_loss,
                "val_bpb": result.val_bpb,
                "eval_time_ms": 1000.0 * (time.perf_counter() - t0),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
