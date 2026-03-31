from __future__ import annotations

import io
import time
import zlib
import zstandard
import zstandard

import sentencepiece as spm
import torch

import train_merged_gpt_gold6 as g


def build_model(args: g.Hyperparameters, device: torch.device) -> g.GPT:
    model = g.GPT(
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
        mtp_num_heads=0,
        mtp_loss_weight=0.0,
        bigram_vocab_size=args.bigram_vocab_size,
        bigram_dim=args.bigram_dim,
        xsa_last_n=args.xsa_last_n,
        rope_dims=args.rope_dims,
        ln_scale=args.ln_scale,
        dtg=args.dtg_enabled,
        ve_enabled=args.ve_enabled,
        ve_dim=args.ve_dim,
        ve_layers=args.ve_layers,
    ).to(device).bfloat16()
    for module in model.modules():
        if isinstance(module, g.CastedLinear):
            module.float()
    g.restore_low_dim_params_to_fp32(model)
    model.eval()
    return model


def timed_eval(
    args: g.Hyperparameters,
    model: torch.nn.Module,
    device: torch.device,
    val_tokens: torch.Tensor,
    base_bytes_lut: torch.Tensor,
    has_leading_space_lut: torch.Tensor,
    is_boundary_token_lut: torch.Tensor,
    eval_seq_len: int,
) -> tuple[float, float, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    val_loss, val_bpb = g.eval_val(
        args,
        model,
        0,
        1,
        device,
        1,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        eval_seq_len=eval_seq_len,
    )
    torch.cuda.synchronize()
    eval_ms = 1000.0 * (time.perf_counter() - start)
    return val_loss, val_bpb, eval_ms


def main() -> None:
    args = g.Hyperparameters()
    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    effective_eval_seq_len = args.eval_seq_len if args.eval_seq_len > 0 else args.train_seq_len
    val_seq_len = max(args.train_seq_len, effective_eval_seq_len)
    val_tokens = g.load_validation_tokens(args.val_files, val_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = g.build_sentencepiece_luts(
        sp, args.vocab_size, device
    )

    bf16_model = build_model(args, device)
    bf16_state = torch.load("final_model_gold6.pt", map_location="cpu")
    bf16_model.load_state_dict(bf16_state, strict=True)
    bf16_loss, bf16_bpb, bf16_ms = timed_eval(
        args,
        bf16_model,
        device,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        effective_eval_seq_len,
    )

    with open("final_model_gold6.int6.ptz", "rb") as f:
        quant_blob = f.read()
    quant_payload = zstandard.ZstdDecompressor().decompress(quant_blob)
    quant_state = torch.load(io.BytesIO(quant_payload), map_location="cpu")
    deq_state = g.dequantize_mixed_int6(quant_state["w"], quant_state["m"], bf16_state)

    int6_model = build_model(args, device)
    int6_model.load_state_dict(deq_state, strict=True)
    int6_loss, int6_bpb, int6_ms = timed_eval(
        args,
        int6_model,
        device,
        val_tokens,
        base_bytes_lut,
        has_leading_space_lut,
        is_boundary_token_lut,
        effective_eval_seq_len,
    )

    print(f"timed_eval_bf16 val_loss:{bf16_loss:.8f} val_bpb:{bf16_bpb:.8f} eval_time:{bf16_ms:.0f}ms")
    print(f"timed_eval_int6 val_loss:{int6_loss:.8f} val_bpb:{int6_bpb:.8f} eval_time:{int6_ms:.0f}ms")


if __name__ == "__main__":
    main()
