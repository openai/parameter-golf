import os
import torch
import torch.distributed as dist
from train_gpt import GPT, Hyperparameters, eval_val, dequantize_state_dict_int8, build_sentencepiece_luts
import sentencepiece as spm
import zlib
import io
from pathlib import Path

def main():
    args = Hyperparameters()
    
    # Setup distributed
    distributed = "RANK" in os.environ and "WORLD_SIZE" in os.environ
    rank = int(os.environ.get("RANK", "0"))
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    device = torch.device("cuda", local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend="nccl", device_id=device)
    
    # Load model
    if not os.path.exists("final_model.int8.ptz"):
        print("ERROR: final_model.int8.ptz not found. Run training first.")
        return

    model = GPT(
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
        qk_gain_init=args.qk_gain_init
    ).to(device)

    with open("final_model.int8.ptz", "rb") as f:
        quant_blob = f.read()
    quant_state = torch.load(io.BytesIO(zlib.decompress(quant_blob)), map_location="cpu")
    model.load_state_dict(dequantize_state_dict_int8(quant_state), strict=True)
    
    # Load tokenizer
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = build_sentencepiece_luts(sp, args.vocab_size, device)
    
    # Load val data
    from train_gpt import load_validation_tokens
    val_tokens = load_validation_tokens(args.val_files, args.train_seq_len)
    
    # Eval
    grad_accum_steps = 8 // world_size
    val_loss, val_bpb = eval_val(
        args, model, rank, world_size, device, grad_accum_steps, 
        val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut
    )
    
    if rank == 0:
        print(f"Evaluation Results:")
        print(f"Val Loss: {val_loss:.4f}")
        print(f"Val BPB: {val_bpb:.4f}")

    if distributed:
        dist.destroy_process_group()

if __name__ == "__main__":
    main()
