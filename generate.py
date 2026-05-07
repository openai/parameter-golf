"""
Generate text from a trained parameter-golf model checkpoint.

Usage:
  python generate.py --checkpoint final_model.pt --prompt "The" --max_tokens 200

Requires the train_gpt.py (or train_gpt_submission.py) in the same directory
for model class definitions.
"""
import argparse
import sys
import os

# Mock flash_attn before importing train script
import types
mock_fa = types.ModuleType("flash_attn")
def _mock_flash(q, k, v, causal=False):
    import torch
    import torch.nn.functional as F
    B, T, H, D = q.shape
    Hkv = k.shape[2]
    group = H // Hkv
    if group > 1:
        k = k.unsqueeze(3).expand(B, T, Hkv, group, D).reshape(B, T, H, D)
        v = v.unsqueeze(3).expand(B, T, Hkv, group, D).reshape(B, T, H, D)
    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    v = v.transpose(1, 2)
    scale = 1.0 / (D ** 0.5)
    attn = torch.matmul(q * scale, k.transpose(-2, -1))
    mask = torch.triu(torch.ones(T, T, device=q.device, dtype=torch.bool), diagonal=1)
    attn = attn.masked_fill(mask, float("-inf"))
    attn = torch.softmax(attn.float(), dim=-1).to(q.dtype)
    out = torch.matmul(attn, v)
    return out.transpose(1, 2)

mock_fa.flash_attn_func = _mock_flash
sys.modules["flash_attn"] = mock_fa
sys.modules["flash_attn_interface"] = mock_fa

import torch
import sentencepiece as spm


def load_model_and_tokenizer(checkpoint_path, script_path="train_gpt_submission.py",
                              tokenizer_path=None):
    sys.path.insert(0, os.path.dirname(os.path.abspath(script_path)))
    spec = __import__(os.path.splitext(os.path.basename(script_path))[0])

    args = spec.Hyperparameters
    if tokenizer_path is None:
        tokenizer_path = args.tokenizer_path

    model = spec.GPT(
        vocab_size=args.vocab_size, num_layers=args.num_layers, model_dim=args.model_dim,
        num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, mlp_mult=args.mlp_mult,
        tie_embeddings=args.tie_embeddings, tied_embed_init_std=args.tied_embed_init_std,
        logit_softcap=args.logit_softcap, rope_base=args.rope_base, qk_gain_init=args.qk_gain_init,
        bigram_vocab_size=args.bigram_vocab_size, bigram_dim=args.bigram_dim,
        value_residual=args.value_residual, gated_attention=args.gated_attention,
    )

    state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state_dict, strict=True)
    model.eval()

    sp = spm.SentencePieceProcessor(model_file=tokenizer_path)

    return model, sp, args


@torch.no_grad()
def generate(model, sp, prompt, max_tokens=200, temperature=0.8, top_k=50, device="cpu"):
    model = model.to(device).float()

    token_ids = sp.encode(prompt)
    tokens = torch.tensor([token_ids], dtype=torch.long, device=device)

    print(f"\n--- Prompt: \"{prompt}\" ---\n")
    print(prompt, end="", flush=True)

    for _ in range(max_tokens):
        x = tokens[:, -2048:]  # context window
        logits = model.forward_logits(x)
        logits = logits[:, -1, :] / temperature

        if top_k > 0:
            v, _ = torch.topk(logits, top_k)
            logits[logits < v[:, [-1]]] = float("-inf")

        probs = torch.softmax(logits.float(), dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        tokens = torch.cat([tokens, next_token], dim=1)

        decoded = sp.decode([next_token.item()])
        print(decoded, end="", flush=True)

    print("\n\n--- Done ---")


def main():
    parser = argparse.ArgumentParser(description="Generate text from parameter-golf model")
    parser.add_argument("--checkpoint", required=True, help="Path to final_model.pt")
    parser.add_argument("--script", default="train_gpt_submission.py", help="Training script for model defs")
    parser.add_argument("--tokenizer", default=None, help="Path to tokenizer .model file")
    parser.add_argument("--prompt", default="The", help="Text prompt")
    parser.add_argument("--max_tokens", type=int, default=200)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_k", type=int, default=50)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    model, sp, hparams = load_model_and_tokenizer(args.checkpoint, args.script, args.tokenizer)
    generate(model, sp, args.prompt, args.max_tokens, args.temperature, args.top_k, args.device)


if __name__ == "__main__":
    main()
