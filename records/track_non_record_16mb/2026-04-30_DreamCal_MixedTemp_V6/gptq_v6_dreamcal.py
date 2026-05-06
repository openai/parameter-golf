"""
gptq_v6_dreamcal.py — V6 GPTQ with mixed-temperature dream-state calibration.

Diffs from gptq_v6.py:
  1. AR self-gen uses temperature sampling (multinomial), not argmax.
  2. --mixed-temp splits sequences across two temperatures (think + dream).
     Default: half at temp 0.5 (think — focused), half at temp 1.5 (dream — diffuse).
  3. --bos-seed seeds generation from BOS-only (id=1) instead of training data.
  4. --calib-temp single-temperature mode (default 0.8 = leader's recipe baseline).

Usage:
  # leader baseline (single temp 0.8, sampling, BOS seed)
  python gptq_v6_dreamcal.py --self-gen --calib-temp 0.8 --bos-seed --calib-seqs 64 --seq-len 2048 --attn6
  # our hypothesis (mixed temp, BOS seed, dream+think split)
  python gptq_v6_dreamcal.py --self-gen --mixed-temp --bos-seed --calib-seqs 64 --seq-len 2048 --attn6

The model + GPTQ + compression code is reused unchanged from gptq_v6.py.
"""
import os, sys, time, argparse, glob
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, '.')
import gptq_v6
from gptq_v6 import GPT, build_rope, gptq_quantize_model, dim, ROPE_DIMS
from gptq import compress_artifact, dequantize_gptq_model


def collect_activations_local(model, tokens, n_seqs, seq_len, device):
    # Reuse the project's collect_activations from gptq_v6 module
    from gptq_v6 import collect_activations
    return collect_activations(model, tokens, n_seqs=n_seqs, seq_len=seq_len, device=device)


def sample_next(logits, temperature):
    """Multinomial sample with temperature. logits: (B, vocab)."""
    if temperature <= 0:
        return logits.argmax(dim=-1, keepdim=True)
    probs = torch.softmax(logits / temperature, dim=-1)
    return torch.multinomial(probs, num_samples=1)


def gen_calibration(model, n_seqs, seq_len, device, mixed_temp, calib_temp,
                    temp_low, temp_high, bos_seed, seed_tokens, gen_seed=42):
    torch.manual_seed(gen_seed)
    gen_seqs = []
    BOS_ID = 1  # sp4096 standard BOS

    with torch.no_grad():
        for i in range(n_seqs):
            # Choose temperature for this sequence
            if mixed_temp:
                # First half: low (think), second half: high (dream)
                temp = temp_low if i < n_seqs // 2 else temp_high
            else:
                temp = calib_temp

            # Seed
            if bos_seed:
                tokens = torch.tensor([[BOS_ID]], dtype=torch.long, device=device)
                seed_len = 1
            else:
                seed_len = 32
                seed = seed_tokens[i * seed_len:(i + 1) * seed_len].long().unsqueeze(0).to(device)
                tokens = seed

            # Autoregressive generation with temperature sampling
            for _ in range(seq_len - seed_len):
                logits = model(tokens[:, -seq_len:])
                next_tok = sample_next(logits[:, -1, :], temp)
                tokens = torch.cat([tokens, next_tok], dim=1)

            gen_seqs.append(tokens.squeeze(0).cpu())
            if (i + 1) % 16 == 0:
                kind = ('mix' if mixed_temp else f't{temp:.2f}')
                print(f'  Generated {i+1}/{n_seqs} seqs [{kind}]', flush=True)

    return torch.cat(gen_seqs)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='best_model_v6_ema.pt')
    parser.add_argument('--bits', type=int, default=4, choices=[4, 6])
    parser.add_argument('--mixed', action='store_true')
    parser.add_argument('--attn6', action='store_true')
    parser.add_argument('--emb6', action='store_true')
    parser.add_argument('--calib-seqs', type=int, default=64)
    parser.add_argument('--seq-len', type=int, default=2048)
    parser.add_argument('--block-size', type=int, default=128)
    parser.add_argument('--damp', type=float, default=0.005)
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--self-gen', action='store_true')
    # NEW: dream-state calibration knobs
    parser.add_argument('--calib-temp', type=float, default=0.8,
                        help='Single-temperature for self-gen sampling (default 0.8 = leader baseline)')
    parser.add_argument('--mixed-temp', action='store_true',
                        help='Split sequences across temp-low (think) and temp-high (dream)')
    parser.add_argument('--temp-low', type=float, default=0.5)
    parser.add_argument('--temp-high', type=float, default=1.5)
    parser.add_argument('--bos-seed', action='store_true',
                        help='Seed from BOS only (avoids training-data legality issue)')
    parser.add_argument('--gen-seed', type=int, default=42)
    parser.add_argument('--suffix-tag', type=str, default='dreamcal',
                        help='Tag added to artifact filename')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    seq_len = args.seq_len

    # Build RoPE module-level globals
    gptq_v6.rope_cos, gptq_v6.rope_sin = build_rope(seq_len, dim // 8, ROPE_DIMS, device=device)

    # Load model
    print(f'Loading {args.model}...', flush=True)
    model = GPT(11, 3).to(device)
    state = torch.load(args.model, map_location=device, weights_only=False)
    model.load_state_dict(state, strict=False)
    model.eval()
    print(f'Model: {sum(p.numel() for p in model.parameters()):,} params on {device}', flush=True)

    # Calibration
    if args.self_gen:
        if args.mixed_temp:
            print(f'\nMIXED-TEMP self-gen calib: {args.calib_seqs//2} seqs at temp={args.temp_low} '
                  f'(think) + {args.calib_seqs//2} at temp={args.temp_high} (dream), seq_len={seq_len}', flush=True)
        else:
            print(f'\nSELF-GEN calib (sampled): {args.calib_seqs} seqs at temp={args.calib_temp}, '
                  f'seq_len={seq_len}', flush=True)

        # Optional training-data seed (only if --bos-seed not set)
        seed_tokens = None
        if not args.bos_seed:
            train_files = sorted(glob.glob('data/datasets/fineweb10B_sp4096/fineweb_train_*.bin'))
            if train_files:
                seed_tokens = torch.from_numpy(
                    np.fromfile(Path(train_files[0]), dtype='<u2', offset=256*4).astype(np.uint16)
                )

        t0 = time.time()
        calib_tokens = gen_calibration(
            model, args.calib_seqs, seq_len, device,
            mixed_temp=args.mixed_temp,
            calib_temp=args.calib_temp,
            temp_low=args.temp_low,
            temp_high=args.temp_high,
            bos_seed=args.bos_seed,
            seed_tokens=seed_tokens,
            gen_seed=args.gen_seed,
        )
        print(f'Calibration: {calib_tokens.numel():,} tokens in {time.time()-t0:.1f}s', flush=True)
    else:
        train_files = sorted(glob.glob('data/datasets/fineweb10B_sp4096/fineweb_train_*.bin'))
        calib_tokens = torch.from_numpy(
            np.fromfile(Path(train_files[0]), dtype='<u2', offset=256*4).astype(np.uint16)
        )
        print(f'Calibration tokens: {calib_tokens.numel():,} from {train_files[0]}', flush=True)

    # Activations
    print(f'\nCollecting activations ({args.calib_seqs} seqs, seq_len={seq_len})...', flush=True)
    t0 = time.time()
    activations = collect_activations_local(model, calib_tokens, args.calib_seqs, seq_len, device)
    print(f'Activations collected in {time.time()-t0:.1f}s\n', flush=True)

    # Quantize
    bits = args.bits
    state_dict = {k: v for k, v in model.state_dict().items()}
    if args.attn6:
        print(f'GPTQ int{bits} + int6 attention q/k/v/o (Hessian)', flush=True)
    elif args.mixed:
        print(f'Mixed GPTQ: int4 MLP + int6 attention+proj (Hessian)', flush=True)
    elif args.emb6:
        print(f'GPTQ int{bits} + int6 embedding (Hessian)', flush=True)
    else:
        print(f'Uniform GPTQ int{bits} (Hessian)', flush=True)

    t0 = time.time()
    quant = gptq_quantize_model(state_dict, activations, bits=bits, mixed=args.mixed,
                                 emb6=args.emb6, attn6=args.attn6,
                                 block_size=args.block_size, damp=args.damp)
    print(f'\nQuantized in {time.time()-t0:.1f}s', flush=True)

    # Compress
    compressed = compress_artifact(quant)
    code_est = 50000
    ngram_est = 800000
    total = len(compressed) + code_est + ngram_est
    print(f'\nLZMA: {len(compressed)/1e6:.3f} MB', flush=True)
    print(f'Total (model + code + ngram): {total/1e6:.3f} MB', flush=True)
    print(f'{"OK" if total < 16e6 else "OVER"} (headroom: {(16e6-total)/1024:.0f} KB)', flush=True)

    # Save
    base_suffix = 'mixed' if args.mixed else ('attn6' if args.attn6 else (f'{bits}bit_emb6' if args.emb6 else f'{bits}bit'))
    suffix = f'{base_suffix}_{args.suffix_tag}'
    artifact_path = args.model.replace('.pt', f'.gptq_{suffix}_hessian.lzma')
    with open(artifact_path, 'wb') as f:
        f.write(compressed)
    print(f'Artifact: {artifact_path}', flush=True)

    rt_state = dequantize_gptq_model(quant)
    rt_path = args.model.replace('.pt', f'_gptq_{suffix}_hessian_roundtrip.pt')
    torch.save(rt_state, rt_path)
    print(f'Roundtrip model: {rt_path}', flush=True)
    print(f'\nNext: python sliding_window_eval_v6.py {rt_path} --gpu 0', flush=True)
