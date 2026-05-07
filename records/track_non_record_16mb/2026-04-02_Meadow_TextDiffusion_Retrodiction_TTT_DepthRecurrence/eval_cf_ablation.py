#!/usr/bin/env python3
"""
CF dual-mode evaluation for the 6-run ablation. Loads the model class
dynamically from a patched train_cdm.py so it can handle arbitrary
num_layers/model_dim/vocab_size without a hard-coded import.
"""
import argparse, os, sys, math, time, json, importlib.util
import numpy as np
import torch
import sentencepiece as spm


def load_module_from_path(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    # The module may try to run argparse / main(); protect it
    sys.argv = [path]  # empty argv so argparse doesn't complain if main is invoked
    spec.loader.exec_module(mod)
    return mod


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", type=str, required=True)
    ap.add_argument("--train_module_path", type=str, required=True)
    ap.add_argument("--num_layers", type=int, required=True)
    ap.add_argument("--model_dim", type=int, required=True)
    ap.add_argument("--vocab_size", type=int, required=True)
    ap.add_argument("--bigram_dim", type=int, required=True)
    ap.add_argument("--xsa_last_n", type=int, required=True)
    ap.add_argument("--n_seqs", type=int, default=500)
    ap.add_argument("--seq_len", type=int, default=1024)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--rounds", type=int, default=2)
    ap.add_argument("--n_random", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--data_dir", type=str, required=True)
    ap.add_argument("--tokenizer_path", type=str, required=True)
    ap.add_argument("--log_path", type=str, default="")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    log_fh = open(args.log_path, "w") if args.log_path else None
    def log(msg=""):
        print(msg, flush=True)
        if log_fh:
            log_fh.write(msg + "\n")
            log_fh.flush()

    log("=" * 70)
    log(f"CF ablation eval")
    log(f"  ckpt: {args.ckpt}")
    log(f"  model: L={args.num_layers} d={args.model_dim} vocab={args.vocab_size}")
    log(f"  eval: N={args.n_seqs} seq={args.seq_len} stride={args.stride} rounds={args.rounds}")
    log("=" * 70)

    # Dynamically import the patched training module to get the GPTv2 class
    # The patched module was created by train_ablation_runner.py and has the
    # correct module-level constants baked in.
    log(f"Loading model class from {args.train_module_path}")
    # Temporarily block the module's main() from running
    with open(args.train_module_path) as f:
        src = f.read()
    # Neuter the `if __name__ == "__main__"` block to prevent training from running
    # during import
    src = src.replace('if __name__ == "__main__"', 'if False')

    mod_globals = {"__name__": "train_cdm_ablation_class_source", "__file__": args.train_module_path}
    exec(compile(src, args.train_module_path, "exec"), mod_globals)

    GPTv2 = mod_globals["GPTv2"]

    # Sanity check: the patched module's constants should match our args
    assert mod_globals["NUM_LAYERS"] == args.num_layers, f"layer mismatch: mod={mod_globals['NUM_LAYERS']} arg={args.num_layers}"
    assert mod_globals["MODEL_DIM"] == args.model_dim, f"dim mismatch"
    assert mod_globals["VOCAB_SIZE"] == args.vocab_size, f"vocab mismatch"
    log(f"  module constants verified: {args.num_layers}L d={args.model_dim} vocab={args.vocab_size}")

    # Instantiate and load state dict
    model = GPTv2().to(device)
    log(f"  loading state dict from {args.ckpt}")
    if args.ckpt.endswith(".npz"):
        # Final-state save written by train_cdm.py at end of training (np.savez of
        # raw_model.state_dict(), with bf16 weights expanded to float32). Load each
        # array as a torch tensor; load_state_dict(strict=False) will cast back.
        log("  detected .npz final-state checkpoint")
        npz = np.load(args.ckpt)
        sd = {k: torch.from_numpy(np.array(npz[k])) for k in npz.files}
    else:
        sd = torch.load(args.ckpt, map_location=device, weights_only=False)
        # The checkpoint might be wrapped
        if isinstance(sd, dict):
            for k in ("model", "state_dict", "raw_model"):
                if k in sd and isinstance(sd[k], dict):
                    sd = sd[k]
                    break
    # Strip common DDP / compile prefixes
    clean = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("module."): k2 = k2[7:]
        if k2.startswith("_orig_mod."): k2 = k2[10:]
        clean[k2] = v
    # Cast each tensor to match the model parameter dtype, so loading float32
    # weights from .npz into a bf16 model works without silent precision loss.
    model_dtypes = {n: p.dtype for n, p in model.named_parameters()}
    model_dtypes.update({n: b.dtype for n, b in model.named_buffers()})
    for k in list(clean.keys()):
        if k in model_dtypes and clean[k].dtype != model_dtypes[k]:
            clean[k] = clean[k].to(model_dtypes[k])
    missing, unexpected = model.load_state_dict(clean, strict=False)
    log(f"  loaded {len(clean)} keys, missing={len(missing)}, unexpected={len(unexpected)}")
    if len(missing) > 10:
        log(f"  WARNING: many missing keys, first 5: {list(missing)[:5]}")
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    log(f"  params: {n_params:,}")

    # Load tokenizer + val data
    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    header = np.fromfile(os.path.join(args.data_dir, "fineweb_val_000000.bin"), dtype="<i4", count=256)
    val = np.fromfile(os.path.join(args.data_dir, "fineweb_val_000000.bin"),
                      dtype="<u2", count=int(header[2]), offset=256*4).astype(np.int32)
    log(f"  val tokens: {len(val):,}")

    # Byte LUTs
    sz = int(sp.vocab_size())
    bb = np.zeros(sz, dtype=np.int16)
    hs = np.zeros(sz, dtype=np.bool_)
    ib = np.ones(sz, dtype=np.bool_)
    for t in range(sz):
        if sp.is_control(t) or sp.is_unknown(t) or sp.is_unused(t): continue
        ib[t] = False
        if sp.is_byte(t): bb[t] = 1; continue
        p = sp.id_to_piece(t)
        if p.startswith("\u2581"): hs[t] = True; p = p[1:]
        bb[t] = len(p.encode("utf-8"))

    def count_bytes(tokens, prev_tokens):
        total = 0.0
        for i in range(len(tokens)):
            b = float(bb[tokens[i]])
            if hs[tokens[i]] and not ib[prev_tokens[i]]:
                b += 1.0
            total += max(b, 1.0)
        return total

    def split_rounds(cdm_positions, cdm_rounds):
        rounds = [[] for _ in range(cdm_rounds)]
        for i, pos in enumerate(cdm_positions):
            rounds[i % cdm_rounds].append(pos)
        return rounds

    @torch.no_grad()
    def get_log_probs(input_ids, is_causal):
        h = model.forward_hidden(input_ids, is_causal=is_causal)
        logits = model.softcap(torch.nn.functional.linear(h, model.tok_emb.weight.to(h.dtype)))
        return torch.log_softmax(logits.float(), dim=-1)

    # Pure AR eval (the shared model in causal mode)
    log("\n[1] Pure AR baseline (is_causal=True)")
    total_nll, total_bytes = 0.0, 0.0
    rng = np.random.RandomState(args.seed)
    t0 = time.time()
    for s in range(args.n_seqs):
        idx = rng.randint(0, len(val) - args.seq_len - 1)
        seq = val[idx:idx+args.seq_len+1]
        inp = torch.from_numpy(seq[:-1].reshape(1,-1).astype(np.int64)).to(device)
        lp = get_log_probs(inp, is_causal=True)[0].cpu().numpy()
        tgt = seq[1:]
        for t in range(args.seq_len):
            total_nll -= lp[t, int(tgt[t])]
        total_bytes += count_bytes(tgt, seq[:-1])
        if (s+1) % max(1, args.n_seqs//5) == 0:
            log(f"    AR {s+1}/{args.n_seqs} | BPB:{total_nll/total_bytes/math.log(2):.4f} | {time.time()-t0:.0f}s")
    pure_ar_bpb = total_nll / total_bytes / math.log(2)
    log(f"  Pure AR BPB: {pure_ar_bpb:.4f}")

    # CF eval
    log(f"\n[2] CF eval (stride={args.stride}, rounds={args.rounds})")
    total_ar_nll = total_cdm_nll = total_bytes = 0.0
    rng = np.random.RandomState(args.seed + 1)
    t0 = time.time()
    for s in range(args.n_seqs):
        idx = rng.randint(0, len(val) - args.seq_len - 1)
        seq = val[idx:idx + args.seq_len + 1]
        x = seq[1:]
        prev = seq[:-1]
        total_bytes += count_bytes(x, prev)

        ar_positions = list(range(0, args.seq_len, args.stride))
        cdm_positions = [i for i in range(args.seq_len) if i not in ar_positions]
        round_groups = split_rounds(cdm_positions, args.rounds)

        input_t = torch.from_numpy(prev.reshape(1, -1).astype(np.int64)).to(device)
        ar_lp = get_log_probs(input_t, is_causal=True)[0].cpu().numpy()
        for pos in ar_positions:
            total_ar_nll -= ar_lp[pos, int(x[pos])]

        for ridx, current_round in enumerate(round_groups):
            if not current_round: continue
            unresolved = set()
            for g in round_groups[ridx:]:
                unresolved.update(g)
            avg_round_nll = np.zeros(len(current_round))
            for r in range(args.n_random):
                cdm_input = x.copy()
                for pos in unresolved:
                    cdm_input[pos] = rng.randint(0, args.vocab_size)
                cdm_t = torch.from_numpy(cdm_input.reshape(1,-1).astype(np.int64)).to(device)
                cdm_lp = get_log_probs(cdm_t, is_causal=False)[0].cpu().numpy()
                for i, pos in enumerate(current_round):
                    avg_round_nll[i] -= cdm_lp[pos, int(x[pos])] / args.n_random
            total_cdm_nll += avg_round_nll.sum()

        if (s+1) % max(1, args.n_seqs//5) == 0:
            ar_bpb = total_ar_nll / total_bytes / math.log(2)
            cdm_bpb = total_cdm_nll / total_bytes / math.log(2)
            total_bpb = ar_bpb + cdm_bpb
            log(f"    CF {s+1}/{args.n_seqs} | AR:{ar_bpb:.4f} CDM:{cdm_bpb:.4f} Total:{total_bpb:.4f} | {time.time()-t0:.0f}s")

    cf_ar = total_ar_nll / total_bytes / math.log(2)
    cf_cdm = total_cdm_nll / total_bytes / math.log(2)
    cf_total = cf_ar + cf_cdm
    delta = (cf_total / pure_ar_bpb - 1) * 100

    log("\n" + "=" * 70)
    log(f"  Pure AR:       {pure_ar_bpb:.4f}")
    log(f"  CF AR part:    {cf_ar:.4f}")
    log(f"  CF CDM part:   {cf_cdm:.4f}")
    log(f"  CF Total:      {cf_total:.4f}")
    log(f"  CF vs Pure AR: {delta:+.2f}%  ({'−' if delta<0 else '+'}{abs(cf_total-pure_ar_bpb):.4f} BPB)")
    log("=" * 70)

    result = {
        "ckpt": args.ckpt, "num_layers": args.num_layers, "model_dim": args.model_dim,
        "vocab_size": args.vocab_size, "n_params": n_params,
        "n_seqs": args.n_seqs, "seq_len": args.seq_len,
        "stride": args.stride, "rounds": args.rounds,
        "pure_ar_bpb": float(pure_ar_bpb),
        "cf_ar_part": float(cf_ar),
        "cf_cdm_part": float(cf_cdm),
        "cf_total": float(cf_total),
        "cf_vs_ar_pct": float(delta),
    }
    log("\nJSON:")
    log(json.dumps(result, indent=2))

    if log_fh: log_fh.close()


if __name__ == "__main__":
    main()
