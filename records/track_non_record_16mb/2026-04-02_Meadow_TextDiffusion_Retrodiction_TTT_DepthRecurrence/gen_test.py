#!/usr/bin/env python3
"""
Greedy bidirectional fill generation test across the 6 ablation checkpoints.

For each model:
  1. Pick 4 real FineWeb val sequences
  2. Mask a 16-token middle span
  3. Run bidirectional (is_causal=False) forward, greedy decode the masked positions
  4. Also run causal (is_causal=True) for comparison on the same masked input
  5. Print prefix / original / bidir_fill / causal_fill / suffix

Also compute an "infill exact match" metric: how many of the 16 masked tokens
the model recovers exactly.
"""
import os, sys, argparse, importlib.util
import numpy as np
import torch
import sentencepiece as spm


def load_model_from_patched(patched_path, ckpt_path, device):
    # Block the module from running its main() training loop at import time
    with open(patched_path) as f:
        src = f.read()
    src = src.replace('if __name__ == "__main__"', 'if False')

    ns = {"__name__": "gen_test_module", "__file__": patched_path}
    exec(compile(src, patched_path, "exec"), ns)

    GPTv2 = ns["GPTv2"]
    model = GPTv2().to(device)

    sd = torch.load(ckpt_path, map_location=device, weights_only=False)
    if isinstance(sd, dict):
        for k in ("model", "state_dict", "raw_model"):
            if k in sd and isinstance(sd[k], dict):
                sd = sd[k]
                break
    clean = {}
    for k, v in sd.items():
        k2 = k
        if k2.startswith("module."): k2 = k2[7:]
        if k2.startswith("_orig_mod."): k2 = k2[10:]
        clean[k2] = v
    model.load_state_dict(clean, strict=False)
    model.eval()
    return model, ns["VOCAB_SIZE"], ns["NUM_LAYERS"], ns["MODEL_DIM"]


@torch.no_grad()
def get_logits(model, input_ids, is_causal):
    h = model.forward_hidden(input_ids, is_causal=is_causal)
    logits = model.softcap(torch.nn.functional.linear(h, model.tok_emb.weight.to(h.dtype)))
    return logits


def run_model(tag, ckpt_path, patched_path, sp, val_tokens, device):
    print(f"\n{'='*76}")
    print(f"  {tag}")
    print(f"{'='*76}")
    try:
        model, vocab, L, D = load_model_from_patched(patched_path, ckpt_path, device)
    except Exception as e:
        print(f"  [skip] load error: {e}")
        return None
    print(f"  model: {L}L d={D} vocab={vocab}")

    SEQ_LEN = 96
    MASK_START = 40
    MASK_END = 56  # 16 tokens masked
    N_SEQS = 4

    rng = np.random.RandomState(7)
    total_bidir_em = 0
    total_causal_em = 0
    total_tokens = 0

    for idx in range(N_SEQS):
        # pick a sequence with actual English content
        attempts = 0
        while attempts < 20:
            start = rng.randint(0, len(val_tokens) - SEQ_LEN - 1)
            seq = val_tokens[start:start + SEQ_LEN].copy()
            text = sp.decode(seq.tolist())
            if len(text) > 200 and "\\x" not in repr(text):
                break
            attempts += 1

        prefix_tok = seq[:MASK_START]
        middle_true = seq[MASK_START:MASK_END]
        suffix_tok = seq[MASK_END:]

        # Build masked input (random noise at middle)
        masked = seq.copy()
        masked[MASK_START:MASK_END] = rng.randint(0, vocab, MASK_END - MASK_START)

        x = torch.from_numpy(masked.reshape(1, -1).astype(np.int64)).to(device)

        # Bidirectional greedy fill
        lg_bi = get_logits(model, x, is_causal=False)
        bi_fill = lg_bi[0, MASK_START:MASK_END].argmax(dim=-1).cpu().numpy()

        # Causal greedy fill (for comparison on same input)
        lg_ca = get_logits(model, x, is_causal=True)
        ca_fill = lg_ca[0, MASK_START:MASK_END].argmax(dim=-1).cpu().numpy()

        bi_em = int((bi_fill == middle_true).sum())
        ca_em = int((ca_fill == middle_true).sum())
        total_bidir_em += bi_em
        total_causal_em += ca_em
        total_tokens += (MASK_END - MASK_START)

        def safe(txt):
            return txt.replace("\n", " \\n ").replace("\r", " \\r ").strip()

        print(f"\n  [seq {idx+1}/{N_SEQS}]")
        print(f"  PREFIX:    {safe(sp.decode(prefix_tok.tolist()))}")
        print(f"  ORIGINAL:  [{safe(sp.decode(middle_true.tolist()))}]")
        print(f"  BIDIR:     [{safe(sp.decode(bi_fill.astype(np.int32).tolist()))}]  EM {bi_em}/16")
        print(f"  CAUSAL:    [{safe(sp.decode(ca_fill.astype(np.int32).tolist()))}]  EM {ca_em}/16")
        print(f"  SUFFIX:    {safe(sp.decode(suffix_tok.tolist()))}")

    print(f"\n  TOTAL EM: bidir={total_bidir_em}/{total_tokens} = {100*total_bidir_em/total_tokens:.1f}%, "
          f"causal={total_causal_em}/{total_tokens} = {100*total_causal_em/total_tokens:.1f}%")

    return {"tag": tag, "bidir_em": total_bidir_em, "causal_em": total_causal_em, "total": total_tokens}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", type=str, required=True,
                    help="directory containing fineweb_val_000000.bin")
    ap.add_argument("--tokenizer_path", type=str, required=True,
                    help="path to bpe_v4096.model")
    ap.add_argument("--ckpt_root", type=str, required=True,
                    help="directory containing per-run ckpt subdirs (5L_w0, 5L_w03, ...) "
                         "each holding a final .npz checkpoint from train_ablation_runner.py")
    ap.add_argument("--patched_dir", type=str, default="/tmp",
                    help="directory containing the patched training scripts produced "
                         "by train_ablation_runner.py (default: /tmp, matches runner output)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    sp = spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    header = np.fromfile(os.path.join(args.data_dir, "fineweb_val_000000.bin"), dtype="<i4", count=256)
    val = np.fromfile(os.path.join(args.data_dir, "fineweb_val_000000.bin"),
                      dtype="<u2", count=int(header[2]), offset=256*4).astype(np.int32)
    print(f"Val tokens: {len(val):,}")

    MODELS = [
        # (tag, ckpt subdir name, patched script layer tag)
        ("5L_w0",  "5L_w0",  "5L_w0.0"),
        ("5L_w03", "5L_w03", "5L_w0.3"),
        ("5L_w1",  "5L_w1",  "5L_w1.0"),
        ("11L_w0",  "11L_w0",  "11L_w0.0"),
        ("11L_w03", "11L_w03", "11L_w0.3"),
        ("11L_w1",  "11L_w1",  "11L_w1.0"),
    ]

    results = []
    for tag, ckpt_subdir, patch_tag in MODELS:
        import glob
        ckpt_dir = os.path.join(args.ckpt_root, ckpt_subdir)
        ckpts = sorted(glob.glob(os.path.join(ckpt_dir, "step_*.pt")),
                       key=lambda p: int(p.split("step_")[-1].split(".")[0]))
        if not ckpts:
            print(f"\n[skip] no checkpoint in {ckpt_dir}")
            continue
        ckpt = ckpts[-1]
        patched = os.path.join(args.patched_dir, f"train_cdm_patched_{patch_tag}.py")
        if not os.path.exists(patched):
            print(f"\n[skip] no patched script at {patched}")
            continue

        r = run_model(tag, ckpt, patched, sp, val, device)
        if r:
            results.append(r)

    print(f"\n{'='*76}\n  SUMMARY\n{'='*76}")
    print(f"{'tag':<12} {'bidir EM':<12} {'causal EM':<12}")
    for r in results:
        bi_pct = 100 * r["bidir_em"] / r["total"]
        ca_pct = 100 * r["causal_em"] / r["total"]
        print(f"  {r['tag']:<10} {r['bidir_em']}/{r['total']} ({bi_pct:.1f}%)   {r['causal_em']}/{r['total']} ({ca_pct:.1f}%)")


if __name__ == "__main__":
    main()
