#!/usr/bin/env python3
"""
Wrapper that runs train_cdm.py with configurable num_layers / model_dim /
vocab_size / cdm_weight by patching its module-level constants at source level
before exec.

Usage on pod:
    python3 train_ablation_runner.py \
        --train_script ./train_cdm.py \
        --num_layers 5 --model_dim 256 --vocab_size 4096 \
        --bigram_dim 128 --xsa_last_n 2 \
        --cdm_weight 0.3 \
        -- \
        --steps 7000 --train_budget_secs 540 \
        --data_dir /workspace/data_v4096 \
        --tokenizer_path /workspace/bpe_v4096.model \
        --save_path /workspace/out/5L_w0.3.npz \
        --save_int6_path /workspace/out/5L_w0.3_int6.lzma \
        --checkpoint_dir /workspace/ckpt/5L_w0.3

The double-dash separates runner args from passthrough args for train_cdm.py.
If --cdm_weight 0.0, the entire CDM block is wrapped in `if False:` so the
forward/backward for the denoising pass is skipped.
"""
import argparse
import os
import re
import sys
import textwrap


def main():
    ap = argparse.ArgumentParser(description="Ablation runner for train_cdm.py")
    ap.add_argument("--train_script", type=str, required=True,
                    help="path to the base train_cdm.py")
    ap.add_argument("--num_layers", type=int, required=True)
    ap.add_argument("--model_dim", type=int, required=True)
    ap.add_argument("--vocab_size", type=int, required=True)
    ap.add_argument("--bigram_dim", type=int, required=True)
    ap.add_argument("--xsa_last_n", type=int, required=True)
    ap.add_argument("--cdm_weight", type=float, required=True,
                    help="Weight on the denoising loss. 0 = disable CDM (causal-only control).")
    ap.add_argument("--seed", type=int, default=None,
                    help="Override module-level SEED constant in train_cdm.py "
                         "(default: leave the script's baked-in seed, 1337, unchanged).")

    # Split on -- to separate runner args from passthrough args
    if "--" in sys.argv:
        sep_idx = sys.argv.index("--")
        runner_args = sys.argv[1:sep_idx]
        passthrough = sys.argv[sep_idx + 1:]
    else:
        runner_args = sys.argv[1:]
        passthrough = []

    args = ap.parse_args(runner_args)

    with open(args.train_script, "r") as f:
        src = f.read()

    # --- Patch module-level constants ---
    # Use regex so we match "NUM_LAYERS = <number>" on a line by itself
    def patch(src, name, new_val):
        pat = re.compile(rf"^{re.escape(name)}\s*=\s*[^\n#]+", re.MULTILINE)
        m = pat.search(src)
        assert m is not None, f"could not find constant {name} in train script"
        return pat.sub(f"{name} = {new_val}", src, count=1)

    src = patch(src, "VOCAB_SIZE", args.vocab_size)
    src = patch(src, "NUM_LAYERS", args.num_layers)
    src = patch(src, "MODEL_DIM", args.model_dim)
    src = patch(src, "BIGRAM_DIM", args.bigram_dim)
    src = patch(src, "XSA_LAST_N", args.xsa_last_n)
    if args.seed is not None:
        src = patch(src, "SEED", args.seed)

    # --- Patch the CDM loss weight inline ---
    # Original line (around line 1023):
    #     cdm_loss = (per_tok * mask.reshape(-1).float()).sum() / (mask.sum() + 1e-8) * 0.3 / args.grad_accum
    new_cdm_line = (
        f"cdm_loss = (per_tok * mask.reshape(-1).float()).sum() / "
        f"(mask.sum() + 1e-8) * {args.cdm_weight} / args.grad_accum"
    )
    cdm_pat = re.compile(
        r"cdm_loss\s*=\s*\(per_tok \* mask\.reshape\(-1\)\.float\(\)\)\.sum\(\) "
        r"/ \(mask\.sum\(\) \+ 1e-8\) \* [0-9.]+ / args\.grad_accum"
    )
    n = len(cdm_pat.findall(src))
    assert n == 1, f"expected 1 cdm_loss line, found {n}"
    src = cdm_pat.sub(new_cdm_line, src, count=1)

    # --- If weight == 0, wrap the entire CDM block in `if False:` so no FLOPs ---
    if args.cdm_weight == 0.0:
        marker_start = "import numpy as np_cdm"
        marker_end = "cdm_loss.backward()"
        # Find the START of the LINE containing the marker (include leading whitespace)
        i_marker = src.find(marker_start)
        assert i_marker != -1, "could not find CDM block start marker"
        i0 = src.rfind("\n", 0, i_marker) + 1  # beginning of the line
        i1_marker = src.find(marker_end, i_marker)
        assert i1_marker != -1, "could not find CDM block end marker"
        i1 = src.find("\n", i1_marker) + 1  # end of the cdm_loss.backward() line, inclusive

        block = src[i0:i1]
        # Now first line retains its original indent
        first_line = block.split("\n", 1)[0]
        orig_indent = len(first_line) - len(first_line.lstrip())
        wrapper_indent = " " * orig_indent
        extra = "    "

        new_block_lines = [f"{wrapper_indent}if False:  # --cdm_weight=0 (causal-only control)"]
        for line in block.splitlines():
            if line.strip():
                new_block_lines.append(extra + line)
            else:
                new_block_lines.append(line)
        # Add a stub cdm_loss after the disabled block so references later don't crash
        tail = src[i1:i1 + 400]
        if "cdm_loss" in tail:
            new_block_lines.append(f"{wrapper_indent}cdm_loss = torch.zeros((), device=x.device, dtype=torch.float32)")

        new_block = "\n".join(new_block_lines) + "\n"
        src = src[:i0] + new_block + src[i1:]

    # --- Write patched source to a temp file for reproducibility ---
    seed_suffix = f"_s{args.seed}" if args.seed is not None else ""
    out_script = f"/tmp/train_cdm_patched_{args.num_layers}L_w{args.cdm_weight}{seed_suffix}.py"
    with open(out_script, "w") as f:
        f.write(src)
    print(f"[ablation runner] patched script written to {out_script}")
    print(f"[ablation runner] config: {args.num_layers}L d={args.model_dim} "
          f"vocab={args.vocab_size} xsa={args.xsa_last_n} "
          f"bigram={args.bigram_dim} cdm_weight={args.cdm_weight}")

    # --- Exec the patched source with passthrough argv ---
    sys.argv = [args.train_script] + passthrough
    code = compile(src, out_script, "exec")
    exec(code, {"__name__": "__main__", "__file__": out_script})


if __name__ == "__main__":
    main()
