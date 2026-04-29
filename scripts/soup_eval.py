#!/usr/bin/env python3
"""soup_eval.py — N2 model-soup evaluator.

Averages N checkpoints saved by run_experiment.sh (records/ckpts/<label>.pt),
loads the averaged state into a fresh model, and runs the standard pre-quant
+ post-quant (GPTQ) eval used by train_gpt_sota_decoded.py so BPB numbers are
directly comparable to single-seed results in logs/sweep/results.csv.

Usage:
  python3 scripts/soup_eval.py \\
      --ckpts records/ckpts/s34_soup_seed1337.pt records/ckpts/s34_soup_seed42.pt records/ckpts/s34_soup_seed314.pt \\
      --label s34_soup_3seed_avg \\
      [--weights 1.0 1.0 1.0]

Env vars matching train_gpt_sota_decoded.py hyperparameters must match the
training config that produced the checkpoints (QK_GAIN_INIT, NUM_LAYERS, etc).
Defaults assume the standard s32/s34 config:
  QK_GAIN_INIT=5.5 WARMDOWN_FRAC=0.64 TTT_ENABLED=1 SLIDING_WINDOW_ENABLED=1
  TTT_EPOCHS=1 EMA_DECAY=0.995 LOGIT_SOFTCAP=20 MATRIX_LR=0.042

Outputs:
  - prints pre_bpb, quant_bpb, delta
  - appends a CSV row to logs/sweep/results.csv with label=<label>
"""

import argparse
import io
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402

# Import the canonical training module so model, eval, and GPTQ paths are
# byte-for-byte identical to what produced the checkpoints.
import train_gpt_sota_decoded as tg  # noqa: E402


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpts', nargs='+', required=True, help='Paths to state_dict .pt files')
    ap.add_argument('--label', required=True, help='Row label for CSV + log file')
    ap.add_argument('--weights', nargs='+', type=float, default=None,
                    help='Optional per-ckpt weights (default: uniform)')
    ap.add_argument('--skip-quant', action='store_true', help='Only run pre-quant eval (faster)')
    return ap.parse_args()


def average_state_dicts(ckpt_paths, weights=None):
    n = len(ckpt_paths)
    if weights is None:
        weights = [1.0] * n
    if len(weights) != n:
        raise ValueError(f'weights len {len(weights)} != ckpts len {n}')
    w_sum = float(sum(weights))
    if w_sum <= 0:
        raise ValueError('weights must sum > 0')

    print(f'[soup] averaging {n} checkpoints with weights {weights} (sum={w_sum})')
    acc = None
    for i, (p, w) in enumerate(zip(ckpt_paths, weights)):
        sd = torch.load(p, map_location='cpu', weights_only=False)
        # Normalize: some saves wrap state_dict in {'state_dict': ...} or {'w': ...}
        if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
            sd = sd['state_dict']
        if acc is None:
            acc = {k: torch.zeros_like(v, dtype=torch.float32) for k, v in sd.items()}
            ref_keys = set(acc.keys())
        else:
            if set(sd.keys()) != ref_keys:
                missing = ref_keys - set(sd.keys())
                extra = set(sd.keys()) - ref_keys
                raise ValueError(f'ckpt {p} keys mismatch; missing={list(missing)[:5]} extra={list(extra)[:5]}')
        for k, v in sd.items():
            acc[k].add_(v.to(torch.float32) * (w / w_sum))
        print(f'[soup]   [{i+1}/{n}] {p}  ({sum(v.numel() for v in sd.values())/1e6:.1f}M params)')

    # Cast back to each tensor's original dtype (use first ckpt as reference)
    ref_sd = torch.load(ckpt_paths[0], map_location='cpu', weights_only=False)
    if isinstance(ref_sd, dict) and 'state_dict' in ref_sd:
        ref_sd = ref_sd['state_dict']
    out = {k: acc[k].to(dtype=ref_sd[k].dtype) for k in acc}
    return out


def append_results_row(label, pre_bpb, quant_bpb, extra_cols):
    csv_path = REPO_ROOT / 'logs' / 'sweep' / 'results.csv'
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # Header must match run_experiment.sh
    header = 'label,status,exit_code,wall_s,train_loss,pre_quant_bpb,quant_bpb,sliding_bpb,ttt_bpb,delta_bpb,tok_s,peak_mem_gb,failure_class,timestamp,hostname,script,script_sha8,seed,iterations,sliding_enabled,ttt_enabled,fast_smoke,overrides,notes\n'
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        csv_path.write_text(header)
    delta = f'{(quant_bpb - pre_bpb):+.5f}' if (quant_bpb is not None and pre_bpb is not None) else ''
    pre_s = f'{pre_bpb:.8f}' if pre_bpb is not None else ''
    quant_s = f'{quant_bpb:.8f}' if quant_bpb is not None else ''
    ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    host = os.uname().nodename
    overrides = extra_cols.get('overrides', '')
    # Escape commas/quotes in overrides
    if ',' in overrides or '"' in overrides:
        overrides = '"' + overrides.replace('"', '""') + '"'
    row = f'{label},ok,0,0,,{pre_s},{quant_s},,,{delta},,,,{ts},{host},soup_eval.py,,,,,,,{overrides},{extra_cols.get("notes", "")}\n'
    with open(csv_path, 'a', encoding='utf-8') as f:
        f.write(row)
    print(f'[soup] appended row to {csv_path}')


def main():
    args = parse_args()
    ckpts = [Path(p).resolve() for p in args.ckpts]
    missing = [p for p in ckpts if not p.exists()]
    if missing:
        raise FileNotFoundError(f'checkpoints not found: {missing}')

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required')

    # Reuse canonical logging setup so log() writes to logs/<run_id>.txt
    os.environ.setdefault('RUN_ID', f'soup_{args.label}')
    h = tg.Hyperparameters()
    # Point model/quant paths to label-scoped files so we don't clobber final_model.pt
    h.model_path = f'records/ckpts/_soup_tmp_{args.label}.pt'
    h.quantized_model_path = f'records/ckpts/_soup_tmp_{args.label}.ptz'
    Path(h.model_path).parent.mkdir(parents=True, exist_ok=True)
    tg.set_logging_hparams(h)
    os.makedirs('logs', exist_ok=True)

    device = torch.device('cuda', h.local_rank)
    torch.cuda.set_device(device)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')
    from torch.backends.cuda import enable_cudnn_sdp, enable_flash_sdp, enable_math_sdp, enable_mem_efficient_sdp
    enable_cudnn_sdp(False); enable_flash_sdp(True); enable_mem_efficient_sdp(False); enable_math_sdp(False)
    torch._dynamo.config.optimize_ddp = False

    tg.log(f'[soup] label={args.label}')
    tg.log(f'[soup] ckpts={[str(p) for p in ckpts]}')
    tg.log(f'[soup] weights={args.weights}')

    # 1) Average checkpoints on CPU
    avg_sd = average_state_dicts([str(p) for p in ckpts], weights=args.weights)

    # 2) Build fresh model on GPU, load averaged weights
    model = tg.GPT(h).to(device).bfloat16()
    tg.restore_fp32_params(model)
    # restore_fp32_params may have re-cast some params; load avg_sd strictly
    model.load_state_dict(avg_sd, strict=True)

    # 3) Validation data
    val_data = tg.ValidationData(h, device)
    tg.log(f'[soup] val_tokens: {val_data.val_tokens.numel()-1}')

    # 4) Pre-quant eval (compile for speed)
    compiled_model = torch.compile(model, dynamic=False, fullgraph=True)
    torch._dynamo.reset()
    pre_loss, pre_bpb = tg.timed_eval('soup:pre-quant', tg.eval_val, h, device, val_data, compiled_model)

    quant_bpb = None
    if not args.skip_quant:
        # 5) GPTQ quantize + compress (reuses canonical serialize())
        code = Path(tg.__file__).read_text(encoding='utf-8')
        tg.serialize(h, model, code)
        # 6) Deserialize + quant eval
        eval_model = tg.deserialize(h, device)
        if h.num_loops > 0:
            eval_model.looping_active = True
        compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
        _, quant_bpb = tg.timed_eval('soup:quantized', tg.eval_val, h, device, val_data, compiled_eval)

    # 7) Report
    print('=' * 60)
    print(f'[soup] RESULT label={args.label}')
    print(f'[soup]   pre_bpb   = {pre_bpb:.6f}')
    if quant_bpb is not None:
        print(f'[soup]   quant_bpb = {quant_bpb:.6f}')
        print(f'[soup]   delta     = {quant_bpb - pre_bpb:+.6f}')
    print('=' * 60)

    overrides = f'CKPTS={",".join(p.name for p in ckpts)}'
    if args.weights is not None:
        overrides += f' WEIGHTS={",".join(str(w) for w in args.weights)}'
    notes = f'N={len(ckpts)}_soup'
    append_results_row(args.label, pre_bpb, quant_bpb, {'overrides': overrides, 'notes': notes})


if __name__ == '__main__':
    main()
