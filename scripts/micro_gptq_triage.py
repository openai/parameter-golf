#!/usr/bin/env python3
"""micro_gptq_triage.py — Tier A micro-test harness for GPTQ knobs.

Loads ONE checkpoint, evals pre-quant once, then loops over GPTQ knob
variations (calibration batches, matrix/embed clip sigmas) and re-runs
only the quant serialize/deserialize path for each. Appends rows to
logs/sweep/micro_gptq.csv.

Each knob test costs ~30-90s (just GPTQ + eval, no training).

Usage:
  python3 scripts/micro_gptq_triage.py --ckpt records/ckpts/s34_soup_seed42.pt \\
      --tag seed42_baseline

Baseline row uses env defaults (GPTQ_CALIBRATION_BATCHES=64,
MATRIX_CLIP_SIGMAS=12.85, EMBED_CLIP_SIGMAS=20).
"""

import argparse
import os
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
os.chdir(REPO_ROOT)
sys.path.insert(0, str(REPO_ROOT))

import torch  # noqa: E402
import train_gpt_sota_decoded as tg  # noqa: E402


KNOB_MATRIX = [
    # (knob_name, value) — univariate only, all else at baseline defaults
    # Calibration batch count
    ('calib',    32),
    ('calib',    64),      # baseline anchor (run first, reference)
    ('calib',    96),
    ('calib',    128),
    ('calib',    192),
    ('calib',    256),
    # Matrix clip sigmas
    ('matclip',  10.0),
    ('matclip',  11.0),
    ('matclip',  12.0),
    ('matclip',  12.85),   # baseline anchor
    ('matclip',  14.0),
    ('matclip',  16.0),
    # Embed clip sigmas
    ('embclip',  12.0),
    ('embclip',  16.0),
    ('embclip',  20.0),    # baseline anchor
    ('embclip',  24.0),
    ('embclip',  28.0),
]


def apply_knob(h, knob, value):
    """Return a human config string, mutate h in place."""
    # Reset to defaults first
    h.gptq_calibration_batches = 64
    h.matrix_clip_sigmas = 12.85
    h.embed_clip_sigmas = 20.0
    if knob == 'calib':
        h.gptq_calibration_batches = int(value)
    elif knob == 'matclip':
        h.matrix_clip_sigmas = float(value)
    elif knob == 'embclip':
        h.embed_clip_sigmas = float(value)
    else:
        raise ValueError(f'unknown knob {knob}')
    return f'{knob}={value}'


def append_csv(csv_path, row):
    header = 'timestamp,tag,ckpt,knob,value,pre_bpb,quant_bpb,delta,quant_bytes,gptq_s,eval_s\n'
    if not csv_path.exists() or csv_path.stat().st_size == 0:
        csv_path.write_text(header)
    with open(csv_path, 'a', encoding='utf-8') as f:
        f.write(row + '\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--csv', default='logs/sweep/micro_gptq.csv')
    ap.add_argument('--knob', default=None, help='Optional: only run knob with this name (calib|matclip|embclip)')
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt).resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(ckpt_path)

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA required')

    os.environ.setdefault('RUN_ID', f'micro_gptq_{args.tag}')
    h = tg.Hyperparameters()
    h.model_path = f'records/ckpts/_micro_tmp_{args.tag}.pt'
    h.quantized_model_path = f'records/ckpts/_micro_tmp_{args.tag}.ptz'
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

    tg.log(f'[micro] tag={args.tag} ckpt={ckpt_path}')

    # Load ckpt into fresh model
    sd = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    if isinstance(sd, dict) and 'state_dict' in sd and isinstance(sd['state_dict'], dict):
        sd = sd['state_dict']
    model = tg.GPT(h).to(device).bfloat16()
    tg.restore_fp32_params(model)
    model.load_state_dict(sd, strict=True)

    val_data = tg.ValidationData(h, device)
    tg.log(f'[micro] val_tokens: {val_data.val_tokens.numel()-1}')

    # One-time pre-quant eval (identical for all knob values, so cache)
    compiled_model = torch.compile(model, dynamic=False, fullgraph=True)
    torch._dynamo.reset()
    pre_loss, pre_bpb = tg.timed_eval('micro:pre-quant', tg.eval_val, h, device, val_data, compiled_model)
    tg.log(f'[micro] pre_bpb={pre_bpb:.6f}')

    csv_path = REPO_ROOT / args.csv
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    code_text = Path(tg.__file__).read_text(encoding='utf-8')

    seen_baseline = False
    for knob, value in KNOB_MATRIX:
        if args.knob and knob != args.knob:
            continue
        # Skip duplicate "baseline" rows (64/12.85/20 appears in all 3 sweeps)
        is_baseline = (knob == 'calib' and value == 64) or \
                      (knob == 'matclip' and abs(value - 12.85) < 1e-6) or \
                      (knob == 'embclip' and abs(value - 20.0) < 1e-6)
        if is_baseline and seen_baseline:
            tg.log(f'[micro] SKIP duplicate baseline: {knob}={value}')
            continue
        if is_baseline:
            seen_baseline = True

        cfg = apply_knob(h, knob, value)
        tg.log(f'[micro] === {cfg} ===')
        t0 = time.perf_counter()
        try:
            tg.serialize(h, model, code_text)
            serialize_s = time.perf_counter() - t0
            quant_bytes = Path(h.quantized_model_path).stat().st_size if Path(h.quantized_model_path).exists() else 0

            t1 = time.perf_counter()
            eval_model = tg.deserialize(h, device)
            if h.num_loops > 0:
                eval_model.looping_active = True
            compiled_eval = torch.compile(eval_model, dynamic=False, fullgraph=True)
            torch._dynamo.reset()
            _, quant_bpb = tg.timed_eval(f'micro:{cfg}', tg.eval_val, h, device, val_data, compiled_eval)
            eval_s = time.perf_counter() - t1

            delta = quant_bpb - pre_bpb
            tg.log(f'[micro] RESULT {cfg}: quant_bpb={quant_bpb:.6f} delta={delta:+.6f} bytes={quant_bytes} gptq={serialize_s:.1f}s eval={eval_s:.1f}s')
            ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            row = f'{ts},{args.tag},{ckpt_path.name},{knob},{value},{pre_bpb:.6f},{quant_bpb:.6f},{delta:+.6f},{quant_bytes},{serialize_s:.1f},{eval_s:.1f}'
            append_csv(csv_path, row)

            # Free intermediate model memory
            del eval_model, compiled_eval
            torch.cuda.empty_cache()
        except Exception as e:
            tg.log(f'[micro] FAIL {cfg}: {e!r}')
            ts = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
            row = f'{ts},{args.tag},{ckpt_path.name},{knob},{value},{pre_bpb:.6f},FAIL,,0,,'
            append_csv(csv_path, row)
            torch.cuda.empty_cache()

    tg.log(f'[micro] DONE — results in {csv_path}')


if __name__ == '__main__':
    main()
