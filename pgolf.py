import argparse
import json
import subprocess
import os
import sys
import time
from pathlib import Path

DEFAULT_SCRIPT = "records/track_10min_16mb/working/train_gpt.py"
RESULTS_FILE = "records/track_10min_16mb/working/results.tsv"

# Brev hosts typically ship a CUDA 12.x driver; default pip stacks often pull torch+cu130, which
# will not initialize CUDA. Pin cu128 torch, then FA3 wheel with --no-deps so pip does not
# replace torch again.
BREV_TORCH_INDEX_URL = "https://download.pytorch.org/whl/cu128"
BREV_FLASH_ATTN_3_WHEEL = (
    "https://download.pytorch.org/whl/cu128/"
    "flash_attn_3-3.0.0-cp39-abi3-manylinux_2_28_x86_64.whl"
)


def resolve_ssh_config_path(args):
    if getattr(args, "provider", "runpod") != "brev":
        return None
    p = args.ssh_config or os.environ.get("BREV_SSH_CONFIG") or "~/.brev/ssh_config"
    return os.path.expanduser(p)


def get_ssh_base_cmd(args):
    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new"]
    cfg = resolve_ssh_config_path(args)
    if cfg:
        cmd.extend(["-F", cfg])
    if args.key:
        cmd.extend(["-i", os.path.expanduser(args.key)])
    if args.port:
        cmd.extend(["-p", str(args.port)])
    cmd.append(args.server)
    return cmd


def get_scp_base_cmd(args):
    cmd = ["scp", "-o", "StrictHostKeyChecking=accept-new"]
    cfg = resolve_ssh_config_path(args)
    if cfg:
        cmd.extend(["-F", cfg])
    if args.key:
        cmd.extend(["-i", os.path.expanduser(args.key)])
    if args.port:
        cmd.extend(["-P", str(args.port)])
    return cmd

def run_ssh(args, cmd, capture=False):
    ssh_cmd = get_ssh_base_cmd(args) + [cmd]
    if capture:
        result = subprocess.run(ssh_cmd, capture_output=True, text=True)
        return result.stdout.strip(), result.stderr.strip(), result.returncode
    else:
        result = subprocess.run(ssh_cmd)
        return "", "", result.returncode

def cmd_setup(args):
    print(f"Setting up remote server: {args.server}")
    hf_token = getattr(args, 'hf_token', None) or os.environ.get('HF_TOKEN', '')
    hf_export = f'export HF_TOKEN="{hf_token}"' if hf_token else ''
    
    ws = args.workspace_dir
    clone_parent = args.clone_parent
    if args.provider == "brev":
        brev_stack = f"""
    export PATH="$HOME/.local/bin:$PATH"
    pip install --user -q --upgrade torch --index-url {BREV_TORCH_INDEX_URL}
    pip install --user -q --no-deps --upgrade "{BREV_FLASH_ATTN_3_WHEEL}"
    python3 -c "import torch; from flash_attn_interface import flash_attn_func; assert torch.cuda.is_available(), 'CUDA required for setup check'"
    """
    else:
        brev_stack = ""
    setup_cmd = f"""set -e
    {hf_export}
    if [ ! -f "{ws}/data/cached_challenge_fineweb.py" ]; then
        rm -rf {ws}
        cd {clone_parent} && git clone https://github.com/openai/parameter-golf.git
    fi
    cd {ws}
    pip install --break-system-packages -q -r requirements.txt brotli 2>/dev/null || pip install -q -r requirements.txt brotli || true
    {brev_stack}
    if [ ! -f "data/tokenizers/fineweb_8192_bpe.model" ]; then
        MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant sp8192
        mkdir -p data/tokenizers
        if ls data/fineweb_8192_bpe.* 1>/dev/null 2>&1; then
            mv data/fineweb_8192_bpe.* data/tokenizers/
        fi
    fi
    echo "SETUP_OK"
    """
    _, _, rc = run_ssh(args, setup_cmd)
    if rc == 0:
        print("Setup completed successfully.")
    else:
        print("Setup failed.")
        sys.exit(rc)

def cmd_deploy(args):
    script_path = args.script
    
    if not os.path.exists(script_path):
        print(f"Error: Local script {script_path} not found.")
        sys.exit(1)
        
    ws = args.workspace_dir
    print(f"Deploying {script_path} to {args.server}:{ws}/train_gpt_remote.py")
    
    with open(script_path, "r") as f:
        script_content = f.read()
        
    ssh_cmd = get_ssh_base_cmd(args) + [f"cat > {ws}/train_gpt_remote.py"]
    process = subprocess.Popen(ssh_cmd, stdin=subprocess.PIPE)
    process.communicate(input=script_content.encode('utf-8'))
    
    if process.returncode == 0:
        print("Deploy completed successfully.")
    else:
        print("Deploy failed.")
        sys.exit(process.returncode)

def _fmt_elapsed(t):
    m, s = divmod(int(t), 60)
    return f"{m:02d}:{s:02d}"

def cmd_train(args):
    iteration = getattr(args, 'iteration', None) or time.strftime("%Y%m%d_%H%M%S")
    local_log_dir = Path("records/track_10min_16mb/working/artifacts") / iteration
    local_log_dir.mkdir(parents=True, exist_ok=True)
    local_log_path = local_log_dir / "run.log"

    print(f"Training on {args.server}...")
    print(f"Local log: {local_log_path}")
    ws = args.workspace_dir
    run_ssh(args, f"rm -f {ws}/train_log.json {ws}/run.log")

    # Brev / user pip installs often put torchrun in ~/.local/bin; non-interactive SSH has a minimal PATH.
    path_prefix = (
        'export PATH="$HOME/.local/bin:$PATH" && '
        if args.provider == "brev"
        else ""
    )
    env_vars = f"PYTHONUNBUFFERED=1 MAX_WALLCLOCK_SECONDS={args.wallclock}"
    env_vars += f" SLIDING_WINDOW_ENABLED={int(args.sliding_window)}"
    env_vars += f" TTT_ENABLED={int(args.ttt)}"
    train_cmd = (
        f"{path_prefix}set -o pipefail && cd {ws} && {env_vars} "
        f"torchrun --standalone --nproc_per_node=1 train_gpt_remote.py "
        f"2>&1 | stdbuf -oL tee run.log"
    )
    ssh_cmd = get_ssh_base_cmd(args) + [train_cmd]
    process = subprocess.Popen(ssh_cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    start_time = time.time()
    with open(local_log_path, 'wb') as log_f:
        for raw_line in process.stdout:
            log_f.write(raw_line)
            log_f.flush()
            ts = _fmt_elapsed(time.time() - start_time)
            text = raw_line.decode('utf-8', errors='replace').rstrip('\n\r')
            print(f"[{ts}] {text}", flush=True)

    process.wait()
    ts = _fmt_elapsed(time.time() - start_time)

    # SSH pipe can break before all output arrives; sync final log from remote
    final_log, _, _ = run_ssh(args, f"cat {ws}/run.log 2>/dev/null", capture=True)
    if final_log:
        with open(local_log_path, 'w') as log_f:
            log_f.write(final_log)
        # Show any lines we missed during streaming
        streamed = set()
        for raw_line in open(local_log_path, 'r'):
            pass  # just need the full file on disk
        print(f"[{ts}] Synced full run.log ({len(final_log)} bytes)")

    # Determine real success: train_log.json existing means training completed
    out, _, rc = run_ssh(args, f"test -f {ws}/train_log.json && echo OK", capture=True)
    if out.strip() == "OK":
        print(f"[{ts}] Training completed (train_log.json found).")
        return 0
    elif process.returncode != 0:
        print(f"[{ts}] Training FAILED (exit code {process.returncode})")
        return process.returncode
    else:
        print(f"[{ts}] Training completed.")
        return 0

def fetch_train_log(args):
    """Fetch and parse train_log.json from the remote."""
    ws = args.workspace_dir
    out, _, rc = run_ssh(args, f"cat {ws}/train_log.json 2>/dev/null", capture=True)
    if rc != 0 or not out:
        return None
    return json.loads(out)

def cmd_validate_submission(args):
    print(f"Validating submission on {args.server}...")
    report = fetch_train_log(args)
    if report is None:
        print("No train_log.json found on remote (training likely crashed).")
        return None
    
    total = report['total_bytes']
    print(f"Script size: {report['code_bytes']:,} bytes")
    print(f"Model size:  {report['model_bytes']:,} bytes")
    print(f"Total size:  {total:,} bytes")
    if total > 16_000_000:
        print(f"FAIL: {total:,} bytes exceeds the 16,000,000 byte limit!")
    else:
        print(f"PASS: under 16MB ({total/16_000_000*100:.1f}% used).")
    
    if getattr(args, 'download_artifact', False) and getattr(args, 'iteration', None):
        d = Path("records/track_10min_16mb/working/artifacts") / args.iteration
        d.mkdir(parents=True, exist_ok=True)
        print(f"Downloading artifacts to {d}...")
        scp = get_scp_base_cmd(args)
        for remote, local in [("train_gpt_remote.py","train_gpt.py"),("final_model.int6.ptz","final_model.int6.ptz"),("run.log","run.log"),("train_log.json","train_log.json")]:
            subprocess.run(
                scp + [f"{args.server}:{args.workspace_scp}/{remote}", str(d / local)],
                capture_output=True,
            )
        print("Artifacts downloaded.")
    
    return report

EVAL_ORDER = ['quantized_ttt', 'quantized_sliding_window', 'quantized', 'pre-quantization post-ema']

def best_eval(report):
    """Pick the best val_bpb from evals whose full submission size is < 16MB."""
    if not report:
        return None, None
    evals = report.get('evals', {})
    total = report['total_bytes']
    for key in EVAL_ORDER:
        if key in evals and total <= 16_000_000:
            return key, evals[key]
    return None, None

FIXED_COLS = [
    'commit', 'best_val_bpb', 'best_eval', 'steps', 'train_time_s',
    'memory_gb', 'total_bytes', 'code_bytes', 'model_bytes',
]
TAIL_COLS = ['status', 'description']


def _build_tsv_row(row, eval_cols, all_cols):
    """Build a TSV line from row dict + eval_cols dict, ordered by all_cols."""
    vals = {**row, **eval_cols}
    return '\t'.join(str(vals.get(c, '')) for c in all_cols)


def _read_tsv(path):
    """Read existing TSV into (header_list, [row_dicts])."""
    if not os.path.isfile(path):
        return None, []
    with open(path, 'r') as f:
        lines = [l.rstrip('\n') for l in f if l.strip()]
    if not lines:
        return None, []
    header = lines[0].split('\t')
    rows = []
    for line in lines[1:]:
        fields = line.split('\t')
        rows.append(dict(zip(header, fields)))
    return header, rows


def _write_tsv(path, header, rows):
    """Write header + rows to TSV, filling missing columns with ''."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, 'w') as f:
        f.write('\t'.join(header) + '\n')
        for r in rows:
            f.write('\t'.join(str(r.get(c, '')) for c in header) + '\n')


def _merge_header(existing, new_cols):
    """Merge new columns into existing header, keeping TAIL_COLS last."""
    merged = list(existing)
    tail_set = set(TAIL_COLS)
    insert_pos = len(merged)
    for c in reversed(merged):
        if c in tail_set:
            insert_pos -= 1
        else:
            break
    for c in new_cols:
        if c not in merged:
            merged.insert(insert_pos, c)
            insert_pos += 1
    return merged


def cmd_eval(args):
    iteration = args.iteration
    description = args.description
    
    print(f"{'='*60}")
    print(f"  Eval Iteration: {iteration}")
    print(f"  wallclock={args.wallclock}s  sliding_window={args.sliding_window}  ttt={args.ttt}")
    print(f"{'='*60}")
    
    cmd_deploy(args)
    rc = cmd_train(args)
    
    args.download_artifact = True
    report = cmd_validate_submission(args)
    
    row = dict(commit=iteration, description=description, status="crash",
               steps=0, train_time_s=0, peak_vram_mib=0,
               total_bytes=0, code_bytes=0, model_bytes=0)
    eval_cols = {}
    best_key, best_val = None, None
    
    if rc == 0 and report:
        row.update(
            peak_vram_mib=report['peak_vram_mib'],
            total_bytes=report['total_bytes'],
            code_bytes=report['code_bytes'],
            model_bytes=report['model_bytes'],
            steps=report.get('steps', 0),
            train_time_s=f"{report.get('train_time_ms', 0) / 1000:.1f}",
            status="keep",
        )
        for label, vals in report.get('evals', {}).items():
            safe = label.replace('-', '_').replace(' ', '_')
            eval_cols[f"{safe}_val_bpb"] = f"{vals['val_bpb']:.6f}"
            eval_cols[f"{safe}_val_loss"] = f"{vals['val_loss']:.6f}"
            eval_cols[f"{safe}_eval_time_ms"] = f"{vals['eval_time_ms']:.0f}"
        best_key, best_val = best_eval(report)
    
    row['best_eval'] = best_key or ""
    row['best_val_bpb'] = f"{best_val['val_bpb']:.6f}" if best_val else "0.000000"
    row['memory_gb'] = f"{row['peak_vram_mib'] / 1024:.1f}"
    
    eval_col_names = sorted(eval_cols.keys())
    new_cols = FIXED_COLS + eval_col_names + TAIL_COLS
    
    existing_header, existing_rows = _read_tsv(RESULTS_FILE)
    if existing_header:
        all_cols = _merge_header(existing_header, new_cols)
    else:
        all_cols = new_cols
    
    existing_rows.append({**row, **eval_cols})
    _write_tsv(RESULTS_FILE, all_cols, existing_rows)
    
    print(f"\n{'='*60}")
    print(f"  Results for {iteration}:")
    print(f"    best_eval:  {best_key or 'N/A'}")
    print(f"    best_bpb:   {row['best_val_bpb']}")
    for k in eval_col_names:
        if k.endswith('_val_bpb'):
            print(f"    {k}: {eval_cols[k]}")
    print(f"    steps:      {row['steps']}")
    print(f"    train_time: {row['train_time_s']}s")
    print(f"    memory_gb:  {row['memory_gb']}")
    print(f"    total_size: {row['total_bytes']:,} bytes")
    print(f"    status:     {row['status']}")
    print(f"  Recorded to {RESULTS_FILE}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Parameter Golf Autoresearch CLI")
    
    parser.add_argument("--key", default="~/.ssh/id_ed25519", help="SSH identity file")
    parser.add_argument("--port", type=int, help="SSH port")
    parser.add_argument(
        "--provider",
        choices=("runpod", "brev"),
        default="runpod",
        help="Remote environment: runpod uses /workspace/parameter-golf; brev uses $HOME/parameter-golf and SSH -F",
    )
    parser.add_argument(
        "--ssh-config",
        default=None,
        help="SSH config file for --provider brev (default: ~/.brev/ssh_config or $BREV_SSH_CONFIG)",
    )
    parser.add_argument(
        "--wallclock", type=int, default=1200,
        help="Training wall-clock budget in seconds (default: 1200 = 20 min)",
    )
    parser.add_argument(
        "--sliding-window", action=argparse.BooleanOptionalAction, default=True,
        help="Enable sliding-window eval (default: on)",
    )
    parser.add_argument(
        "--ttt", action=argparse.BooleanOptionalAction, default=False,
        help="Enable TTT eval (default: off)",
    )
    
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    p_setup = subparsers.add_parser("setup", help="Setup remote server")
    p_setup.add_argument("server", help="SSH server address (e.g. root@host)")
    p_setup.add_argument("--hf-token", help="HuggingFace token for gated dataset downloads")
    
    p_deploy = subparsers.add_parser("deploy", help="Deploy script to server")
    p_deploy.add_argument("server", help="SSH server address")
    p_deploy.add_argument("--script", default=DEFAULT_SCRIPT, help="Local script to deploy")
    
    p_train = subparsers.add_parser("train", help="Train on remote server")
    p_train.add_argument("server", help="SSH server address")
    
    p_validate = subparsers.add_parser("validate-submission", help="Validate submission size")
    p_validate.add_argument("server", help="SSH server address")
    p_validate.add_argument("--download-artifact", action="store_true", help="Download artifacts locally")
    
    p_eval = subparsers.add_parser("eval", help="Deploy, train, validate, and record results")
    p_eval.add_argument("server", help="SSH server address")
    p_eval.add_argument("iteration", help="Iteration ID or commit hash")
    p_eval.add_argument("--script", default=DEFAULT_SCRIPT, help="Local script to deploy")
    p_eval.add_argument("--description", default="autoresearch run", help="Description of the run")
    
    args = parser.parse_args()

    # Remote shell uses $HOME/... on brev; scp does not expand $HOME in paths, so use a home-relative prefix for brev.
    if args.provider == "brev":
        args.workspace_dir = "$HOME/parameter-golf"
        args.workspace_scp = "parameter-golf"
        args.clone_parent = "$HOME"
    else:
        args.workspace_dir = "/workspace/parameter-golf"
        args.workspace_scp = "/workspace/parameter-golf"
        args.clone_parent = "/workspace"

    if args.command == "setup":
        cmd_setup(args)
    elif args.command == "deploy":
        cmd_deploy(args)
    elif args.command == "train":
        cmd_train(args)
    elif args.command == "validate-submission":
        if getattr(args, 'download_artifact', False) and not hasattr(args, 'iteration'):
            args.iteration = "manual_validation"
        cmd_validate_submission(args)
    elif args.command == "eval":
        cmd_eval(args)

if __name__ == "__main__":
    main()
