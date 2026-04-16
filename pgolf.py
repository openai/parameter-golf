import argparse
import json
import subprocess
import os
import sys
import time
from pathlib import Path

WORKSPACE_DIR = "/workspace/parameter-golf"
DEFAULT_SCRIPT = "records/track_10min_16mb/working/train_gpt.py"
RESULTS_FILE = "records/track_10min_16mb/working/results.tsv"

def get_ssh_base_cmd(args):
    cmd = ["ssh", "-o", "StrictHostKeyChecking=accept-new"]
    if args.key:
        cmd.extend(["-i", os.path.expanduser(args.key)])
    if args.port:
        cmd.extend(["-p", str(args.port)])
    cmd.append(args.server)
    return cmd

def get_scp_base_cmd(args):
    cmd = ["scp", "-o", "StrictHostKeyChecking=accept-new"]
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
    
    setup_cmd = f"""set -e
    {hf_export}
    if [ ! -f "{WORKSPACE_DIR}/data/cached_challenge_fineweb.py" ]; then
        rm -rf {WORKSPACE_DIR}
        cd /workspace && git clone https://github.com/openai/parameter-golf.git
    fi
    cd {WORKSPACE_DIR}
    pip install --break-system-packages -q -r requirements.txt brotli 2>/dev/null || pip install -q -r requirements.txt brotli || true
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
        
    print(f"Deploying {script_path} to {args.server}:{WORKSPACE_DIR}/train_gpt_remote.py")
    
    with open(script_path, "r") as f:
        script_content = f.read()
        
    ssh_cmd = get_ssh_base_cmd(args) + [f"cat > {WORKSPACE_DIR}/train_gpt_remote.py"]
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
    run_ssh(args, f"rm -f {WORKSPACE_DIR}/train_log.json {WORKSPACE_DIR}/run.log")

    train_cmd = (
        f"cd {WORKSPACE_DIR} && PYTHONUNBUFFERED=1 "
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
    final_log, _, _ = run_ssh(args, f"cat {WORKSPACE_DIR}/run.log 2>/dev/null", capture=True)
    if final_log:
        with open(local_log_path, 'w') as log_f:
            log_f.write(final_log)
        # Show any lines we missed during streaming
        streamed = set()
        for raw_line in open(local_log_path, 'r'):
            pass  # just need the full file on disk
        print(f"[{ts}] Synced full run.log ({len(final_log)} bytes)")

    # Determine real success: train_log.json existing means training completed
    out, _, rc = run_ssh(args, f"test -f {WORKSPACE_DIR}/train_log.json && echo OK", capture=True)
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
    out, _, rc = run_ssh(args, f"cat {WORKSPACE_DIR}/train_log.json 2>/dev/null", capture=True)
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
            subprocess.run(scp + [f"{args.server}:{WORKSPACE_DIR}/{remote}", str(d / local)], capture_output=True)
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

def cmd_eval(args):
    iteration = args.iteration
    description = args.description
    
    print(f"{'='*60}")
    print(f"  Eval Iteration: {iteration}")
    print(f"{'='*60}")
    
    cmd_deploy(args)
    rc = cmd_train(args)
    
    args.download_artifact = True
    report = cmd_validate_submission(args)
    
    # Defaults for crash
    row = dict(commit=iteration, description=description, status="crash",
               peak_vram_mib=0, total_bytes=0, code_bytes=0, model_bytes=0)
    eval_cols = {}
    best_key, best_val = None, None
    
    if rc == 0 and report:
        row.update(peak_vram_mib=report['peak_vram_mib'], total_bytes=report['total_bytes'],
                   code_bytes=report['code_bytes'], model_bytes=report['model_bytes'], status="keep")
        for label, vals in report.get('evals', {}).items():
            safe = label.replace('-','_').replace(' ','_')
            eval_cols[f"{safe}_val_bpb"] = f"{vals['val_bpb']:.6f}"
            eval_cols[f"{safe}_val_loss"] = f"{vals['val_loss']:.6f}"
            eval_cols[f"{safe}_eval_time_ms"] = f"{vals['eval_time_ms']:.0f}"
        best_key, best_val = best_eval(report)
    
    row['best_eval'] = best_key or ""
    row['best_val_bpb'] = f"{best_val['val_bpb']:.6f}" if best_val else "0.000000"
    row['memory_gb'] = f"{row['peak_vram_mib']/1024:.1f}"
    
    # Build TSV with stable column order
    fixed_cols = ['commit','best_val_bpb','best_eval','memory_gb','total_bytes','code_bytes','model_bytes']
    eval_col_names = sorted(eval_cols.keys())
    all_cols = fixed_cols + eval_col_names + ['status','description']
    
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    
    # Read existing header if present
    existing_header = None
    if os.path.isfile(RESULTS_FILE):
        with open(RESULTS_FILE, 'r') as f:
            existing_header = f.readline().strip().split('\t')
    
    if existing_header:
        # Merge any new columns
        for c in all_cols:
            if c not in existing_header:
                existing_header.insert(-2, c)  # insert before status,description
        all_cols = existing_header
    
    vals = {**row, **eval_cols}
    line = '\t'.join(str(vals.get(c, '')) for c in all_cols)
    
    with open(RESULTS_FILE, 'w' if not existing_header else 'a') as f:
        if not existing_header:
            f.write('\t'.join(all_cols) + '\n')
        f.write(line + '\n')
    
    print(f"\n{'='*60}")
    print(f"  Results for {iteration}:")
    print(f"    best_eval:  {best_key or 'N/A'}")
    print(f"    best_bpb:   {row['best_val_bpb']}")
    print(f"    memory_gb:  {row['memory_gb']}")
    print(f"    total_size: {row['total_bytes']:,} bytes")
    print(f"    status:     {row['status']}")
    print(f"  Recorded to {RESULTS_FILE}")
    print(f"{'='*60}")

def main():
    parser = argparse.ArgumentParser(description="Parameter Golf Autoresearch CLI")
    
    parser.add_argument("--key", default="~/.ssh/id_ed25519", help="SSH identity file")
    parser.add_argument("--port", type=int, help="SSH port")
    
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
