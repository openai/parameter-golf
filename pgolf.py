import argparse
import json
import subprocess
import os
import sys
import time
from pathlib import Path

DEFAULT_SCRIPT = "records/track_10min_16mb/2026-04-29_EBT_EnergyRefinement/train_gpt.py"

# Env vars forwarded from the local invocation environment to the remote training process.
# Only contains knobs read by the records-folder EBT train_gpt.py (and the shared GPT/Muon scaffolding).
TRAIN_ENV_FORWARD = (
    # Run identification / reproducibility
    "RUN_ID", "SEED",
    # Training schedule
    "ITERATIONS", "WARMUP_STEPS", "WARMDOWN_ITERS", "MAX_WALLCLOCK_SECONDS",
    "VAL_LOSS_EVERY", "VAL_BATCH_SIZE", "TRAIN_LOG_EVERY",
    # Data / tokenizer
    "DATA_PATH", "TOKENIZER_PATH", "VOCAB_SIZE",
    "TRAIN_BATCH_TOKENS", "TRAIN_SEQ_LEN",
    # Model architecture
    "NUM_LAYERS", "MODEL_DIM", "NUM_HEADS", "NUM_KV_HEADS", "MLP_MULT",
    # Optimizer
    "MUON_MOMENTUM", "MUON_MOMENTUM_WARMUP_START", "MUON_MOMENTUM_WARMUP_STEPS",
    "MATRIX_LR", "MATRIX_CLIP_SIGMAS", "MATRIX_WEIGHT_DECAY",
    "SCALAR_LR", "EMBED_LR", "GRAD_CLIP_NORM",
    # EBT-specific
    "ENERGY_RANK", "REFINE_STEPS_TRAIN", "REFINE_STEPS_EVAL",
    "REFINE_ETA_INIT", "AUX_LOSS_WEIGHT", "H0_NOISE_STD",
)

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


def _control_path(args):
    """Stable per-server ControlMaster socket path under /tmp."""
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in str(args.server))
    return f"/tmp/pgolf-ssh-{safe}-{os.getuid()}.sock"


def _ssh_common_opts(args):
    """Shared SSH options: keepalive + ControlMaster multiplexing.

    ControlMaster avoids creating a new TCP connection per ssh invocation, which
    otherwise triggers Cloudflare tunnel / gateway throttling on some providers
    (e.g. Brev). All subsequent ssh/scp calls reuse the single control socket.
    """
    return [
        "-o", "StrictHostKeyChecking=accept-new",
        "-o", "ServerAliveInterval=30",
        "-o", "ServerAliveCountMax=240",
        "-o", "TCPKeepAlive=yes",
        "-o", "ControlMaster=auto",
        "-o", f"ControlPath={_control_path(args)}",
        "-o", "ControlPersist=2h",
    ]


def get_ssh_base_cmd(args):
    cmd = ["ssh"] + _ssh_common_opts(args)
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
    cmd = ["scp"] + _ssh_common_opts(args)
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
    print(f"Setting up remote server: {args.server} (dataset variant: {args.dataset_variant})")
    hf_token = getattr(args, 'hf_token', None) or os.environ.get('HF_TOKEN', '')
    hf_export = f'export HF_TOKEN="{hf_token}"' if hf_token else ''

    ws = args.workspace_dir
    clone_parent = args.clone_parent
    variant = args.dataset_variant
    # variant tag like "sp1024" -> tokenizer file "fineweb_1024_bpe.model"
    vocab_tag = variant.removeprefix("sp")
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
    if [ ! -f "data/tokenizers/fineweb_{vocab_tag}_bpe.model" ]; then
        MATCHED_FINEWEB_REPO_ID=kevclark/parameter-golf python3 data/cached_challenge_fineweb.py --variant {variant}
        mkdir -p data/tokenizers
        if ls data/fineweb_{vocab_tag}_bpe.* 1>/dev/null 2>&1; then
            mv data/fineweb_{vocab_tag}_bpe.* data/tokenizers/
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

    # Preflight: refuse to deploy if remote disk is too full (training needs headroom for
    # final_model.pt ~130MB, run.log, train_log.json, and pytorch compile cache).
    df_out, _, df_rc = run_ssh(args, "df --output=avail -BG / | tail -1", capture=True)
    if df_rc == 0 and df_out.strip():
        try:
            avail_gb = int(df_out.strip().rstrip("G"))
            if avail_gb < 5:
                print(f"Deploy refused: only {avail_gb}G free on remote /. Need >=5G. Clean up before retrying.")
                sys.exit(1)
            print(f"Remote disk free: {avail_gb}G")
        except ValueError:
            print(f"Warning: couldn't parse df output ({df_out!r}); skipping preflight check")

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
    run_ssh(args, f"rm -f {ws}/train_log.json {ws}/run.log {ws}/train.pid {ws}/train.done")

    # Brev / user pip installs often put torchrun in ~/.local/bin; non-interactive SSH has a minimal PATH.
    path_prefix = (
        'export PATH="$HOME/.local/bin:$PATH" && '
        if args.provider == "brev"
        else ""
    )
    env_vars = f"PYTHONUNBUFFERED=1 MAX_WALLCLOCK_SECONDS={args.wallclock}"
    forwarded = []
    for k in TRAIN_ENV_FORWARD:
        v = os.environ.get(k)
        if v is not None:
            env_vars += f" {k}={v}"
            forwarded.append(f"{k}={v}")
    if forwarded:
        print(f"Forwarding {len(forwarded)} env vars: {' '.join(forwarded)}")

    nproc = max(1, int(args.nproc_per_node))
    print(f"torchrun --nproc_per_node={nproc}")

    # Detached launch: training runs in a nohup'd subshell that persists beyond SSH disconnects.
    # Writes train.done sentinel on exit so we can reliably detect completion via polling.
    launch_cmd = (
        f"cd {ws} && "
        f"nohup bash -c '{path_prefix}set -o pipefail; "
        f"{env_vars} torchrun --standalone --nproc_per_node={nproc} train_gpt_remote.py "
        f"2>&1 | stdbuf -oL tee run.log; "
        f"echo $? > train.done' "
        f">/dev/null 2>&1 & echo $! > train.pid; sleep 1; cat train.pid"
    )
    pid_out, _, rc = run_ssh(args, launch_cmd, capture=True)
    pid_str = pid_out.strip().splitlines()[-1] if pid_out.strip() else ""
    if rc != 0 or not pid_str.isdigit():
        print(f"Failed to launch training detached. rc={rc} output={pid_out!r}")
        return rc or 1
    print(f"Launched detached training (bash pid {pid_str}). Polling for completion...")

    # Poll loop: robust to SSH drops. Check every POLL_INTERVAL seconds for:
    #   - train.done sentinel (clean exit, captures return code)
    #   - train_log.json (canonical success marker)
    #   - Dead bash pid AND no train.done (process crashed or was killed externally)
    # Also tail run.log so we surface progress to the user.
    start_time = time.time()
    poll_interval = 60
    last_log_size = 0
    max_wall = args.wallclock + 600  # training budget + eval/quant/flush margin
    last_seen_ts = start_time
    consecutive_ssh_errs = 0

    while True:
        time.sleep(poll_interval)
        elapsed = time.time() - start_time
        ts = _fmt_elapsed(elapsed)

        if elapsed > max_wall:
            print(f"[{ts}] Poll timeout exceeded ({max_wall}s > wallclock+buffer). Giving up.")
            break

        status_cmd = (
            f"cd {ws} && "
            f"echo '---DONE---'; cat train.done 2>/dev/null; "
            f"echo '---PID_ALIVE---'; kill -0 {pid_str} 2>/dev/null && echo yes || echo no; "
            f"echo '---LOG_SIZE---'; stat -c %s run.log 2>/dev/null || echo 0; "
            f"echo '---LOG_TAIL---'; tail -c 4096 run.log 2>/dev/null; "
            f"echo '---TRAINLOG---'; test -f train_log.json && echo yes || echo no"
        )
        out, err, rc = run_ssh(args, status_cmd, capture=True)
        if rc != 0:
            consecutive_ssh_errs += 1
            print(f"[{ts}] SSH probe failed (rc={rc}, #{consecutive_ssh_errs}). Will retry.")
            if consecutive_ssh_errs >= 20:
                print(f"[{ts}] Too many SSH failures ({consecutive_ssh_errs}). Giving up.")
                break
            continue
        consecutive_ssh_errs = 0

        def _section(name):
            key = f"---{name}---"
            if key not in out:
                return ""
            rest = out.split(key, 1)[1]
            for nxt in ("---DONE---", "---PID_ALIVE---", "---LOG_SIZE---", "---LOG_TAIL---", "---TRAINLOG---"):
                if nxt != key and nxt in rest:
                    rest = rest.split(nxt, 1)[0]
            return rest.strip()

        done_rc = _section("DONE")
        pid_alive = _section("PID_ALIVE")
        log_size = _section("LOG_SIZE")
        log_tail = _section("LOG_TAIL")
        trainlog_exists = _section("TRAINLOG") == "yes"

        try:
            cur_size = int(log_size)
        except ValueError:
            cur_size = 0
        if cur_size > last_log_size:
            new_chunk = log_tail[-(cur_size - last_log_size):] if last_log_size else log_tail
            for line in new_chunk.strip().splitlines():
                if line.strip():
                    print(f"[{ts}] {line}", flush=True)
            last_log_size = cur_size
            last_seen_ts = time.time()

        if trainlog_exists:
            print(f"[{ts}] train_log.json present on remote. Training succeeded.")
            break
        if done_rc:
            print(f"[{ts}] train.done sentinel found (exit={done_rc}). Training ended without train_log.json.")
            break
        if pid_alive == "no":
            print(f"[{ts}] Detached bash pid {pid_str} no longer alive and no train.done yet; waiting one more cycle for sentinel.")
            time.sleep(poll_interval)
            out2, _, rc2 = run_ssh(args, f"cd {ws} && cat train.done 2>/dev/null; test -f train_log.json && echo TRAINLOG_YES || echo TRAINLOG_NO", capture=True)
            if "TRAINLOG_YES" in out2:
                print(f"[{ts}] train_log.json appeared. Training succeeded.")
                break
            print(f"[{ts}] Process dead and no sentinel/train_log. Treating as crashed.")
            break

    # Sync final run.log
    final_log, _, _ = run_ssh(args, f"cat {ws}/run.log 2>/dev/null", capture=True)
    if final_log:
        with open(local_log_path, 'w') as log_f:
            log_f.write(final_log)
        print(f"Synced full run.log ({len(final_log)} bytes)")

    out, _, _ = run_ssh(args, f"test -f {ws}/train_log.json && echo OK", capture=True)
    if out.strip() == "OK":
        print("Training completed (train_log.json found).")
        return 0
    print("Training FAILED (no train_log.json on remote).")
    return 1

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
        for remote, local in [
            ("train_gpt_remote.py", "train_gpt.py"),
            ("final_model.int8.ptz", "final_model.int8.ptz"),
            ("run.log", "run.log"),
            ("train_log.json", "train_log.json"),
        ]:
            subprocess.run(
                scp + [f"{args.server}:{args.workspace_scp}/{remote}", str(d / local)],
                capture_output=True,
            )
        print("Artifacts downloaded.")

    return report


def cmd_eval(args):
    """Convenience: deploy + train + validate + summary print."""
    iteration = args.iteration
    print(f"{'='*60}")
    print(f"  Eval Iteration: {iteration}")
    print(f"  wallclock={args.wallclock}s nproc_per_node={args.nproc_per_node}")
    if args.description:
        print(f"  description: {args.description}")
    print(f"{'='*60}")

    cmd_deploy(args)
    rc = cmd_train(args)

    args.download_artifact = True
    report = cmd_validate_submission(args)

    print(f"\n{'='*60}")
    print(f"  Results for {iteration}:")
    if rc != 0 or not report:
        print(f"    FAILED (rc={rc})")
        print(f"{'='*60}")
        return rc

    print(f"    total_bytes:  {report['total_bytes']:,}")
    print(f"    code_bytes:   {report['code_bytes']:,}")
    print(f"    model_bytes:  {report['model_bytes']:,}")
    print(f"    steps:        {report.get('steps', 0)}")
    print(f"    train_time:   {report.get('train_time_ms', 0)/1000:.1f}s")
    print(f"    peak_vram:    {report.get('peak_vram_mib', 0)} MiB")
    for label, vals in report.get("evals", {}).items():
        print(f"    {label:>14}: val_bpb={vals['val_bpb']:.6f} val_loss={vals['val_loss']:.6f}")
    if "ebt" in report:
        ebt = report["ebt"]
        print(f"    ebt: rank={ebt['energy_rank']} K_train={ebt['refine_steps_train']} K_eval={ebt['refine_steps_eval']} eta_init={ebt['refine_eta_init']} relchange={ebt['final_refine_relchange']:.4f}")
    print(f"{'='*60}")
    return 0

def main():
    parser = argparse.ArgumentParser(description="Parameter Golf remote training CLI (EBT submission)")

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
        "--nproc-per-node", type=int, default=1,
        help="GPUs per node for torchrun (default: 1; use 8 for full 8xH100 runs)",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    p_setup = subparsers.add_parser("setup", help="Setup remote server (clone repo, install deps, fetch data)")
    p_setup.add_argument("server", help="SSH server address (e.g. root@host)")
    p_setup.add_argument("--hf-token", help="HuggingFace token for gated dataset downloads")
    p_setup.add_argument(
        "--dataset-variant", default="sp1024", choices=("sp1024", "sp8192"),
        help="Tokenizer/data variant to download (default: sp1024 for the EBT submission)",
    )

    p_deploy = subparsers.add_parser("deploy", help="Deploy training script to server")
    p_deploy.add_argument("server", help="SSH server address")
    p_deploy.add_argument("--script", default=DEFAULT_SCRIPT, help="Local script to deploy")

    p_train = subparsers.add_parser("train", help="Train on remote server (detached, polled)")
    p_train.add_argument("server", help="SSH server address")

    p_validate = subparsers.add_parser("validate-submission", help="Validate submission size + download artifacts")
    p_validate.add_argument("server", help="SSH server address")
    p_validate.add_argument("--download-artifact", action="store_true", help="Download artifacts locally")
    p_validate.add_argument("--iteration", default=None, help="Iteration ID for artifact destination dir")

    p_eval = subparsers.add_parser("eval", help="Deploy, train, validate, and print summary")
    p_eval.add_argument("server", help="SSH server address")
    p_eval.add_argument("iteration", help="Iteration ID (used as artifact subdir name)")
    p_eval.add_argument("--script", default=DEFAULT_SCRIPT, help="Local script to deploy")
    p_eval.add_argument("--description", default="", help="Free-form description (printed only)")

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
        if args.download_artifact and not args.iteration:
            args.iteration = "manual_validation"
        cmd_validate_submission(args)
    elif args.command == "eval":
        cmd_eval(args)

if __name__ == "__main__":
    main()
