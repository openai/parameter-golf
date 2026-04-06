import json
import os
import re
import shutil
import subprocess
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pandas as pd
import pytz
import streamlit as st

import run_tracker

PST = pytz.timezone('US/Pacific')


def fmt_timestamp(iso_str):
    """Convert ISO timestamp to PST, formatted as 'Mar 27 10:15 PM'."""
    if not iso_str or iso_str == "?" or not isinstance(iso_str, str):
        return "?"
    try:
        dt = datetime.fromisoformat(iso_str)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(PST).strftime("%b %d %I:%M %p")
    except Exception:
        return iso_str[:16].replace("T", " ") if len(iso_str) > 16 else iso_str

st.set_page_config(page_title="Parameter Golf Command Center", layout="wide")
st.title("Parameter Golf: Command Center")

tab1, tab2, tab3, tab4 = st.tabs([
    "Launch Run", "Run History", "Autoresearch", "System Info",
])

# ──────────────────────────────────────────────
# TAB 1: Launch Run
# ──────────────────────────────────────────────
with tab1:
    run_mode = st.radio("Run Mode", ["Quick Test", "Overnight Run"], horizontal=True)
    if run_mode == "Quick Test":
        default_iters, default_wall = 200, 600.0
    else:
        default_iters, default_wall = 10000, 0.0

    st.sidebar.header("Hyperparameters")
    num_layers = st.sidebar.slider("NUM_LAYERS", 6, 12, 9)
    model_dim = st.sidebar.selectbox("MODEL_DIM", [384, 512, 640], index=1)
    mlp_mult = st.sidebar.selectbox("MLP_MULT", [2, 3], index=0)
    matrix_lr = st.sidebar.number_input("MATRIX_LR", value=0.04, format="%.4f", step=0.005)
    scalar_lr = st.sidebar.number_input("SCALAR_LR", value=0.04, format="%.4f", step=0.005)
    tied_embed_lr = st.sidebar.number_input("TIED_EMBED_LR", value=0.05, format="%.4f", step=0.005)
    muon_momentum = st.sidebar.number_input("MUON_MOMENTUM", value=0.95, format="%.4f", step=0.01)
    warmdown_iters = st.sidebar.number_input("WARMDOWN_ITERS", value=1200, step=100)
    seed = st.sidebar.number_input("SEED", value=1337, step=1)

    st.sidebar.markdown("---")
    iterations = st.sidebar.number_input("ITERATIONS", value=default_iters, step=100)
    wallclock = st.sidebar.number_input("MAX_WALLCLOCK_SECONDS", value=default_wall, step=60.0,
                                        help="0 = unlimited")
    val_loss_every = st.sidebar.number_input("VAL_LOSS_EVERY", value=100, step=50)

    col_launch, col_stop = st.columns([3, 1])
    with col_launch:
        launch_btn = st.button("Launch Training Run", type="primary")
    with col_stop:
        stop_btn = st.button("Stop Current Run", type="secondary")

    if stop_btn:
        # Kill any running torchrun process
        subprocess.run(["bash", "-c", "pkill -f torchrun || true"], capture_output=True)
        st.warning("Sent stop signal to training process.")

    if launch_btn:
        run_id = f"manual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        st.info(f"Run ID: `{run_id}`")

        env_vars = os.environ.copy()
        params = {
            "NUM_LAYERS": int(num_layers),
            "MODEL_DIM": int(model_dim),
            "MLP_MULT": int(mlp_mult),
            "MATRIX_LR": float(matrix_lr),
            "SCALAR_LR": float(scalar_lr),
            "TIED_EMBED_LR": float(tied_embed_lr),
            "MUON_MOMENTUM": float(muon_momentum),
            "WARMDOWN_ITERS": int(warmdown_iters),
            "SEED": int(seed),
        }
        for k, v in params.items():
            env_vars[k] = str(v)
        env_vars["RUN_ID"] = run_id
        env_vars["ITERATIONS"] = str(int(iterations))
        env_vars["MAX_WALLCLOCK_SECONDS"] = str(float(wallclock))
        env_vars["VAL_LOSS_EVERY"] = str(int(val_loss_every))
        env_vars["TRAIN_LOG_EVERY"] = "10"

        st.subheader("Live Training Output")
        progress_bar = st.progress(0, text="Starting...")
        chart_placeholder = st.empty()
        terminal_placeholder = st.empty()
        train_data = []

        train_loss_pat = re.compile(
            r"step:(\d+)/(\d+) train_loss:([\d.]+) train_time:(\d+)ms step_avg:([\d.]+)ms"
        )

        t0 = time.time()
        max_wall = float(wallclock) if float(wallclock) > 0 else None
        process = subprocess.Popen(
            ["bash", "start_baseline.sh"],
            env=env_vars,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )

        terminal_output = ""
        for line in iter(process.stdout.readline, ""):
            terminal_output += line
            lines = terminal_output.split("\n")
            terminal_placeholder.code("\n".join(lines[-20:]), language="bash")

            m = train_loss_pat.search(line)
            if m:
                step = int(m.group(1))
                total_steps = int(m.group(2))
                step_avg_ms = float(m.group(5))
                train_data.append({
                    "Step": step,
                    "train_loss": float(m.group(3)),
                })

                # Progress: use wallclock if set, otherwise step count
                elapsed = time.time() - t0
                if max_wall and max_wall > 0:
                    pct = min(elapsed / max_wall, 1.0)
                    remaining = max(max_wall - elapsed, 0)
                    eta_str = f"{remaining/60:.1f} min left"
                else:
                    pct = min(step / total_steps, 1.0) if total_steps > 0 else 0
                    remaining_steps = total_steps - step
                    remaining = remaining_steps * step_avg_ms / 1000
                    eta_str = f"{remaining/60:.1f} min left"
                progress_bar.progress(pct, text=f"Step {step}/{total_steps} | {eta_str} | {step_avg_ms:.0f}ms/step")

                if len(train_data) >= 2:
                    chart_placeholder.line_chart(
                        pd.DataFrame(train_data).set_index("Step")
                    )

        process.stdout.close()
        process.wait()
        duration = time.time() - t0
        progress_bar.progress(1.0, text="Done")

        # Parse log — start_baseline.sh already saves to run_history.json
        log_path = f"./logs/{run_id}.txt"
        parsed = run_tracker.parse_log(log_path)

        if process.returncode == 0 and parsed["final_bpb"] is not None:
            st.success(
                f"Training complete! final_bpb={parsed['final_bpb']:.4f} "
                f"| {parsed['steps']} steps | {duration/60:.1f} min "
                f"| step_avg={parsed['step_avg']:.1f}ms"
            )
        elif process.returncode == 0:
            st.warning("Run finished but no final_bpb found (wallclock cap hit before eval?).")
        else:
            st.error(f"Run failed with exit code {process.returncode}")

# ──────────────────────────────────────────────
# TAB 2: Run History
# ──────────────────────────────────────────────
with tab2:
    st.header("Run History")
    runs = run_tracker.load_runs()

    if not runs:
        st.info("No runs recorded yet. Launch a training run first.")
    else:
        sort_option = st.selectbox(
            "Sort by",
            ["Date (newest)", "BPB (best)", "Steps", "Duration"],
            index=0,
        )

        def _parse_finished(r):
            raw = r.get("finished_at")
            if raw and isinstance(raw, str):
                try:
                    return datetime.fromisoformat(raw)
                except Exception:
                    pass
            return datetime.min.replace(tzinfo=timezone.utc)

        if sort_option == "Date (newest)":
            runs.sort(key=_parse_finished, reverse=True)
        elif sort_option == "BPB (best)":
            runs.sort(key=lambda r: r.get("final_bpb") if r.get("final_bpb") is not None else float("inf"))
        elif sort_option == "Steps":
            runs.sort(key=lambda r: r.get("steps_completed", 0) or 0, reverse=True)
        elif sort_option == "Duration":
            runs.sort(key=lambda r: r.get("duration_seconds", 0) or 0, reverse=True)

        # Table
        rows = []
        for r in runs:
            p = r.get("params", {})
            bpb = r.get("final_bpb")
            bpb_str = f"{bpb:.4f}" if isinstance(bpb, (int, float)) and bpb is not None else "N/A"
            dur = r.get("duration_seconds", 0)
            dur_str = f"{dur/60:.1f} min" if dur else "?"
            art = r.get("artifact_size_bytes")
            if art is not None:
                art_mb = art / 1_000_000
                art_ok = "Y" if art <= 16_000_000 else "X"
                art_str = f"{art_mb:.1f}MB {art_ok}"
            else:
                art_str = "?"
            started_raw = r.get("started_at")
            finished_raw = r.get("finished_at")
            # Estimate started_at from finished_at - duration if missing
            if not started_raw and finished_raw and dur:
                try:
                    fin_dt = datetime.fromisoformat(finished_raw)
                    started_raw = (fin_dt - timedelta(seconds=dur)).isoformat()
                except Exception:
                    pass
            started = fmt_timestamp(started_raw)
            finished = fmt_timestamp(finished_raw)
            rows.append({
                "Run ID": r.get("run_id", "?"),
                "Layers": p.get("NUM_LAYERS", "?"),
                "Matrix LR": p.get("MATRIX_LR", "?"),
                "Final BPB": bpb_str,
                "Artifact": art_str,
                "Steps": r.get("steps_completed", "?"),
                "Duration": dur_str,
                "Started": started,
                "Finished": finished,
                "Status": r.get("status", "?"),
            })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Charts
        scored_runs = [r for r in runs if r.get("final_bpb") is not None]
        if scored_runs:
            col1, col2 = st.columns(2)

            with col1:
                st.subheader("BPB by Trial")
                chart_data = pd.DataFrame([
                    {"Trial": i + 1, "val_bpb": r["final_bpb"]}
                    for i, r in enumerate(scored_runs)
                ]).set_index("Trial")
                st.scatter_chart(chart_data)

            with col2:
                st.subheader("BPB Comparison")
                bar_data = pd.DataFrame([
                    {"Run": r["run_id"][:20], "val_bpb": r["final_bpb"]}
                    for r in scored_runs[:15]
                ]).set_index("Run")
                st.bar_chart(bar_data)

            # Best run details
            best = scored_runs[0]
            with st.expander(f"Best Run: {best['run_id']} (BPB: {best['final_bpb']:.4f})"):
                st.json(best.get("params", {}))

# ──────────────────────────────────────────────
# TAB 3: Autoresearch
# ──────────────────────────────────────────────
with tab3:
    st.header("Autoresearch (Optuna)")
    st.markdown(
        "Launches headless Optuna search as a background process. "
        "Results are saved to run_history.json and viewable in the Run History tab."
    )

    col1, col2 = st.columns(2)
    with col1:
        n_trials = st.number_input("Number of trials", min_value=1, value=5, key="auto_trials")
    with col2:
        mins = st.number_input(
            "Minutes per trial",
            min_value=10, value=80, key="auto_mins",
            help="At ~9s/step: 80 min = ~530 steps. Matches rough training dynamics.",
        )

    col_start, col_stop_auto = st.columns([3, 1])
    with col_start:
        start_auto = st.button("Start Autoresearch", type="primary")
    with col_stop_auto:
        stop_auto = st.button("Stop Autoresearch", type="secondary", key="stop_auto")

    if stop_auto:
        Path("STOP_AUTORESEARCH").touch()
        st.warning("Stop signal sent. Autoresearch will stop after the current trial finishes.")

    if start_auto:
        Path("STOP_AUTORESEARCH").unlink(missing_ok=True)
        subprocess.Popen(
            ["python3", "autoresearch.py",
             "--trials", str(int(n_trials)),
             "--minutes-per-trial", str(int(mins))],
            stdout=open("logs/autoresearch_stdout.log", "a"),
            stderr=subprocess.STDOUT,
        )
        st.success(
            f"Autoresearch launched in background: {int(n_trials)} trials x {int(mins)} min. "
            "Check the Run History tab for results as they come in."
        )

    # Show Optuna study stats if DB exists
    if Path("optuna_study.db").exists():
        st.subheader("Optuna Study Summary")
        try:
            import optuna
            study = optuna.load_study(
                study_name="parameter_golf_sota",
                storage="sqlite:///optuna_study.db",
            )
            st.metric("Total Trials", len(study.trials))
            completed = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
            if completed:
                st.metric("Best BPB", f"{study.best_value:.4f}")
                with st.expander("Best Parameters"):
                    st.json(study.best_params)
        except Exception as e:
            st.warning(f"Could not load Optuna study: {e}")

    # Check if autoresearch is actively running
    ps_auto = subprocess.run(
        ["bash", "-c", "ps aux | grep 'autoresearch.py' | grep -v grep"],
        capture_output=True, text=True,
    )
    if ps_auto.stdout.strip():
        st.info("Autoresearch is currently running.")
        if Path("optuna_study.db").exists():
            try:
                import optuna as _optuna
                _study = _optuna.load_study(
                    study_name="parameter_golf_sota",
                    storage="sqlite:///optuna_study.db",
                )
                done = len([t for t in _study.trials if t.state == _optuna.trial.TrialState.COMPLETE])
                # Estimate: last requested n_trials from the log
                st.progress(min(done / max(int(n_trials), 1), 1.0),
                            text=f"Completed {done} trials")
            except Exception:
                pass
    elif Path("STOP_AUTORESEARCH").exists():
        st.warning("Stop signal is pending (autoresearch may have already stopped).")
        Path("STOP_AUTORESEARCH").unlink(missing_ok=True)

    # Refresh button for run history
    if st.button("Refresh Results"):
        st.rerun()

    # Show autoresearch log tail
    auto_log = Path("logs/autoresearch_stdout.log")
    auto_log_content = ""
    if auto_log.exists():
        auto_log_content = auto_log.read_text().strip()
    # If stdout log is empty, show the latest individual autoresearch/sota log
    if not auto_log_content:
        import glob as _glob
        candidates = sorted(_glob.glob("logs/sota_optuna_*.txt") + _glob.glob("logs/optuna_trial_*.txt") + _glob.glob("logs/autoresearch_*.txt"))
        candidates = [c for c in candidates if not c.endswith("_stdout.log")]
        if candidates:
            latest = candidates[-1]
            auto_log_content = Path(latest).read_text().strip()
            auto_log_content = f"[Showing tail of {latest}]\n" + "\n".join(auto_log_content.split("\n")[-30:])
    if auto_log_content:
        with st.expander("Autoresearch Log (last 30 lines)"):
            lines = auto_log_content.split("\n")
            st.code("\n".join(lines[-30:]), language="bash")
    else:
        with st.expander("Autoresearch Log (last 30 lines)"):
            st.info("No autoresearch logs found yet.")

# ──────────────────────────────────────────────
# TAB 4: System Info
# ──────────────────────────────────────────────
with tab4:
    st.header("System Info")

    if st.button("Refresh", key="sysinfo_refresh"):
        st.rerun()

    # GPU temperature and power
    st.subheader("GPU Thermal")
    try:
        smi_out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True,
        )
        if smi_out.returncode == 0 and smi_out.stdout.strip():
            parts = [p.strip() for p in smi_out.stdout.strip().split(",")]
            gpu_temp = int(parts[0])
            gpu_power = float(parts[1])
            col_t, col_p = st.columns(2)
            col_t.metric("GPU Temp", f"{gpu_temp} C")
            col_p.metric("Power Draw", f"{gpu_power:.1f} W")
            if gpu_temp > 80:
                st.error(f"GPU temperature {gpu_temp} C is CRITICAL (>80 C). Risk of thermal throttling!")
            elif gpu_temp > 70:
                st.warning(f"GPU temperature {gpu_temp} C is elevated (>70 C). Monitor closely.")
    except Exception as e:
        st.warning(f"Could not read GPU temp: {e}")

    # GPU info
    st.subheader("GPU Details")
    try:
        import torch
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            cc = torch.cuda.get_device_capability(0)
            st.metric("GPU", gpu_name)
            st.metric("Compute Capability", f"sm_{cc[0]}{cc[1]}")
            st.metric("PyTorch Version", torch.__version__)
            st.metric("CUDA Version", torch.version.cuda or "N/A")
            st.metric("torch.compile", "Available")
        else:
            st.error("CUDA not available")
    except Exception as e:
        st.error(f"Could not get GPU info: {e}")

    # Disk usage
    st.subheader("Disk Usage")
    data_dir = Path("./data/datasets")
    if data_dir.exists():
        usage = shutil.disk_usage(data_dir)
        total_gb = usage.total / (1024 ** 3)
        used_gb = usage.used / (1024 ** 3)
        free_gb = usage.free / (1024 ** 3)
        st.text(f"Total: {total_gb:.1f} GB | Used: {used_gb:.1f} GB | Free: {free_gb:.1f} GB")

        # Size of dataset dir
        dataset_size = sum(f.stat().st_size for f in data_dir.rglob("*") if f.is_file())
        st.text(f"Dataset dir size: {dataset_size / (1024**2):.0f} MB")
    else:
        st.warning("Data directory not found")

    # Active training processes
    st.subheader("Active Processes")
    try:
        ps_out = subprocess.run(
            ["bash", "-c", "ps aux | grep -E 'torchrun|train_gpt|autoresearch' | grep -v grep"],
            capture_output=True, text=True,
        )
        if ps_out.stdout.strip():
            st.code(ps_out.stdout.strip(), language="bash")
        else:
            st.info("No training processes running")
    except Exception:
        st.info("Could not check processes")

    # Run history stats
    st.subheader("Run Statistics")
    runs = run_tracker.load_runs()
    st.text(f"Total runs recorded: {len(runs)}")
    completed = [r for r in runs if r.get("status") == "completed" and r.get("final_bpb")]
    if completed:
        best = completed[0]
        st.text(f"Best BPB: {best['final_bpb']:.4f} ({best['run_id']})")
