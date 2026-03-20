#!/usr/bin/env python3
"""Read-only progress monitor for Parameter Golf.

Queries GPU state LIVE via nvidia-smi over SSH (not from cached CSVs).
Auto-discovers any active training by finding the most recent log file.
Also reads search harness JSON if available.

Usage:
    # Remote (queries GPU directly, finds active logs automatically):
    python3 tools/monitor.py --ssh "ssh -p 27846 root@ssh7.vast.ai" \
        --workdir /workspace/parameter-golf

    # Local:
    python3 tools/monitor.py --workdir /home/spark-advantage/parameter-golf
"""

import argparse
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def run_remote(ssh_cmd: str, cmd: str, timeout: int = 10) -> str | None:
    try:
        result = subprocess.run(
            ssh_cmd.split() + [cmd],
            capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, Exception):
        pass
    return None


def run_local(cmd: str, timeout: int = 10) -> str | None:
    try:
        result = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout,
        )
        if result.returncode == 0:
            return result.stdout
    except (subprocess.TimeoutExpired, Exception):
        pass
    return None


def run_cmd(ssh_cmd: str | None, cmd: str, timeout: int = 10) -> str | None:
    if ssh_cmd:
        return run_remote(ssh_cmd, cmd, timeout)
    return run_local(cmd, timeout)


def read_json_file(ssh_cmd: str | None, path: str) -> dict | None:
    raw = run_cmd(ssh_cmd, f"cat {path}")
    if raw:
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            pass
    return None


def format_bytes(n: int | float) -> str:
    if n < 1024:
        return f"{n} B"
    if n < 1024 * 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n / (1024 * 1024):.1f} MB"


def format_duration(seconds: float) -> str:
    if seconds < 60:
        return f"{seconds:.0f}s"
    if seconds < 3600:
        return f"{seconds / 60:.1f}m"
    return f"{seconds / 3600:.1f}h"


def make_bar(fraction: float, width: int = 30) -> str:
    filled = int(fraction * width)
    return f"[{'=' * filled}{' ' * (width - filled)}]"


STEP_RE = re.compile(
    r"step:(\d+)/(\d+)\s+(?:train_loss:[\d.]+\s+)?(?:val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+)?train_time:(\d+)ms\s+step_avg:([\d.]+)ms"
)
QUANT_RE = re.compile(r"quant_summary\s+int8_bpb:([\d.]+)\s+int6_bpb:([\d.]+)\s+int8_sz:(\d+)\s+int6_sz:(\d+)")
ROUNDTRIP_RE = re.compile(r"roundtrip_(int\d+\+\w+)\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)\s+eval_time:(\d+)ms")
SLIDING_RE = re.compile(r"sliding_window\s+val_loss:([\d.]+)\s+val_bpb:([\d.]+)")
SLIDING_PROG_RE = re.compile(r"sliding_window:progress.*approx_pct:([\d.]+).*windows_per_sec:([\d.]+)")
STOP_RE = re.compile(r"stopping_early:")
PARAMS_RE = re.compile(r"model_params:(\d+)")
WD_RE = re.compile(r"weight_decay\s+token:([\d.]+)\s+head:([\d.]+)\s+muon:([\d.]+)\s+scalar:([\d.]+)")


def parse_training_log(text: str) -> dict:
    """Parse key metrics from a training log."""
    info: dict = {}
    last_step = None
    last_val = None
    for line in text.splitlines():
        m = STEP_RE.search(line)
        if m:
            step, total = int(m.group(1)), int(m.group(2))
            last_step = {"step": step, "total": total, "time_ms": int(m.group(5)), "avg_ms": float(m.group(6))}
            if m.group(3):
                last_val = {"val_loss": float(m.group(3)), "val_bpb": float(m.group(4)), "step": step, "total": total}
        m = QUANT_RE.search(line)
        if m:
            info["quant"] = {"int8_bpb": float(m.group(1)), "int6_bpb": float(m.group(2)),
                             "int8_sz": int(m.group(3)), "int6_sz": int(m.group(4))}
        m = ROUNDTRIP_RE.search(line)
        if m:
            info.setdefault("roundtrips", {})[m.group(1)] = {"bpb": float(m.group(3)), "time_ms": int(m.group(4))}
        m = SLIDING_RE.search(line)
        if m and "sliding_window:progress" not in line:
            info["sliding"] = {"bpb": float(m.group(2)), "complete": True}
        m = SLIDING_PROG_RE.search(line)
        if m:
            info["sliding_progress"] = {"pct": float(m.group(1)), "wps": float(m.group(2))}
        if STOP_RE.search(line):
            info["stopped_early"] = True
        m = PARAMS_RE.search(line)
        if m:
            info["params"] = int(m.group(1))
        m = WD_RE.search(line)
        if m:
            info["wd"] = {"token": m.group(1), "muon": m.group(3), "scalar": m.group(4)}
    if last_step:
        info["step"] = last_step
    if last_val:
        info["val"] = last_val
    return info


def render_status(ssh_cmd: str | None, workdir: str) -> str:
    lines = []
    now_str = datetime.now().strftime("%H:%M:%S")
    mode = "remote" if ssh_cmd else "local"
    lines.append(f"\033[1mParameter Golf Monitor\033[0m  [{mode}]  {now_str}")
    lines.append(f"  workdir: {workdir}")
    lines.append("")

    # --- LIVE GPU STATUS (direct nvidia-smi query, not cached) ---
    gpu_raw = run_cmd(ssh_cmd,
        "nvidia-smi --query-gpu=index,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,clocks.current.sm,clocks.max.sm "
        "--format=csv,noheader,nounits")
    if gpu_raw:
        lines.append(f"\033[1;35mGPU Status (live):\033[0m")
        temps, powers, utils, clocks_cur, clocks_max = [], [], [], [], []
        for row in gpu_raw.strip().splitlines():
            parts = [p.strip() for p in row.split(",")]
            if len(parts) >= 8:
                idx, util, mem_used, mem_total, temp, power, clk, clk_max = parts[:8]
                throttle = ""
                try:
                    c, cm = int(float(clk)), int(float(clk_max))
                    if cm > 0 and c < cm - 30:
                        throttle = f"  \033[33m-{cm - c}MHz\033[0m"
                    clocks_cur.append(c)
                    clocks_max.append(cm)
                except ValueError:
                    pass
                lines.append(f"  GPU {idx}: {util:>3}% util  {mem_used:>6}/{mem_total} MB  {temp:>3}C  {power:>7}W  clk {clk}/{clk_max}{throttle}")
                try: temps.append(float(temp))
                except ValueError: pass
                try: powers.append(float(power))
                except ValueError: pass
                try: utils.append(float(util))
                except ValueError: pass

        if temps:
            t_min, t_max, t_avg = min(temps), max(temps), sum(temps) / len(temps)
            t_spread = t_max - t_min
            spread_color = "\033[31m" if t_spread >= 10 else "\033[33m" if t_spread >= 7 else "\033[32m"
            max_color = "\033[31m" if t_max >= 80 else "\033[33m" if t_max >= 75 else "\033[32m"
            lines.append(f"  -------")
            lines.append(f"  Temp:  min={t_min:.0f}C  max={max_color}{t_max:.0f}C\033[0m  "
                         f"avg={t_avg:.1f}C  spread={spread_color}{t_spread:.0f}C\033[0m"
                         f"{'  \033[31m!! RMA RISK (spread>=10C)\033[0m' if t_spread >= 10 else ''}")
        if powers:
            lines.append(f"  Power: total={sum(powers):.0f}W  avg={sum(powers)/len(powers):.0f}W/GPU")
        if utils:
            avg_util = sum(utils) / len(utils)
            if avg_util < 5:
                lines.append(f"  \033[31;1m!! GPUs IDLE — {avg_util:.0f}% avg util !!\033[0m")
        lines.append("")
    else:
        lines.append("\033[33mGPU: unable to query nvidia-smi\033[0m\n")

    # --- ECC check ---
    ecc_raw = run_cmd(ssh_cmd,
        "nvidia-smi --query-gpu=index,ecc.errors.corrected.volatile.total,ecc.errors.uncorrected.volatile.total "
        "--format=csv,noheader,nounits")
    if ecc_raw:
        ecc_issues = []
        for row in ecc_raw.strip().splitlines():
            parts = [p.strip() for p in row.split(",")]
            if len(parts) >= 3:
                idx, corr, uncorr = parts[:3]
                try:
                    c, u = int(corr), int(uncorr)
                    if c > 0 or u > 0:
                        color = "\033[31m" if u > 0 else "\033[33m"
                        ecc_issues.append(f"  {color}GPU {idx}: {c} corrected, {u} uncorrected ECC errors\033[0m")
                except ValueError:
                    pass
        if ecc_issues:
            lines.append(f"\033[1;31mECC Errors:\033[0m")
            lines.extend(ecc_issues)
            lines.append("")

    # --- Active training processes ---
    procs_raw = run_cmd(ssh_cmd,
        "ps -eo pid,etimes,pcpu,rss,cmd --no-headers | grep train_gpt | grep -v grep")
    if procs_raw and procs_raw.strip():
        proc_lines = procs_raw.strip().splitlines()
        # Group by unique command to show rank count instead of listing every DDP rank
        cmd_groups: dict[str, list] = {}
        for pl in proc_lines:
            parts = pl.split(None, 4)
            if len(parts) >= 5:
                pid, elapsed, cpu, rss, cmd = parts
                cmd_key = cmd.split("/")[-1].split()[0] if "/" in cmd else cmd.split()[0]
                cmd_groups.setdefault(cmd_key, []).append({"pid": pid, "elapsed": elapsed, "cpu": float(cpu), "rss": int(rss)})
        lines.append(f"\033[1;36mActive Processes ({len(proc_lines)} total):\033[0m")
        for cmd_key, procs in cmd_groups.items():
            total_cpu = sum(p["cpu"] for p in procs)
            total_rss = sum(p["rss"] for p in procs) // 1024
            max_elapsed = max(p["elapsed"] for p in procs)
            pids = ", ".join(p["pid"] for p in procs[:4])
            if len(procs) > 4:
                pids += f" +{len(procs)-4}"
            lines.append(f"  {cmd_key}: {len(procs)} ranks  {total_cpu:.0f}% CPU  {total_rss}MB RSS  {max_elapsed}s  PIDs [{pids}]")
        lines.append("")
    else:
        lines.append("\033[33mNo active train_gpt.py processes\033[0m\n")

    # --- Find and parse the most recent training log ---
    log_raw = run_cmd(ssh_cmd, f"ls -t {workdir}/logs/*.txt 2>/dev/null | head -5")
    if log_raw and log_raw.strip():
        log_files = log_raw.strip().splitlines()
        lines.append(f"\033[1;33mRecent Logs:\033[0m")
        for lf in log_files:
            name = lf.split("/")[-1]
            # Check if this log is still being written to (active)
            stat_raw = run_cmd(ssh_cmd, f"stat -c '%Y' {lf} 2>/dev/null")
            age_str = ""
            active = False
            if stat_raw:
                try:
                    mtime = int(stat_raw.strip())
                    now_ts = int(time.time())
                    age = now_ts - mtime
                    age_str = f" ({format_duration(age)} ago)"
                    active = age < 30
                except ValueError:
                    pass
            marker = "\033[32m[ACTIVE]\033[0m" if active else "\033[2m[done]\033[0m"
            lines.append(f"  {marker} {name}{age_str}")

        # Parse the most recent active log (or just the most recent)
        active_log = None
        for lf in log_files:
            stat_raw = run_cmd(ssh_cmd, f"stat -c '%Y' {lf} 2>/dev/null")
            if stat_raw:
                try:
                    age = int(time.time()) - int(stat_raw.strip())
                    if age < 30:
                        active_log = lf
                        break
                except ValueError:
                    pass
        target_log = active_log or log_files[0]
        log_text = run_cmd(ssh_cmd, f"tail -200 {target_log}", timeout=15)
        if log_text:
            info = parse_training_log(log_text)
            log_name = target_log.split("/")[-1]
            lines.append("")
            label = "\033[1;32mActive Run" if active_log else "\033[1;33mLast Run"
            lines.append(f"{label}:\033[0m {log_name}")

            if "params" in info:
                lines.append(f"  Model:     {info['params']:,} params")
            if "wd" in info:
                wd = info["wd"]
                lines.append(f"  WD:        muon={wd['muon']}  scalar={wd['scalar']}  token={wd['token']}")

            if "step" in info:
                s = info["step"]
                frac = s["step"] / s["total"] if s["total"] > 0 else 0
                bar = make_bar(frac)
                elapsed = format_duration(s["time_ms"] / 1000)
                lines.append(f"  Training:  {bar} {s['step']}/{s['total']} ({frac*100:.1f}%)  {s['avg_ms']:.1f}ms/step  elapsed={elapsed}")

            if "val" in info:
                v = info["val"]
                lines.append(f"  Val:       bpb={v['val_bpb']:.4f}  loss={v['val_loss']:.4f}  (step {v['step']})")

            if info.get("stopped_early"):
                lines.append(f"  \033[33mStopped early (wallclock cap)\033[0m")

            if "quant" in info:
                q = info["quant"]
                lines.append(f"  Quant:     int8={q['int8_bpb']:.4f} ({format_bytes(q['int8_sz'])})  "
                             f"int6={q['int6_bpb']:.4f} ({format_bytes(q['int6_sz'])})")

            for label, rt in info.get("roundtrips", {}).items():
                lines.append(f"  Roundtrip: {label}  bpb={rt['bpb']:.4f}  eval={format_duration(rt['time_ms']/1000)}")

            if "sliding" in info:
                sl = info["sliding"]
                lines.append(f"  Sliding:   \033[1;32mCOMPLETE\033[0m  bpb={sl['bpb']:.4f}")
            elif "sliding_progress" in info:
                sp = info["sliding_progress"]
                bar = make_bar(sp["pct"] / 100)
                lines.append(f"  Sliding:   {bar} {sp['pct']:.1f}%  {sp['wps']:.0f} w/s")

        lines.append("")

    # --- Search harness state (if any) ---
    search_dirs_raw = run_cmd(ssh_cmd, f"find {workdir} -name 'current_run.json' -newer /tmp/.pg_monitor_epoch 2>/dev/null; "
                                       f"find {workdir} -name 'runs.jsonl' 2>/dev/null | head -5")
    if not search_dirs_raw:
        # Try without the -newer filter
        search_dirs_raw = run_cmd(ssh_cmd, f"find {workdir} -name 'runs.jsonl' 2>/dev/null | head -5")

    if search_dirs_raw:
        jsonl_files = [f for f in search_dirs_raw.strip().splitlines() if f.endswith("runs.jsonl")]
        for jf in jsonl_files:
            runs_raw = run_cmd(ssh_cmd, f"cat {jf}")
            if runs_raw and runs_raw.strip():
                completed = []
                for line in runs_raw.strip().split("\n"):
                    try:
                        completed.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
                if completed:
                    search_name = jf.split("/")[-2] if "/" in jf else "?"
                    lines.append(f"\033[1;33mSearch Runs ({search_name}, {len(completed)} completed):\033[0m")
                    lines.append(f"  {'Run ID':<35} {'Status':<12} {'Pre bpb':>8} {'int6':>8} {'Slide':>8} {'Size':>10}")
                    lines.append(f"  {'-'*35} {'-'*12} {'-'*8} {'-'*8} {'-'*8} {'-'*10}")
                    for r in completed:
                        rid = r.get("run_id", "?")[:35]
                        st = r.get("status", "?")[:12]
                        pre = r.get("terminal_prequant_bpb")
                        i6 = r.get("final_int6_bpb")
                        sl = r.get("sliding_window_bpb")
                        sz = r.get("int6_artifact_bytes")
                        color = "\033[32m" if st == "completed" else "\033[31m"
                        pre_s = f"{pre:.4f}" if pre else "N/A"
                    i6_s = f"{i6:.4f}" if i6 else "N/A"
                    sl_s = f"{sl:.4f}" if sl else "N/A"
                    sz_s = format_bytes(sz) if sz else "N/A"
                    lines.append(f"  {rid:<35} {color}{st:<12}\033[0m {pre_s:>8} {i6_s:>8} {sl_s:>8} {sz_s:>10}")
                    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Parameter Golf monitor (read-only, live GPU queries)")
    parser.add_argument("--ssh", default=None, help="SSH command for remote (e.g., 'ssh -p 27846 root@ssh7.vast.ai')")
    parser.add_argument("--workdir", default="/home/spark-advantage/parameter-golf", help="Parameter golf workdir")
    parser.add_argument("--interval", type=int, default=10, help="Refresh interval in seconds (default: 10)")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    if args.once:
        print(render_status(args.ssh, args.workdir))
        return

    try:
        while True:
            os.system("clear")
            print(render_status(args.ssh, args.workdir))
            print(f"\n\033[2mRefreshing every {args.interval}s. Ctrl+C to quit.\033[0m")
            time.sleep(args.interval)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
