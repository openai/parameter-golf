"""
Shepherd Monitor — demo telemetry writer.
Writes a live-looking telemetry.json into /workspace/telemetry_room every 2 seconds
so the dashboard has something to display before the real training loop is wired in.

Replace with shepherd_telemetry.py output when you're ready to wire real runs.
"""
import json
import math
import random
import time
from datetime import datetime, timezone

OUT = "/workspace/telemetry_room/telemetry.json"

step = 0
while True:
    step += 1
    gpu_count = 3
    payload = {
        "run_id": "demo_heartbeat",
        "run_name": "Shepherd Monitor Demo",
        "hardware": "demo / CPU pod",
        "step": step,
        "iteration_max": 6000,
        "batch_size": 48,
        "total_params": 6363306,
        "train_loss": round(6.9 - min(step * 0.01, 3.2) + random.uniform(-0.03, 0.03), 4),
        "val_loss":   round(6.8 - min(step * 0.009, 3.0) + random.uniform(-0.03, 0.03), 4),
        "est_bpb":    round(2.1 - min(step * 0.003, 1.1) + random.uniform(-0.01, 0.01), 4),
        "elapsed_sec": round(step * 2.0, 1),
        "step_avg_ms": 120.0,
        "gpu_count": gpu_count,
        "gpu_name": "demo-gpu",
        "gpu_util_pct":      [int(50 + 30 * math.sin((step + i) / 7)) for i in range(gpu_count)],
        "gpu_vram_used_mb":  [2000 + (step % 50) + i * 10 for i in range(gpu_count)],
        "gpu_vram_total_mb": [24564] * gpu_count,
        "gpu_temp_c":        [40 + (step % 7) + i for i in range(gpu_count)],
        "gpu_power_w":       [round(70 + (step % 9) * 1.2 + i, 2) for i in range(gpu_count)],
        "probe_init": 0.22,
        "probe_d1":   round(0.45 + 0.08 * math.sin(step / 13), 4),
        "probe_d2":   round(0.48 + 0.08 * math.sin(step / 11), 4),
        "probe_d3":   round(0.51 + 0.08 * math.sin(step / 9), 4),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "log_line":   f"demo heartbeat step={step}",
    }
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    time.sleep(2)
