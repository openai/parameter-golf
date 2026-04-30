#!/usr/bin/env bash
# Shepherd Monitor — single-paste bootstrap for pod 0fyktt2ztu6151.
# Paste this entire file into the pod's terminal (Jupyter > New > Terminal, or web terminal).
# Idempotent: re-running it kills old listeners and restarts clean.

set -e
ROOM="/workspace/telemetry_room"
mkdir -p "$ROOM"

########################  dashboard.html  ########################
cat > "$ROOM/dashboard.html" <<'HTML_EOF'
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>SHEPHERD — Mission Control</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { font-family: "JetBrains Mono", Consolas, monospace; background:#05070f; color:#d9e4ff; padding:20px; }
  h1 { font-family: Arial, sans-serif; font-weight:900; letter-spacing:.08em; font-size:26px; color:#8fb7ff; margin-bottom:14px; }
  .sub { font-size:12px; color:#6b7da0; margin-bottom:18px; }
  .grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:12px; }
  .card { background:#0d1326; border:1px solid #1e2a50; border-radius:12px; padding:14px; }
  .label { font-size:10px; letter-spacing:.12em; text-transform:uppercase; color:#6b7da0; margin-bottom:6px; }
  .big { font-size:28px; font-weight:700; color:#e8eefc; }
  .mid { font-size:16px; font-weight:600; color:#c7d4f0; }
  .row { display:grid; grid-template-columns: repeat(3, 1fr); gap:12px; margin-top:12px; }
  .gpu-grid { display:grid; grid-template-columns: repeat(4, 1fr); gap:10px; margin-top:10px; font-size:12px; }
  .gpu-box { background:#07101f; border:1px solid #1a2647; border-radius:8px; padding:8px; }
  .probe-row { display:flex; gap:16px; margin-top:8px; font-size:14px; }
  .probe-row span { padding:4px 10px; background:#07101f; border:1px solid #1a2647; border-radius:6px; }
  pre { white-space:pre-wrap; word-break:break-word; font-size:12px; line-height:1.4; color:#8fa3c9; }
  .dot { display:inline-block; width:8px; height:8px; border-radius:50%; margin-right:6px; vertical-align:middle; }
  .dot.live  { background:#6ee7a8; box-shadow:0 0 8px #6ee7a8; }
  .dot.stale { background:#ffd166; }
  .dot.dead  { background:#ff7a7a; }
</style>
</head>
<body>
  <h1><span class="dot live" id="livedot"></span>SHEPHERD &mdash; MISSION CONTROL</h1>
  <div class="sub" id="sub">waiting for telemetry...</div>
  <div class="grid">
    <div class="card"><div class="label">Run</div><div class="mid" id="run_id">&mdash;</div></div>
    <div class="card"><div class="label">Hardware</div><div class="mid" id="hardware">&mdash;</div></div>
    <div class="card"><div class="label">Step</div><div class="big" id="step">&mdash;</div></div>
    <div class="card"><div class="label">Elapsed (s)</div><div class="big" id="elapsed">&mdash;</div></div>
  </div>
  <div class="row">
    <div class="card"><div class="label">Train Loss</div><div class="big" id="train_loss">&mdash;</div></div>
    <div class="card"><div class="label">Val Loss</div><div class="big" id="val_loss">&mdash;</div></div>
    <div class="card"><div class="label">Est BPB</div><div class="big" id="est_bpb">&mdash;</div></div>
  </div>
  <div class="card" style="margin-top:12px;">
    <div class="label">Probes</div>
    <div class="probe-row">
      <span>init: <b id="probe_init">&mdash;</b></span>
      <span>d1: <b id="probe_d1">&mdash;</b></span>
      <span>d2: <b id="probe_d2">&mdash;</b></span>
      <span>d3: <b id="probe_d3">&mdash;</b></span>
    </div>
  </div>
  <div class="card" style="margin-top:12px;"><div class="label">GPUs</div><div class="gpu-grid" id="gpu_grid">&mdash;</div></div>
  <div class="card" style="margin-top:12px;"><div class="label">Last Log Line</div><pre id="log_line">&mdash;</pre></div>
  <div class="card" style="margin-top:12px;"><div class="label">Raw Payload</div><pre id="raw">&mdash;</pre></div>
<script>
let lastUpdate = 0;
function fmt(v) { return (v===null||v===undefined) ? "—" : v; }
async function refresh() {
  try {
    const res = await fetch('./telemetry.json?ts=' + Date.now(), { cache: 'no-store' });
    if (!res.ok) throw new Error('http ' + res.status);
    const t = await res.json();
    lastUpdate = Date.now();
    document.getElementById('run_id').textContent     = fmt(t.run_id || t.run_name);
    document.getElementById('hardware').textContent   = fmt(t.hardware);
    document.getElementById('step').textContent       = fmt(t.step);
    document.getElementById('elapsed').textContent    = fmt(t.elapsed_sec);
    document.getElementById('train_loss').textContent = fmt(t.train_loss ?? t.loss);
    document.getElementById('val_loss').textContent   = fmt(t.val_loss);
    document.getElementById('est_bpb').textContent    = fmt(t.est_bpb);
    document.getElementById('probe_init').textContent = fmt(t.probe_init);
    document.getElementById('probe_d1').textContent   = fmt(t.probe_d1);
    document.getElementById('probe_d2').textContent   = fmt(t.probe_d2);
    document.getElementById('probe_d3').textContent   = fmt(t.probe_d3);
    document.getElementById('log_line').textContent   = fmt(t.log_line);
    document.getElementById('raw').textContent        = JSON.stringify(t, null, 2);
    document.getElementById('sub').textContent        = 'updated ' + (t.updated_at || new Date().toISOString());
    const grid = document.getElementById('gpu_grid');
    grid.innerHTML = '';
    const n = t.gpu_count || (Array.isArray(t.gpu_util_pct) ? t.gpu_util_pct.length : 0);
    if (n > 0) {
      for (let i = 0; i < n; i++) {
        const box = document.createElement('div');
        box.className = 'gpu-box';
        const util = Array.isArray(t.gpu_util_pct) ? t.gpu_util_pct[i] : '—';
        const vram = Array.isArray(t.gpu_vram_used_mb) ? t.gpu_vram_used_mb[i] : '—';
        const temp = Array.isArray(t.gpu_temp_c) ? t.gpu_temp_c[i] : '—';
        const pwr  = Array.isArray(t.gpu_power_w) ? t.gpu_power_w[i] : '—';
        box.innerHTML = `<b>GPU ${i}</b> (${t.gpu_name || ''})<br>util: ${util}%<br>vram: ${vram} MB<br>temp: ${temp} &deg;C<br>power: ${pwr} W`;
        grid.appendChild(box);
      }
    } else { grid.textContent = 'no gpu data'; }
  } catch (e) { document.getElementById('log_line').textContent = 'fetch error: ' + e; }
}
function heartbeat() {
  const dot = document.getElementById('livedot');
  const age = Date.now() - lastUpdate;
  if (lastUpdate === 0 || age > 15000) dot.className = 'dot dead';
  else if (age > 5000) dot.className = 'dot stale';
  else dot.className = 'dot live';
}
refresh();
setInterval(refresh, 2000);
setInterval(heartbeat, 1000);
</script>
</body>
</html>
HTML_EOF

########################  telemetry_demo.py  ########################
cat > "$ROOM/telemetry_demo.py" <<'PY_EOF'
import json, math, random, time
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
PY_EOF

########################  seed telemetry.json  ########################
cat > "$ROOM/telemetry.json" <<'JSON_EOF'
{"run_id":"boot_test","run_name":"Monitor Boot Probe","step":0,"updated_at":"not started","log_line":"dashboard booted, waiting for writer","gpu_count":0,"gpu_util_pct":[],"gpu_vram_used_mb":[],"gpu_vram_total_mb":[],"gpu_temp_c":[],"gpu_power_w":[]}
JSON_EOF

########################  restart cleanly  ########################
pkill -f "http.server 8080" 2>/dev/null || true
pkill -f "telemetry_demo.py" 2>/dev/null || true
sleep 1

nohup python3 "$ROOM/telemetry_demo.py" > "$ROOM/telemetry_demo.log" 2>&1 &
echo "demo writer pid: $!"

nohup python3 -m http.server 8080 --directory "$ROOM" > "$ROOM/http8080.log" 2>&1 &
echo "http server pid: $!"

sleep 2
echo ""
echo "==== listeners on 8080 ===="
ss -ltnp 2>/dev/null | grep 8080 || netstat -ltnp 2>/dev/null | grep 8080 || true
echo ""
echo "==== local probe ===="
curl -s http://127.0.0.1:8080/telemetry.json | head -c 400
echo ""
echo ""
echo "Open this in your browser:"
echo "  https://0fyktt2ztu6151-8080.proxy.runpod.net/dashboard.html"
