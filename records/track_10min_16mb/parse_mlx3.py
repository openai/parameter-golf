import re

with open('2026-03-25_Ternary_Feedback_TTT/logs/mlx_reasoner.txt', 'rb') as fp:
    content = fp.read().decode('utf-8', errors='ignore').replace('\r', '\n')

lines = content.split('\n')
runs = []
curr_run = []
for line in lines:
    if "param" in line or "model_params" in line or "RUNNING" in line:
        if curr_run:
            runs.append(curr_run)
            curr_run = []
    curr_run.append(line)
if curr_run:
    runs.append(curr_run)

res = []
for idx, run_lines in enumerate(runs):
    max_train_time = 0
    final_bpb = None
    final_step = None
    run_info = ""
    for line in run_lines:
        if "param" in line or "arch" in line or "model_params" in line:
            run_info += line + " "
        m = re.search(r'train_time:(\d+)ms', line)
        if m:
            t = int(m.group(1))
            if t > max_train_time: max_train_time = t
            
        m_bpb = re.search(r'val_bpb\s*[:=]\s*([\d\.]+)', line)
        if m_bpb:
            final_bpb = float(m_bpb.group(1))
            final_step = line
            
    if 550000 <= max_train_time <= 650000 and final_bpb is not None:
        res.append((final_bpb, max_train_time, run_info.strip()))

res.sort()
print("Top 10-minute MLX runs in mlx_reasoner.txt:")
for i, r in enumerate(res):
    print(f"{i+1}. BPB: {r[0]:.4f} | Time: {r[1]/1000:.1f}s | Config: {r[2][:150]}")
