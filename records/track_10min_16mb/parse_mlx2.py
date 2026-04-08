import os, re

res = []
# get list of logs
from glob import glob
files = glob('*/logs/mlx_*.txt') + glob('*/hybrid_mac*.log') + glob('*/*.log')

for f in files:
    try:
        with open(f, 'rb') as fp:
            content = fp.read().decode('utf-8', errors='ignore').replace('\r', '\n')
            
        lines = content.split('\n')
        max_train_time = 0
        final_bpb = None
        
        for line in lines:
            m = re.search(r'train_time:(\d+)ms', line)
            if m:
                t = int(m.group(1))
                if t > max_train_time: max_train_time = t
                
            m_bpb = re.search(r'val_bpb\s*[:=]\s*([\d\.]+)', line)
            if m_bpb:
                final_bpb = float(m_bpb.group(1))
                
        # we found tka-h was exactly 601142ms. Other models might be around 600s
        # some benchmark logs are exact 10 mins (600,000ms).
        if 500000 <= max_train_time <= 700000 and final_bpb is not None:
            res.append((final_bpb, f, max_train_time))
    except Exception as e:
        pass

res.sort()
for r in res:
    print(f"BPB: {r[0]:.4f} | Time: {r[2]/1000:.1f}s | File: {r[1]}")
