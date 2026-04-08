import os, re

res = []

for root, dirs, files in os.walk('.'):
    for fn in files:
        if ('mlx' in fn and fn.endswith('.txt')) or ('hybrid_mac' in fn and fn.endswith('.log')):
            f = os.path.join(root, fn)
            with open(f, 'rb') as fp:
                content = fp.read().decode('utf-8', errors='ignore')
            
            # replace \r with \n
            content = content.replace('\r', '\n')
            lines = content.split('\n')
            
            is_10min = False
            max_train_time = 0
            final_bpb = None
            
            for line in lines:
                if "EXACTLY 10 MINS" in line or "10-MIN RUN START" in line or "1hr_marathon" in f:
                    pass # just rely on traintime
                
                m = re.search(r'train_time:(\d+)ms', line)
                if m:
                    t = int(m.group(1))
                    if t > max_train_time:
                        max_train_time = t
                
                m_bpb = re.search(r'val_bpb\s*[:=]\s*([\d\.]+)', line)
                if m_bpb:
                    final_bpb = float(m_bpb.group(1))
                    
            if 580000 <= max_train_time <= 650000:
                is_10min = True
                
            if is_10min and final_bpb is not None:
                res.append((final_bpb, max_train_time, f))

res.sort()
for i, r in enumerate(res):
    print(f"{i+1}. BPB: {r[0]:.4f} | Time: {r[1]/1000:.1f}s | File: {r[2]}")
