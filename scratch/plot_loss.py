import re
import matplotlib.pyplot as plt
import os

log_file = '/Users/akhileshgogikar/parameter-golf/logs/remote_run_20260413/spec11x512_run.txt'
output_plot = '/Users/akhileshgogikar/parameter-golf/logs/remote_run_20260413/loss_convergence.png'

steps = []
losses = []

# Regex for "step:156 loss:9.243 (frac=0.010) ..." format
pattern = re.compile(r'step:(\d+)\s+loss:([\d\.]+)')

if os.path.exists(log_file):
    with open(log_file, 'r', errors='ignore') as f:
        for line in f:
            match = pattern.search(line)
            if match:
                steps.append(int(match.group(1)))
                losses.append(float(match.group(2)))

if steps:
    plt.figure(figsize=(10, 6))
    plt.plot(steps, losses, label='Training Loss')
    plt.xlabel('Step')
    plt.ylabel('Loss')
    plt.title('Loss Convergence - 11x512 spec + Engram')
    plt.grid(True)
    plt.legend()
    plt.savefig(output_plot)
    print(f"Plot saved to {output_plot}")
    print(f"Points found: {len(steps)}")
    print(f"Final step: {steps[-1]}, Final loss: {losses[-1]}")
else:
    print("No loss data found in log file.")
