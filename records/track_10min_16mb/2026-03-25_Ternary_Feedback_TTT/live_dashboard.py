import time
import subprocess
import threading
from collections import deque
from rich.live import Live
from rich.layout import Layout
from rich.panel import Panel
from rich.text import Text

MAC_LOG = "hybrid_mac_benchmark.log"
WIN_CMD = [
    "ssh", "winlaptop", 
    "C:\\Python312\\python.exe", "-u", "-c", 
    "\"import time, sys, os; log='C:\\\\Users\\\\Public\\\\parameter-golf\\\\records\\\\track_10min_16mb\\\\2026-03-25_Ternary_Feedback_TTT\\\\hybrid_cuda_benchmark.log'; sys.exit(1) if not os.path.exists(log) else None; f=open(log,'r'); f.seek(max(0, os.path.getsize(log)-4096), 0); lines=f.readlines(); sys.stdout.write(''.join(lines[-30:])) if lines else None; [(sys.stdout.write(l), sys.stdout.flush()) if l else (time.sleep(0.5), f.seek(f.tell())) for _ in iter(int, 1) for l in [f.readline()]]\""
]

mac_lines = deque(maxlen=25)
win_lines = deque(maxlen=25)

def tail_mac():
    while True:
        try:
            proc = subprocess.Popen(["tail", "-F", "-n", "30", MAC_LOG], stdout=subprocess.PIPE, text=True, bufsize=1)
            for line in iter(proc.stdout.readline, ''):
                if line: mac_lines.append(line.strip())
        except:
            pass
        time.sleep(1)

def tail_win():
    while True:
        try:
            proc = subprocess.Popen(WIN_CMD, stdout=subprocess.PIPE, text=True, bufsize=1)
            for line in iter(proc.stdout.readline, ''):
                if line: win_lines.append(line.strip())
        except:
            pass
        time.sleep(1)

threading.Thread(target=tail_mac, daemon=True).start()
threading.Thread(target=tail_win, daemon=True).start()

def make_layout() -> Layout:
    layout = Layout()
    layout.split_row(
        Layout(name="mac"),
        Layout(name="win")
    )
    return layout

def update_layout(layout: Layout):
    mac_text = Text("\n".join(mac_lines), style="cyan")
    win_text = Text("\n".join(win_lines), style="green")
    
    layout["mac"].update(Panel(mac_text, title="🤖 Local MacBook (Apple MLX)", border_style="cyan"))
    layout["win"].update(Panel(win_text, title="🚀 Remote WinLaptop (NVIDIA CUDA)", border_style="green"))

layout = make_layout()

with Live(layout, refresh_per_second=2, screen=True):
    try:
        while True:
            update_layout(layout)
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
