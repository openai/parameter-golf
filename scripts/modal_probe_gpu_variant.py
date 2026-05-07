import subprocess
import modal

app = modal.App("parameter-golf-gpu-variant-probe")

image = modal.Image.from_registry("pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime")


@app.function(image=image, gpu="H100:8", timeout=30 * 60, cpu=4, memory=8192)
def probe() -> None:
    cmds = [
        ["bash", "-lc", "nvidia-smi -L"],
        [
            "bash",
            "-lc",
            "nvidia-smi --query-gpu=name,gpu_bus_id,pci.bus_id,pci.device_id,driver_version --format=csv,noheader",
        ],
        ["bash", "-lc", "nvidia-smi topo -m"],
    ]
    for cmd in cmds:
        print("\n===", " ".join(cmd), "===")
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        print(proc.stdout)


@app.local_entrypoint()
def main() -> None:
    probe.remote()
