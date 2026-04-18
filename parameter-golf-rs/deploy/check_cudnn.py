import modal

app = modal.App("apt-cache-test")

image = modal.Image.from_registry("nvidia/cuda:12.6.3-devel-ubuntu22.04")

@app.function(image=image)
def run_apt_cache():
    import subprocess
    result = subprocess.run(["apt-get", "update"], capture_output=True, text=True)
    result = subprocess.run(["apt-cache", "search", "libcudnn"], capture_output=True, text=True)
    print(result.stdout)

@app.local_entrypoint()
def main():
    run_apt_cache.remote()
