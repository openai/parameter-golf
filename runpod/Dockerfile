# Parameter Golf custom image. Extends the community PyTorch+CUDA base so we
# skip the 3-5 minute "install everything" step on every cold pod. Build once,
# push to Docker Hub, point RunPod Template at it.
#
# Build:
#   docker build -t <dockerhub_user>/parameter-golf:2026-04-25 -f runpod/Dockerfile .
#   docker push  <dockerhub_user>/parameter-golf:2026-04-25
#
# In the RunPod Template UI, set Container Image to the tag above and
# Container Disk to >= 40 GB (torch + CUDA alone is ~15 GB).

FROM runpod/pytorch:2.9.1-cu128-py311

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_NO_CACHE_DIR=1 \
    PYTHONUNBUFFERED=1

# Core deps match parameter-golf/requirements.txt + competition-common extras.
# Pinning the important ones so spin-up is deterministic.
RUN pip install --upgrade pip && pip install \
        numpy tqdm \
        huggingface-hub==0.26.* \
        kernels setuptools \
        "typing-extensions==4.15.0" \
        datasets tiktoken sentencepiece \
        zstandard brotli \
        triton

# Flash-Attention 3 wheel for H100 (sm90). --no-build-isolation is critical or
# it tries to recompile CUDA from source and takes 30+ min.
RUN pip install "flash-attn>=3.0" --no-build-isolation || echo "flash-attn install failed — continue without"

# Pre-seed torch.compile cache with a trivial graph so that cold pods already
# have the Inductor bootstrap done. Real kernels still recompile on first run,
# but the ~20s interpreter/dynamo startup is amortized.
RUN python -c "import torch; f=torch.compile(lambda x: x*2 + 1); f(torch.zeros(4))" || true

WORKDIR /workspace
CMD ["/bin/bash"]
