FROM nvcr.io/nvidia/pytorch:25.12-py3

WORKDIR /workspace/parameter-golf

# 25.12 ships PyTorch 2.10+ with native Blackwell sm_120/sm_121 support.
# Do NOT install a different PyTorch. Just add our extra deps.
ENV TORCH_CUDA_ARCH_LIST="12.0"

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
