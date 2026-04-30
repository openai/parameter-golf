FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    FLASH_ATTN_WHEEL_INDEX=https://windreamer.github.io/flash-attention3-wheels/cu128_torch291/ \
    RUNPOD_VOLUME_ROOT=/runpod-volume \
    RUNPOD_PERSIST_ROOT=/runpod-volume/parameter-golf \
    RUNPOD_DATA_ROOT=/runpod-volume/parameter-golf/data \
    RUNPOD_RESULTS_ROOT=/runpod-volume/parameter-golf/endpoint_results/2026-04-29_SP8192_AttnGate_PhasedTTT_LoRA_LaCT \
    RUNPOD_ENDPOINT_TIMEOUT_SECONDS=1400 \
    RUNPOD_GPU_COUNT_REQUIRED=8

RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 \
    python3-dev \
    python3-pip \
    python3-venv \
    build-essential \
    ca-certificates \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/* \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3 1

WORKDIR /opt/parameter-golf

COPY requirements.txt ./requirements.txt
COPY records/track_10min_16mb/2026-04-29_SP8192_AttnGate_PhasedTTT_LoRA_LaCT/requirements.txt ./records/track_10min_16mb/2026-04-29_SP8192_AttnGate_PhasedTTT_LoRA_LaCT/requirements.txt

RUN python -m pip install --upgrade pip setuptools wheel \
    && python -m pip install --index-url https://download.pytorch.org/whl/cu128 torch==2.9.1 \
    && python -m pip install -r requirements.txt \
    && python -m pip install -r records/track_10min_16mb/2026-04-29_SP8192_AttnGate_PhasedTTT_LoRA_LaCT/requirements.txt \
    && python -m pip install runpod \
    && python -m pip install flash_attn_3 --no-deps --find-links "${FLASH_ATTN_WHEEL_INDEX}"

COPY . .

WORKDIR /opt/parameter-golf/records/track_10min_16mb/2026-04-29_SP8192_AttnGate_PhasedTTT_LoRA_LaCT

CMD ["python", "handler.py"]
