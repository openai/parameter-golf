# RunPod Baseline

This path is meant to reproduce the March 18 baseline as closely as possible on a real VM:

- historical HF dataset revision: `1f2782522e6326a78ca5f1ed8edfb2eeeaf08d11`
- historical train shard count: `25`
- root `train_gpt.py`
- `8` local processes with `torchrun`
- `OMP_NUM_THREADS=1`
- `TORCH_NCCL_ASYNC_ERROR_HANDLING=1`
- `NCCL_IB_DISABLE=1`

## Recommended Pod

Use the official Parameter Golf RunPod template from the repo README:

`https://console.runpod.io/deploy?template=y5cejece4j&ref=nl2r56th`

Prefer `8x H100 80GB SXM` if you want the closest public environment.

## Bring-Up

```bash
cd /workspace
git clone https://github.com/openai/parameter-golf.git
cd parameter-golf
```

Verify the image already has the expected basics:

```bash
python3 -c "import torch, sentencepiece, huggingface_hub; print(torch.__version__)"
which torchrun
nvidia-smi
```

If the template is missing packages, install the repo requirements:

```bash
python3 -m pip install -r requirements.txt
```

## Prefetch The Historical Snapshot

```bash
python3 runpod_baseline.py prefetch
```

This downloads the pinned baseline snapshot into:

`./data/snapshots/1f2782522e6326a78ca5f1ed8edfb2eeeaf08d11/`

## Run The Baseline

```bash
python3 runpod_baseline.py run --run-id runpod_baseline_seed1337
```

The script will print the exact dataset revision, shard counts, and local paths before launching `torchrun`.

## Useful Overrides

Use the current dataset head instead of the pinned historical snapshot:

```bash
python3 runpod_baseline.py prefetch --hf-revision head --train-shards 80
python3 runpod_baseline.py run --hf-revision head --train-shards 80
```

Run the archived record copy instead of the root script:

```bash
python3 runpod_baseline.py run --script-variant record
```
