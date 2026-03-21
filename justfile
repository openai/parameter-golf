set shell := ["bash", "-lc"]

_default:
    @just --list

setup:
    uv sync --extra mlx

setup-cuda:
    uv sync --extra cuda

test:
    uv run python3 -m unittest discover -s tests -p "test_*.py"

download-data train_shards="10" variant="sp1024":
    uv run python3 data/cached_challenge_fineweb.py --variant {{variant}} --train-shards {{train_shards}}

mlx-smoke:
    RUN_ID=mlx_smoke \
    ITERATIONS=200 \
    TRAIN_BATCH_TOKENS=8192 \
    VAL_LOSS_EVERY=0 \
    VAL_BATCH_SIZE=8192 \
    uv run python3 train_gpt_mlx.py

mlx-train run_id="mlx_run" iterations="2000" train_batch_tokens="524288" val_loss_every="0" val_batch_size="524288":
    RUN_ID={{run_id}} \
    ITERATIONS={{iterations}} \
    TRAIN_BATCH_TOKENS={{train_batch_tokens}} \
    VAL_LOSS_EVERY={{val_loss_every}} \
    VAL_BATCH_SIZE={{val_batch_size}} \
    uv run python3 train_gpt_mlx.py

torch-train run_id="baseline_sp1024" nproc="1" data_path="./data/datasets/fineweb10B_sp1024/" tokenizer_path="./data/tokenizers/fineweb_1024_bpe.model" vocab_size="1024":
    RUN_ID={{run_id}} \
    DATA_PATH={{data_path}} \
    TOKENIZER_PATH={{tokenizer_path}} \
    VOCAB_SIZE={{vocab_size}} \
    uv run torchrun --standalone --nproc_per_node={{nproc}} train_gpt.py

autoresearch-mlx trials="5" seed="1337":
    uv run python3 autoresearch/run_search.py --backend mlx --trials {{trials}} --seed {{seed}} --baseline-first

autoresearch-cuda trials="5" nproc="1" seed="1337":
    uv run python3 autoresearch/run_search.py --backend cuda --trials {{trials}} --nproc {{nproc}} --seed {{seed}} --baseline-first

autoresearch-resume backend="mlx" trials="5" nproc="1" seed="1337":
    uv run python3 autoresearch/run_search.py --backend {{backend}} --trials {{trials}} --nproc {{nproc}} --seed {{seed}} --resume

autoresearch-preset-mlx trials="5" seed="1337" preset="":
    @if [ -n "{{preset}}" ]; then \
        uv run python3 autoresearch/run_search.py --backend mlx --mode preset --trials {{trials}} --seed {{seed}} --preset {{preset}}; \
    else \
        uv run python3 autoresearch/run_search.py --backend mlx --mode preset --trials {{trials}} --seed {{seed}}; \
    fi

autoresearch-preset-cuda trials="5" nproc="1" seed="1337" preset="":
    @if [ -n "{{preset}}" ]; then \
        uv run python3 autoresearch/run_search.py --backend cuda --mode preset --trials {{trials}} --nproc {{nproc}} --seed {{seed}} --preset {{preset}}; \
    else \
        uv run python3 autoresearch/run_search.py --backend cuda --mode preset --trials {{trials}} --nproc {{nproc}} --seed {{seed}}; \
    fi

autoresearch-evolution-mlx trials="5" seed="1337" population="6":
    uv run python3 autoresearch/run_search.py --backend mlx --mode evolution --trials {{trials}} --seed {{seed}} --population {{population}} --resume

autoresearch-evolution-cuda trials="5" nproc="1" seed="1337" population="6":
    uv run python3 autoresearch/run_search.py --backend cuda --mode evolution --trials {{trials}} --nproc {{nproc}} --seed {{seed}} --population {{population}} --resume

autoresearch-code-mlx trials="5" seed="1337" mutation="":
    @if [ -n "{{mutation}}" ]; then \
        uv run python3 autoresearch/run_search.py --backend mlx --mode code --trials {{trials}} --seed {{seed}} --code-mutation {{mutation}} --resume; \
    else \
        uv run python3 autoresearch/run_search.py --backend mlx --mode code --trials {{trials}} --seed {{seed}} --resume; \
    fi

modal-upload-data:
    modal run autoresearch/modal_search.py::upload_data

modal-sweep-preset trials="8" gpu="A10G" seed="1337" preset="":
    @if [ -n "{{preset}}" ]; then \
        modal run autoresearch/modal_search.py --mode preset --trials {{trials}} --gpu {{gpu}} --seed {{seed}} --preset {{preset}}; \
    else \
        modal run autoresearch/modal_search.py --mode preset --trials {{trials}} --gpu {{gpu}} --seed {{seed}}; \
    fi

modal-sweep-random trials="10" gpu="A10G" seed="1337":
    modal run autoresearch/modal_search.py --mode random --trials {{trials}} --gpu {{gpu}} --seed {{seed}}

modal-sweep-evolution trials="6" gpu="A10G" seed="1337" population="6":
    modal run autoresearch/modal_search.py --mode evolution --trials {{trials}} --gpu {{gpu}} --seed {{seed}} --population {{population}}

modal-sweep-h100 trials="8" seed="1337":
    modal run autoresearch/modal_search.py --mode preset --trials {{trials}} --gpu h100 --seed {{seed}} --baseline-first

autoresearch-code-cuda trials="5" nproc="1" seed="1337" mutation="":
    @if [ -n "{{mutation}}" ]; then \
        uv run python3 autoresearch/run_search.py --backend cuda --mode code --trials {{trials}} --nproc {{nproc}} --seed {{seed}} --code-mutation {{mutation}} --resume; \
    else \
        uv run python3 autoresearch/run_search.py --backend cuda --mode code --trials {{trials}} --nproc {{nproc}} --seed {{seed}} --resume; \
    fi
