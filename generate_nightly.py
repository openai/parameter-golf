import os

CONFIGS = [
    # H1: QK Gain
    {"QK_GAIN_INIT": "2.0"},
    {"QK_GAIN_INIT": "3.0"},
    {"QK_GAIN_INIT": "4.0"},
    {"QK_GAIN_INIT": "5.0"},
    {"QK_GAIN_INIT": "6.0"},
    
    # H2: Tied Embeddings
    {"TIED_EMBED_LR": "0.01"},
    {"TIED_EMBED_LR": "0.02"},
    {"TIED_EMBED_LR": "0.08"},
    {"TIED_EMBED_INIT_STD": "0.003"},
    {"TIED_EMBED_INIT_STD": "0.008"},
    {"TIED_EMBED_INIT_STD": "0.01"},
    {"TIED_EMBED_LR": "0.02", "TIED_EMBED_INIT_STD": "0.003"},
    {"TIED_EMBED_LR": "0.08", "TIED_EMBED_INIT_STD": "0.008"},
    {"TIED_EMBED_LR": "0.02", "TIED_EMBED_INIT_STD": "0.008"},
    
    # H3: Warmdown
    {"WARMDOWN_ITERS": "800"},
    {"WARMDOWN_ITERS": "1600"},
    {"WARMDOWN_ITERS": "2000"},
    
    # H4: PTQ Clipping
    {"INT8_CLIP_PERCENTILE": "99.5"},
    {"INT8_CLIP_PERCENTILE": "99.7"},
    {"INT8_CLIP_PERCENTILE": "99.9"},
    {"INT8_CLIP_PERCENTILE": "99.99"},
    
    # H5: RoPE Base
    {"ROPE_BASE": "5000"},
    {"ROPE_BASE": "50000"},
    {"ROPE_BASE": "100000"},
    
    # H6: Seq Len & Batch
    {"TRAIN_SEQ_LEN": "512"},
    {"TRAIN_SEQ_LEN": "768"},
    {"TRAIN_BATCH_TOKENS": "262144"},
    {"TRAIN_BATCH_TOKENS": "1048576"},
    {"TRAIN_SEQ_LEN": "512", "TRAIN_BATCH_TOKENS": "262144"},
    {"TRAIN_SEQ_LEN": "768", "TRAIN_BATCH_TOKENS": "262144"},
    
    # H7: Mixed PTQ
    {"PTQ_BITS": "8"}, # Attn in int8
    {"INT6_LAYER_START": "18", "INT6_LAYER_END": "20"}, # Last 2 layers in int6
    {"INT6_LAYER_START": "0", "INT6_LAYER_END": "2"},   # First 2 layers in int6
    
    # H8: Combinations
    {"QK_GAIN_INIT": "3.0", "TRAIN_SEQ_LEN": "512"},
    {"QK_GAIN_INIT": "4.0", "TRAIN_SEQ_LEN": "512"},
    {"QK_GAIN_INIT": "3.0", "TRAIN_SEQ_LEN": "768"},
    {"QK_GAIN_INIT": "4.0", "TRAIN_SEQ_LEN": "768"},
    
    {"QK_GAIN_INIT": "3.0", "TIED_EMBED_LR": "0.02"},
    {"QK_GAIN_INIT": "4.0", "TIED_EMBED_LR": "0.02"},
    {"QK_GAIN_INIT": "3.0", "TIED_EMBED_LR": "0.08"},
    {"QK_GAIN_INIT": "4.0", "TIED_EMBED_LR": "0.08"},
    
    {"QK_GAIN_INIT": "3.0", "WARMDOWN_ITERS": "800"},
    {"QK_GAIN_INIT": "4.0", "WARMDOWN_ITERS": "800"},
    {"QK_GAIN_INIT": "3.0", "WARMDOWN_ITERS": "1600"},
    {"QK_GAIN_INIT": "4.0", "WARMDOWN_ITERS": "1600"},
    
    {"INT8_CLIP_PERCENTILE": "99.9", "PTQ_BITS": "8"},
    {"INT8_CLIP_PERCENTILE": "99.99", "PTQ_BITS": "8"},
    
    {"QK_GAIN_INIT": "3.0", "INT8_CLIP_PERCENTILE": "99.9"},
    {"QK_GAIN_INIT": "4.0", "INT8_CLIP_PERCENTILE": "99.9"},
    
    {"ROPE_BASE": "50000", "TRAIN_SEQ_LEN": "512"},
    {"ROPE_BASE": "50000", "TRAIN_SEQ_LEN": "768"},
    
    {"WARMDOWN_ITERS": "800", "TRAIN_SEQ_LEN": "512"},
    {"WARMDOWN_ITERS": "1600", "TRAIN_SEQ_LEN": "512"},
    
    {"INT8_CLIP_PERCENTILE": "99.9", "INT6_LAYER_START": "18", "INT6_LAYER_END": "20"},
    {"QK_GAIN_INIT": "4.0", "INT6_LAYER_START": "18", "INT6_LAYER_END": "20"},
    
    {"QK_GAIN_INIT": "3.0", "PTQ_BITS": "8"},
    {"TIED_EMBED_LR": "0.02", "PTQ_BITS": "8"},
    {"TRAIN_SEQ_LEN": "512", "PTQ_BITS": "8"},
    {"TRAIN_SEQ_LEN": "768", "PTQ_BITS": "8"},
    {"WARMDOWN_ITERS": "800", "PTQ_BITS": "8"},
]

TEMPLATE = """#!/bin/bash
export RUN_ID="{run_id}"
export EXPERIMENT_NAME="{exp_name}"

# Base config from ptq_int5mlp_L20_d288.sh
export NUM_LAYERS=20
export MODEL_DIM=288
export MLP_MULT=4
export TERNARY_ENABLED=0
export QAT_BITS=0
export PTQ_BITS=6
export PTQ_MLP_BITS=5
export TRAIN_BATCH_TOKENS=524288
export TRAIN_SEQ_LEN=1024
export OPTIMIZER=muon_adam
export LR_SCHEDULE=trapezoid

# Hypotheses overrides
{overrides}

# Validation overrides (controlled by run_all_nightly.sh)
export ITERATIONS="${{NIGHTLY_ITERATIONS:-20000}}"
export MAX_WALLCLOCK_SECONDS="${{NIGHTLY_WALLCLOCK:-600}}"
if [ -n "$NIGHTLY_COMET_KEY" ]; then
    export COMET_API_KEY="$NIGHTLY_COMET_KEY"
fi

SCRIPT_DIR="$(cd "$(dirname "${{BASH_SOURCE[0]}}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

torchrun --nproc_per_node=8 train_gpt.py
"""

RUN_ALL_TEMPLATE = """#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# ==============================================================================
# НАСТРОЙКИ ДЛЯ БЫСТРОЙ ВАЛИДАЦИИ
# Раскомментируй эти строки, чтобы прогнать скрипты на 10 итерациях и проверить,
# что ничего не падает (OOM, ошибки компиляции и т.д.)
# ==============================================================================
# export NIGHTLY_ITERATIONS=10
# export NIGHTLY_WALLCLOCK=30
# export NIGHTLY_COMET_KEY="your_api_key_here"

echo "Starting 60 nightly experiments..."
for script in "$SCRIPT_DIR"/exp_*.sh; do
    echo "========================================================="
    echo "Running $script..."
    echo "========================================================="
    bash "$script"
done
echo "All done!"
"""

def main():
    out_dir = "exps/nightly_ptq"
    os.makedirs(out_dir, exist_ok=True)
    
    for i, conf in enumerate(CONFIGS, 1):
        # Format overrides
        overrides_str = "\n".join(f'export {k}="{v}"' for k, v in conf.items())
        
        # Format name
        parts = []
        for k, v in conf.items():
            short_k = k.replace("_INIT", "").replace("TIED_EMBED_", "emb_").replace("TRAIN_", "").replace("INT8_CLIP_PERCENTILE", "clip").replace("WARMDOWN_ITERS", "wd").replace("INT6_LAYER_", "L6_")
            parts.append(f"{short_k}_{v}")
        
        short_name = "__".join(parts)
        run_id = f"nightly_{i:02d}_{short_name}"
        exp_name = f"Nightly {i:02d}: {short_name}"
        
        script_content = TEMPLATE.format(
            run_id=run_id,
            exp_name=exp_name,
            overrides=overrides_str
        )
        
        script_path = os.path.join(out_dir, f"exp_{i:02d}.sh")
        with open(script_path, "w") as f:
            f.write(script_content)
            
    with open(os.path.join(out_dir, "run_all_nightly.sh"), "w") as f:
        f.write(RUN_ALL_TEMPLATE)
        
    print(f"Generated {len(CONFIGS)} scripts in {out_dir}/")

if __name__ == "__main__":
    main()
