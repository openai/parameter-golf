# Parallel Eval Framework

## How it works

1. Train model once (all 8 GPUs, PR #834 script)
2. Save `final_model.int6.ptz` + `final_model_pre_ttt.pt`
3. Run 8 single-GPU eval variants simultaneously
4. Compare results → winner becomes new baseline
5. Repeat

## Running a round

```bash
# Step 1: Train (once)
torchrun --nproc_per_node=8 pr834_train_gpt.py

# Step 2: Run 8 parallel evals (each on 1 GPU)
for gpu in 0 1 2 3 4 5 6 7; do
    CUDA_VISIBLE_DEVICES=$gpu python3 parallel_eval.py \
        --model final_model.int6.ptz \
        --method config_gpu${gpu}.json \
        --output results_gpu${gpu}.json &
done
wait

# Step 3: Compare results
python3 compare_results.py results_gpu*.json
```

## Method configs

Each `config_gpuN.json` specifies one eval method variant to test.
Winner's config becomes the baseline for next round.
