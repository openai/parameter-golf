#!/bin/bash
cd ~/parameter-golf/parameter-golf || exit 1

mkdir -p logs
mkdir -p results
mkdir -p models

pkill -f train_gpt_optimized.py || true

EXP1="RUN_ID=exp_small NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=4"
EXP2="RUN_ID=exp_medium NUM_LAYERS=6 MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=6"
EXP3="RUN_ID=exp_large NUM_LAYERS=8 MODEL_DIM=512 NUM_HEADS=8 NUM_KV_HEADS=8"

COMMON="ITERATIONS=100000 MAX_WALLCLOCK_SECONDS=14400 DATA_PATH=./data/datasets/fineweb10B_sp1024 PYTHONUNBUFFERED=1"

echo "Running EXP1..."
eval $EXP1 $COMMON python3 train_gpt_optimized.py | tee logs/exp_small.txt
cp final_model.int8.ptz models/exp_small.ptz

echo "Running EXP2..."
eval $EXP2 $COMMON python3 train_gpt_optimized.py | tee logs/exp_medium.txt
cp final_model.int8.ptz models/exp_medium.ptz

echo "Running EXP3..."
eval $EXP3 $COMMON python3 train_gpt_optimized.py | tee logs/exp_large.txt
cp final_model.int8.ptz models/exp_large.ptz

echo "EXPERIMENT RESULTS" > results/summary.txt
echo "==================" >> results/summary.txt

for exp in small medium large
do
    echo "" >> results/summary.txt
    echo "EXP_$exp" >> results/summary.txt
    tail -n 20 logs/exp_${exp}.txt | grep -E "val_loss|val_bpb" >> results/summary.txt
    echo "Model size:" >> results/summary.txt
    ls -lh models/exp_${exp}.ptz | awk '{print $5}' >> results/summary.txt
done

cat results/summary.txt
