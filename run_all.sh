cd ~/parameter-golf/parameter-golf || exit
pkill -f train_gpt_optimized.py || true
rm -f logs/exp1.txt logs/exp2.txt
mkdir -p logs

RUN_ID=exp1 NUM_LAYERS=4 MODEL_DIM=256 NUM_HEADS=4 NUM_KV_HEADS=4 ITERATIONS=100000 MAX_WALLCLOCK_SECONDS=3600 DATA_PATH=./data/datasets/fineweb10B_sp1024 PYTHONUNBUFFERED=1 python3 train_gpt_optimized.py | tee logs/exp1.txt

tail -n 50 logs/exp1.txt

RUN_ID=exp2 NUM_LAYERS=6 MODEL_DIM=384 NUM_HEADS=6 NUM_KV_HEADS=6 ITERATIONS=100000 MAX_WALLCLOCK_SECONDS=3600 DATA_PATH=./data/datasets/fineweb10B_sp1024 PYTHONUNBUFFERED=1 python3 train_gpt_optimized.py | tee logs/exp2.txt

tail -n 50 logs/exp2.txt

echo "==== EXP1 ===="
tail -n 10 logs/exp1.txt
echo "==== EXP2 ===="
tail -n 10 logs/exp2.txt
