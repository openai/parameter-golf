

#CUDA_VISIBLE_DEVICES=0 VOCAB_SIZE=1024 ITERATIONS=120 SEED=314 QK_GAIN_INIT=5.25 TTT_ENABLED=1 TTT_LR=0.005 TTT_EPOCHS=3 \
#  torchrun --standalone --nproc_per_node=1 train_gpt_encode.py
torchrun --standalone --nproc_per_node=1 train_baseline.py
