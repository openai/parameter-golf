#!/bin/bash
set -e
cd /workspace/meadow-golf/experiments/2026-04-09_matched_ablation
mkdir -p /workspace/out /workspace/ckpt /workspace/logs /workspace/eval

tag=11L_w0_s1337
echo "================================================================"
echo "== TRAIN: $tag (control, seed 1337, w=0.0)"
echo "================================================================"
python3 train_ablation_runner.py \
  --train_script ./train_cdm.py \
  --num_layers 11 --model_dim 512 --vocab_size 4096 \
  --bigram_dim 128 --xsa_last_n 4 \
  --cdm_weight 0.0 --seed 1337 \
  -- \
  --train_budget_secs 540 --steps 9999 \
  --data_dir /workspace/gv4096/data \
  --tokenizer_path /workspace/gv4096/bpe_v4096.model \
  --save_path /workspace/out/${tag}.npz \
  --save_int6_path /workspace/out/${tag}_int6.lzma \
  --checkpoint_dir /workspace/ckpt/${tag} \
  --val_every 500 --val_tokens 1000000 \
  > /workspace/logs/${tag}_train.log 2>&1
echo "  done -> $(grep 'FINAL val_bpb' /workspace/logs/${tag}_train.log | tail -1)"
ls -la /workspace/ckpt/${tag}/step_final.pt /workspace/out/${tag}.npz

echo
echo "== EVAL: $tag"
python3 eval_cf_ablation.py \
  --ckpt /workspace/ckpt/${tag}/step_final.pt \
  --train_module_path /tmp/train_cdm_patched_11L_w0.0_s1337.py \
  --num_layers 11 --model_dim 512 --vocab_size 4096 \
  --bigram_dim 128 --xsa_last_n 4 \
  --n_seqs 500 --seq_len 1024 --stride 2 --rounds 2 --seed 42 \
  --data_dir /workspace/gv4096/data \
  --tokenizer_path /workspace/gv4096/bpe_v4096.model \
  --log_path /workspace/eval/${tag}_cf.log \
  > /workspace/eval/${tag}_eval.out 2>&1
echo "  cf -> $(grep 'cf_total' /workspace/eval/${tag}_cf.log | tail -1)"
echo "  pure_ar -> $(grep 'pure_ar_bpb' /workspace/eval/${tag}_cf.log | tail -1)"
echo PHASE_B_DONE
