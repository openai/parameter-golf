#!/bin/bash
# ═══════════════════════════════════════════════════════════════
# Parameter Golf — RunPod 원클릭 실행 스크립트
# 8xH100 SXM에서 실행
# ═══════════════════════════════════════════════════════════════
set -e

echo "=== 1. 환경 설정 ==="
pip install sentencepiece huggingface-hub datasets tqdm zstandard

echo "=== 2. Repo + 데이터 ==="
cd /workspace
git clone https://github.com/openai/parameter-golf && cd parameter-golf
python -c "
from huggingface_hub import snapshot_download
snapshot_download('openai/parameter-golf-data', repo_type='dataset', local_dir='./data')
"

echo "=== 3. PR #398 코드 다운로드 ==="
# PR #398의 train_gpt.py (11L + EMA + TTT)
gh api repos/felipe-parodi/parameter-golf/contents/train_gpt.py?ref=submission/11L-EMA-TTT20ep-1.1213 \
  --jq '.content' | base64 -d > train_gpt_pr398.py

echo "=== 4. 우리 TTT 패치 적용 ==="
# ttt_adapt 함수에 TENT + Dynamic Eval 추가
cp train_gpt_pr398.py train_gpt_ours.py
# 패치는 아래 Python으로 적용

python3 << 'PATCH_EOF'
import re

with open("train_gpt_ours.py", "r") as f:
    code = f.read()

# 1. Hyperparameters에 새 TTT 옵션 추가
old_ttt_params = '''    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))'''
new_ttt_params = '''    ttt_freeze_blocks = int(os.environ.get("TTT_FREEZE_BLOCKS", 0))

    # Our TTT improvements
    ttt_tent_enabled = bool(int(os.environ.get("TTT_TENT_ENABLED", "1")))
    ttt_tent_epochs = int(os.environ.get("TTT_TENT_EPOCHS", 30))
    ttt_tent_lr = float(os.environ.get("TTT_TENT_LR", 0.01))
    ttt_dyneval = bool(int(os.environ.get("TTT_DYNEVAL", "0")))
    ttt_dyneval_lr = float(os.environ.get("TTT_DYNEVAL_LR", 0.001))'''

code = code.replace(old_ttt_params, new_ttt_params)

# 2. ttt_adapt 함수 뒤에 TENT + Dynamic Eval 함수 추가
tent_code = '''

# --- TENT: Norm recalibration before TTT ---
NORM_PARAM_PATTERNS = ("attn_scale", "mlp_scale", "q_gain", "skip_weight")

def tent_norm_recalib(args, base_model, device, val_tokens, rank, world_size, log_fn=None):
    """TENT-style norm/scale recalibration before main TTT."""
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    batch_seqs = args.ttt_batch_seqs

    # Freeze all, unfreeze only norm/scale params
    for p in base_model.parameters():
        p.requires_grad_(False)
    norm_params = []
    for name, p in base_model.named_parameters():
        if any(k in name for k in NORM_PARAM_PATTERNS):
            p.requires_grad_(True)
            norm_params.append(p)

    if log_fn:
        n_params = sum(p.numel() for p in norm_params)
        log_fn(f"tent:start params={n_params} epochs={args.ttt_tent_epochs} lr={args.ttt_tent_lr}")

    optimizer = torch.optim.Adam(norm_params, lr=args.ttt_tent_lr)

    my_start = (total_seqs * rank) // world_size
    my_end = (total_seqs * (rank + 1)) // world_size

    base_model.train()
    t0 = time.perf_counter()

    for epoch in range(args.ttt_tent_epochs):
        for batch_start in range(my_start, my_end, batch_seqs):
            batch_end = min(batch_start + batch_seqs, my_end)
            raw_start = batch_start * seq_len
            raw_end = batch_end * seq_len + 1
            local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
            x = local[:-1].reshape(-1, seq_len)
            y = local[1:].reshape(-1, seq_len)

            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                loss = base_model(x, y)
            loss.backward()

            if world_size > 1:
                for p in norm_params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
            optimizer.step()

        if log_fn and (epoch + 1) % max(1, args.ttt_tent_epochs // 5) == 0:
            elapsed = time.perf_counter() - t0
            log_fn(f"tent_epoch:{epoch+1}/{args.ttt_tent_epochs} time:{elapsed:.1f}s")

    # Restore requires_grad for main TTT
    for p in base_model.parameters():
        p.requires_grad_(True)

    if log_fn:
        log_fn(f"tent:done elapsed={time.perf_counter()-t0:.1f}s")


# --- Dynamic Evaluation: adapt during scoring ---
def eval_val_dynamic(args, model, device, val_tokens, rank=0, world_size=1, log_fn=None):
    """Dynamic eval: score each window, then adapt weights for next window."""
    seq_len = args.train_seq_len
    total_seqs = (val_tokens.numel() - 1) // seq_len
    lr = args.ttt_dyneval_lr

    my_start = (total_seqs * rank) // world_size
    my_end = (total_seqs * (rank + 1)) // world_size

    model.train()
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.0)

    total_loss = torch.zeros((), device=device, dtype=torch.float64)
    total_tokens = torch.zeros((), device=device, dtype=torch.float64)

    for i in range(my_start, my_end):
        raw_start = i * seq_len
        raw_end = raw_start + seq_len + 1
        local = val_tokens[raw_start:raw_end].to(device=device, dtype=torch.int64, non_blocking=True)
        if local.numel() < 2:
            break
        n = local.numel() - 1
        x = local[:n].unsqueeze(0)
        y = local[1:n+1].unsqueeze(0)

        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = model(x, y)

        # Score BEFORE adaptation
        total_loss += loss.detach().to(torch.float64) * y.numel()
        total_tokens += float(y.numel())

        # Adapt for next window
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        if world_size > 1:
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, op=dist.ReduceOp.AVG)
        optimizer.step()

    if world_size > 1:
        dist.all_reduce(total_loss, op=dist.ReduceOp.SUM)
        dist.all_reduce(total_tokens, op=dist.ReduceOp.SUM)

    import math
    bpb = (total_loss.item() / max(total_tokens.item(), 1)) / math.log(2)
    if log_fn:
        log_fn(f"dyneval:done bpb={bpb:.4f}")
    return bpb

'''

# Insert after ttt_adapt function (before INT6 section)
marker = "# -----------------------------\n# INT6 MIXED QUANTIZATION"
code = code.replace(marker, tent_code + "\n" + marker)

# 3. Modify the main flow to call TENT before TTT
# Find where ttt_adapt is called and add TENT before it
old_ttt_call = "ttt_adapt(args, base_model, device, val_tokens, rank, world_size, log)"
new_ttt_call = """# TENT norm recalibration (Phase 0)
        if args.ttt_tent_enabled:
            tent_norm_recalib(args, base_model, device, val_tokens, rank, world_size, log)
        # Main TTT (Phase 1)
        ttt_adapt(args, base_model, device, val_tokens, rank, world_size, log)"""

code = code.replace(old_ttt_call, new_ttt_call)

with open("train_gpt_ours.py", "w") as f:
    f.write(code)

print("Patch applied successfully!")
PATCH_EOF

echo "=== 5. 학습 + TTT 실행 ==="
# A. PR #398 원본 (비교용)
echo "--- Run A: PR #398 Original (SGD TTT only) ---"
SEED=1337 NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=0 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=0 \
TTT_ENABLED=1 TTT_LR=0.008 TTT_EPOCHS=20 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=0 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt_pr398.py 2>&1 | tee run_a_original.log

echo ""
echo "--- Run B: TENT→SGD TTT (our improvement) ---"
SEED=1337 NUM_LAYERS=11 BIGRAM_VOCAB_SIZE=2048 XSA_LAST_N=0 \
EMA_ENABLED=1 EMA_DECAY=0.997 SWA_ENABLED=0 \
ROPE_DIMS=16 LN_SCALE=1 LATE_QAT=0 \
TTT_ENABLED=1 TTT_LR=0.008 TTT_EPOCHS=20 TTT_MOMENTUM=0.9 TTT_FREEZE_BLOCKS=0 \
TTT_TENT_ENABLED=1 TTT_TENT_EPOCHS=30 TTT_TENT_LR=0.01 \
MUON_WD=0.04 ADAM_WD=0.04 \
MATRIX_LR=0.025 SCALAR_LR=0.025 TIED_EMBED_LR=0.035 \
MUON_MOMENTUM=0.99 MUON_MOMENTUM_WARMUP_START=0.92 \
MUON_MOMENTUM_WARMUP_STEPS=1500 WARMDOWN_ITERS=3000 \
ITERATIONS=9000 MAX_WALLCLOCK_SECONDS=600 EVAL_STRIDE=64 \
torchrun --standalone --nproc_per_node=8 train_gpt_ours.py 2>&1 | tee run_b_tent_sgd.log

echo ""
echo "=== DONE ==="
echo "Compare run_a_original.log vs run_b_tent_sgd.log"
echo "Look for val_bpb lines"
grep "val_bpb" run_a_original.log | tail -5
grep "val_bpb" run_b_tent_sgd.log | tail -5
