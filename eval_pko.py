"""Test PKO (Partial Key Offset) on existing PR#1493 quantized model at eval time."""
import io, os, time, types, torch, torch.distributed as dist, torch.nn.functional as F

from train_pr1493 import (
    Hyperparameters, GPT, ValidationData,
    restore_fp32_params, eval_val_sliding, set_logging_hparams, log,
    deserialize, flash_attn_3_func, apply_rotary_emb,
)

torch._dynamo.config.cache_size_limit = 32


def pko_attn_forward(self, x):
    bsz, seqlen, dim = x.shape
    q = self.c_q(x).reshape(bsz, seqlen, self.num_heads, self.head_dim)
    k = self.c_k(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
    v = self.c_v(x).reshape(bsz, seqlen, self.num_kv_heads, self.head_dim)
    q = F.rms_norm(q, (q.size(-1),))
    k = F.rms_norm(k, (k.size(-1),))
    cos, sin = self.rotary(seqlen, x.device, q.dtype)
    q = apply_rotary_emb(q, cos, sin, self.rope_dims)
    k = apply_rotary_emb(k, cos, sin, self.rope_dims)
    # PKO: shift non-RoPE key dims by 1 position
    rd = self.rope_dims if self.rope_dims > 0 and self.rope_dims < self.head_dim else self.head_dim // 2
    if seqlen > 1:
        k = k.clone()
        k[:, 1:, :, rd:] = k[:, :-1, :, rd:]
    q = q * self.q_gain.to(dtype=q.dtype)[None, None, :, None]
    y = flash_attn_3_func(q, k, v, causal=True)
    if self.use_xsa:
        y = self._xsa_efficient(y, v)
    y = y.reshape(bsz, seqlen, dim)
    return self.proj(y)


def patch_pko(model):
    for block in model.blocks:
        block.attn.forward = types.MethodType(pko_attn_forward, block.attn)


def main():
    distributed = 'RANK' in os.environ and 'WORLD_SIZE' in os.environ
    local_rank = int(os.environ.get('LOCAL_RANK', '0'))
    device = torch.device('cuda', local_rank)
    torch.cuda.set_device(device)
    if distributed:
        dist.init_process_group(backend='nccl', device_id=device)
        dist.barrier()
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.set_float32_matmul_precision('high')

    h = Hyperparameters()
    set_logging_hparams(h)
    val_data = ValidationData(h, device)

    # Test 1: baseline (no PKO)
    log("=== Baseline (no PKO) ===")
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    torch._dynamo.reset()
    t0 = time.perf_counter()
    val_loss, val_bpb = eval_val_sliding(h, device, val_data, eval_model)
    log(f"Baseline sliding: val_bpb={val_bpb:.8f} time={time.perf_counter()-t0:.1f}s")
    del eval_model
    torch.cuda.empty_cache()

    # Test 2: with PKO on all layers
    log("\n=== PKO all layers ===")
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    patch_pko(eval_model)
    torch._dynamo.reset()
    t0 = time.perf_counter()
    val_loss, val_bpb = eval_val_sliding(h, device, val_data, eval_model)
    log(f"PKO all layers sliding: val_bpb={val_bpb:.8f} time={time.perf_counter()-t0:.1f}s")
    del eval_model
    torch.cuda.empty_cache()

    # Test 3: PKO only on non-XSA layers (if any without XSA)
    # In PR#1493, xsa_last_n=11 means ALL layers have XSA
    # So let's try PKO only on encoder layers (0-7 with recurrence)
    log("\n=== PKO encoder layers only ===")
    eval_model = deserialize(h, device)
    if h.num_loops > 0:
        eval_model.looping_active = True
    for i in range(len(eval_model.blocks)):
        if i < eval_model.num_encoder_layers:
            eval_model.blocks[i].attn.forward = types.MethodType(
                pko_attn_forward, eval_model.blocks[i].attn)
    torch._dynamo.reset()
    t0 = time.perf_counter()
    val_loss, val_bpb = eval_val_sliding(h, device, val_data, eval_model)
    log(f"PKO encoder-only sliding: val_bpb={val_bpb:.8f} time={time.perf_counter()-t0:.1f}s")

    if distributed:
        dist.destroy_process_group()


if __name__ == '__main__':
    main()
