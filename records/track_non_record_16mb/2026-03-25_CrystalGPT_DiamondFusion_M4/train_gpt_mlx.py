#!/usr/bin/env python3
from __future__ import annotations

import train_gpt_mlx as base
import mlx.core as mx
import mlx.nn as nn
from crystal_model import CrystalGPT, CrystalMoE

class Hyperparameters(base.Hyperparameters):
    model_type: str = __import__('os').environ.get('MODEL_TYPE', 'crystal_gpt')
    crystal_iters: int = int(__import__('os').environ.get('CRYSTAL_ITERS', 12))
    dim: int = int(__import__('os').environ.get('DIM', 512))
    num_experts: int = int(__import__('os').environ.get('NUM_EXPERTS', 4))
    fusion_every: int = int(__import__('os').environ.get('FUSION_EVERY', 3))

class WrappedCrystalGPT(CrystalGPT):
    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        logits = self(input_ids).astype(mx.float32)
        return nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), reduction='mean')

class WrappedCrystalMoE(CrystalMoE):
    def loss(self, input_ids: mx.array, target_ids: mx.array) -> mx.array:
        logits = self(input_ids).astype(mx.float32)
        return nn.losses.cross_entropy(logits.reshape(-1, logits.shape[-1]), target_ids.reshape(-1), reduction='mean')

class SplitOptimizers(base.SplitOptimizers):
    def __init__(self, model, args):
        self.args = args
        params = dict(base.tree_flatten(model.parameters()))
        self.embed_key = 'embed.weight'
        self.matrix_keys = [
            k for k, p in params.items()
            if p.ndim == 2 and k != self.embed_key and not any(pattern in k for pattern in base.CONTROL_TENSOR_NAME_PATTERNS)
        ]
        self.scalar_keys = [
            k for k, p in params.items()
            if k != self.embed_key and (p.ndim < 2 or any(pattern in k for pattern in base.CONTROL_TENSOR_NAME_PATTERNS))
        ]
        self.muon = base.Muon(self.matrix_keys, params, args)
        self.adam_embed = base.optim.Adam(
            learning_rate=args.tied_embed_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )
        self.adam_scalar = base.optim.Adam(
            learning_rate=args.scalar_lr,
            betas=[args.beta1, args.beta2],
            eps=args.adam_eps,
            bias_correction=True,
        )

def main():
    args = Hyperparameters()
    out_dir = base.Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logfile = out_dir / f"{args.run_id}.txt"
    print(logfile)

    def log(msg: str, console: bool = True) -> None:
        if console:
            print(msg)
        with logfile.open('a', encoding='utf-8') as f:
            print(msg, file=f)

    sp = base.spm.SentencePieceProcessor(model_file=args.tokenizer_path)
    if int(sp.vocab_size()) != args.vocab_size:
        raise ValueError(f'VOCAB_SIZE={args.vocab_size} does not match tokenizer vocab_size={int(sp.vocab_size())}')
    dataset_name, actual_train_files, expected_train_files = base.validate_dataset_tokenizer_pair(args.data_path, args.tokenizer_path)
    val_tokens = base.load_validation_tokens(args.val_files, args.train_seq_len)
    base_bytes_lut, has_leading_space_lut, is_boundary_token_lut = base.build_sentencepiece_luts(sp, args.vocab_size)
    mx.random.seed(args.seed)
    train_loader = base.TokenLoader(args.train_files, log_fn=log, dataset_name=dataset_name)

    if args.model_type == 'crystal_moe':
        model = WrappedCrystalMoE(vocab_size=args.vocab_size, crystal_iters=args.crystal_iters, dim=args.dim, num_experts=args.num_experts, fusion_every=args.fusion_every, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, rope_dims=16, rope_base=args.rope_base, qk_gain=args.qk_gain_init, softcap=args.logit_softcap)
    else:
        model = WrappedCrystalGPT(vocab_size=args.vocab_size, crystal_iters=args.crystal_iters, dim=args.dim, num_heads=args.num_heads, num_kv_heads=args.num_kv_heads, rope_dims=16, rope_base=args.rope_base, qk_gain=args.qk_gain_init, softcap=args.logit_softcap)

    opt = SplitOptimizers(model, args)
    compiled_loss = mx.compile(lambda x, y: model.loss(x, y), inputs=model.state, outputs=model.state)
    compiled_loss_and_grad = mx.compile(nn.value_and_grad(model, lambda x, y: model.loss(x, y)), inputs=model.state, outputs=model.state)

    n_params = sum(int(__import__('numpy').prod(p.shape)) for _, p in base.tree_flatten(model.parameters()))
    log(f'run_id:{args.run_id}')
    log(f'model_type:{args.model_type} params:{n_params} crystal_iters:{args.crystal_iters} dim:{args.dim}')

    step = 0
    train_time_ms = 0.0
    t0 = base.time.perf_counter()
    while True:
        last_step = step == args.iterations
        if last_step or (args.val_loss_every > 0 and step % args.val_loss_every == 0):
            train_time_ms += 1000.0 * (base.time.perf_counter() - t0)
            val_loss, val_bpb = base.eval_val(args, compiled_loss, val_tokens, base_bytes_lut, has_leading_space_lut, is_boundary_token_lut, log_fn=log)
            log(f'step:{step}/{args.iterations} val_loss:{val_loss:.4f} val_bpb:{val_bpb:.4f} train_time:{train_time_ms:.0f}ms')
            t0 = base.time.perf_counter()
        if last_step:
            break
        lr_mul = args.lr_mul(step, train_time_ms + 1000.0 * (base.time.perf_counter() - t0))
        step_t0 = base.time.perf_counter()
        accum = None
        train_loss = mx.array(0.0, dtype=mx.float32)
        grad_scale = 1.0 / args.grad_accum_steps
        for _ in range(args.grad_accum_steps):
            loss, grads = base.loss_and_grad_chunked(args, train_loader, compiled_loss_and_grad)
            accum = base.accumulate_flat_grads(accum, grads, grad_scale)
            train_loss = train_loss + loss.astype(mx.float32) * grad_scale
            if args.mlx_eager_eval:
                mx.eval(train_loss, accum)
        grads = base.tree_unflatten(list(accum.items()))
        grads = base.clip_grad_tree(grads, args.grad_clip_norm)
        train_loss_value = float(train_loss.item())
        opt.step(model, grads, step=step, lr_mul=lr_mul)
        mx.synchronize()
        step_ms = 1000.0 * (base.time.perf_counter() - step_t0)
        step += 1
        if args.train_log_every > 0 and (step <= 10 or step % args.train_log_every == 0):
            tok_s = args.train_batch_tokens / (step_ms / 1000.0)
            log(f'step:{step}/{args.iterations} train_loss:{train_loss_value:.4f} tok_s:{tok_s:.0f}')

if __name__ == '__main__':
    main()
