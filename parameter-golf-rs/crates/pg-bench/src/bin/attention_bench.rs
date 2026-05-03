use std::time::Instant;

use cudarc::driver::CudaContext;
use pg_core::{DType, GpuTensor, PgResult};
use pg_kernels::flash_attn::CudnnFrontendAttention;

#[derive(Debug, Clone, Copy)]
struct Args {
    batch: usize,
    seq_len: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    warmup: usize,
    iters: usize,
    backward: bool,
}

impl Default for Args {
    fn default() -> Self {
        Self {
            batch: 48,
            seq_len: 2048,
            heads: 8,
            kv_heads: 8,
            head_dim: 64,
            warmup: 2,
            iters: 8,
            backward: true,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let result = real_main();
    if let Err(err) = &result {
        println!("status=attention_bench_failed");
        println!("error={err}");
    }
    result.map_err(Into::into)
}

fn real_main() -> PgResult<()> {
    let args = parse_args()?;
    let tokens = args.batch * args.seq_len;
    let q_shape = [tokens, args.heads, args.head_dim];
    let kv_shape = [tokens, args.kv_heads, args.head_dim];

    let ctx = CudaContext::new(0)
        .map_err(|e| pg_core::PgError::InvalidOp(format!("CUDA init failed: {e:?}")))?;
    let stream = ctx.default_stream();
    let attention = CudnnFrontendAttention::new(stream.clone())?;

    let q = GpuTensor::zeros_gpu(stream.clone(), &q_shape, DType::F32)?;
    let k = GpuTensor::zeros_gpu(stream.clone(), &kv_shape, DType::F32)?;
    let v = GpuTensor::zeros_gpu(stream.clone(), &kv_shape, DType::F32)?;
    let out = GpuTensor::zeros_gpu(stream.clone(), &q_shape, DType::F32)?;
    let grad_out = GpuTensor::zeros_gpu(stream.clone(), &q_shape, DType::F32)?;
    let grad_q = GpuTensor::zeros_gpu(stream.clone(), &q_shape, DType::F32)?;
    let grad_k = GpuTensor::zeros_gpu(stream.clone(), &kv_shape, DType::F32)?;
    let grad_v = GpuTensor::zeros_gpu(stream.clone(), &kv_shape, DType::F32)?;

    let q_ptr = q.cu_ptr(&stream)?;
    let k_ptr = k.cu_ptr(&stream)?;
    let v_ptr = v.cu_ptr(&stream)?;
    let out_ptr = out.cu_ptr(&stream)?;
    let grad_out_ptr = grad_out.cu_ptr(&stream)?;
    let grad_q_ptr = grad_q.cu_ptr(&stream)?;
    let grad_k_ptr = grad_k.cu_ptr(&stream)?;
    let grad_v_ptr = grad_v.cu_ptr(&stream)?;

    for _ in 0..args.warmup {
        attention.forward(
            q_ptr,
            k_ptr,
            v_ptr,
            out_ptr,
            tokens,
            args.seq_len,
            args.heads,
            args.kv_heads,
            args.head_dim,
        )?;
        if args.backward {
            attention.backward(
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                grad_out_ptr,
                grad_q_ptr,
                grad_k_ptr,
                grad_v_ptr,
                tokens,
                args.seq_len,
                args.heads,
                args.kv_heads,
                args.head_dim,
            )?;
        }
    }
    stream
        .synchronize()
        .map_err(|e| pg_core::PgError::InvalidOp(format!("CUDA sync failed: {e:?}")))?;

    let mut forward_ms = 0.0;
    let mut backward_ms = 0.0;
    for _ in 0..args.iters {
        let start = Instant::now();
        attention.forward(
            q_ptr,
            k_ptr,
            v_ptr,
            out_ptr,
            tokens,
            args.seq_len,
            args.heads,
            args.kv_heads,
            args.head_dim,
        )?;
        stream
            .synchronize()
            .map_err(|e| pg_core::PgError::InvalidOp(format!("CUDA sync failed: {e:?}")))?;
        forward_ms += start.elapsed().as_secs_f64() * 1000.0;

        if args.backward {
            let start = Instant::now();
            attention.backward(
                q_ptr,
                k_ptr,
                v_ptr,
                out_ptr,
                grad_out_ptr,
                grad_q_ptr,
                grad_k_ptr,
                grad_v_ptr,
                tokens,
                args.seq_len,
                args.heads,
                args.kv_heads,
                args.head_dim,
            )?;
            stream
                .synchronize()
                .map_err(|e| pg_core::PgError::InvalidOp(format!("CUDA sync failed: {e:?}")))?;
            backward_ms += start.elapsed().as_secs_f64() * 1000.0;
        }
    }

    let iters = args.iters.max(1) as f64;
    let forward_avg_ms = forward_ms / iters;
    let backward_avg_ms = if args.backward {
        backward_ms / iters
    } else {
        0.0
    };
    println!("status=attention_bench_ok");
    println!("backend={}", attention.backend_name());
    println!("batch={}", args.batch);
    println!("seq_len={}", args.seq_len);
    println!("tokens={tokens}");
    println!("heads={}", args.heads);
    println!("kv_heads={}", args.kv_heads);
    println!("head_dim={}", args.head_dim);
    println!("warmup={}", args.warmup);
    println!("iters={}", args.iters);
    println!("backward={}", args.backward);
    println!("forward_avg_ms={forward_avg_ms:.3}");
    println!("backward_avg_ms={backward_avg_ms:.3}");
    println!(
        "forward_backward_avg_ms={:.3}",
        forward_avg_ms + backward_avg_ms
    );
    Ok(())
}

fn parse_args() -> PgResult<Args> {
    let mut args = Args::default();
    let mut iter = std::env::args().skip(1);
    while let Some(flag) = iter.next() {
        match flag.as_str() {
            "--batch" => args.batch = parse_next(&mut iter, "--batch")?,
            "--seq-len" => args.seq_len = parse_next(&mut iter, "--seq-len")?,
            "--heads" => args.heads = parse_next(&mut iter, "--heads")?,
            "--kv-heads" => args.kv_heads = parse_next(&mut iter, "--kv-heads")?,
            "--head-dim" => args.head_dim = parse_next(&mut iter, "--head-dim")?,
            "--warmup" => args.warmup = parse_next(&mut iter, "--warmup")?,
            "--iters" => args.iters = parse_next(&mut iter, "--iters")?,
            "--forward-only" => args.backward = false,
            "--backward" => args.backward = true,
            other => {
                return Err(pg_core::PgError::InvalidOp(format!(
                    "unknown attention-bench argument {other}"
                )));
            }
        }
    }
    if args.batch == 0 || args.seq_len == 0 || args.heads == 0 || args.head_dim == 0 {
        return Err(pg_core::PgError::InvalidOp(
            "batch, seq-len, heads, and head-dim must be nonzero".into(),
        ));
    }
    if args.kv_heads == 0 || args.heads % args.kv_heads != 0 {
        return Err(pg_core::PgError::InvalidOp(
            "kv-heads must be nonzero and divide heads".into(),
        ));
    }
    Ok(args)
}

fn parse_next<T: std::str::FromStr>(
    iter: &mut impl Iterator<Item = String>,
    flag: &str,
) -> PgResult<T> {
    let raw = iter
        .next()
        .ok_or_else(|| pg_core::PgError::InvalidOp(format!("{flag} requires a value")))?;
    raw.parse::<T>()
        .map_err(|_| pg_core::PgError::InvalidOp(format!("invalid value for {flag}: {raw}")))
}
