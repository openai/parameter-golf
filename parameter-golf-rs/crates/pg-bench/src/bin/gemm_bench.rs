use cudarc::driver::CudaContext;
use half::bf16;
use pg_kernels::gemm::GemmEngine;

/// Competition-relevant GEMM shapes.
/// (label, m, n, k) — m is the batch*seq dimension.
const SHAPES: &[(&str, usize, usize, usize)] = &[
    // Attention Q/K/V/O projections
    ("attn_qvo  384x512x512", 384, 512, 512),
    // Attention K/V (d_head=256)
    ("attn_kv   384x256x512", 384, 256, 512),
    // MLP up projection
    ("mlp_up    384x1536x512", 384, 1536, 512),
    // MLP down projection
    ("mlp_down  384x512x1536", 384, 512, 1536),
    // Larger batch
    ("attn_qvo  768x512x512", 768, 512, 512),
    ("mlp_up    768x1536x512", 768, 1536, 512),
    // Record-shaped local batch: B=48, T=2048 => M=98,304 per rank.
    ("record_qkv 98304x1024x512", 98_304, 1024, 512),
    ("record_o   98304x512x512", 98_304, 512, 512),
    ("record_up  98304x1536x512", 98_304, 1536, 512),
    ("record_dn  98304x512x1536", 98_304, 512, 1536),
    // Newton-Schulz square (per-layer, not batched here)
    ("ns_square 512x512x512", 512, 512, 512),
];

const WARMUP_ITERS: usize = 20;
const BENCH_ITERS: usize = 100;

fn main() {
    env_logger::init();

    println!(
        "=== Parameter Golf F32/TF32 GEMM Benchmark ({}) ===\n",
        pg_kernels::gemm::f32_compute_mode_label()
    );

    let ctx = CudaContext::new(0).expect("Failed to create CUDA context");
    let stream = ctx.default_stream();
    let name = ctx.name().unwrap_or_else(|_| "unknown".into());
    println!("Device: {name}\n");

    let engine = GemmEngine::new(stream.clone()).expect("Failed to create GemmEngine");

    println!(
        "{:<28} {:>6} {:>6} {:>6}  {:>10}  {:>8}",
        "Shape", "M", "N", "K", "TFLOPS", "us/iter"
    );
    println!("{}", "-".repeat(76));

    for &(label, m, n, k) in SHAPES {
        let a = stream.alloc_zeros::<f32>(m * k).expect("alloc A failed");
        let b = stream.alloc_zeros::<f32>(k * n).expect("alloc B failed");
        let c = stream.alloc_zeros::<f32>(m * n).expect("alloc C failed");

        // Warmup
        for _ in 0..WARMUP_ITERS {
            unsafe {
                engine
                    .matmul_f32(
                        cudarc::driver::DevicePtr::device_ptr(&a, engine.stream()).0,
                        cudarc::driver::DevicePtr::device_ptr(&b, engine.stream()).0,
                        cudarc::driver::DevicePtr::device_ptr(&c, engine.stream()).0,
                        m,
                        n,
                        k,
                        1.0,
                        0.0,
                    )
                    .expect("warmup gemm failed");
            }
        }

        // Timed region using CUDA events
        let start = ctx
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event create failed");
        let end = ctx
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event create failed");

        start.record(&stream).expect("event record failed");
        for _ in 0..BENCH_ITERS {
            unsafe {
                engine
                    .matmul_f32(
                        cudarc::driver::DevicePtr::device_ptr(&a, engine.stream()).0,
                        cudarc::driver::DevicePtr::device_ptr(&b, engine.stream()).0,
                        cudarc::driver::DevicePtr::device_ptr(&c, engine.stream()).0,
                        m,
                        n,
                        k,
                        1.0,
                        0.0,
                    )
                    .expect("bench gemm failed");
            }
        }
        end.record(&stream).expect("event record failed");
        stream.synchronize().expect("sync failed");

        let total_ms = start.elapsed_ms(&end).expect("elapsed_ms failed");
        let us_per_iter = (total_ms as f64 * 1000.0) / BENCH_ITERS as f64;

        // FLOPS = 2 * M * N * K per GEMM
        let flops_per_iter = 2.0 * m as f64 * n as f64 * k as f64;
        let tflops = flops_per_iter * BENCH_ITERS as f64 / (total_ms as f64 * 1e-3) / 1e12;

        println!(
            "{:<28} {:>6} {:>6} {:>6}  {:>10.2}  {:>8.1}",
            label, m, n, k, tflops, us_per_iter
        );
    }

    println!("\n--- BF16 Tensor-Core GEMM, F32 Output ---\n");
    println!(
        "{:<28} {:>6} {:>6} {:>6}  {:>10}  {:>8}",
        "Shape", "M", "N", "K", "TFLOPS", "us/iter"
    );
    println!("{}", "-".repeat(76));

    for &(label, m, n, k) in SHAPES {
        let a = stream.alloc_zeros::<bf16>(m * k).expect("alloc A failed");
        let b = stream.alloc_zeros::<bf16>(n * k).expect("alloc B failed");
        let c = stream.alloc_zeros::<f32>(m * n).expect("alloc C failed");

        for _ in 0..WARMUP_ITERS {
            unsafe {
                engine
                    .matmul_bf16_bt_to_f32(
                        cudarc::driver::DevicePtr::device_ptr(&a, engine.stream()).0,
                        cudarc::driver::DevicePtr::device_ptr(&b, engine.stream()).0,
                        cudarc::driver::DevicePtr::device_ptr(&c, engine.stream()).0,
                        m,
                        n,
                        k,
                        1.0,
                        0.0,
                    )
                    .expect("warmup bf16 gemm failed");
            }
        }

        let start = ctx
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event create failed");
        let end = ctx
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event create failed");

        start.record(&stream).expect("event record failed");
        for _ in 0..BENCH_ITERS {
            unsafe {
                engine
                    .matmul_bf16_bt_to_f32(
                        cudarc::driver::DevicePtr::device_ptr(&a, engine.stream()).0,
                        cudarc::driver::DevicePtr::device_ptr(&b, engine.stream()).0,
                        cudarc::driver::DevicePtr::device_ptr(&c, engine.stream()).0,
                        m,
                        n,
                        k,
                        1.0,
                        0.0,
                    )
                    .expect("bench bf16 gemm failed");
            }
        }
        end.record(&stream).expect("event record failed");
        stream.synchronize().expect("sync failed");

        let total_ms = start.elapsed_ms(&end).expect("elapsed_ms failed");
        let us_per_iter = (total_ms as f64 * 1000.0) / BENCH_ITERS as f64;
        let flops_per_iter = 2.0 * m as f64 * n as f64 * k as f64;
        let tflops = flops_per_iter * BENCH_ITERS as f64 / (total_ms as f64 * 1e-3) / 1e12;

        println!(
            "{:<28} {:>6} {:>6} {:>6}  {:>10.2}  {:>8.1}",
            label, m, n, k, tflops, us_per_iter
        );
    }

    // Batched GEMM for Newton-Schulz
    println!("\n--- Strided Batched GEMM (Newton-Schulz) ---\n");
    println!(
        "{:<28} {:>5} {:>6} {:>6} {:>6}  {:>10}  {:>8}",
        "Shape", "B", "M", "N", "K", "TFLOPS", "us/iter"
    );
    println!("{}", "-".repeat(82));

    let batched_shapes: &[(&str, usize, usize, usize, usize)] = &[
        // X @ X^T for NS5: [22, 512, 512] x [22, 512, 512]
        ("ns5_XXT 22x512x512x512", 22, 512, 512, 512),
        // B @ X for NS5
        ("ns5_BX  22x512x512x512", 22, 512, 512, 512),
        // Smaller: just qo_bank half (11 layers)
        ("ns5_half 11x512x512x512", 11, 512, 512, 512),
    ];

    for &(label, batch, m, n, k) in batched_shapes {
        let a = stream
            .alloc_zeros::<f32>(batch * m * k)
            .expect("alloc A failed");
        let b = stream
            .alloc_zeros::<f32>(batch * k * n)
            .expect("alloc B failed");
        let c = stream
            .alloc_zeros::<f32>(batch * m * n)
            .expect("alloc C failed");

        for _ in 0..WARMUP_ITERS {
            unsafe {
                engine
                    .batched_matmul_f32(
                        cudarc::driver::DevicePtr::device_ptr(&a, engine.stream()).0,
                        cudarc::driver::DevicePtr::device_ptr(&b, engine.stream()).0,
                        cudarc::driver::DevicePtr::device_ptr(&c, engine.stream()).0,
                        batch,
                        m,
                        n,
                        k,
                        1.0,
                        0.0,
                    )
                    .expect("warmup batched gemm failed");
            }
        }

        let start = ctx
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event create failed");
        let end = ctx
            .new_event(Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event create failed");

        start.record(&stream).expect("event record failed");
        for _ in 0..BENCH_ITERS {
            unsafe {
                engine
                    .batched_matmul_f32(
                        cudarc::driver::DevicePtr::device_ptr(&a, engine.stream()).0,
                        cudarc::driver::DevicePtr::device_ptr(&b, engine.stream()).0,
                        cudarc::driver::DevicePtr::device_ptr(&c, engine.stream()).0,
                        batch,
                        m,
                        n,
                        k,
                        1.0,
                        0.0,
                    )
                    .expect("bench batched gemm failed");
            }
        }
        end.record(&stream).expect("event record failed");
        stream.synchronize().expect("sync failed");

        let total_ms = start.elapsed_ms(&end).expect("elapsed_ms failed");
        let us_per_iter = (total_ms as f64 * 1000.0) / BENCH_ITERS as f64;
        let flops_per_iter = 2.0 * batch as f64 * m as f64 * n as f64 * k as f64;
        let tflops = flops_per_iter * BENCH_ITERS as f64 / (total_ms as f64 * 1e-3) / 1e12;

        println!(
            "{:<28} {:>5} {:>6} {:>6} {:>6}  {:>10.2}  {:>8.1}",
            label, batch, m, n, k, tflops, us_per_iter
        );
    }

    println!("\nGEMM benchmark complete.");
}
