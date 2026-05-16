use cudarc::driver::{CudaContext, sys as driver_sys};
use cudarc::nccl::{Comm, ReduceOp};
use half::bf16;
use std::sync::Arc;

const WARMUP_ITERS: usize = 10;
const BENCH_ITERS: usize = 50;

// Message sizes to sweep (in bf16 elements)
const SIZES: &[(usize, &str)] = &[
    (256 * 1024, "512 KB"),       // 256K * 2B = 512KB
    (1024 * 1024, "2 MB"),        // 1M * 2B = 2MB
    (4 * 1024 * 1024, "8 MB"),    // 4M * 2B = 8MB
    (16 * 1024 * 1024, "32 MB"),  // 16M * 2B = 32MB
    (64 * 1024 * 1024, "128 MB"), // 64M * 2B = 128MB
];

fn main() {
    env_logger::init();

    println!("=== Parameter Golf NCCL Benchmark ===\n");

    let n_devices = CudaContext::device_count().expect("Failed to query device count") as usize;
    println!("Devices: {n_devices}\n");

    if n_devices < 2 {
        println!("Need at least 2 GPUs for NCCL benchmarks. Skipping.");
        return;
    }

    // Create one stream per device for Comm::from_devices
    let streams: Vec<Arc<cudarc::driver::CudaStream>> = (0..n_devices)
        .map(|i| {
            let ctx = CudaContext::new(i).expect("Failed to create context");
            ctx.default_stream()
        })
        .collect();

    let comms = Comm::from_devices(streams.clone()).expect("Failed to create NCCL comms");

    // Print device info
    for (i, stream) in streams.iter().enumerate() {
        let name = stream.context().name().unwrap_or_else(|_| "unknown".into());
        println!("  GPU {i}: {name}");
    }
    println!();

    // --- AllReduce ---
    bench_all_reduce(&comms, &streams, n_devices);

    // --- ReduceScatter ---
    bench_reduce_scatter(&comms, &streams, n_devices);

    // --- AllGather ---
    bench_all_gather(&comms, &streams, n_devices);

    println!("\nNCCL benchmark complete.");
}

fn bench_all_reduce(comms: &[Comm], streams: &[Arc<cudarc::driver::CudaStream>], n_devices: usize) {
    println!("--- AllReduce (bf16, Sum) ---\n");
    println!(
        "{:>10}  {:>10}  {:>10}  {:>10}",
        "Size", "Time (ms)", "BusBW", "AlgBW"
    );
    println!("{}", "-".repeat(50));

    for &(n_elems, label) in SIZES {
        // Allocate per-device
        let send_bufs: Vec<_> = streams
            .iter()
            .map(|s| s.alloc_zeros::<bf16>(n_elems).expect("alloc failed"))
            .collect();
        let mut recv_bufs: Vec<_> = streams
            .iter()
            .map(|s| s.alloc_zeros::<bf16>(n_elems).expect("alloc failed"))
            .collect();

        // Warmup
        for _ in 0..WARMUP_ITERS {
            cudarc::nccl::group_start().expect("group_start");
            for i in 0..n_devices {
                comms[i]
                    .all_reduce::<_, _, bf16>(&send_bufs[i], &mut recv_bufs[i], &ReduceOp::Sum)
                    .expect("all_reduce failed");
            }
            cudarc::nccl::group_end().expect("group_end");
            for s in streams {
                s.synchronize().expect("sync");
            }
        }

        // Timed
        let ctx = streams[0].context();
        let start = ctx
            .new_event(Some(driver_sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event");
        let end = ctx
            .new_event(Some(driver_sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event");

        start.record(&streams[0]).expect("record");
        for _ in 0..BENCH_ITERS {
            cudarc::nccl::group_start().expect("group_start");
            for i in 0..n_devices {
                comms[i]
                    .all_reduce::<_, _, bf16>(&send_bufs[i], &mut recv_bufs[i], &ReduceOp::Sum)
                    .expect("all_reduce failed");
            }
            cudarc::nccl::group_end().expect("group_end");
        }
        // Sync all devices
        for s in streams {
            s.synchronize().expect("sync");
        }
        end.record(&streams[0]).expect("record");
        streams[0].synchronize().expect("sync");

        let total_ms = start.elapsed_ms(&end).expect("elapsed_ms");
        let ms_per_iter = total_ms as f64 / BENCH_ITERS as f64;
        let bytes = n_elems as f64 * 2.0; // bf16 = 2 bytes

        // Algorithm bandwidth = data_size / time
        let alg_bw_gbps = bytes / (ms_per_iter * 1e-3) / 1e9;
        // Bus bandwidth for all_reduce = 2 * (n-1)/n * data_size / time
        let bus_bw_gbps =
            2.0 * (n_devices - 1) as f64 / n_devices as f64 * bytes / (ms_per_iter * 1e-3) / 1e9;

        println!(
            "{:>10}  {:>10.3}  {:>8.1} GB/s  {:>8.1} GB/s",
            label, ms_per_iter, bus_bw_gbps, alg_bw_gbps
        );
    }
    println!();
}

fn bench_reduce_scatter(
    comms: &[Comm],
    streams: &[Arc<cudarc::driver::CudaStream>],
    n_devices: usize,
) {
    println!("--- ReduceScatter (bf16, Sum) ---\n");
    println!(
        "{:>10}  {:>10}  {:>10}  {:>10}",
        "Size", "Time (ms)", "BusBW", "AlgBW"
    );
    println!("{}", "-".repeat(50));

    for &(n_elems, label) in SIZES {
        // For reduce_scatter: send n_elems, recv n_elems/n_devices per rank
        let recv_elems = n_elems / n_devices;
        if recv_elems == 0 {
            continue;
        }

        let send_bufs: Vec<_> = streams
            .iter()
            .map(|s| s.alloc_zeros::<bf16>(n_elems).expect("alloc"))
            .collect();
        let mut recv_bufs: Vec<_> = streams
            .iter()
            .map(|s| s.alloc_zeros::<bf16>(recv_elems).expect("alloc"))
            .collect();

        for _ in 0..WARMUP_ITERS {
            cudarc::nccl::group_start().expect("group_start");
            for i in 0..n_devices {
                comms[i]
                    .reduce_scatter::<_, _, bf16>(&send_bufs[i], &mut recv_bufs[i], &ReduceOp::Sum)
                    .expect("reduce_scatter");
            }
            cudarc::nccl::group_end().expect("group_end");
            for s in streams {
                s.synchronize().expect("sync");
            }
        }

        let ctx = streams[0].context();
        let start = ctx
            .new_event(Some(driver_sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event");
        let end = ctx
            .new_event(Some(driver_sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event");

        start.record(&streams[0]).expect("record");
        for _ in 0..BENCH_ITERS {
            cudarc::nccl::group_start().expect("group_start");
            for i in 0..n_devices {
                comms[i]
                    .reduce_scatter::<_, _, bf16>(&send_bufs[i], &mut recv_bufs[i], &ReduceOp::Sum)
                    .expect("reduce_scatter");
            }
            cudarc::nccl::group_end().expect("group_end");
        }
        for s in streams {
            s.synchronize().expect("sync");
        }
        end.record(&streams[0]).expect("record");
        streams[0].synchronize().expect("sync");

        let total_ms = start.elapsed_ms(&end).expect("elapsed_ms");
        let ms_per_iter = total_ms as f64 / BENCH_ITERS as f64;
        let bytes = n_elems as f64 * 2.0;
        let alg_bw_gbps = bytes / (ms_per_iter * 1e-3) / 1e9;
        // Bus BW for reduce_scatter = (n-1)/n * data_size / time
        let bus_bw_gbps =
            (n_devices - 1) as f64 / n_devices as f64 * bytes / (ms_per_iter * 1e-3) / 1e9;

        println!(
            "{:>10}  {:>10.3}  {:>8.1} GB/s  {:>8.1} GB/s",
            label, ms_per_iter, bus_bw_gbps, alg_bw_gbps
        );
    }
    println!();
}

fn bench_all_gather(comms: &[Comm], streams: &[Arc<cudarc::driver::CudaStream>], n_devices: usize) {
    println!("--- AllGather (bf16) ---\n");
    println!(
        "{:>10}  {:>10}  {:>10}  {:>10}",
        "Size", "Time (ms)", "BusBW", "AlgBW"
    );
    println!("{}", "-".repeat(50));

    for &(n_elems, label) in SIZES {
        // For all_gather: each rank sends n_elems/n_devices, receives n_elems
        let send_elems = n_elems / n_devices;
        if send_elems == 0 {
            continue;
        }

        let send_bufs: Vec<_> = streams
            .iter()
            .map(|s| s.alloc_zeros::<bf16>(send_elems).expect("alloc"))
            .collect();
        let mut recv_bufs: Vec<_> = streams
            .iter()
            .map(|s| s.alloc_zeros::<bf16>(n_elems).expect("alloc"))
            .collect();

        for _ in 0..WARMUP_ITERS {
            cudarc::nccl::group_start().expect("group_start");
            for i in 0..n_devices {
                comms[i]
                    .all_gather::<_, _, bf16>(&send_bufs[i], &mut recv_bufs[i])
                    .expect("all_gather");
            }
            cudarc::nccl::group_end().expect("group_end");
            for s in streams {
                s.synchronize().expect("sync");
            }
        }

        let ctx = streams[0].context();
        let start = ctx
            .new_event(Some(driver_sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event");
        let end = ctx
            .new_event(Some(driver_sys::CUevent_flags::CU_EVENT_DEFAULT))
            .expect("event");

        start.record(&streams[0]).expect("record");
        for _ in 0..BENCH_ITERS {
            cudarc::nccl::group_start().expect("group_start");
            for i in 0..n_devices {
                comms[i]
                    .all_gather::<_, _, bf16>(&send_bufs[i], &mut recv_bufs[i])
                    .expect("all_gather");
            }
            cudarc::nccl::group_end().expect("group_end");
        }
        for s in streams {
            s.synchronize().expect("sync");
        }
        end.record(&streams[0]).expect("record");
        streams[0].synchronize().expect("sync");

        let total_ms = start.elapsed_ms(&end).expect("elapsed_ms");
        let ms_per_iter = total_ms as f64 / BENCH_ITERS as f64;
        let bytes = n_elems as f64 * 2.0;
        let alg_bw_gbps = bytes / (ms_per_iter * 1e-3) / 1e9;
        // Bus BW for all_gather = (n-1)/n * data_size / time
        let bus_bw_gbps =
            (n_devices - 1) as f64 / n_devices as f64 * bytes / (ms_per_iter * 1e-3) / 1e9;

        println!(
            "{:>10}  {:>10.3}  {:>8.1} GB/s  {:>8.1} GB/s",
            label, ms_per_iter, bus_bw_gbps, alg_bw_gbps
        );
    }
    println!();
}
