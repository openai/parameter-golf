use cudarc::driver::{CudaContext, sys};
use half::bf16;
use std::time::Instant;

fn main() {
    env_logger::init();

    println!("=== Parameter Golf GPU Smoke Test ===\n");

    let n_devices = CudaContext::device_count().expect("Failed to query device count");
    println!("CUDA devices found: {n_devices}\n");

    for i in 0..n_devices as usize {
        let ctx = CudaContext::new(i).expect("Failed to create context");
        let name = ctx.name().unwrap_or_else(|_| "unknown".into());

        let sm_count = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT)
            .unwrap_or(0);
        let clock_mhz = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_CLOCK_RATE)
            .unwrap_or(0)
            / 1000;
        let mem_clock_mhz = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_MEMORY_CLOCK_RATE)
            .unwrap_or(0)
            / 1000;
        let bus_width = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_GLOBAL_MEMORY_BUS_WIDTH)
            .unwrap_or(0);
        let cc_major = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR)
            .unwrap_or(0);
        let cc_minor = ctx
            .attribute(sys::CUdevice_attribute::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR)
            .unwrap_or(0);

        println!("Device {i}: {name}");
        println!("  Compute capability: {cc_major}.{cc_minor}");
        println!("  SMs: {sm_count}");
        println!("  Core clock: {clock_mhz} MHz");
        println!("  Memory clock: {mem_clock_mhz} MHz, bus width: {bus_width}-bit");

        // Memory bandwidth estimate: 2 * mem_clock_MHz * (bus_width/8) bytes/s
        let bw_gbps = 2.0 * mem_clock_mhz as f64 * (bus_width as f64 / 8.0) / 1000.0;
        println!("  Theoretical mem BW: {bw_gbps:.0} GB/s");

        // Alloc + H2D + D2H test
        let stream = ctx.default_stream();
        let n_elems = 1024 * 1024; // 1M bf16 = 2MB

        let t0 = Instant::now();
        let host_data: Vec<bf16> = (0..n_elems)
            .map(|i| bf16::from_f32((i % 1000) as f32 * 0.001))
            .collect();
        let dev = stream.memcpy_stod(&host_data).expect("H2D copy failed");
        let roundtrip = stream.memcpy_dtov(&dev).expect("D2H copy failed");
        let dt = t0.elapsed();

        assert_eq!(roundtrip.len(), n_elems);
        assert_eq!(roundtrip[0], host_data[0]);
        assert_eq!(roundtrip[999], host_data[999]);

        let mbytes = (n_elems * 2) as f64 / 1e6;
        println!(
            "  H2D + D2H roundtrip: {:.2} MB in {:.2} ms ({:.1} GB/s effective)",
            mbytes,
            dt.as_secs_f64() * 1000.0,
            2.0 * mbytes / dt.as_secs_f64() / 1000.0
        );

        // Alloc zeros test
        let zeros = stream
            .alloc_zeros::<bf16>(n_elems)
            .expect("alloc_zeros failed");
        let zhost = stream.memcpy_dtov(&zeros).expect("D2H zeros failed");
        assert!(zhost.iter().all(|&v| v == bf16::ZERO));
        println!("  alloc_zeros: OK ({n_elems} bf16 elements)");

        println!();
    }

    println!("Smoke test PASSED");
}
