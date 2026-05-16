use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_cuda_cpp)");
    println!("cargo:rustc-check-cfg=cfg(has_cudnn_frontend_sdpa)");
    println!("cargo:rustc-check-cfg=cfg(has_fused_output_ce)");

    // Only attempt to build CUDA/C++ extensions if the 'cuda' feature is enabled.
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    // Check if nvcc is available
    if Command::new("nvcc").arg("--version").status().is_err() {
        println!("cargo:warning=nvcc not found. Skipping CUDA/C++ F32 SDPA compilation.");
        return;
    }

    // We always compile the CUDA/C++ F32 SDPA backend when nvcc is found. This
    // remains a parity/debug backend, not a record-grade attention path.
    let mut build = cc::Build::new();

    add_cuda_arch_flags(&mut build);
    build
        .cuda(true)
        .flag("-O3")
        // Allow C++17 for CUDA/C++ attention sources.
        .flag("-std=c++17")
        .file("cpp/sdpa.cu")
        .compile("naive_sdpa_f32");

    println!("cargo:rustc-cfg=has_cuda_cpp");

    if fused_output_ce_probe() {
        let mut ce_build = cc::Build::new();
        add_cuda_arch_flags(&mut ce_build);
        ce_build
            .cuda(true)
            .flag("-O3")
            .flag("-std=c++17")
            .file("cpp/fused_output_ce.cu")
            .compile("fused_output_ce");
        println!("cargo:rustc-cfg=has_fused_output_ce");
        println!("cargo:rustc-link-lib=cublas");
        println!("cargo:rerun-if-changed=cpp/fused_output_ce.cu");
    } else {
        println!(
            "cargo:warning=cuBLAS headers not found or did not compile. Skipping fused output CE backend."
        );
    }

    if cudnn_frontend_probe() {
        let mut cudnn_build = cc::Build::new();
        add_cuda_arch_flags(&mut cudnn_build);
        cudnn_build
            .cuda(true)
            .flag("-O3")
            .flag("-std=c++17")
            .file("cpp/cudnn_sdpa.cu")
            .compile("cudnn_frontend_sdpa");
        println!("cargo:rustc-cfg=has_cudnn_frontend_sdpa");
        println!("cargo:rustc-link-lib=cudnn");
        println!("cargo:rustc-link-lib=nvrtc");
        println!("cargo:rerun-if-changed=cpp/cudnn_sdpa.cu");
    } else {
        println!(
            "cargo:warning=cudnn_frontend.h not found or did not compile. Skipping cuDNN frontend SDPA backend."
        );
    }

    // Link against cudart. The optional cuDNN path also links libcudnn above.
    println!("cargo:rustc-link-lib=cudart");

    // Re-run if the C++ file changes
    println!("cargo:rerun-if-changed=cpp/sdpa.cu");
    println!("cargo:rerun-if-changed=build.rs");
}

fn add_cuda_arch_flags(build: &mut cc::Build) {
    for arch in cuda_arches() {
        build.flag(&format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
    }
}

fn add_cuda_arch_args(command: &mut Command) {
    for arch in cuda_arches() {
        command.arg(format!("-gencode=arch=compute_{arch},code=sm_{arch}"));
    }
}

fn cuda_arches() -> Vec<String> {
    let raw = env::var("PG_CUDA_NVCC_ARCH")
        .or_else(|_| env::var("CUDAARCHS"))
        .unwrap_or_else(|_| "90".to_string());
    let mut any = false;
    let mut arches = Vec::new();
    for arch in raw
        .split([';', ',', ' '])
        .map(str::trim)
        .filter(|arch| !arch.is_empty())
    {
        let arch = arch
            .strip_prefix("sm_")
            .or_else(|| arch.strip_prefix("compute_"))
            .unwrap_or(arch);
        if !arch.chars().all(|ch| ch.is_ascii_digit()) {
            println!("cargo:warning=ignoring invalid CUDA architecture {arch:?}");
            continue;
        }
        arches.push(arch.to_string());
        any = true;
    }
    if !any {
        arches.push("90".to_string());
    }
    arches
}

fn fused_output_ce_probe() -> bool {
    if env::var("PG_DISABLE_FUSED_OUTPUT_CE")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
    {
        return false;
    }

    let out_dir = match env::var("OUT_DIR") {
        Ok(out_dir) => out_dir,
        Err(_) => return false,
    };
    let probe_src = Path::new(&out_dir).join("fused_output_ce_probe.cu");
    let probe_obj = Path::new(&out_dir).join("fused_output_ce_probe.o");
    let source = r#"
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
int main() {
    cublasHandle_t handle = nullptr;
    return handle == nullptr ? 0 : 1;
}
"#;
    if std::fs::write(&probe_src, source).is_err() {
        return false;
    }
    let mut command = Command::new("nvcc");
    command.arg("-std=c++17");
    add_cuda_arch_args(&mut command);
    command
        .arg("-c")
        .arg(&probe_src)
        .arg("-o")
        .arg(&probe_obj)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}

fn cudnn_frontend_probe() -> bool {
    if env::var("PG_DISABLE_CUDNN_FRONTEND_SDPA")
        .map(|value| matches!(value.as_str(), "1" | "true" | "TRUE" | "yes" | "YES"))
        .unwrap_or(false)
    {
        return false;
    }

    let out_dir = match env::var("OUT_DIR") {
        Ok(out_dir) => out_dir,
        Err(_) => return false,
    };
    let probe_src = Path::new(&out_dir).join("cudnn_frontend_probe.cu");
    let probe_obj = Path::new(&out_dir).join("cudnn_frontend_probe.o");
    let source = r#"
#include <cudnn.h>
#include <cudnn_frontend.h>
int main() {
    auto graph = cudnn_frontend::graph::Graph();
    graph.set_io_data_type(cudnn_frontend::DataType_t::BFLOAT16)
         .set_intermediate_data_type(cudnn_frontend::DataType_t::FLOAT)
         .set_compute_data_type(cudnn_frontend::DataType_t::FLOAT);
    return 0;
}
"#;
    if std::fs::write(&probe_src, source).is_err() {
        return false;
    }
    let mut command = Command::new("nvcc");
    command.arg("-std=c++17");
    add_cuda_arch_args(&mut command);
    command
        .arg("-c")
        .arg(&probe_src)
        .arg("-o")
        .arg(&probe_obj)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}
