use std::env;
use std::path::Path;
use std::process::Command;

fn main() {
    println!("cargo:rustc-check-cfg=cfg(has_cuda_cpp)");
    println!("cargo:rustc-check-cfg=cfg(has_cudnn_frontend_sdpa)");

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

    build
        .cuda(true)
        .flag("-O3")
        // Allow C++17 for CUDA/C++ attention sources.
        .flag("-std=c++17")
        .file("cpp/sdpa.cu")
        .compile("naive_sdpa_f32");

    println!("cargo:rustc-cfg=has_cuda_cpp");

    if cudnn_frontend_probe() {
        let mut cudnn_build = cc::Build::new();
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
    Command::new("nvcc")
        .arg("-std=c++17")
        .arg("-c")
        .arg(&probe_src)
        .arg("-o")
        .arg(&probe_obj)
        .status()
        .map(|status| status.success())
        .unwrap_or(false)
}
