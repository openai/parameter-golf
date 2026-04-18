use std::env;
use std::process::Command;
use std::path::PathBuf;

fn main() {
    // Only attempt to build CUDA/C++ extensions if the 'cuda' feature is enabled.
    if env::var("CARGO_FEATURE_CUDA").is_err() {
        return;
    }

    // Check if nvcc is available
    if Command::new("nvcc").arg("--version").status().is_err() {
        println!("cargo:warning=nvcc not found. Skipping cuDNN SDPA wrapper compilation.");
        return;
    }

    // We only compile the C++ cuDNN wrapper if nvcc is found.
    // Ensure `cc` is listed in build-dependencies.
    let mut build = cc::Build::new();
    
    build.cuda(true)
         .flag("-O3")
         .flag("-use_fast_math")
         // Allow C++17 for cuDNN headers
         .flag("-std=c++17")
         .file("cpp/sdpa.cu")
         .compile("flash_attn_cudnn");

    // Link against cudnn and cudart
    println!("cargo:rustc-link-lib=cudnn");
    println!("cargo:rustc-link-lib=cudart");
    
    // Re-run if the C++ file changes
    println!("cargo:rerun-if-changed=cpp/sdpa.cu");
    println!("cargo:rerun-if-changed=build.rs");
}
