// pg-bench: GPU microbenchmarks for Parameter Golf on Modal H100:8
//
// Binaries:
//   smoke      — device enumeration, H2D/D2H sanity, alloc test
//   gemm-bench — bf16 GEMM TFLOPS sweep across competition shapes
//   nccl-bench — NCCL collective bandwidth (all_reduce, reduce_scatter, all_gather)
