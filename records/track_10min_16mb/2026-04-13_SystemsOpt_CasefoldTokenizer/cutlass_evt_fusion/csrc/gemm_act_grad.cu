// CUTLASS 3.x EVT kernel: fused GEMM * elementwise multiply
// Computes: dpre = (go @ down_w.T) * act_grad
// Where act_grad = f'(pre) is pre-computed in the forward pass.
//
// Layout convention:
//   go:       (M, K) bf16 row-major
//   down_w:   (K, N) bf16 row-major -- CUTLASS B(N,K) with RowMajor layout
//   act_grad: (M, N) bf16 row-major
//   dpre:     (M, N) bf16 row-major output

#include "cutlass/cutlass.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_load_tma_warpspecialized.hpp"
#include "cutlass/epilogue/fusion/sm90_visitor_compute_tma_warpspecialized.hpp"
#include "cute/tensor.hpp"
#include "cutlass/util/packed_stride.hpp"
#include <iostream>

using namespace cute;

using ElementAcc = float;
using ElementCompute = float;
using ElementOutput = cutlass::bfloat16_t;
using ElementAux = cutlass::bfloat16_t;

using namespace cutlass::epilogue::fusion;

using TileShape = Shape<_128, _256, _64>;
using ClusterShape = Shape<_1, _1, _1>;
using EpilogueTile = cutlass::epilogue::collective::EpilogueTileAuto;
using EpilogueSchedule = cutlass::epilogue::TmaWarpSpecializedCooperative;

using EpiDesc = cutlass::epilogue::collective::detail::EpilogueDescriptor<
    TileShape, EpilogueTile, ElementOutput, ElementOutput, EpilogueSchedule>;

using AuxDesc = cutlass::epilogue::collective::detail::AuxLoadDescriptor<
    EpiDesc, cutlass::layout::RowMajor, ElementAux>;

using AuxLoad = Sm90AuxLoad<
    AuxDesc::Stages,
    typename EpiDesc::EpilogueTile,
    typename AuxDesc::Element,
    typename AuxDesc::Stride,
    typename AuxDesc::SmemLayoutAtom,
    typename AuxDesc::CopyOpS2R>;

using Compute = Sm90Compute<
    cutlass::multiplies,
    ElementOutput,
    ElementCompute,
    cutlass::FloatRoundStyle::round_to_nearest>;

using EVT = Sm90EVT<Compute, Sm90AccFetch, AuxLoad>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    TileShape,
    ClusterShape,
    EpilogueTile,
    ElementAcc, ElementCompute,
    ElementOutput, cutlass::layout::RowMajor, 8,
    ElementOutput, cutlass::layout::RowMajor, 8,
    EpilogueSchedule,
    EVT
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90,
    cutlass::arch::OpClassTensorOp,
    ElementOutput, cutlass::layout::RowMajor, 8,
    ElementOutput, cutlass::layout::RowMajor, 8,
    ElementAcc,
    TileShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        sizeof(typename CollectiveEpilogue::SharedStorage)>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue>;

using GemmOp = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

void launch_gemm_mul(
    void const* ptr_go,
    void const* ptr_down_w,
    void const* ptr_act_grad,
    void* ptr_dpre,
    int M, int N, int K,
    cudaStream_t stream)
{
    using StrideA = cutlass::gemm::TagToStrideA_t<cutlass::layout::RowMajor>;
    using StrideB = cutlass::gemm::TagToStrideB_t<cutlass::layout::RowMajor>;
    using StrideC = cutlass::gemm::TagToStrideC_t<cutlass::layout::RowMajor>;

    int L = 1;
    auto prob_shape = make_shape(M, N, K, L);

    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, L));
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, L));
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, L));
    auto stride_Aux = cutlass::make_cute_packed_stride(
        typename AuxDesc::Stride{}, cute::make_shape(M, N, L));

    typename EVT::Arguments evt_args{
        {},
        {
            static_cast<ElementAux const*>(ptr_act_grad),
            ElementAux(0),
            stride_Aux
        },
        {}
    };

    typename GemmOp::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        prob_shape,
        {
            static_cast<ElementOutput const*>(ptr_go),
            stride_A,
            static_cast<ElementOutput const*>(ptr_down_w),
            stride_B,
        },
        {
            evt_args,
            static_cast<ElementOutput const*>(ptr_dpre),
            stride_C,
            static_cast<ElementOutput*>(ptr_dpre),
            stride_C,
        }
    };

    GemmOp gemm_op;
    size_t workspace_size = GemmOp::get_workspace_size(args);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    auto status = gemm_op.initialize(args, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        std::cerr << "CUTLASS initialize failed: " << cutlassGetStatusString(status) << std::endl;
        if (workspace) cudaFree(workspace);
        exit(EXIT_FAILURE);
    }

    status = gemm_op.run(stream);
    if (status != cutlass::Status::kSuccess) {
        cudaError_t cuda_err = cudaStreamSynchronize(stream);
        std::cerr << "CUTLASS run failed: " << cutlassGetStatusString(status)
                  << " CUDA: " << cudaGetErrorString(cuda_err) << std::endl;
        if (workspace) cudaFree(workspace);
        exit(EXIT_FAILURE);
    }

    if (workspace) cudaFree(workspace);
}
