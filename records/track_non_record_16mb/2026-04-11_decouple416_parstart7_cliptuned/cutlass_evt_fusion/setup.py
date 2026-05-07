from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension
import os

CUTLASS_PATH = os.environ.get("CUTLASS_PATH", "/opt/cutlass")

setup(
    name="cutlass_evt_fusion",
    ext_modules=[
        CUDAExtension(
            name="cutlass_evt_fusion",
            sources=[
                "csrc/gemm_act_grad.cu",
                "csrc/torch_binding.cpp",
            ],
            include_dirs=[
                f"{CUTLASS_PATH}/include",
                f"{CUTLASS_PATH}/tools/util/include",
            ],
            extra_compile_args={
                "nvcc": [
                    "-std=c++17",
                    "-arch=sm_90a",
                    "-O3",
                    "--use_fast_math",
                    "--expt-relaxed-constexpr",
                    "-DNDEBUG",
                    "-DCUTLASS_ENABLE_TENSOR_CORE_MMA=1",
                ],
            },
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
