cmake_minimum_required(VERSION 3.15...3.27)
project(fast_ngram LANGUAGES CXX)

find_package(Python 3.8
  REQUIRED COMPONENTS Interpreter Development.Module
  OPTIONAL_COMPONENTS Development.SABIModule)

execute_process(
  COMMAND "${Python_EXECUTABLE}" -m nanobind --cmake_dir
  OUTPUT_STRIP_TRAILING_WHITESPACE OUTPUT_VARIABLE NB_DIR)
list(APPEND CMAKE_PREFIX_PATH "${NB_DIR}")

find_package(nanobind CONFIG REQUIRED)

nanobind_add_module(fused_expert_ext fused_expert_blend.cpp)
target_compile_features(fused_expert_ext PRIVATE cxx_std_17)
target_compile_options(fused_expert_ext PRIVATE
    -O3
    -march=native
    -funroll-loops
    -fno-math-errno
    -ffinite-math-only
)
install(TARGETS fused_expert_ext LIBRARY DESTINATION .)
