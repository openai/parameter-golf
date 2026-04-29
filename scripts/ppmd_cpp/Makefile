# Makefile for the `_ppmd_cpp` and `_ppmd_cuda` pybind11 extensions.
#
# No cmake required. Produces `_ppmd_cpp$(EXT_SUFFIX)` (and optionally
# `_ppmd_cuda$(EXT_SUFFIX)`) next to this Makefile, so
# `sys.path.insert(0, scripts/ppmd_cpp/)` finds them.
#
# Override PYTHON to target a different interpreter, e.g.:
#   make PYTHON=/some/other/python
# Default is the repo's `.venv-smoke/bin/python` (Python 3.12).

PYTHON ?= ../../.venv-smoke/bin/python

# Derive Python include / pybind11 includes / extension suffix from the
# selected interpreter. Evaluated lazily so `make clean` works without Python.
PY_INCLUDE  = $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_path("include"))')
PYBIND_INCS = $(shell $(PYTHON) -m pybind11 --includes)
EXT_SUFFIX  = $(shell $(PYTHON) -c 'import sysconfig; print(sysconfig.get_config_var("EXT_SUFFIX"))')

CXX      ?= g++
CXXFLAGS ?= -O3 -march=native -fopenmp -fPIC -std=c++17 -Wall

SRCS    = src/module.cpp src/ppmd.cpp src/virtual_ppmd.cpp src/sha256.cpp \
          src/trie.cpp src/scorer.cpp
OBJS    = $(SRCS:.cpp=.o)
TARGET  = _ppmd_cpp$(EXT_SUFFIX)

# ---------------------------------------------------------------------------
# CUDA target variables (overridable)
# ---------------------------------------------------------------------------
NVCC        ?= nvcc
CUDA_ARCHES ?= -gencode arch=compute_80,code=sm_80 -gencode arch=compute_90,code=sm_90
# No --use_fast_math, no -ffast-math, no --ftz=true, no -prec-div=false,
# no -prec-sqrt=false.  FMA fusion disabled for double-precision determinism.
NVCC_FLAGS_BASE = -O3 -std=c++17 -Xcompiler -fPIC -Xcompiler -fopenmp \
                  -Xcompiler -Wall --prec-div=true --prec-sqrt=true --fmad=false

# Glob all .cu files under src/cuda/ so new kernels are picked up automatically.
CUDA_SRCS = $(wildcard src/cuda/*.cu)
CUDA_OBJS = $(CUDA_SRCS:.cu=.o)

# CPU objects needed by _ppmd_cuda (all except module.o, which belongs to _ppmd_cpp).
CPU_SRCS_FOR_CUDA = src/ppmd.cpp src/virtual_ppmd.cpp src/sha256.cpp \
                    src/trie.cpp src/scorer.cpp
CPU_OBJS_FOR_CUDA = $(CPU_SRCS_FOR_CUDA:.cpp=.o)

CUDA_MODULE_OBJ = src/cuda_module.o
CUDA_TARGET     = _ppmd_cuda$(EXT_SUFFIX)

.PHONY: all clean cuda cuda-clean print-config

all: $(TARGET)

# ---------------------------------------------------------------------------
# _ppmd_cpp build rules
# ---------------------------------------------------------------------------

# Per-object compile rule (no -shared here; we link below).
src/%.o: src/%.cpp
	$(CXX) $(CXXFLAGS) -I$(PY_INCLUDE) $(PYBIND_INCS) -Isrc -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(CXXFLAGS) -shared $(OBJS) -o $(TARGET)

# ---------------------------------------------------------------------------
# _ppmd_cuda build rules
# ---------------------------------------------------------------------------

# Skip cleanly when nvcc is not in PATH.
cuda:
	@if ! command -v $(NVCC) >/dev/null 2>&1; then \
	    echo "nvcc not found, skipping CUDA build"; \
	    exit 0; \
	fi; \
	$(MAKE) $(CUDA_TARGET)

# CUDA object compilation rule.
src/cuda/%.o: src/cuda/%.cu
	mkdir -p src/cuda
	$(NVCC) $(NVCC_FLAGS_BASE) $(CUDA_ARCHES) -I$(PY_INCLUDE) $(PYBIND_INCS) -Isrc -c $< -o $@

# Link _ppmd_cuda with nvcc so CUDA runtime linkage is correct.
# Pass -lgomp so the CPU objects' OpenMP symbols resolve at runtime.
$(CUDA_TARGET): $(CPU_OBJS_FOR_CUDA) $(CUDA_MODULE_OBJ) $(CUDA_OBJS)
	$(NVCC) $(CUDA_ARCHES) -shared $^ -Xcompiler -fopenmp -lgomp -o $@

clean:
	rm -f _ppmd_cpp*.so src/*.o

cuda-clean:
	rm -f _ppmd_cuda*.so src/cuda/*.o

print-config:
	@echo "PYTHON           = $(PYTHON)"
	@echo "PY_INCLUDE       = $(PY_INCLUDE)"
	@echo "PYBIND_INCS      = $(PYBIND_INCS)"
	@echo "EXT_SUFFIX       = $(EXT_SUFFIX)"
	@echo "CXX              = $(CXX)"
	@echo "CXXFLAGS         = $(CXXFLAGS)"
	@echo "SRCS             = $(SRCS)"
	@echo "TARGET           = $(TARGET)"
	@echo "NVCC             = $(NVCC)"
	@echo "CUDA_ARCHES      = $(CUDA_ARCHES)"
	@echo "NVCC_FLAGS_BASE  = $(NVCC_FLAGS_BASE)"
	@echo "CUDA_SRCS        = $(CUDA_SRCS)"
	@echo "CUDA_TARGET      = $(CUDA_TARGET)"
