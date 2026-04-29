// Pybind11 module entry point for _ppmd_cpp.
//
// All binding logic lives in cpp_bindings.hpp (shared with _ppmd_cuda).
// `version()` returns "0.0.1" to preserve existing smoke tests.

#include <pybind11/pybind11.h>

#include "cpp_bindings.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_ppmd_cpp, m) {
    m.doc() = "Path A PPM-D C++ backend (Phase 3: trie + scorer)";
    register_cpp_bindings(m);
}
