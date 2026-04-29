// cuda_module.cpp — pybind11 entry point for _ppmd_cuda.
//
// Re-exports the full CPU surface via register_cpp_bindings(), then adds
// a `cuda` submodule exposing CUDA runtime probe functions, the batched
// byte-prob kernel, the trie scoring kernel, and the full Path A array scorer.

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "cpp_bindings.hpp"
#include "cuda/flat_ctx.hpp"
#include "cuda/ppmd_cuda.hpp"
#include "cuda/trie_kernel.hpp"
#include "ppmd.hpp"
#include "scorer.hpp"
#include "trie.hpp"
#include "virtual_ppmd.hpp"

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Minimal RAII wrapper for device buffers (uses our CXX-friendly wrappers).
// ---------------------------------------------------------------------------
namespace {
struct CudaBuf {
    void* ptr = nullptr;
    explicit CudaBuf(size_t bytes) { ppmd_cuda::device_alloc(&ptr, bytes); }
    ~CudaBuf() { if (ptr) { ppmd_cuda::device_free(ptr); ptr = nullptr; } }
    CudaBuf(const CudaBuf&)            = delete;
    CudaBuf& operator=(const CudaBuf&) = delete;
    template <typename T> T* as() const { return static_cast<T*>(ptr); }
};
}  // anonymous namespace

PYBIND11_MODULE(_ppmd_cuda, m) {
    m.doc() = "Path A PPM-D CUDA backend (Phase 3: byte-prob kernel)";
    register_cpp_bindings(m);

    auto cuda = m.def_submodule("cuda", "CUDA runtime probes and kernels");
    cuda.def("available",                &ppmd_cuda::cuda_available);
    cuda.def("device_count",             &ppmd_cuda::cuda_device_count);
    cuda.def("runtime_version",          &ppmd_cuda::cuda_runtime_version);
    cuda.def("driver_version",           &ppmd_cuda::cuda_driver_version);
    cuda.def("device_name",              &ppmd_cuda::cuda_device_name,
             py::arg("device") = 0);
    cuda.def("compute_capability_major", &ppmd_cuda::cuda_compute_capability_major,
             py::arg("device") = 0);
    cuda.def("compute_capability_minor", &ppmd_cuda::cuda_compute_capability_minor,
             py::arg("device") = 0);

    // -----------------------------------------------------------------------
    // byte_probs_batched(state, windows) -> ndarray shape (n_windows, 256)
    //
    // state   : PPMDState or VirtualPPMDState
    // windows : list of bytes objects, each being the update bytes to apply
    //           via fork_and_update before computing byte_probs.
    // Returns : numpy float64 array of shape (n_windows, 256).
    // -----------------------------------------------------------------------
    cuda.def("byte_probs_batched",
        [](py::object state_obj, py::list win_list) -> py::array_t<double> {
            // --- Determine state type and build device table ---------------
            ppmd_cuda::DeviceCtxTableHandle handle;
            int order = 5;
            bool handle_valid = false;

            if (py::isinstance<ppmd::PPMDState>(state_obj)) {
                const auto& state = state_obj.cast<const ppmd::PPMDState&>();
                handle = ppmd_cuda::build_device_ctx_table(state);
                order  = state.order();
                handle_valid = true;
            } else if (py::isinstance<ppmd::VirtualPPMDState>(state_obj)) {
                const auto& vstate = state_obj.cast<const ppmd::VirtualPPMDState&>();
                handle = ppmd_cuda::build_device_ctx_table(vstate);
                order  = vstate.order();
                handle_valid = true;
            } else {
                throw std::invalid_argument(
                    "state_obj must be PPMDState or VirtualPPMDState");
            }

            // RAII guard: free device ctx table on exit.
            struct HandleGuard {
                ppmd_cuda::DeviceCtxTableHandle& h;
                bool active;
                ~HandleGuard() { if (active) ppmd_cuda::free_device_ctx_table(h); }
            } guard{handle, handle_valid};

            int n_windows = static_cast<int>(win_list.size());

            if (n_windows == 0) {
                // Return empty (0, 256) array.
                std::vector<py::ssize_t> shape = {0, 256};
                return py::array_t<double>(shape);
            }

            // --- Flatten windows into a contiguous byte buffer ------------
            std::vector<uint8_t> wins_flat;
            std::vector<int32_t> offsets(n_windows);
            std::vector<int32_t> lens(n_windows);

            for (int i = 0; i < n_windows; ++i) {
                py::bytes win = win_list[i].cast<py::bytes>();
                char*     buf = nullptr;
                Py_ssize_t len = 0;
                if (PYBIND11_BYTES_AS_STRING_AND_SIZE(win.ptr(), &buf, &len) != 0)
                    throw py::error_already_set();
                offsets[i] = static_cast<int32_t>(wins_flat.size());
                lens[i]    = static_cast<int32_t>(len);
                wins_flat.insert(wins_flat.end(),
                                 reinterpret_cast<const uint8_t*>(buf),
                                 reinterpret_cast<const uint8_t*>(buf) + len);
            }

            // --- Allocate device buffers and copy --------------------------
            size_t wins_bytes    = wins_flat.empty() ? 1u : wins_flat.size();
            size_t offsets_bytes = static_cast<size_t>(n_windows) * sizeof(int32_t);
            size_t lens_bytes    = static_cast<size_t>(n_windows) * sizeof(int32_t);
            size_t out_bytes     = static_cast<size_t>(n_windows) * 256 * sizeof(double);

            CudaBuf d_wins(wins_bytes);
            CudaBuf d_offs(offsets_bytes);
            CudaBuf d_lens(lens_bytes);
            CudaBuf d_out(out_bytes);

            if (!wins_flat.empty())
                ppmd_cuda::device_memcpy_h2d(d_wins.ptr, wins_flat.data(), wins_flat.size());
            ppmd_cuda::device_memcpy_h2d(d_offs.ptr, offsets.data(), offsets_bytes);
            ppmd_cuda::device_memcpy_h2d(d_lens.ptr, lens.data(), lens_bytes);

            // --- Launch kernel --------------------------------------------
            {
                py::gil_scoped_release release;
                ppmd_cuda::launch_byte_prob_kernel(
                    handle,
                    d_wins.as<uint8_t>(),
                    d_offs.as<int32_t>(),
                    d_lens.as<int32_t>(),
                    n_windows, order,
                    d_out.as<double>());
            }

            // --- Copy results back to host --------------------------------
            std::vector<py::ssize_t> shape = {n_windows, 256};
            py::array_t<double> result(shape);
            ppmd_cuda::device_memcpy_d2h(
                result.mutable_data(), d_out.ptr, out_bytes);

            return result;
        },
        py::arg("state"),
        py::arg("windows"),
        "Compute byte_probs for each window via the CUDA kernel.\n\n"
        "Equivalent to: for each w in windows, clone_virtual() then\n"
        "fork_and_update each byte, then byte_probs().\n"
        "Returns ndarray float64 shape (len(windows), 256).");

    // -----------------------------------------------------------------------
    // trie_partial_z_and_target_batched(state, trie, target_ids)
    //
    // Given a fixed PPM state and a Trie, compute (z, target_q) for each
    // target_id in the batch.  z is the same for all target_ids.
    // Returns: ndarray float64 shape (n, 2) where row i = (z, target_q_i).
    // -----------------------------------------------------------------------
    cuda.def("trie_partial_z_and_target_batched",
        [](py::object state_obj, const ppmd::Trie& trie,
           py::object target_ids_obj) -> py::array_t<double> {

            ppmd_cuda::DeviceCtxTableHandle ctx_handle;
            int order = 5;
            bool ctx_valid = false;

            if (py::isinstance<ppmd::PPMDState>(state_obj)) {
                const auto& st = state_obj.cast<const ppmd::PPMDState&>();
                ctx_handle = ppmd_cuda::build_device_ctx_table(st);
                order      = st.order();
                ctx_valid  = true;
            } else if (py::isinstance<ppmd::VirtualPPMDState>(state_obj)) {
                const auto& vs = state_obj.cast<const ppmd::VirtualPPMDState&>();
                ctx_handle = ppmd_cuda::build_device_ctx_table(vs);
                order      = vs.order();
                ctx_valid  = true;
            } else {
                throw std::invalid_argument(
                    "state_obj must be PPMDState or VirtualPPMDState");
            }
            struct CtxGuard {
                ppmd_cuda::DeviceCtxTableHandle& h; bool active;
                ~CtxGuard() { if (active) ppmd_cuda::free_device_ctx_table(h); }
            } ctx_guard{ctx_handle, ctx_valid};

            // Parse target_ids.
            std::vector<int32_t> tids;
            try {
                auto arr = target_ids_obj.cast<
                    py::array_t<int32_t, py::array::c_style | py::array::forcecast>>();
                auto buf = arr.request();
                auto* ptr = static_cast<int32_t*>(buf.ptr);
                tids.assign(ptr, ptr + static_cast<size_t>(buf.size));
            } catch (...) {
                for (auto h : target_ids_obj.cast<py::sequence>())
                    tids.push_back(static_cast<int32_t>(h.cast<int>()));
            }

            int n = static_cast<int>(tids.size());
            std::vector<py::ssize_t> shape = {n, 2};
            py::array_t<double> result(shape);
            if (n == 0) return result;

            ppmd_cuda::DeviceTrieHandle dtrie = ppmd_cuda::build_device_trie(trie);
            struct TrieGuard {
                ppmd_cuda::DeviceTrieHandle& h;
                ~TrieGuard() { ppmd_cuda::free_device_trie(h); }
            } trie_guard{dtrie};

            int vocab_size = 0;
            for (int32_t t : tids)
                if (t + 1 > vocab_size) vocab_size = t + 1;
            if (vocab_size < 1) vocab_size = 1;

            constexpr int STACK_CAP = 8192;
            CudaBuf d_stack (static_cast<size_t>(STACK_CAP) * sizeof(ppmd_cuda::TrieFrame));
            CudaBuf d_tprobs(static_cast<size_t>(vocab_size) * sizeof(double));
            CudaBuf d_z     (sizeof(double));

            ppmd_cuda::device_memset_zero(d_tprobs.ptr,
                static_cast<size_t>(vocab_size) * sizeof(double));

            {
                py::gil_scoped_release release;
                ppmd_cuda::launch_trie_score_kernel(
                    ctx_handle, dtrie,
                    vocab_size,
                    0, INT32_MAX,
                    order,
                    d_stack.ptr, STACK_CAP,
                    d_tprobs.as<double>(), d_z.as<double>());
            }

            double z_val = 0.0;
            ppmd_cuda::device_memcpy_d2h(&z_val, d_z.ptr, sizeof(double));
            std::vector<double> h_tprobs(static_cast<size_t>(vocab_size), 0.0);
            ppmd_cuda::device_memcpy_d2h(
                h_tprobs.data(), d_tprobs.ptr,
                static_cast<size_t>(vocab_size) * sizeof(double));

            auto* out = result.mutable_data();
            for (int i = 0; i < n; ++i) {
                int32_t tid = tids[i];
                double tq = (tid >= 0 && tid < vocab_size) ? h_tprobs[tid] : 0.0;
                out[i * 2 + 0] = z_val;
                out[i * 2 + 1] = tq;
            }
            return result;
        },
        py::arg("state"), py::arg("trie"), py::arg("target_ids"),
        "Compute (z, target_q) per target_id via device trie DFS.\n"
        "state: PPMDState or VirtualPPMDState (fixed for all target_ids).\n"
        "Returns float64 ndarray shape (n, 2).");

    // -----------------------------------------------------------------------
    // score_path_a_arrays_cuda(...)  — same signature as _ppmd_cpp.score_path_a_arrays
    // -----------------------------------------------------------------------
    cuda.def("score_path_a_arrays_cuda",
        [](
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> target_ids,
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> prev_ids,
            py::array_t<double,  py::array::c_style | py::array::forcecast> nll_nats,
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast> boundary_bytes,
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> boundary_offsets,
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast> nonboundary_bytes,
            py::array_t<int32_t, py::array::c_style | py::array::forcecast> nonboundary_offsets,
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast> emittable,
            py::array_t<uint8_t, py::array::c_style | py::array::forcecast> is_boundary_lut,
            py::dict hyperparams
        ) -> py::dict {

            if (target_ids.ndim() != 1 || prev_ids.ndim() != 1 || nll_nats.ndim() != 1)
                throw std::invalid_argument("arrays must be 1-D");
            int64_t n_positions = target_ids.shape(0);
            if (prev_ids.shape(0) != n_positions || nll_nats.shape(0) != n_positions)
                throw std::invalid_argument("length mismatch");
            if (boundary_offsets.ndim() != 1 || nonboundary_offsets.ndim() != 1)
                throw std::invalid_argument("offset arrays must be 1-D");
            int64_t v1 = boundary_offsets.shape(0);
            if (v1 < 1) throw std::invalid_argument("boundary_offsets empty");
            if (nonboundary_offsets.shape(0) != v1)
                throw std::invalid_argument("offset length mismatch");
            int32_t vocab_size = static_cast<int32_t>(v1 - 1);
            if (emittable.shape(0) != vocab_size || is_boundary_lut.shape(0) != vocab_size)
                throw std::invalid_argument("emittable/is_boundary length");

            auto gd = [&](const char* k, double d) -> double {
                return hyperparams.contains(k) ? hyperparams[k].cast<double>() : d;
            };
            auto gi = [&](const char* k, int d) -> int {
                return hyperparams.contains(k) ? hyperparams[k].cast<int>() : d;
            };
            auto gb = [&](const char* k, bool d) -> bool {
                return hyperparams.contains(k) ? hyperparams[k].cast<bool>() : d;
            };
            int    order          = gi("order", 5);
            double lambda_hi      = gd("lambda_hi", 0.9);
            double lambda_lo      = gd("lambda_lo", 0.05);
            double conf_threshold = gd("conf_threshold", 0.9);
            bool   update_after   = gb("update_after_score", true);

            const uint8_t* bnd_bytes  = boundary_bytes.data();
            const int32_t* bnd_off    = boundary_offsets.data();
            const uint8_t* nbnd_bytes = nonboundary_bytes.data();
            const int32_t* nbnd_off   = nonboundary_offsets.data();
            const uint8_t* emit_arr   = emittable.data();
            const uint8_t* isb_arr    = is_boundary_lut.data();
            const int32_t* tids       = target_ids.data();
            const int32_t* pids       = prev_ids.data();
            const double*  nlls       = nll_nats.data();

            // Build tries.
            ppmd::Trie bnd_trie, nbnd_trie;
            for (int32_t tid = 0; tid < vocab_size; ++tid) {
                if (emit_arr[tid] == 0) continue;
                { int32_t o = bnd_off[tid], e = bnd_off[tid+1];
                  bnd_trie.insert(tid, bnd_bytes + o, static_cast<size_t>(e - o)); }
                { int32_t o = nbnd_off[tid], e = nbnd_off[tid+1];
                  nbnd_trie.insert(tid, nbnd_bytes + o, static_cast<size_t>(e - o)); }
            }

            ppmd_cuda::DeviceTrieHandle d_bnd  = ppmd_cuda::build_device_trie(bnd_trie);
            ppmd_cuda::DeviceTrieHandle d_nbnd = ppmd_cuda::build_device_trie(nbnd_trie);
            struct TG {
                ppmd_cuda::DeviceTrieHandle& b; ppmd_cuda::DeviceTrieHandle& n;
                ~TG() { ppmd_cuda::free_device_trie(b); ppmd_cuda::free_device_trie(n); }
            } tg{d_bnd, d_nbnd};

            constexpr int STACK_CAP = 8192;
            CudaBuf d_stack (static_cast<size_t>(STACK_CAP) * sizeof(ppmd_cuda::TrieFrame));
            CudaBuf d_tprobs(static_cast<size_t>(vocab_size) * sizeof(double));
            CudaBuf d_z     (sizeof(double));

            ppmd::PPMDState state(order);
            std::string start_digest = state.state_digest();
            double  total_bits  = 0.0;
            int64_t total_bytes_count = 0;

            for (int64_t pos = 0; pos < n_positions; ++pos) {
                int32_t target_id = tids[pos];
                int32_t prev_id   = pids[pos];
                bool boundary = (prev_id < 0) ||
                    (prev_id >= vocab_size ? true : isb_arr[prev_id] != 0);

                const ppmd_cuda::DeviceTrieHandle& dtrie = boundary ? d_bnd : d_nbnd;

                ppmd_cuda::DeviceCtxTableHandle ctx_h =
                    ppmd_cuda::build_device_ctx_table(state);
                struct CG {
                    ppmd_cuda::DeviceCtxTableHandle& h;
                    ~CG() { ppmd_cuda::free_device_ctx_table(h); }
                } cg{ctx_h};

                ppmd_cuda::device_memset_zero(d_tprobs.ptr,
                    static_cast<size_t>(vocab_size) * sizeof(double));

                ppmd_cuda::launch_trie_score_kernel(
                    ctx_h, dtrie, vocab_size, 0, INT32_MAX, order,
                    d_stack.ptr, STACK_CAP,
                    d_tprobs.as<double>(), d_z.as<double>());

                double z_val = 0.0;
                ppmd_cuda::device_memcpy_d2h(&z_val, d_z.ptr, sizeof(double));
                double target_q = 0.0;
                if (target_id >= 0 && target_id < vocab_size)
                    ppmd_cuda::device_memcpy_d2h(
                        &target_q,
                        d_tprobs.as<double>() + target_id,
                        sizeof(double));

                if (z_val <= 0.0)
                    throw std::runtime_error(
                        "CUDA Path A: Z non-positive at pos " +
                        std::to_string(pos));

                double p_ppm = target_q / z_val;
                double p_nn  = std::exp(-nlls[pos]);
                double conf  = state.confidence();
                double lam   = (conf >= conf_threshold) ? lambda_lo : lambda_hi;
                double p_mix = lam * p_nn + (1.0 - lam) * p_ppm;
                if (p_mix <= 0.0)
                    throw std::runtime_error(
                        "CUDA Path A: p_mix zero at pos " + std::to_string(pos));

                total_bits += -std::log(p_mix) / std::log(2.0);

                int32_t bo   = boundary ? bnd_off[target_id]   : nbnd_off[target_id];
                int32_t be   = boundary ? bnd_off[target_id+1] : nbnd_off[target_id+1];
                int32_t blen = be - bo;
                const uint8_t* bbytes = (boundary ? bnd_bytes : nbnd_bytes) + bo;
                total_bytes_count += blen;

                if (update_after)
                    state.update_bytes(bbytes, static_cast<size_t>(blen));
            }

            double bpb = (total_bytes_count > 0)
                ? total_bits / static_cast<double>(total_bytes_count) : 0.0;

            py::dict out;
            out["positions"]          = static_cast<int64_t>(n_positions);
            out["total_bits"]         = total_bits;
            out["total_bytes"]        = static_cast<int64_t>(total_bytes_count);
            out["bpb"]                = bpb;
            out["start_state_digest"] = start_digest;
            out["end_state_digest"]   = state.state_digest();
            return out;
        },
        py::arg("target_ids"), py::arg("prev_ids"), py::arg("nll_nats"),
        py::arg("boundary_bytes"), py::arg("boundary_offsets"),
        py::arg("nonboundary_bytes"), py::arg("nonboundary_offsets"),
        py::arg("emittable"), py::arg("is_boundary"), py::arg("hyperparams"),
        "Full Path A scorer via device trie DFS.\n"
        "Same signature as _ppmd_cpp.score_path_a_arrays.");
}

