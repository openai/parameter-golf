// cpp_bindings.hpp — shared pybind11 bindings used by both _ppmd_cpp and _ppmd_cuda.
//
// Both modules call register_cpp_bindings(m) inside their PYBIND11_MODULE entry
// points.  All helper functions are static (internal linkage per TU); the
// register function is inline so it satisfies ODR when included from two TUs
// in the same link step.

#pragma once

#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <array>
#include <climits>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

#include "ppmd.hpp"
#include "scorer.hpp"
#include "trie.hpp"
#include "virtual_ppmd.hpp"

namespace py = pybind11;

namespace _cpp_bind_detail {

using ppmd::PartialResult;
using ppmd::PPMDState;
using ppmd::ScoreArraysResult;
using ppmd::ScoreArraysVocab;
using ppmd::ScoreHyperparams;
using ppmd::Trie;
using ppmd::VirtualPPMDState;

static py::array_t<double> probs_array(const std::array<double, 256>& probs) {
    py::array_t<double> out(256);
    auto buf = out.request();
    std::memcpy(buf.ptr, probs.data(), sizeof(double) * 256);
    return out;
}

static void update_bytes_from_pybytes(PPMDState& self, py::bytes data) {
    char* buf = nullptr;
    Py_ssize_t len = 0;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buf, &len) != 0) {
        throw py::error_already_set();
    }
    self.update_bytes(reinterpret_cast<const uint8_t*>(buf), static_cast<size_t>(len));
}

static void trie_insert_from_pybytes(Trie& self, int token_id, py::bytes data) {
    char* buf = nullptr;
    Py_ssize_t len = 0;
    if (PYBIND11_BYTES_AS_STRING_AND_SIZE(data.ptr(), &buf, &len) != 0) {
        throw py::error_already_set();
    }
    self.insert(static_cast<int32_t>(token_id),
                reinterpret_cast<const uint8_t*>(buf),
                static_cast<size_t>(len));
}

static py::dict partial_to_dict(const PartialResult& p) {
    py::dict d;
    d["z"] = p.z;
    d["target_q"] = p.target_q;
    d["terminal_count"] = static_cast<int64_t>(p.terminal_count);
    return d;
}

static double dict_get_double(const py::dict& d, const char* key, double def) {
    if (d.contains(key)) return py::cast<double>(d[key]);
    return def;
}

static int dict_get_int(const py::dict& d, const char* key, int def) {
    if (d.contains(key)) return py::cast<int>(d[key]);
    return def;
}

static bool dict_get_bool(const py::dict& d, const char* key, bool def) {
    if (d.contains(key)) return py::cast<bool>(d[key]);
    return def;
}

static py::dict trie_partial_py(const VirtualPPMDState& virt, const Trie& trie,
                                 int target_id, int shard_start, int shard_end) {
    int32_t end = (shard_end < 0) ? INT32_MAX : static_cast<int32_t>(shard_end);
    PartialResult r = ppmd::trie_partial_z_and_target(
        virt, trie, static_cast<int32_t>(target_id),
        static_cast<int32_t>(shard_start), end);
    return partial_to_dict(r);
}

static py::dict combine_partials_py(const py::iterable& partials) {
    PartialResult acc;
    for (auto h : partials) {
        py::handle item = h;
        if (py::isinstance<py::dict>(item)) {
            py::dict d = py::reinterpret_borrow<py::dict>(item);
            acc.z += py::cast<double>(d["z"]);
            acc.target_q += py::cast<double>(d["target_q"]);
            acc.terminal_count += py::cast<int64_t>(d["terminal_count"]);
        } else {
            py::sequence s = py::reinterpret_borrow<py::sequence>(item);
            if (py::len(s) != 3) {
                throw std::invalid_argument("partial must be (z, target_q, count)");
            }
            acc.z += py::cast<double>(s[0]);
            acc.target_q += py::cast<double>(s[1]);
            acc.terminal_count += py::cast<int64_t>(s[2]);
        }
    }
    return partial_to_dict(acc);
}

static py::dict score_arrays_py(
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> target_ids,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> prev_ids,
    py::array_t<double, py::array::c_style | py::array::forcecast> nll_nats,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> boundary_bytes,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> boundary_offsets,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> nonboundary_bytes,
    py::array_t<int32_t, py::array::c_style | py::array::forcecast> nonboundary_offsets,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> emittable,
    py::array_t<uint8_t, py::array::c_style | py::array::forcecast> is_boundary,
    py::dict hyperparams) {
    if (target_ids.ndim() != 1 || prev_ids.ndim() != 1 || nll_nats.ndim() != 1) {
        throw std::invalid_argument("target_ids/prev_ids/nll_nats must be 1-D");
    }
    int64_t n_positions = target_ids.shape(0);
    if (prev_ids.shape(0) != n_positions || nll_nats.shape(0) != n_positions) {
        throw std::invalid_argument("target_ids/prev_ids/nll_nats length mismatch");
    }
    if (boundary_offsets.ndim() != 1 || nonboundary_offsets.ndim() != 1) {
        throw std::invalid_argument("offsets arrays must be 1-D");
    }
    int64_t v_plus_1 = boundary_offsets.shape(0);
    if (v_plus_1 < 1) {
        throw std::invalid_argument("boundary_offsets must have length >= 1");
    }
    if (nonboundary_offsets.shape(0) != v_plus_1) {
        throw std::invalid_argument("nonboundary_offsets length mismatch");
    }
    int32_t vocab_size = static_cast<int32_t>(v_plus_1 - 1);
    if (emittable.shape(0) != vocab_size || is_boundary.shape(0) != vocab_size) {
        throw std::invalid_argument("emittable/is_boundary length must equal vocab_size");
    }

    ScoreArraysVocab vocab;
    vocab.boundary_bytes = boundary_bytes.data();
    vocab.boundary_offsets = boundary_offsets.data();
    vocab.nonboundary_bytes = nonboundary_bytes.data();
    vocab.nonboundary_offsets = nonboundary_offsets.data();
    vocab.emittable = emittable.data();
    vocab.is_boundary = is_boundary.data();
    vocab.vocab_size = vocab_size;

    ScoreHyperparams hyper;
    hyper.order = dict_get_int(hyperparams, "order", 5);
    hyper.lambda_hi = dict_get_double(hyperparams, "lambda_hi", 0.9);
    hyper.lambda_lo = dict_get_double(hyperparams, "lambda_lo", 0.05);
    hyper.conf_threshold = dict_get_double(hyperparams, "conf_threshold", 0.9);
    hyper.update_after_score = dict_get_bool(hyperparams, "update_after_score", true);

    ScoreArraysResult r;
    {
        py::gil_scoped_release release;
        r = ppmd::score_path_a_arrays(
            target_ids.data(), prev_ids.data(), nll_nats.data(),
            n_positions, vocab, hyper);
    }

    py::dict out;
    out["positions"] = static_cast<int64_t>(r.positions);
    out["total_bits"] = r.total_bits;
    out["total_bytes"] = static_cast<int64_t>(r.total_bytes);
    out["bpb"] = r.bpb;
    out["start_state_digest"] = r.start_state_digest;
    out["end_state_digest"] = r.end_state_digest;
    return out;
}

}  // namespace _cpp_bind_detail

// Register all CPU-surface bindings onto module m.
inline void register_cpp_bindings(py::module_& m) {
    using namespace _cpp_bind_detail;
    using ppmd::PPMDState;
    using ppmd::Trie;
    using ppmd::VirtualPPMDState;

    m.def("version", []() { return std::string("0.0.1"); },
          "Return the C++ backend version string.");

    py::class_<VirtualPPMDState>(m, "VirtualPPMDState", py::module_local())
        .def("byte_prob",
             [](const VirtualPPMDState& self, int b) { return self.byte_prob(b); },
             py::arg("b"))
        .def("byte_probs",
             [](const VirtualPPMDState& self) { return probs_array(self.byte_probs()); })
        .def("fork_and_update",
             [](const VirtualPPMDState& self, int b) { return self.fork_and_update(b); },
             py::arg("b"));

    py::class_<PPMDState>(m, "PPMDState", py::module_local())
        .def(py::init<int>(), py::arg("order") = 5)
        .def("update_byte", &PPMDState::update_byte, py::arg("b"))
        .def("update_bytes", &update_bytes_from_pybytes, py::arg("data"))
        .def("byte_prob", &PPMDState::byte_prob, py::arg("b"))
        .def("byte_probs",
             [](const PPMDState& self) { return probs_array(self.byte_probs()); })
        .def("clone_virtual", &PPMDState::clone_virtual)
        .def("state_digest", &PPMDState::state_digest)
        .def("confidence", &PPMDState::confidence);

    py::class_<Trie>(m, "Trie", py::module_local())
        .def(py::init<>())
        .def("insert", &trie_insert_from_pybytes,
             py::arg("token_id"), py::arg("bytes"))
        .def("num_nodes", &Trie::num_nodes)
        .def("num_edges", &Trie::num_edges)
        .def("num_terminals", &Trie::num_terminals);

    m.def("trie_partial_z_and_target", &trie_partial_py,
          py::arg("virtual_state"), py::arg("trie"),
          py::arg("target_id"), py::arg("shard_start"), py::arg("shard_end"));

    m.def("combine_path_a_partials", &combine_partials_py,
          py::arg("partials"));

    m.def("score_path_a_arrays", &score_arrays_py,
          py::arg("target_ids"), py::arg("prev_ids"), py::arg("nll_nats"),
          py::arg("boundary_bytes"), py::arg("boundary_offsets"),
          py::arg("nonboundary_bytes"), py::arg("nonboundary_offsets"),
          py::arg("emittable"), py::arg("is_boundary"),
          py::arg("hyperparams"));

    m.def("set_num_threads", &ppmd::set_num_threads, py::arg("n"));
    m.def("get_max_threads", &ppmd::get_max_threads);
}
