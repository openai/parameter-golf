// VirtualPPMDState — overlay over a base PPMDState's ctx_counts. Holds a
// const pointer to the base map (lifetime managed by the owning PPMDState
// or by Python keeping the parent alive). `fork_and_update` deep-copies the
// overlay only.

#pragma once

#include "ppmd.hpp"

#include <array>
#include <cstdint>
#include <functional>
#include <unordered_map>
#include <vector>

namespace ppmd {

class VirtualPPMDState {
public:
    VirtualPPMDState() = default;
    VirtualPPMDState(const std::unordered_map<uint64_t, CtxCounts>* base,
                     std::vector<uint8_t> window,
                     int order)
        : base_counts_(base),
          window_(std::move(window)),
          order_(order) {}

    int order() const { return order_; }
    const std::vector<uint8_t>& window() const { return window_; }
    const std::unordered_map<uint64_t, CtxCounts>& overlay_counts() const { return overlay_counts_; }

    std::array<double, 256> byte_probs() const;
    double byte_prob(int target_b) const;

    VirtualPPMDState fork_and_update(int b) const;

    // Iterate over every merged (base+overlay) context, invoking fn(key, merged_cc).
    // Additive; does not modify state. Used by CUDA table builder.
    void each_context(std::function<void(uint64_t, const CtxCounts&)> fn) const;

private:
    friend class PPMDState;
    const CtxCounts* combined_counts(uint64_t key) const;

    const std::unordered_map<uint64_t, CtxCounts>* base_counts_ = nullptr;
    std::vector<uint8_t> window_;
    int order_ = 5;
    std::unordered_map<uint64_t, CtxCounts> overlay_counts_;
};

}  // namespace ppmd
