// Internal header: shared backoff templates for PPMDState and VirtualPPMDState.
// Both byte_probs and byte_prob must be bit-for-bit identical between the
// base and virtual variants modulo the count provider.

#pragma once

#include "ppmd.hpp"

#include <algorithm>
#include <array>
#include <bitset>
#include <cstdint>
#include <vector>

namespace ppmd {

template <typename Provider>
inline std::array<double, 256> backoff_byte_probs(
    const Provider& provider,
    const std::vector<uint8_t>& window,
    int order) {
    std::array<double, 256> probs{};
    std::bitset<256> assigned;
    double escape_mass = 1.0;
    int wlen = static_cast<int>(window.size());
    int kmax = std::min(order, wlen);
    for (int k = kmax; k >= 0; --k) {
        uint64_t key = pack_ctx(window.data() + (wlen - k), k);
        const CtxCounts* cc = provider(key);
        if (cc == nullptr) continue;

        int active_unique = 0;
        uint64_t active_total = 0;
        int active_bytes[256];
        for (int b = 0; b < 256; ++b) {
            uint32_t c = cc->counts[b];
            if (c == 0) continue;
            if (assigned.test(b)) continue;
            active_bytes[active_unique] = b;
            ++active_unique;
            active_total += c;
        }
        if (active_unique == 0) continue;

        int active_alphabet_size = 256 - static_cast<int>(assigned.count());
        if (active_unique == active_alphabet_size) {
            if (active_total == 0) continue;
            double total_d = static_cast<double>(active_total);
            for (int i = 0; i < active_unique; ++i) {
                int b = active_bytes[i];
                double c = static_cast<double>(cc->counts[b]);
                probs[b] = escape_mass * (c / total_d);
                assigned.set(b);
            }
            escape_mass = 0.0;
            break;
        }

        double total_d = static_cast<double>(active_total);
        double unique_d = static_cast<double>(active_unique);
        double denom = total_d + unique_d;
        for (int i = 0; i < active_unique; ++i) {
            int b = active_bytes[i];
            double c = static_cast<double>(cc->counts[b]);
            probs[b] = escape_mass * (c / denom);
            assigned.set(b);
        }
        escape_mass *= unique_d / denom;
    }

    int remaining = 256 - static_cast<int>(assigned.count());
    if (remaining > 0) {
        double per_byte = escape_mass / static_cast<double>(remaining);
        for (int b = 0; b < 256; ++b) {
            if (!assigned.test(b)) probs[b] = per_byte;
        }
    }
    return probs;
}

template <typename Provider>
inline double backoff_byte_prob(
    const Provider& provider,
    const std::vector<uint8_t>& window,
    int order,
    int target_b) {
    std::bitset<256> assigned;
    double escape_mass = 1.0;
    int wlen = static_cast<int>(window.size());
    int kmax = std::min(order, wlen);
    for (int k = kmax; k >= 0; --k) {
        uint64_t key = pack_ctx(window.data() + (wlen - k), k);
        const CtxCounts* cc = provider(key);
        if (cc == nullptr) continue;

        int active_unique = 0;
        uint64_t active_total = 0;
        int active_bytes[256];
        for (int b = 0; b < 256; ++b) {
            uint32_t c = cc->counts[b];
            if (c == 0) continue;
            if (assigned.test(b)) continue;
            active_bytes[active_unique] = b;
            ++active_unique;
            active_total += c;
        }
        if (active_unique == 0) continue;
        if (active_total == 0) continue;

        int active_alphabet_size = 256 - static_cast<int>(assigned.count());
        if (active_unique == active_alphabet_size) {
            uint32_t tc = cc->counts[target_b];
            if (tc > 0 && !assigned.test(target_b)) {
                return escape_mass * (static_cast<double>(tc) / static_cast<double>(active_total));
            }
            return 0.0;
        }
        double total_d = static_cast<double>(active_total);
        double unique_d = static_cast<double>(active_unique);
        double denom = total_d + unique_d;
        uint32_t tc = cc->counts[target_b];
        if (tc > 0 && !assigned.test(target_b)) {
            return escape_mass * (static_cast<double>(tc) / denom);
        }
        for (int i = 0; i < active_unique; ++i) {
            assigned.set(active_bytes[i]);
        }
        escape_mass *= unique_d / denom;
    }

    if (assigned.test(target_b)) return 0.0;
    int remaining = 256 - static_cast<int>(assigned.count());
    if (remaining <= 0) return 0.0;
    return escape_mass / static_cast<double>(remaining);
}

}  // namespace ppmd
