// Minimal SHA-256 implementation for state_digest equivalence with Python's
// hashlib.sha256. Single translation unit; no external dependencies.
//
// Produces a 32-byte digest and a 64-char lowercase hex string.

#pragma once

#include <array>
#include <cstdint>
#include <cstddef>
#include <string>

namespace ppmd {

class Sha256 {
public:
    Sha256() { reset(); }

    void reset();
    void update(const uint8_t* data, size_t len);
    void update(const std::string& s) {
        update(reinterpret_cast<const uint8_t*>(s.data()), s.size());
    }
    // Finalize and return 32-byte digest.
    std::array<uint8_t, 32> digest();
    // Finalize and return 64-char lowercase hex string.
    std::string hexdigest();

private:
    void transform(const uint8_t* block);

    uint32_t state_[8];
    uint64_t bit_count_;
    uint8_t buffer_[64];
    size_t buffer_len_;
};

}  // namespace ppmd
