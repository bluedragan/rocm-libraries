#pragma once
#if defined(__PPC64__) || defined(__powerpc64__)
    #include <hip/hip_fp16.h>
    inline bool operator==(half a, half b) { return static_cast<float>(a) == static_cast<float>(b); }
    inline bool operator!=(half a, half b) { return static_cast<float>(a) != static_cast<float>(b); }
    inline bool operator<(half a, half b)  { return static_cast<float>(a) < static_cast<float>(b); }
    inline bool operator>(half a, half b)  { return static_cast<float>(a) > static_cast<float>(b); }
    inline bool operator<=(half a, half b) { return static_cast<float>(a) <= static_cast<float>(b); }
    inline bool operator>=(half a, half b) { return static_cast<float>(a) >= static_cast<float>(b); }

    // Also include the unary minus you found
    inline half operator-(half a) {
        auto r = static_cast<__half_raw>(a);
        r.x ^= 0x8000u;
        return r;
    }

    #include <hip/hip_runtime_api.h> // Define the structs first
    #ifndef warpSize
        static constexpr int warpSize = 64; // Define the constant second
    #endif
#endif
