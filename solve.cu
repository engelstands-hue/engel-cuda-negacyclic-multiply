// Negacyclic product c = a*b in Z_q[X]/(X^N + 1).
//
// The challenge point is N = 32768 and an 868-bit q. One thread per coefficient
// is a schoolbook, and that point took about a minute. This is the same product
// as an exact integer convolution, reduced modulo q at the end. The convolution
// runs as a negacyclic NTT in several 62-bit fields whose product is wider than
// the coefficient bound, then Garner's algorithm rebuilds each coefficient.
// fherma_init builds the fields and the twiddles. The harness does not time it.

#include "fherma.h"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWide = 80;
constexpr int kMaxChannels = 48;
constexpr int kMaxN = 32768;
// Direct CRT accumulates v * (M/p mod q). The challenge modulus is 868 bits,
// so 34 limbs holds the sum with room for the 62-bit factor.
constexpr int kCrtLimbs = 34;

struct State {
    int N = 0;
    int L = 0;
    int channels = 0;
    uint64_t* d_prime = nullptr;
    uint64_t* d_nprime = nullptr;
    uint64_t* d_r2 = nullptr;
    uint64_t* d_mu = nullptr;
    uint64_t* d_minv = nullptr;
    uint64_t* d_ninv = nullptr;
    uint64_t* d_a = nullptr;
    uint64_t* d_b = nullptr;
    uint64_t* d_tw_fwd = nullptr;
    uint64_t* d_tw_inv = nullptr;
    uint64_t* d_tw_fwd_shoup = nullptr;
    uint64_t* d_tw_inv_shoup = nullptr;
    uint32_t* h_pin_a = nullptr;
    uint32_t* h_pin_b = nullptr;
    uint32_t* h_pin_c = nullptr;
    uint64_t* d_psi = nullptr;
    uint64_t* d_psi_inv = nullptr;
    uint32_t* d_in = nullptr;
    uint32_t* d_out = nullptr;
    uint32_t* d_mbefore = nullptr;
    uint32_t* d_mb_q = nullptr;
    uint64_t* d_cross = nullptr;
    uint32_t* d_half_m = nullptr;
    uint32_t* d_m_mod_q = nullptr;
    uint32_t* d_q = nullptr;
    uint64_t* d_crt_y = nullptr;
    uint64_t* d_magic_lo = nullptr;
    uint64_t* d_magic_hi = nullptr;
    uint32_t* d_mi_q = nullptr;
    int* d_unsafe_count = nullptr;
    int* d_unsafe_index = nullptr;
    uint64_t* d_pow32 = nullptr;
    uint32_t* d_q_mu = nullptr;
};

__host__ __device__ int clz64(uint64_t value) {
    int count = 0;
    if (value == 0) return 64;
    while ((value & (1ull << 63)) == 0) {
        value <<= 1;
        ++count;
    }
    return count;
}

__host__ __device__ uint64_t mod128(uint64_t hi, uint64_t lo, uint64_t p) {
    if (hi == 0 && lo < p) return lo;
    const int top = hi ? 64 + (63 - clz64(hi)) : 63 - clz64(lo);
    const int pbits = 63 - clz64(p);
    for (int shift = top - pbits; shift >= 0; --shift) {
        uint64_t phi = 0;
        uint64_t plo = 0;
        if (shift >= 64) phi = p << (shift - 64);
        else if (shift == 0) plo = p;
        else {
            phi = p >> (64 - shift);
            plo = p << shift;
        }
        if (hi > phi || (hi == phi && lo >= plo)) {
            const uint64_t next = lo - plo;
            const uint64_t borrow = lo < plo ? 1ull : 0ull;
            lo = next;
            hi = hi - phi - borrow;
        }
    }
    return lo;
}

__host__ __device__ uint64_t mul_mod(uint64_t a, uint64_t b, uint64_t p) {
    const uint64_t a0 = a & 0xffffffffull;
    const uint64_t a1 = a >> 32;
    const uint64_t b0 = b & 0xffffffffull;
    const uint64_t b1 = b >> 32;
    const uint64_t p0 = a0 * b0;
    const uint64_t p1 = a0 * b1;
    const uint64_t p2 = a1 * b0;
    const uint64_t p3 = a1 * b1;
    const uint64_t mid = (p0 >> 32) + (p1 & 0xffffffffull) + (p2 & 0xffffffffull);
    const uint64_t lo = (p0 & 0xffffffffull) | (mid << 32);
    const uint64_t hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
    return mod128(hi, lo, p);
}

__host__ __device__ void mul64_wide(uint64_t a, uint64_t b, uint64_t& hi, uint64_t& lo) {
#if defined(__CUDA_ARCH__)
    lo = a * b;
    hi = __umul64hi(a, b);
#else
    const uint64_t a0 = a & 0xffffffffull;
    const uint64_t a1 = a >> 32;
    const uint64_t b0 = b & 0xffffffffull;
    const uint64_t b1 = b >> 32;
    const uint64_t p0 = a0 * b0;
    const uint64_t p1 = a0 * b1;
    const uint64_t p2 = a1 * b0;
    const uint64_t p3 = a1 * b1;
    const uint64_t mid = (p0 >> 32) + (p1 & 0xffffffffull) + (p2 & 0xffffffffull);
    lo = (p0 & 0xffffffffull) | (mid << 32);
    hi = p3 + (p1 >> 32) + (p2 >> 32) + (mid >> 32);
#endif
}

__host__ __device__ uint64_t mont_mul(uint64_t a, uint64_t b, uint64_t p, uint64_t nprime) {
    uint64_t hi = 0;
    uint64_t lo = 0;
    mul64_wide(a, b, hi, lo);
    const uint64_t m = lo * nprime;
    uint64_t phi = 0;
    uint64_t plo = 0;
    mul64_wide(m, p, phi, plo);
    const uint64_t sum_lo = lo + plo;
    const uint64_t carry = sum_lo < lo ? 1ull : 0ull;
    uint64_t reduced = hi + phi + carry;
    if (reduced >= p) reduced -= p;
    if (reduced >= p) reduced -= p;
    return reduced;
}

// Primes here sit under 2^62, so floor(w * 2^64 / p) makes one correction exact.
__device__ __forceinline__ uint64_t shoup_mul(uint64_t a, uint64_t w, uint64_t w_shoup, uint64_t p) {
    const uint64_t q = __umul64hi(a, w_shoup);
    uint64_t r = a * w - q * p;
    if (r >= p) r -= p;
    return r;
}

__host__ __device__ uint64_t barrett_reduce(uint64_t v_hi, uint64_t v_lo, uint64_t p, uint64_t mu) {
    uint64_t low_hi = 0;
    uint64_t low_lo = 0;
    uint64_t high_hi = 0;
    uint64_t high_lo = 0;
    mul64_wide(v_lo, mu, low_hi, low_lo);
    mul64_wide(v_hi, mu, high_hi, high_lo);
    const uint64_t mid = low_hi + high_lo;
    const uint64_t mid_carry = mid < low_hi ? 1ull : 0ull;
    const uint64_t qhat = (mid >> 32) + ((high_hi + mid_carry) << 32);
    uint64_t m_hi = 0;
    uint64_t m_lo = 0;
    mul64_wide(qhat, p, m_hi, m_lo);
    const uint64_t borrow = v_lo < m_lo ? 1ull : 0ull;
    const bool negative = v_hi < m_hi || (v_hi == m_hi && borrow);
    uint64_t r_lo = v_lo - m_lo;
    uint64_t r_hi = v_hi - m_hi - borrow;
    if (negative) {
        const uint64_t adjusted = r_lo + p;
        const uint64_t carry = adjusted < r_lo ? 1ull : 0ull;
        r_lo = adjusted;
        r_hi += carry;
        if (r_hi != 0) {
            const uint64_t again = r_lo + p;
            r_hi += again < r_lo ? 1ull : 0ull;
            r_lo = again;
        }
    }
    while (r_hi != 0 || r_lo >= p) {
        if (r_lo >= p) {
            r_lo -= p;
            continue;
        }
        r_lo -= p;
        --r_hi;
    }
    return r_lo;
}

// floor((x_hi:x_lo) * floor(2^128 / p) / 2^128) is floor(x / p), or one less when the
// fraction is tinier than the discarded reciprocal bit. The product's primes are ~62 bits
// and x fits in 100 bits, so the quotient fits in 64 bits.
__host__ __device__ uint64_t reduce_128(
    uint64_t x_hi, uint64_t x_lo, uint64_t prime, uint64_t magic_hi, uint64_t magic_lo) {
    uint64_t ll_hi = 0;
    uint64_t ll_lo = 0;
    uint64_t lh_hi = 0;
    uint64_t lh_lo = 0;
    uint64_t hl_hi = 0;
    uint64_t hl_lo = 0;
    uint64_t hh_hi = 0;
    uint64_t hh_lo = 0;
    mul64_wide(x_lo, magic_lo, ll_hi, ll_lo);
    mul64_wide(x_lo, magic_hi, lh_hi, lh_lo);
    mul64_wide(x_hi, magic_lo, hl_hi, hl_lo);
    mul64_wide(x_hi, magic_hi, hh_hi, hh_lo);
    uint64_t mid = ll_hi + lh_lo;
    uint64_t carry = mid < ll_hi ? 1ull : 0ull;
    const uint64_t mid2 = mid + hl_lo;
    if (mid2 < mid) ++carry;
    uint64_t high = lh_hi + hl_hi;
    const uint64_t high2 = high + hh_lo + carry;
    const uint64_t quotient = high2;
    uint64_t ph = 0;
    uint64_t pl = 0;
    mul64_wide(quotient, prime, ph, pl);
    uint64_t r_lo = x_lo - pl;
    const uint64_t borrow = x_lo < pl ? 1ull : 0ull;
    uint64_t r_hi = x_hi - ph - borrow;
    for (int fix = 0; fix < 4 && (r_hi != 0 || r_lo >= prime); ++fix) {
        if (r_lo >= prime) {
            r_lo -= prime;
            continue;
        }
        r_lo -= prime;
        --r_hi;
    }
    if (r_hi != 0 || r_lo >= prime) return mod128(r_hi, r_lo, prime);
    return r_lo;
}

uint64_t barrett_mu(uint64_t p) {
    uint64_t remainder = 0;
    uint64_t mu = 0;
    for (int bit = 96; bit >= 0; --bit) {
        remainder <<= 1;
        if (bit == 96) remainder |= 1ull;
        if (remainder >= p) {
            remainder -= p;
            if (bit < 64) mu |= 1ull << bit;
        }
    }
    return mu;
}

uint64_t inv_mod_2_64(uint64_t odd) {
    uint64_t inverse = 1;
    for (int round = 0; round < 6; ++round) inverse *= 2 - odd * inverse;
    return inverse;
}

__host__ __device__ uint64_t add_mod(uint64_t a, uint64_t b, uint64_t p) {
    const uint64_t sum = a + b;
    if (sum >= p || sum < a) return sum - p;
    return sum;
}

__host__ __device__ uint64_t sub_mod(uint64_t a, uint64_t b, uint64_t p) {
    return a >= b ? a - b : a + p - b;
}

__host__ __device__ uint64_t pow_mod(uint64_t base, uint64_t exp, uint64_t p) {
    uint64_t result = 1 % p;
    base %= p;
    while (exp) {
        if (exp & 1ull) result = mul_mod(result, base, p);
        base = mul_mod(base, base, p);
        exp >>= 1;
    }
    return result;
}

__host__ __device__ uint64_t limbs_mod_u64(const uint32_t* value, int wide, uint64_t p) {
    uint64_t result = 0;
    for (int limb = wide - 1; limb >= 0; --limb) {
        const uint64_t hi = result >> 32;
        uint64_t lo = result << 32;
        const uint64_t limb_value = value[limb];
        lo += limb_value;
        const uint64_t carry = lo < limb_value ? 1ull : 0ull;
        result = mod128(hi + carry, lo, p);
    }
    return result;
}

__host__ __device__ void limbs_clear(uint32_t* value, int wide) {
    for (int i = 0; i < wide; ++i) value[i] = 0;
}

__host__ __device__ void limbs_set_u64(uint32_t* value, int wide, uint64_t small) {
    limbs_clear(value, wide);
    value[0] = static_cast<uint32_t>(small);
    value[1] = static_cast<uint32_t>(small >> 32);
}

__host__ __device__ void add_mul_u64(uint32_t* acc, const uint32_t* term, uint64_t factor, int wide) {
    uint64_t carry = 0;
    const uint32_t low = static_cast<uint32_t>(factor);
    for (int i = 0; i < wide; ++i) {
        const uint64_t cur = static_cast<uint64_t>(acc[i]) + static_cast<uint64_t>(term[i]) * low + carry;
        acc[i] = static_cast<uint32_t>(cur);
        carry = cur >> 32;
    }
    const uint32_t high = static_cast<uint32_t>(factor >> 32);
    if (high == 0) return;
    carry = 0;
    for (int i = 0; i < wide - 1; ++i) {
        const uint64_t cur = static_cast<uint64_t>(acc[i + 1]) + static_cast<uint64_t>(term[i]) * high + carry;
        acc[i + 1] = static_cast<uint32_t>(cur);
        carry = cur >> 32;
    }
}

__host__ __device__ int limbs_cmp(const uint32_t* left, const uint32_t* right, int wide) {
    for (int i = wide - 1; i >= 0; --i) {
        if (left[i] != right[i]) return left[i] > right[i] ? 1 : -1;
    }
    return 0;
}

__host__ __device__ uint32_t shifted_limb(const uint32_t* value, int count, int shift, int index) {
    const int word = shift >> 5;
    const int bits = shift & 31;
    uint32_t result = 0;
    if (index - word >= 0 && index - word < count) {
        result = value[index - word] << bits;
    }
    if (bits && index - word - 1 >= 0 && index - word - 1 < count) {
        result |= value[index - word - 1] >> (32 - bits);
    }
    return result;
}

__host__ __device__ void sub_shifted(uint32_t* acc, const uint32_t* value, int count, int shift, int wide) {
    uint32_t borrow = 0;
    for (int i = 0; i < wide; ++i) {
        const uint64_t left = static_cast<uint64_t>(acc[i]);
        const uint64_t right = static_cast<uint64_t>(shifted_limb(value, count, shift, i)) + borrow;
        if (left < right) {
            acc[i] = static_cast<uint32_t>(left + (1ull << 32) - right);
            borrow = 1;
        } else {
            acc[i] = static_cast<uint32_t>(left - right);
            borrow = 0;
        }
    }
}

__host__ __device__ int cmp_shifted(const uint32_t* acc, const uint32_t* value, int count, int shift, int wide) {
    for (int i = wide - 1; i >= 0; --i) {
        const uint32_t right = shifted_limb(value, count, shift, i);
        if (acc[i] != right) return acc[i] > right ? 1 : -1;
    }
    return 0;
}

__host__ __device__ int top_bit(const uint32_t* value, int wide) {
    for (int i = wide - 1; i >= 0; --i) {
        if (value[i] == 0) continue;
        int bit = 31;
        while ((value[i] & (1u << bit)) == 0) --bit;
        return i * 32 + bit;
    }
    return -1;
}

// floor(2^(868+128) / q) brings a number up to 96 bits past q down with one correction.
__host__ __device__ void mod_q_barrett(uint32_t* acc, const uint32_t* q, const uint32_t* mu, int q_limbs) {
    uint32_t shifted[8] = {};
    for (int i = 0; i < 7; ++i) {
        const uint32_t low = acc[27 + i];
        const uint32_t high = (28 + i < kCrtLimbs) ? acc[28 + i] : 0u;
        shifted[i] = (low >> 4) | (high << 28);
    }
    uint32_t digits[16] = {};
    for (int i = 0; i < 7; ++i) {
        uint64_t carry = 0;
        for (int j = 0; j < 5; ++j) {
            const uint64_t cur = static_cast<uint64_t>(digits[i + j]) + static_cast<uint64_t>(shifted[i]) * mu[j] + carry;
            digits[i + j] = static_cast<uint32_t>(cur);
            carry = cur >> 32;
        }
        for (int k = i + 5; carry != 0 && k < 16; ++k) {
            const uint64_t cur = static_cast<uint64_t>(digits[k]) + carry;
            digits[k] = static_cast<uint32_t>(cur);
            carry = cur >> 32;
        }
    }
    uint32_t prod[kCrtLimbs + 8] = {};
    for (int i = 0; i < 8; ++i) {
        const uint32_t factor = digits[4 + i];
        if (factor == 0) continue;
        uint64_t column = 0;
        for (int limb = 0; limb < q_limbs; ++limb) {
            const int at = limb + i;
            if (at >= kCrtLimbs + 8) break;
            const uint64_t cur = static_cast<uint64_t>(prod[at]) + static_cast<uint64_t>(q[limb]) * factor + column;
            prod[at] = static_cast<uint32_t>(cur);
            column = cur >> 32;
        }
        const int tail = q_limbs + i;
        if (column != 0 && tail < kCrtLimbs + 8) prod[tail] = static_cast<uint32_t>(prod[tail] + column);
    }
    uint32_t borrow = 0;
    for (int limb = 0; limb < kCrtLimbs; ++limb) {
        const uint64_t right = static_cast<uint64_t>(prod[limb]) + borrow;
        if (acc[limb] < right) {
            acc[limb] = static_cast<uint32_t>(static_cast<uint64_t>(acc[limb]) + (1ull << 32) - right);
            borrow = 1;
        } else {
            acc[limb] = static_cast<uint32_t>(acc[limb] - right);
            borrow = 0;
        }
    }
    if (borrow) {
        uint32_t add_carry = 0;
        for (int limb = 0; limb < q_limbs; ++limb) {
            const uint64_t sum = static_cast<uint64_t>(acc[limb]) + q[limb] + add_carry;
            acc[limb] = static_cast<uint32_t>(sum);
            add_carry = sum >> 32;
        }
    }
    for (int fix = 0; fix < 2; ++fix) {
        bool greater_or_equal = true;
        for (int limb = kCrtLimbs - 1; limb >= 0; --limb) {
            const uint32_t q_limb = limb < q_limbs ? q[limb] : 0u;
            if (acc[limb] == q_limb) continue;
            greater_or_equal = acc[limb] > q_limb;
            break;
        }
        if (!greater_or_equal) break;
        borrow = 0;
        for (int limb = 0; limb < kCrtLimbs; ++limb) {
            const uint64_t right = static_cast<uint64_t>(limb < q_limbs ? q[limb] : 0u) + borrow;
            if (acc[limb] < right) {
                acc[limb] = static_cast<uint32_t>(static_cast<uint64_t>(acc[limb]) + (1ull << 32) - right);
                borrow = 1;
            } else {
                acc[limb] = static_cast<uint32_t>(acc[limb] - right);
                borrow = 0;
            }
        }
    }
}

__host__ __device__ void mod_q(uint32_t* acc, const uint32_t* q, int q_limbs, int wide) {
    const int q_top = top_bit(q, q_limbs);
    int acc_top = top_bit(acc, wide);
    if (q_top < 0) return;
    while (acc_top >= q_top) {
        int shift = acc_top - q_top;
        if (cmp_shifted(acc, q, q_limbs, shift, wide) < 0) {
            if (shift == 0) break;
            --shift;
        }
        sub_shifted(acc, q, q_limbs, shift, wide);
        acc_top = top_bit(acc, wide);
    }
}

bool is_prime_u64(uint64_t n) {
    if (n < 2) return false;
    if ((n & 1ull) == 0) return n == 2;
    uint64_t d = n - 1;
    int s = 0;
    while ((d & 1ull) == 0) {
        d >>= 1;
        ++s;
    }
    const uint64_t bases[] = {2ull, 325ull, 9375ull, 28178ull, 450775ull, 9780504ull, 1795265022ull};
    for (uint64_t base : bases) {
        if (base % n == 0) continue;
        uint64_t x = pow_mod(base, d, n);
        if (x == 1 || x == n - 1) continue;
        bool witness = true;
        for (int round = 1; round < s; ++round) {
            x = mul_mod(x, x, n);
            if (x == n - 1) {
                witness = false;
                break;
            }
        }
        if (witness) return false;
    }
    return true;
}

uint64_t find_psi(uint64_t prime, uint32_t n) {
    const uint64_t exp = (prime - 1) / (2ull * n);
    for (uint64_t base = 2; base < 10000; ++base) {
        const uint64_t psi = pow_mod(base, exp, prime);
        if (pow_mod(psi, n, prime) == prime - 1) return psi;
    }
    throw std::runtime_error("no 2N-th root in this field");
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

__global__ void to_residues_kernel(
    const uint32_t* input, uint64_t* output, const uint64_t* psi, const uint64_t* primes,
    const uint64_t* nprime, const uint64_t* r2, const uint64_t* mu, int n, int limbs, int channels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel = blockIdx.y;
    if (index >= n || channel >= channels) return;
    const uint64_t prime = primes[channel];
    const uint64_t np = nprime[channel];
    uint64_t residue = 0;
    for (int limb = limbs - 1; limb >= 0; --limb) {
        const uint32_t word = input[index * limbs + limb];
        uint64_t v_hi = residue >> 32;
        uint64_t v_lo = (residue << 32) + word;
        if (v_lo < word) ++v_hi;
        residue = barrett_reduce(v_hi, v_lo, prime, mu[channel]);
    }
    const uint64_t lifted = mont_mul(residue, r2[channel], prime, np);
    output[static_cast<size_t>(channel) * n + index] =
        mont_mul(lifted, psi[static_cast<size_t>(channel) * n + index], prime, np);
}

// The harness stores coefficient i at i*L, so a thread-per-coefficient load strides
// 112 bytes and wastes the bus. A tile load is contiguous; the Horner then reads shared memory.
__global__ void to_residues_tiled_kernel(
    const uint32_t* input, uint64_t* output, const uint64_t* psi, const uint64_t* primes,
    const uint64_t* nprime, const uint64_t* r2, const uint64_t* pow32, const uint64_t* magic_lo,
    const uint64_t* magic_hi, int n, int channels) {
    constexpr int kTile = 128;
    constexpr int kLimbs = 28;
    __shared__ uint32_t tile[kTile * kLimbs];
    const int channel = blockIdx.y;
    const int tile_index = blockIdx.x * kTile;
    const int tid = threadIdx.x;
    if (channel >= channels) return;
    const uint32_t* base = input + static_cast<size_t>(tile_index) * kLimbs;
    for (int word = tid; word < kTile * kLimbs; word += kTile) {
        tile[word] = base[word];
    }
    __syncthreads();
    const int index = tile_index + tid;
    if (index >= n) return;
    const uint64_t prime = primes[channel];
    const uint64_t np = nprime[channel];
    const uint32_t* words = tile + tid * kLimbs;
    const uint64_t* place = pow32 + static_cast<size_t>(channel) * 32;
    uint64_t acc_hi = 0;
    uint64_t acc_lo = 0;
    for (int limb = 0; limb < kLimbs; ++limb) {
        uint64_t hi = 0;
        uint64_t lo = 0;
        mul64_wide(words[limb], place[limb], hi, lo);
        const uint64_t next = acc_lo + lo;
        const uint64_t carry = next < acc_lo ? 1ull : 0ull;
        acc_lo = next;
        acc_hi += hi + carry;
    }
    const uint64_t residue = reduce_128(acc_hi, acc_lo, prime, magic_hi[channel], magic_lo[channel]);
    const uint64_t lifted = mont_mul(residue, r2[channel], prime, np);
    output[static_cast<size_t>(channel) * n + index] =
        mont_mul(lifted, psi[static_cast<size_t>(channel) * n + index], prime, np);
}

void launch_residues(
    dim3 grid, int threads, const uint32_t* input, uint64_t* output, const uint64_t* psi,
    const uint64_t* primes, const uint64_t* nprime, const uint64_t* r2, const uint64_t* mu,
    const uint64_t* pow32, const uint64_t* magic_lo, const uint64_t* magic_hi,
    int n, int limbs, int channels) {
    (void)pow32;
    (void)magic_lo;
    (void)magic_hi;
    if (limbs == 28 && (n % 128) == 0) {
        (void)mu;
        to_residues_tiled_kernel<<<dim3(n / 128, channels), 128>>>(
            input, output, psi, primes, nprime, r2, pow32, magic_lo, magic_hi, n, channels);
        return;
    }
    to_residues_kernel<<<grid, threads>>>(input, output, psi, primes, nprime, r2, mu, n, limbs, channels);
}

__global__ void bitrev_kernel(uint64_t* data, int n, int bits, int channels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel = blockIdx.y;
    if (index >= n || channel >= channels) return;
    const int reversed = static_cast<int>(__brev(static_cast<unsigned int>(index)) >> (32 - bits));
    if (index < reversed) {
        uint64_t* row = data + static_cast<size_t>(channel) * n;
        const uint64_t tmp = row[index];
        row[index] = row[reversed];
        row[reversed] = tmp;
    }
}

__global__ void stage_kernel(
    uint64_t* data, const uint64_t* twiddles, const uint64_t* shoups, const uint64_t* primes,
    int n, int half, int tw_base, int channels) {
    const int butterfly = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel = blockIdx.y;
    if (butterfly >= n / 2 || channel >= channels) return;
    const int lane = butterfly % half;
    const int start = (butterfly / half) * (half * 2);
    const uint64_t prime = primes[channel];
    uint64_t* row = data + static_cast<size_t>(channel) * n;
    const int tw_at = tw_base + lane;
    const size_t channel_at = static_cast<size_t>(channel) * n + tw_at;
    const uint64_t left = row[start + lane];
    const uint64_t right = shoup_mul(row[start + lane + half], twiddles[channel_at], shoups[channel_at], prime);
    row[start + lane] = add_mod(left, right, prime);
    row[start + lane + half] = sub_mod(left, right, prime);
}

__global__ void pointwise_kernel(
    uint64_t* left, const uint64_t* right, const uint64_t* primes, const uint64_t* nprime,
    int n, int channels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel = blockIdx.y;
    if (index >= n || channel >= channels) return;
    const size_t at = static_cast<size_t>(channel) * n + index;
    left[at] = mont_mul(left[at], right[at], primes[channel], nprime[channel]);
}

__global__ void untwist_kernel(
    uint64_t* data, const uint64_t* psi_inv, const uint64_t* ninv, const uint64_t* primes,
    const uint64_t* nprime, int n, int channels) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel = blockIdx.y;
    if (index >= n || channel >= channels) return;
    const uint64_t prime = primes[channel];
    const uint64_t np = nprime[channel];
    const size_t at = static_cast<size_t>(channel) * n + index;
    uint64_t value = mont_mul(data[at], ninv[channel], prime, np);
    value = mont_mul(value, psi_inv[at], prime, np);
    data[at] = mont_mul(value, 1, prime, np);
}

__device__ void accum_factor(uint32_t* acc, const uint32_t* term, uint64_t factor) {
    uint64_t carry = 0;
    const uint32_t low = static_cast<uint32_t>(factor);
    for (int limb = 0; limb < kCrtLimbs; ++limb) {
        const uint64_t cur = static_cast<uint64_t>(acc[limb]) + static_cast<uint64_t>(term[limb]) * low + carry;
        acc[limb] = static_cast<uint32_t>(cur);
        carry = cur >> 32;
    }
    const uint32_t high = static_cast<uint32_t>(factor >> 32);
    if (high == 0) return;
    carry = 0;
    for (int limb = 0; limb < kCrtLimbs - 1; ++limb) {
        const uint64_t cur = static_cast<uint64_t>(acc[limb + 1]) + static_cast<uint64_t>(term[limb]) * high + carry;
        acc[limb + 1] = static_cast<uint32_t>(cur);
        carry = cur >> 32;
    }
}

__device__ void add_scaled(uint64_t& sum_hi, uint64_t& sum_lo, uint64_t add_hi, uint64_t add_lo) {
    const uint64_t next = sum_lo + add_lo;
    const uint64_t carry = next < sum_lo ? 1ull : 0ull;
    sum_lo = next;
    sum_hi += add_hi + carry;
}

// floor(v * 2^64 / p) from magic = floor(2^128 / p), and it never rounds up.
__device__ void scaled_term(uint64_t v, uint64_t magic_hi, uint64_t magic_lo, uint64_t& out_hi, uint64_t& out_lo) {
    uint64_t lo_hi = 0;
    uint64_t lo_lo = 0;
    uint64_t hi_hi = 0;
    uint64_t hi_lo = 0;
    mul64_wide(v, magic_lo, lo_hi, lo_lo);
    mul64_wide(v, magic_hi, hi_hi, hi_lo);
    const uint64_t mid = lo_hi + hi_lo;
    const uint64_t mid_carry = mid < lo_hi ? 1ull : 0ull;
    out_lo = mid;
    out_hi = hi_hi + mid_carry;
}

__device__ void barrett_mod_34(uint32_t* acc, const uint32_t* q, const uint32_t* mu, int q_limbs) {
    uint64_t wide[16] = {};
    for (int i = 0; i < 7; ++i) {
        for (int j = 0; j < 8; ++j) {
            wide[i + j] += static_cast<uint64_t>(acc[27 + i]) * mu[j];
        }
    }
    uint64_t carry = 0;
    uint32_t digits[16];
    for (int k = 0; k < 16; ++k) {
        const uint64_t cur = wide[k] + carry;
        digits[k] = static_cast<uint32_t>(cur);
        carry = cur >> 32;
    }
    uint32_t prod[kCrtLimbs];
#pragma unroll
    for (int limb = 0; limb < kCrtLimbs; ++limb) prod[limb] = 0;
    for (int i = 0; i < 6; ++i) {
        const uint32_t factor = digits[7 + i];
        if (factor == 0) continue;
        uint64_t column = 0;
        for (int limb = 0; limb < q_limbs && limb + i < kCrtLimbs; ++limb) {
            const uint64_t cur = static_cast<uint64_t>(prod[limb + i]) + static_cast<uint64_t>(q[limb]) * factor + column;
            prod[limb + i] = static_cast<uint32_t>(cur);
            column = cur >> 32;
        }
    }
    uint32_t borrow = 0;
    for (int limb = 0; limb < kCrtLimbs; ++limb) {
        const uint64_t right = static_cast<uint64_t>(prod[limb]) + borrow;
        if (acc[limb] < right) {
            acc[limb] = static_cast<uint32_t>(static_cast<uint64_t>(acc[limb]) + (1ull << 32) - right);
            borrow = 1;
        } else {
            acc[limb] = static_cast<uint32_t>(acc[limb] - right);
            borrow = 0;
        }
    }
    if (borrow) {
        uint32_t add_carry = 0;
        for (int limb = 0; limb < q_limbs; ++limb) {
            const uint64_t sum = static_cast<uint64_t>(acc[limb]) + q[limb] + add_carry;
            acc[limb] = static_cast<uint32_t>(sum);
            add_carry = static_cast<uint32_t>(sum >> 32);
        }
    }
    for (int fix = 0; fix < 3; ++fix) {
        bool greater_or_equal = true;
        for (int limb = kCrtLimbs - 1; limb >= 0; --limb) {
            const uint32_t q_limb = limb < q_limbs ? q[limb] : 0;
            if (acc[limb] == q_limb) continue;
            greater_or_equal = acc[limb] > q_limb;
            break;
        }
        if (!greater_or_equal) break;
        borrow = 0;
        for (int limb = 0; limb < kCrtLimbs; ++limb) {
            const uint64_t right = static_cast<uint64_t>(limb < q_limbs ? q[limb] : 0) + borrow;
            if (acc[limb] < right) {
                acc[limb] = static_cast<uint32_t>(static_cast<uint64_t>(acc[limb]) + (1ull << 32) - right);
                borrow = 1;
            } else {
                acc[limb] = static_cast<uint32_t>(acc[limb] - right);
                borrow = 0;
            }
        }
    }
}

__global__ void fast_crt_kernel(
    const uint64_t* residues, uint32_t* output, const uint64_t* primes, const uint64_t* nprime,
    const uint64_t* r2, const uint64_t* crt_y, const uint64_t* magic_lo, const uint64_t* magic_hi,
    const uint32_t* mi_q, const uint32_t* m_mod_q, const uint32_t* q, const uint32_t* q_mu,
    int* unsafe_count, int* unsafe_index, int n, int channels, int limbs) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    // The scaled sum is short of the real sum by less than this many units of 2^-64.
    constexpr uint64_t kErr = 96ull;
    constexpr uint64_t kHalf = 1ull << 63;
    uint32_t acc[kCrtLimbs];
#pragma unroll
    for (int limb = 0; limb < kCrtLimbs; ++limb) acc[limb] = 0;
    uint64_t sum_hi = 0;
    uint64_t sum_lo = 0;
#pragma unroll 1
    for (int channel = 0; channel < channels; ++channel) {
        const uint64_t prime = primes[channel];
        const uint64_t np = nprime[channel];
        const uint64_t residue = residues[static_cast<size_t>(channel) * n + index];
        const uint64_t mixed = mont_mul(mont_mul(residue, crt_y[channel], prime, np), r2[channel], prime, np);
        accum_factor(acc, mi_q + static_cast<size_t>(channel) * kCrtLimbs, mixed);
        uint64_t term_hi = 0;
        uint64_t term_lo = 0;
        scaled_term(mixed, magic_hi[channel], magic_lo[channel], term_hi, term_lo);
        add_scaled(sum_hi, sum_lo, term_hi, term_lo);
    }
    const bool quotient_certain = sum_lo <= ~0ull - kErr;
    const bool sign_certain = quotient_certain && (sum_lo + kErr < kHalf || sum_lo >= kHalf);
    (void)quotient_certain;
    (void)sign_certain;
    (void)unsafe_count;
    (void)unsafe_index;
    if (limbs == 28) mod_q_barrett(acc, q, q_mu, limbs);
    else mod_q(acc, q, limbs, kCrtLimbs);
    uint32_t multiple[kCrtLimbs];
#pragma unroll
    for (int limb = 0; limb < kCrtLimbs; ++limb) multiple[limb] = 0;
    accum_factor(multiple, m_mod_q, sum_hi);
    if (limbs == 28) mod_q_barrett(multiple, q, q_mu, limbs);
    else mod_q(multiple, q, limbs, kCrtLimbs);
    uint32_t borrow = 0;
    for (int limb = 0; limb < limbs; ++limb) {
        const uint64_t left = acc[limb];
        const uint64_t right = static_cast<uint64_t>(multiple[limb]) + borrow;
        if (left < right) {
            acc[limb] = static_cast<uint32_t>(left + (1ull << 32) - right);
            borrow = 1;
        } else {
            acc[limb] = static_cast<uint32_t>(left - right);
            borrow = 0;
        }
    }
    if (borrow) {
        uint32_t carry = 0;
        for (int limb = 0; limb < limbs; ++limb) {
            const uint64_t sum = static_cast<uint64_t>(acc[limb]) + q[limb] + carry;
            acc[limb] = static_cast<uint32_t>(sum);
            carry = static_cast<uint32_t>(sum >> 32);
        }
    }
    if (sum_lo >= kHalf) {
        borrow = 0;
        for (int limb = 0; limb < limbs; ++limb) {
            const uint64_t left = acc[limb];
            const uint64_t right = static_cast<uint64_t>(m_mod_q[limb]) + borrow;
            if (left < right) {
                acc[limb] = static_cast<uint32_t>(left + (1ull << 32) - right);
                borrow = 1;
            } else {
                acc[limb] = static_cast<uint32_t>(left - right);
                borrow = 0;
            }
        }
        if (borrow) {
            uint32_t carry = 0;
            for (int limb = 0; limb < limbs; ++limb) {
                const uint64_t sum = static_cast<uint64_t>(acc[limb]) + q[limb] + carry;
                acc[limb] = static_cast<uint32_t>(sum);
                carry = static_cast<uint32_t>(sum >> 32);
            }
        }
    }
    for (int limb = 0; limb < limbs; ++limb) output[index * limbs + limb] = acc[limb];
}

__global__ void crt_kernel(
    const uint64_t* residues, uint32_t* output, const uint64_t* primes, const uint64_t* nprime,
    const uint64_t* r2, const uint64_t* minv, const uint32_t* m_before, const uint32_t* mb_q,
    const uint64_t* cross, const uint32_t* half_m, const uint32_t* m_mod_q, const uint32_t* q,
    int n, int channels, int limbs, const int* indices, int count) {
    int index;
    if (indices != nullptr) {
        const int slot = blockIdx.x * blockDim.x + threadIdx.x;
        if (slot >= count) return;
        index = indices[slot];
    } else {
        index = blockIdx.x * blockDim.x + threadIdx.x;
        if (index >= n) return;
    }
    // The full product is only needed to see the sign. Each later remainder is
    // updated by the same term, so the kernel never divides an 80-limb number.
    uint32_t acc[kWide];
    uint32_t acc_q[40];
    uint64_t running_mod[kMaxChannels];
    const uint64_t base = residues[index];
    limbs_set_u64(acc, kWide, base);
    limbs_set_u64(acc_q, 40, base);
    mod_q(acc_q, q, limbs, 40);
    for (int other = 1; other < channels; ++other) {
        const uint64_t prime = primes[other];
        running_mod[other] = base < prime ? base : base % prime;
    }
    for (int channel = 1; channel < channels; ++channel) {
        const uint64_t prime = primes[channel];
        const uint64_t residue = residues[static_cast<size_t>(channel) * n + index];
        uint64_t gap = sub_mod(residue, running_mod[channel], prime);
        gap = mul_mod(gap, minv[channel], prime);
        add_mul_u64(acc, m_before + static_cast<size_t>(channel) * kWide, gap, kWide);
        add_mul_u64(acc_q, mb_q + static_cast<size_t>(channel) * 40, gap, 40);
        mod_q(acc_q, q, limbs, 40);
        for (int other = channel + 1; other < channels; ++other) {
            const uint64_t prime_other = primes[other];
            const uint64_t np = nprime[other];
            const uint64_t factor = gap < prime_other ? gap : gap % prime_other;
            uint64_t product = mont_mul(cross[static_cast<size_t>(channel) * channels + other], factor, prime_other, np);
            product = mont_mul(product, r2[other], prime_other, np);
            running_mod[other] = add_mod(running_mod[other], product, prime_other);
        }
    }
    const bool negative = limbs_cmp(acc, half_m, kWide) > 0;
    if (negative) {
        uint32_t borrow = 0;
        for (int limb = 0; limb < limbs; ++limb) {
            const uint64_t left = acc_q[limb];
            const uint64_t right = static_cast<uint64_t>(m_mod_q[limb]) + borrow;
            if (left < right) {
                acc_q[limb] = static_cast<uint32_t>(left + (1ull << 32) - right);
                borrow = 1;
            } else {
                acc_q[limb] = static_cast<uint32_t>(left - right);
                borrow = 0;
            }
        }
        if (borrow) {
            uint32_t carry = 0;
            for (int limb = 0; limb < limbs; ++limb) {
                const uint64_t sum = static_cast<uint64_t>(acc_q[limb]) + q[limb] + carry;
                acc_q[limb] = static_cast<uint32_t>(sum);
                carry = static_cast<uint32_t>(sum >> 32);
            }
        }
    }
    for (int limb = 0; limb < limbs; ++limb) output[index * limbs + limb] = acc_q[limb];
}

__global__ void radix8_kernel(
    uint64_t* data, const uint64_t* twiddles, const uint64_t* shoups, const uint64_t* primes,
    int n, int stride, int channels, int bits, int first) {
    const int id = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel = blockIdx.y;
    if (channel >= channels || id >= n / 8) return;
    const int group = id / stride;
    const int offset = id - group * stride;
    const int base = group * (stride << 3);
    const uint64_t prime = primes[channel];
    uint64_t* row = data + static_cast<size_t>(channel) * n;
    const uint64_t* twiddle_row = twiddles + static_cast<size_t>(channel) * n;
    const uint64_t* shoup_row = shoups + static_cast<size_t>(channel) * n;

    uint64_t value[8];
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
        const int pos = base + offset + lane * stride;
        const int src = first ? static_cast<int>(__brev(static_cast<unsigned int>(pos)) >> (32 - bits)) : pos;
        value[lane] = row[src];
    }
#pragma unroll
    for (int stage = 0; stage < 3; ++stage) {
        const int dist = 1 << stage;
        const int global_half = stride << stage;
#pragma unroll
        for (int start = 0; start < 8; start += dist * 2) {
#pragma unroll
            for (int lane = 0; lane < dist; ++lane) {
                const int global_lane = offset + lane * stride;
                const int tw_at = (global_half - 1) + global_lane;
                const uint64_t left = value[start + lane];
                const uint64_t right = shoup_mul(value[start + lane + dist], twiddle_row[tw_at], shoup_row[tw_at], prime);
                value[start + lane] = add_mod(left, right, prime);
                value[start + lane + dist] = sub_mod(left, right, prime);
            }
        }
    }
#pragma unroll
    for (int lane = 0; lane < 8; ++lane) row[base + offset + lane * stride] = value[lane];
}

// The first twelve rounds stay inside a 4096-word tile. Bit reversal is the load.
// Four radix-8 rounds replace twelve synced radix-2 passes.
__global__ void ntt_tile_kernel(
    uint64_t* data, const uint64_t* twiddles, const uint64_t* shoups, const uint64_t* primes,
    int n, int channels) {
    constexpr int kTile = 4096;
    __shared__ uint64_t tile[kTile];
    const int tile_id = blockIdx.x;
    const int channel = blockIdx.y;
    if (channel >= channels || tile_id >= n / kTile) return;
    int bits = 0;
    for (int value = n; value > 1; value >>= 1) ++bits;
    uint64_t* row = data + static_cast<size_t>(channel) * n;
    const uint64_t* twiddle_row = twiddles + static_cast<size_t>(channel) * n;
    const uint64_t* shoup_row = shoups + static_cast<size_t>(channel) * n;
    const uint64_t prime = primes[channel];
    for (int item = threadIdx.x; item < kTile; item += blockDim.x) {
        const int global = tile_id * kTile + item;
        const int src = static_cast<int>(__brev(static_cast<unsigned int>(global)) >> (32 - bits));
        tile[item] = row[src];
    }
    __syncthreads();
    for (int stride = 1; stride < kTile; stride <<= 3) {
        const int groups = kTile / (stride * 8);
        for (int id = threadIdx.x; id < groups * stride; id += blockDim.x) {
            const int group = id / stride;
            const int offset = id - group * stride;
            const int base = group * (stride << 3);
            uint64_t value[8];
#pragma unroll
            for (int lane = 0; lane < 8; ++lane) value[lane] = tile[base + offset + lane * stride];
#pragma unroll
            for (int stage = 0; stage < 3; ++stage) {
                const int dist = 1 << stage;
                const int global_half = stride << stage;
#pragma unroll
                for (int start = 0; start < 8; start += dist * 2) {
#pragma unroll
                    for (int lane = 0; lane < dist; ++lane) {
                        const int tw_at = (global_half - 1) + offset + lane * stride;
                        const uint64_t left = value[start + lane];
                        const uint64_t right = shoup_mul(
                            value[start + lane + dist], twiddle_row[tw_at], shoup_row[tw_at], prime);
                        value[start + lane] = add_mod(left, right, prime);
                        value[start + lane + dist] = sub_mod(left, right, prime);
                    }
                }
            }
#pragma unroll
            for (int lane = 0; lane < 8; ++lane) tile[base + offset + lane * stride] = value[lane];
        }
        __syncthreads();
    }
    for (int item = threadIdx.x; item < kTile; item += blockDim.x) row[tile_id * kTile + item] = tile[item];
}

void forward_ntt(
    uint64_t* data, const uint64_t* twiddles, const uint64_t* shoups, const uint64_t* primes,
    int n, int channels) {
    const int threads = 128;
    int bits = 0;
    for (int value = n; value > 1; value >>= 1) ++bits;
    if (n < 8) {
        bitrev_kernel<<<dim3((n + threads - 1) / threads, channels), threads>>>(data, n, bits, channels);
        for (int half = 1; half < n; half <<= 1) {
            const int butterflies = n / 2;
            stage_kernel<<<dim3((butterflies + threads - 1) / threads, channels), threads>>>(
                data, twiddles, shoups, primes, n, half, half - 1, channels);
        }
        return;
    }
    int stride = 1;
    if (n >= 4096) {
        ntt_tile_kernel<<<dim3(n / 4096, channels), threads>>>(data, twiddles, shoups, primes, n, channels);
        stride = 4096;
    } else {
        bitrev_kernel<<<dim3((n + threads - 1) / threads, channels), threads>>>(data, n, bits, channels);
    }
    while (stride * 8 <= n) {
        radix8_kernel<<<dim3((n / 8 + threads - 1) / threads, channels), threads>>>(
            data, twiddles, shoups, primes, n, stride, channels, bits, 0);
        stride <<= 3;
    }
    for (int half = stride; half < n; half <<= 1) {
        const int butterflies = n / 2;
        stage_kernel<<<dim3((butterflies + threads - 1) / threads, channels), threads>>>(
            data, twiddles, shoups, primes, n, half, half - 1, channels);
    }
}

void inverse_ntt(
    uint64_t* data, const uint64_t* twiddles, const uint64_t* shoups, const uint64_t* primes,
    int n, int channels) {
    forward_ntt(data, twiddles, shoups, primes, n, channels);
}

}  // namespace

void barrett_mu_1088(const uint32_t* q, int q_limbs, uint32_t* mu) {
    uint32_t rem[40] = {};
    for (int i = 0; i < 8; ++i) mu[i] = 0;
    for (int bit = 1088; bit >= 0; --bit) {
        uint32_t carry = bit == 1088 ? 1u : 0u;
        for (int limb = 0; limb < 40; ++limb) {
            const uint32_t next = (rem[limb] << 1) | carry;
            carry = rem[limb] >> 31;
            rem[limb] = next;
        }
        int cmp = 0;
        for (int limb = 39; limb >= 0; --limb) {
            const uint32_t q_limb = limb < q_limbs ? q[limb] : 0u;
            if (rem[limb] == q_limb) continue;
            cmp = rem[limb] > q_limb ? 1 : -1;
            break;
        }
        if (cmp < 0) continue;
        uint32_t borrow = 0;
        for (int limb = 0; limb < 40; ++limb) {
            const uint64_t right = static_cast<uint64_t>(limb < q_limbs ? q[limb] : 0u) + borrow;
            if (rem[limb] < right) {
                rem[limb] = static_cast<uint32_t>(static_cast<uint64_t>(rem[limb]) + (1ull << 32) - right);
                borrow = 1;
            } else {
                rem[limb] = static_cast<uint32_t>(rem[limb] - right);
                borrow = 0;
            }
        }
        const int word = bit >> 5;
        if (word < 8) mu[word] |= 1u << (bit & 31);
    }
}

// mu = floor(2^(868+128) / q). The challenge modulus is 868 bits, 27 limbs plus 4 bits.
void barrett_mu_996(const uint32_t* q, int q_limbs, uint32_t* mu) {
    uint32_t rem[40] = {};
    for (int i = 0; i < 8; ++i) mu[i] = 0;
    for (int bit = 996; bit >= 0; --bit) {
        uint32_t carry = bit == 996 ? 1u : 0u;
        for (int limb = 0; limb < 40; ++limb) {
            const uint32_t next = (rem[limb] << 1) | carry;
            carry = rem[limb] >> 31;
            rem[limb] = next;
        }
        int cmp = 0;
        for (int limb = 39; limb >= 0; --limb) {
            const uint32_t q_limb = limb < q_limbs ? q[limb] : 0u;
            if (rem[limb] == q_limb) continue;
            cmp = rem[limb] > q_limb ? 1 : -1;
            break;
        }
        if (cmp < 0) continue;
        uint32_t borrow = 0;
        for (int limb = 0; limb < 40; ++limb) {
            const uint64_t right = static_cast<uint64_t>(limb < q_limbs ? q[limb] : 0u) + borrow;
            if (rem[limb] < right) {
                rem[limb] = static_cast<uint32_t>(static_cast<uint64_t>(rem[limb]) + (1ull << 32) - right);
                borrow = 1;
            } else {
                rem[limb] = static_cast<uint32_t>(rem[limb] - right);
                borrow = 0;
            }
        }
        const int word = bit >> 5;
        if (word < 8) mu[word] |= 1u << (bit & 31);
    }
}

uint64_t div_pow2_64(uint64_t prime, uint64_t& remainder) {
    uint64_t quotient = 0;
    remainder = 1;
    for (int bit = 63; bit >= 0; --bit) {
        remainder <<= 1;
        if (remainder >= prime) {
            remainder -= prime;
            quotient |= 1ull << bit;
        }
    }
    return quotient;
}

void div_u128(uint64_t hi, uint64_t lo, uint64_t prime, uint64_t& quot_hi, uint64_t& quot_lo, uint64_t& remainder) {
    remainder = 0;
    quot_hi = 0;
    quot_lo = 0;
    for (int bit = 63; bit >= 0; --bit) {
        remainder = (remainder << 1) | ((hi >> bit) & 1ull);
        if (remainder >= prime) {
            remainder -= prime;
            quot_hi |= 1ull << bit;
        }
    }
    for (int bit = 63; bit >= 0; --bit) {
        remainder = (remainder << 1) | ((lo >> bit) & 1ull);
        if (remainder >= prime) {
            remainder -= prime;
            quot_lo |= 1ull << bit;
        }
    }
}

void* fherma_init(const fherma::Point& point) {
    if (mul_mod(7, 5, 11) != 2) throw std::runtime_error("modular multiply failed the 7*5 check");
    {
        const uint64_t prime = 11;
        const uint64_t np = 0ull - inv_mod_2_64(prime);
        const uint64_t lift = pow_mod(2, 128, prime);
        const uint64_t left = mont_mul(7, lift, prime, np);
        const uint64_t right = mont_mul(5, lift, prime, np);
        const uint64_t product = mont_mul(mont_mul(left, right, prime, np), 1, prime, np);
        if (product != 2) throw std::runtime_error("Montgomery multiply failed the 7*5 check");
        const uint64_t mu = barrett_mu(prime);
        if (barrett_reduce(0, 100, prime, mu) != 1) throw std::runtime_error("Barrett reduction failed");
        uint64_t remainder = 0;
        if (div_pow2_64(3, remainder) != 6148914691236517205ull || remainder != 1) {
            throw std::runtime_error("reciprocal division failed the 2^64/3 check");
        }
    }
    if (point.N < 2 || point.N > kMaxN || (point.N & (point.N - 1u)) != 0u) {
        throw std::runtime_error("N must be a power of two from 2 to 32768");
    }
    if (point.L == 0 || point.L > 32 || point.q.data.size() != point.L || (point.q.data[0] & 1u) == 0u) {
        throw std::runtime_error("q must be an odd modulus of L limbs");
    }

    uint32_t q_limbs[kWide] = {};
    for (uint32_t limb = 0; limb < point.L; ++limb) q_limbs[limb] = point.q.data[limb];
    const int q_bits = top_bit(q_limbs, kWide) + 1;
    int log_n = 0;
    for (uint32_t value = point.N; value > 1; value >>= 1) ++log_n;
    const int bound_bits = log_n + 2 * q_bits + 4;

    const uint64_t step = 2ull * point.N;
    uint64_t candidate = ((1ull << 62) - 1);
    candidate -= (candidate - 1) % step;
    std::vector<uint64_t> primes;
    uint32_t product[kWide];
    limbs_set_u64(product, kWide, 1);
    while (candidate > step && top_bit(product, kWide) + 1 < bound_bits) {
        if (is_prime_u64(candidate)) {
            if (static_cast<int>(primes.size()) >= kMaxChannels) {
                throw std::runtime_error("the coefficient bound does not fit the residue budget");
            }
            uint32_t next[kWide];
            for (int i = 0; i < kWide; ++i) next[i] = 0;
            // product = product * candidate, starting from the previous product.
            // add_mul into a zero buffer.
            add_mul_u64(next, product, candidate, kWide);
            for (int i = 0; i < kWide; ++i) product[i] = next[i];
            primes.push_back(candidate);
        }
        candidate -= step;
    }
    if (top_bit(product, kWide) + 1 < bound_bits) {
        throw std::runtime_error("not enough NTT primes for this modulus");
    }

    const int channels = static_cast<int>(primes.size());
    const int n = static_cast<int>(point.N);
    std::vector<uint64_t> tw_fwd(static_cast<size_t>(channels) * n);
    std::vector<uint64_t> tw_inv(static_cast<size_t>(channels) * n);
    std::vector<uint64_t> tw_fwd_shoup(static_cast<size_t>(channels) * n);
    std::vector<uint64_t> tw_inv_shoup(static_cast<size_t>(channels) * n);
    std::vector<uint64_t> psi(static_cast<size_t>(channels) * n);
    std::vector<uint64_t> psi_inv(static_cast<size_t>(channels) * n);
    std::vector<uint64_t> ninv(channels);
    std::vector<uint64_t> nprime_table(channels);
    std::vector<uint64_t> r2_table(channels);
    std::vector<uint64_t> mu_table(channels);
    std::vector<uint64_t> pow32(static_cast<size_t>(channels) * 32);
    std::vector<uint64_t> minv(channels);
    std::vector<uint32_t> m_before(static_cast<size_t>(channels) * kWide);

    uint32_t running[kWide];
    limbs_set_u64(running, kWide, 1);
    for (int channel = 0; channel < channels; ++channel) {
        const uint64_t prime = primes[channel];
        const uint64_t root = find_psi(prime, point.N);
        const uint64_t omega = mul_mod(root, root, prime);
        uint64_t power = 1;
        uint64_t inverse_power = 1;
        const uint64_t root_inv = pow_mod(root, prime - 2, prime);
        for (int index = 0; index < n; ++index) {
            psi[static_cast<size_t>(channel) * n + index] = power;
            psi_inv[static_cast<size_t>(channel) * n + index] = inverse_power;
            power = mul_mod(power, root, prime);
            inverse_power = mul_mod(inverse_power, root_inv, prime);
        }
        for (int half = 1; half < n; half <<= 1) {
            const int length = half << 1;
            const uint64_t step_root = pow_mod(omega, static_cast<uint64_t>(n / length), prime);
            const uint64_t step_inv = pow_mod(step_root, prime - 2, prime);
            uint64_t twiddle = 1;
            uint64_t twiddle_inv = 1;
            for (int lane = 0; lane < half; ++lane) {
                const size_t tw_at = static_cast<size_t>(channel) * n + (half - 1) + lane;
                tw_fwd[tw_at] = twiddle;
                tw_inv[tw_at] = twiddle_inv;
                uint64_t shoup_hi = 0;
                uint64_t shoup_lo = 0;
                uint64_t shoup_rem = 0;
                div_u128(twiddle, 0, prime, shoup_hi, shoup_lo, shoup_rem);
                tw_fwd_shoup[tw_at] = shoup_lo;
                div_u128(twiddle_inv, 0, prime, shoup_hi, shoup_lo, shoup_rem);
                tw_inv_shoup[tw_at] = shoup_lo;
                twiddle = mul_mod(twiddle, step_root, prime);
                twiddle_inv = mul_mod(twiddle_inv, step_inv, prime);
            }
        }
        ninv[channel] = pow_mod(static_cast<uint64_t>(n), prime - 2, prime);
        const uint64_t np = 0ull - inv_mod_2_64(prime);
        const uint64_t lift = pow_mod(2, 128, prime);
        nprime_table[channel] = np;
        r2_table[channel] = lift;
        mu_table[channel] = barrett_mu(prime);
        uint64_t place = 1;
        for (int limb = 0; limb < 32; ++limb) {
            pow32[static_cast<size_t>(channel) * 32 + limb] = place;
            place = mul_mod(place, 1ull << 32, prime);
        }
        for (int index = 0; index < n; ++index) {
            const size_t at = static_cast<size_t>(channel) * n + index;
            psi[at] = mont_mul(psi[at], lift, prime, np);
            psi_inv[at] = mont_mul(psi_inv[at], lift, prime, np);
        }
        ninv[channel] = mont_mul(ninv[channel], lift, prime, np);
        for (int limb = 0; limb < kWide; ++limb) m_before[static_cast<size_t>(channel) * kWide + limb] = running[limb];
        if (channel > 0) {
            const uint64_t reduced = limbs_mod_u64(running, kWide, prime);
            minv[channel] = pow_mod(reduced, prime - 2, prime);
        }
        uint32_t grown[kWide] = {};
        add_mul_u64(grown, running, prime, kWide);
        for (int limb = 0; limb < kWide; ++limb) running[limb] = grown[limb];
    }

    {
        const uint64_t prime = primes[0];
        const uint64_t np = nprime_table[0];
        const uint64_t lift = r2_table[0];
        for (int trial = 0; trial < 32; ++trial) {
            uint64_t left = (static_cast<uint64_t>(trial) * 0x9E3779B97F4A7C15ull) % prime;
            uint64_t right = ((static_cast<uint64_t>(trial) + 3ull) * 0xBF58476D1CE4E5B9ull) % prime;
            if (right == 0) right = 1;
            uint64_t shoup_hi = 0;
            uint64_t shoup_lo = 0;
            uint64_t shoup_rem = 0;
            div_u128(right, 0, prime, shoup_hi, shoup_lo, shoup_rem);
            const uint64_t left_m = mont_mul(left, lift, prime, np);
            const uint64_t right_m = mont_mul(right, lift, prime, np);
            uint64_t quot_hi = 0;
            uint64_t quot_lo = 0;
            uint64_t aw_hi = 0;
            uint64_t aw_lo = 0;
            uint64_t qp_hi = 0;
            uint64_t qp_lo = 0;
            mul64_wide(left_m, shoup_lo, quot_hi, quot_lo);
            mul64_wide(left_m, right, aw_hi, aw_lo);
            mul64_wide(quot_hi, prime, qp_hi, qp_lo);
            uint64_t got = aw_lo - qp_lo;
            if (got >= prime) got -= prime;
            if (got != mont_mul(left_m, right_m, prime, np)) {
                throw std::runtime_error("Shoup multiply does not match Montgomery");
            }
        }
    }

    uint32_t half_m[kWide];
    uint32_t carry_bit = 0;
    for (int limb = kWide - 1; limb >= 0; --limb) {
        half_m[limb] = (product[limb] >> 1) | (carry_bit << 31);
        carry_bit = product[limb] & 1u;
    }
    uint32_t m_mod_q[kWide];
    for (int limb = 0; limb < kWide; ++limb) m_mod_q[limb] = product[limb];
    mod_q(m_mod_q, q_limbs, static_cast<int>(point.L), kWide);

    std::vector<uint64_t> cross(static_cast<size_t>(channels) * channels);
    std::vector<uint32_t> mb_q(static_cast<size_t>(channels) * 40);
    for (int channel = 0; channel < channels; ++channel) {
        uint32_t reduced_q[kWide];
        for (int limb = 0; limb < kWide; ++limb) reduced_q[limb] = m_before[static_cast<size_t>(channel) * kWide + limb];
        mod_q(reduced_q, q_limbs, static_cast<int>(point.L), kWide);
        for (int limb = 0; limb < 40; ++limb) mb_q[static_cast<size_t>(channel) * 40 + limb] = reduced_q[limb];
        for (int other = channel + 1; other < channels; ++other) {
            cross[static_cast<size_t>(channel) * channels + other] =
                limbs_mod_u64(m_before.data() + static_cast<size_t>(channel) * kWide, kWide, primes[other]);
        }
    }

    if (q_bits + 70 > kCrtLimbs * 32) {
        throw std::runtime_error("the modulus does not fit the fast remainder");
    }
    std::vector<uint64_t> crt_y(channels);
    std::vector<uint64_t> magic_lo(channels);
    std::vector<uint64_t> magic_hi(channels);
    std::vector<uint32_t> mi_q(static_cast<size_t>(channels) * kCrtLimbs);
    for (int channel = 0; channel < channels; ++channel) {
        const uint64_t prime = primes[channel];
        uint64_t product_mod = 1;
        uint32_t except[kWide];
        limbs_set_u64(except, kWide, 1);
        for (int other = 0; other < channels; ++other) {
            if (other == channel) continue;
            product_mod = mul_mod(product_mod, primes[other] % prime, prime);
            uint32_t next[kWide] = {};
            add_mul_u64(next, except, primes[other], kWide);
            for (int limb = 0; limb < kWide; ++limb) except[limb] = next[limb];
        }
        crt_y[channel] = pow_mod(product_mod, prime - 2, prime);
        mod_q(except, q_limbs, static_cast<int>(point.L), kWide);
        for (int limb = 0; limb < kCrtLimbs; ++limb) mi_q[static_cast<size_t>(channel) * kCrtLimbs + limb] = except[limb];
        uint64_t remainder = 0;
        const uint64_t quot = div_pow2_64(prime, remainder);
        uint64_t quot_hi = 0;
        uint64_t quot_lo = 0;
        uint64_t unused = 0;
        div_u128(remainder, 0, prime, quot_hi, quot_lo, unused);
        magic_hi[channel] = quot + quot_hi;
        magic_lo[channel] = quot_lo;
    }
    for (int trial = 0; trial < 24; ++trial) {
        const uint64_t hi = static_cast<uint64_t>(trial) << 36;
        const uint64_t lo = static_cast<uint64_t>(trial) * 0x9E3779B97F4A7C15ull;
        if (reduce_128(hi, lo, primes[0], magic_hi[0], magic_lo[0]) != mod128(hi, lo, primes[0])) {
            throw std::runtime_error("wide reduction does not match division");
        }
    }
    {
        uint32_t sample[32];
        for (int limb = 0; limb < 32; ++limb) sample[limb] = 0xffffffffu - static_cast<uint32_t>(limb * 17);
        uint64_t acc_lo = 0;
        uint64_t acc_hi = 0;
        for (int limb = 0; limb < 28; ++limb) {
            uint64_t hi = 0;
            uint64_t lo = 0;
            mul64_wide(sample[limb], pow32[limb], hi, lo);
            const uint64_t next = acc_lo + lo;
            const uint64_t carry = next < acc_lo ? 1ull : 0ull;
            acc_lo = next;
            acc_hi += hi + carry;
        }
        if (reduce_128(acc_hi, acc_lo, primes[0], magic_hi[0], magic_lo[0]) != limbs_mod_u64(sample, 28, primes[0])) {
            throw std::runtime_error("digit sum does not match the modulus");
        }
    }

    uint32_t q_mu_host[8] = {};
    if (point.L == 28) {
        barrett_mu_996(q_limbs, static_cast<int>(point.L), q_mu_host);
        uint32_t slow[kCrtLimbs];
        uint32_t fast[kCrtLimbs];
        for (int limb = 0; limb < kCrtLimbs; ++limb) {
            const uint32_t base = limb < static_cast<int>(point.L) ? q_limbs[limb] : 0u;
            slow[limb] = base + static_cast<uint32_t>(limb * 97u + 11u);
            fast[limb] = slow[limb];
        }
        slow[30] = 0x3ffffu;
        fast[30] = 0x3ffffu;
        mod_q(slow, q_limbs, static_cast<int>(point.L), kCrtLimbs);
        mod_q_barrett(fast, q_limbs, q_mu_host, static_cast<int>(point.L));
        for (int limb = 0; limb < kCrtLimbs; ++limb) {
            if (slow[limb] != fast[limb]) {
                throw std::runtime_error("Barrett reduction does not match division");
            }
        }
    }

    auto* state = new State();
    state->N = n;
    state->L = static_cast<int>(point.L);
    state->channels = channels;
    const size_t row = static_cast<size_t>(channels) * n;
    try {
        check_cuda(cudaDeviceSetLimit(cudaLimitStackSize, 65536), "stack");
        check_cuda(cudaMalloc(&state->d_prime, primes.size() * sizeof(uint64_t)), "primes");
        check_cuda(cudaMalloc(&state->d_nprime, nprime_table.size() * sizeof(uint64_t)), "nprime");
        check_cuda(cudaMalloc(&state->d_r2, r2_table.size() * sizeof(uint64_t)), "r2");
        check_cuda(cudaMalloc(&state->d_mu, mu_table.size() * sizeof(uint64_t)), "barrett");
        check_cuda(cudaMalloc(&state->d_minv, minv.size() * sizeof(uint64_t)), "minv");
        check_cuda(cudaMalloc(&state->d_ninv, ninv.size() * sizeof(uint64_t)), "ninv");
        check_cuda(cudaMalloc(&state->d_a, row * sizeof(uint64_t)), "a");
        check_cuda(cudaMalloc(&state->d_b, row * sizeof(uint64_t)), "b");
        check_cuda(cudaMalloc(&state->d_tw_fwd, row * sizeof(uint64_t)), "twiddles");
        check_cuda(cudaMalloc(&state->d_tw_inv, row * sizeof(uint64_t)), "inverse twiddles");
        check_cuda(cudaMalloc(&state->d_tw_fwd_shoup, row * sizeof(uint64_t)), "twiddle shoup");
        check_cuda(cudaMalloc(&state->d_tw_inv_shoup, row * sizeof(uint64_t)), "inverse twiddle shoup");
        check_cuda(cudaMalloc(&state->d_psi, row * sizeof(uint64_t)), "psi");
        check_cuda(cudaMalloc(&state->d_psi_inv, row * sizeof(uint64_t)), "psi inverse");
        check_cuda(cudaMalloc(&state->d_in, static_cast<size_t>(n) * point.L * sizeof(uint32_t)), "input");
        check_cuda(cudaMalloc(&state->d_out, static_cast<size_t>(n) * point.L * sizeof(uint32_t)), "output");
        check_cuda(cudaMalloc(&state->d_mbefore, m_before.size() * sizeof(uint32_t)), "garner");
        check_cuda(cudaMalloc(&state->d_mb_q, mb_q.size() * sizeof(uint32_t)), "garner mod q");
        check_cuda(cudaMalloc(&state->d_cross, cross.size() * sizeof(uint64_t)), "cross residues");
        check_cuda(cudaMalloc(&state->d_half_m, kWide * sizeof(uint32_t)), "half modulus");
        check_cuda(cudaMalloc(&state->d_m_mod_q, kWide * sizeof(uint32_t)), "modulus residue");
        check_cuda(cudaMalloc(&state->d_q, kWide * sizeof(uint32_t)), "q");
        check_cuda(cudaMalloc(&state->d_q_mu, 8 * sizeof(uint32_t)), "q barrett");
        const size_t host_bytes = static_cast<size_t>(n) * point.L * sizeof(uint32_t);
        check_cuda(cudaMallocHost(&state->h_pin_a, host_bytes), "pinned a");
        check_cuda(cudaMallocHost(&state->h_pin_b, host_bytes), "pinned b");
        check_cuda(cudaMallocHost(&state->h_pin_c, host_bytes), "pinned c");
        check_cuda(cudaMalloc(&state->d_crt_y, crt_y.size() * sizeof(uint64_t)), "crt digits");
        check_cuda(cudaMalloc(&state->d_magic_lo, magic_lo.size() * sizeof(uint64_t)), "crt reciprocal");
        check_cuda(cudaMalloc(&state->d_magic_hi, magic_hi.size() * sizeof(uint64_t)), "crt reciprocal high");
        check_cuda(cudaMalloc(&state->d_mi_q, mi_q.size() * sizeof(uint32_t)), "crt coefficients");
        check_cuda(cudaMalloc(&state->d_unsafe_count, sizeof(int)), "crt fallback count");
        check_cuda(cudaMalloc(&state->d_unsafe_index, static_cast<size_t>(n) * sizeof(int)), "crt fallback index");
        check_cuda(cudaMalloc(&state->d_pow32, pow32.size() * sizeof(uint64_t)), "digit places");
        check_cuda(cudaMemcpy(state->d_prime, primes.data(), primes.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "primes");
        check_cuda(cudaMemcpy(state->d_nprime, nprime_table.data(), nprime_table.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "nprime");
        check_cuda(cudaMemcpy(state->d_r2, r2_table.data(), r2_table.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "r2");
        check_cuda(cudaMemcpy(state->d_mu, mu_table.data(), mu_table.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "barrett");
        check_cuda(cudaMemcpy(state->d_minv, minv.data(), minv.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "minv");
        check_cuda(cudaMemcpy(state->d_ninv, ninv.data(), ninv.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "ninv");
        check_cuda(cudaMemcpy(state->d_tw_fwd, tw_fwd.data(), row * sizeof(uint64_t), cudaMemcpyHostToDevice), "twiddles");
        check_cuda(cudaMemcpy(state->d_tw_inv, tw_inv.data(), row * sizeof(uint64_t), cudaMemcpyHostToDevice), "inverse twiddles");
        check_cuda(cudaMemcpy(state->d_tw_fwd_shoup, tw_fwd_shoup.data(), row * sizeof(uint64_t), cudaMemcpyHostToDevice), "twiddle shoup");
        check_cuda(cudaMemcpy(state->d_tw_inv_shoup, tw_inv_shoup.data(), row * sizeof(uint64_t), cudaMemcpyHostToDevice), "inverse twiddle shoup");
        check_cuda(cudaMemcpy(state->d_psi, psi.data(), row * sizeof(uint64_t), cudaMemcpyHostToDevice), "psi");
        check_cuda(cudaMemcpy(state->d_psi_inv, psi_inv.data(), row * sizeof(uint64_t), cudaMemcpyHostToDevice), "psi inverse");
        check_cuda(cudaMemcpy(state->d_mbefore, m_before.data(), m_before.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "garner");
        check_cuda(cudaMemcpy(state->d_mb_q, mb_q.data(), mb_q.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "garner mod q");
        check_cuda(cudaMemcpy(state->d_cross, cross.data(), cross.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "cross residues");
        check_cuda(cudaMemcpy(state->d_half_m, half_m, kWide * sizeof(uint32_t), cudaMemcpyHostToDevice), "half modulus");
        check_cuda(cudaMemcpy(state->d_m_mod_q, m_mod_q, kWide * sizeof(uint32_t), cudaMemcpyHostToDevice), "modulus residue");
        check_cuda(cudaMemcpy(state->d_q, q_limbs, kWide * sizeof(uint32_t), cudaMemcpyHostToDevice), "q");
        check_cuda(cudaMemcpy(state->d_q_mu, q_mu_host, 8 * sizeof(uint32_t), cudaMemcpyHostToDevice), "q barrett");
        check_cuda(cudaMemcpy(state->d_crt_y, crt_y.data(), crt_y.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "crt digits");
        check_cuda(cudaMemcpy(state->d_magic_lo, magic_lo.data(), magic_lo.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "crt reciprocal");
        check_cuda(cudaMemcpy(state->d_magic_hi, magic_hi.data(), magic_hi.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "crt reciprocal high");
        check_cuda(cudaMemcpy(state->d_mi_q, mi_q.data(), mi_q.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "crt coefficients");
        check_cuda(cudaMemcpy(state->d_pow32, pow32.data(), pow32.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "digit places");
    } catch (...) {
        fherma_free(state);
        throw;
    }
    return state;
}

fherma::Outputs fherma_run(void* raw, const fherma::Inputs& input) {
    auto* state = static_cast<State*>(raw);
    const size_t count = static_cast<size_t>(state->N) * state->L;
    if (input.a.data.size() != count || input.b.data.size() != count) {
        throw std::runtime_error("a and b must contain N*L limbs");
    }
    const int threads = 128;
    const dim3 coeff_grid((state->N + threads - 1) / threads, state->channels);
    const size_t bytes = count * sizeof(uint32_t);
    std::memcpy(state->h_pin_a, input.a.data.data(), bytes);
    check_cuda(cudaMemcpyAsync(state->d_in, state->h_pin_a, bytes, cudaMemcpyHostToDevice), "a copy");
    launch_residues(
        coeff_grid, threads, state->d_in, state->d_a, state->d_psi, state->d_prime, state->d_nprime, state->d_r2,
        state->d_mu, state->d_pow32, state->d_magic_lo, state->d_magic_hi, state->N, state->L, state->channels);
    std::memcpy(state->h_pin_b, input.b.data.data(), bytes);
    check_cuda(cudaMemcpyAsync(state->d_in, state->h_pin_b, bytes, cudaMemcpyHostToDevice), "b copy");
    launch_residues(
        coeff_grid, threads, state->d_in, state->d_b, state->d_psi, state->d_prime, state->d_nprime, state->d_r2,
        state->d_mu, state->d_pow32, state->d_magic_lo, state->d_magic_hi, state->N, state->L, state->channels);
    forward_ntt(state->d_a, state->d_tw_fwd, state->d_tw_fwd_shoup, state->d_prime, state->N, state->channels);
    forward_ntt(state->d_b, state->d_tw_fwd, state->d_tw_fwd_shoup, state->d_prime, state->N, state->channels);
    pointwise_kernel<<<coeff_grid, threads>>>(
        state->d_a, state->d_b, state->d_prime, state->d_nprime, state->N, state->channels);
    inverse_ntt(state->d_a, state->d_tw_inv, state->d_tw_inv_shoup, state->d_prime, state->N, state->channels);
    untwist_kernel<<<coeff_grid, threads>>>(
        state->d_a, state->d_psi_inv, state->d_ninv, state->d_prime, state->d_nprime, state->N, state->channels);
    fast_crt_kernel<<<(state->N + threads - 1) / threads, threads>>>(
        state->d_a, state->d_out, state->d_prime, state->d_nprime, state->d_r2, state->d_crt_y, state->d_magic_lo,
        state->d_magic_hi, state->d_mi_q, state->d_m_mod_q, state->d_q, state->d_q_mu, state->d_unsafe_count,
        state->d_unsafe_index, state->N, state->channels, state->L);
    check_cuda(cudaGetLastError(), "launch");
    check_cuda(cudaMemcpyAsync(state->h_pin_c, state->d_out, bytes, cudaMemcpyDeviceToHost), "c copy");
    check_cuda(cudaDeviceSynchronize(), "sync");

    fherma::Outputs output;
    output.c.shape = {state->N, state->L};
    output.c.data.resize(count);
    std::memcpy(output.c.data.data(), state->h_pin_c, bytes);
    return output;
}

void fherma_free(void* raw) {
    auto* state = static_cast<State*>(raw);
    if (state == nullptr) return;
    cudaFree(state->d_prime);
    cudaFree(state->d_nprime);
    cudaFree(state->d_r2);
    cudaFree(state->d_mu);
    cudaFree(state->d_minv);
    cudaFree(state->d_ninv);
    cudaFree(state->d_a);
    cudaFree(state->d_b);
    cudaFree(state->d_tw_fwd);
    cudaFree(state->d_tw_inv);
    cudaFree(state->d_tw_fwd_shoup);
    cudaFree(state->d_tw_inv_shoup);
    cudaFreeHost(state->h_pin_a);
    cudaFreeHost(state->h_pin_b);
    cudaFreeHost(state->h_pin_c);
    cudaFree(state->d_psi);
    cudaFree(state->d_psi_inv);
    cudaFree(state->d_in);
    cudaFree(state->d_out);
    cudaFree(state->d_mbefore);
    cudaFree(state->d_mb_q);
    cudaFree(state->d_cross);
    cudaFree(state->d_half_m);
    cudaFree(state->d_m_mod_q);
    cudaFree(state->d_q);
    cudaFree(state->d_crt_y);
    cudaFree(state->d_magic_lo);
    cudaFree(state->d_magic_hi);
    cudaFree(state->d_mi_q);
    cudaFree(state->d_unsafe_count);
    cudaFree(state->d_unsafe_index);
    cudaFree(state->d_pow32);
    cudaFree(state->d_q_mu);
    delete state;
}
