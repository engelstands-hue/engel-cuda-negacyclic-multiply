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

#include <stdexcept>
#include <string>
#include <vector>

namespace {

constexpr int kWide = 80;
constexpr int kMaxChannels = 48;
constexpr int kMaxN = 32768;

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
    uint64_t* data, const uint64_t* twiddles, const uint64_t* primes, const uint64_t* nprime,
    int n, int half, int tw_base, int channels) {
    const int butterfly = blockIdx.x * blockDim.x + threadIdx.x;
    const int channel = blockIdx.y;
    if (butterfly >= n / 2 || channel >= channels) return;
    const int lane = butterfly % half;
    const int start = (butterfly / half) * (half * 2);
    const uint64_t prime = primes[channel];
    uint64_t* row = data + static_cast<size_t>(channel) * n;
    const uint64_t twiddle = twiddles[static_cast<size_t>(channel) * n + tw_base + lane];
    const uint64_t left = row[start + lane];
    const uint64_t right = mont_mul(row[start + lane + half], twiddle, prime, nprime[channel]);
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

__global__ void crt_kernel(
    const uint64_t* residues, uint32_t* output, const uint64_t* primes, const uint64_t* nprime,
    const uint64_t* r2, const uint64_t* minv, const uint32_t* m_before, const uint32_t* mb_q,
    const uint64_t* cross, const uint32_t* half_m, const uint32_t* m_mod_q, const uint32_t* q,
    int n, int channels, int limbs) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
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

void forward_ntt(
    uint64_t* data, const uint64_t* twiddles, const uint64_t* primes, const uint64_t* nprime,
    int n, int channels) {
    const int threads = 128;
    int bits = 0;
    for (int value = n; value > 1; value >>= 1) ++bits;
    bitrev_kernel<<<dim3((n + threads - 1) / threads, channels), threads>>>(data, n, bits, channels);
    for (int half = 1; half < n; half <<= 1) {
        const int butterflies = n / 2;
        stage_kernel<<<dim3((butterflies + threads - 1) / threads, channels), threads>>>(
            data, twiddles, primes, nprime, n, half, half - 1, channels);
    }
}

void inverse_ntt(
    uint64_t* data, const uint64_t* twiddles, const uint64_t* primes, const uint64_t* nprime,
    int n, int channels) {
    forward_ntt(data, twiddles, primes, nprime, n, channels);
}

}  // namespace

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
    std::vector<uint64_t> psi(static_cast<size_t>(channels) * n);
    std::vector<uint64_t> psi_inv(static_cast<size_t>(channels) * n);
    std::vector<uint64_t> ninv(channels);
    std::vector<uint64_t> nprime_table(channels);
    std::vector<uint64_t> r2_table(channels);
    std::vector<uint64_t> mu_table(channels);
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
                tw_fwd[static_cast<size_t>(channel) * n + (half - 1) + lane] = twiddle;
                tw_inv[static_cast<size_t>(channel) * n + (half - 1) + lane] = twiddle_inv;
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
        for (int index = 0; index < n; ++index) {
            const size_t at = static_cast<size_t>(channel) * n + index;
            psi[at] = mont_mul(psi[at], lift, prime, np);
            psi_inv[at] = mont_mul(psi_inv[at], lift, prime, np);
            tw_fwd[at] = mont_mul(tw_fwd[at], lift, prime, np);
            tw_inv[at] = mont_mul(tw_inv[at], lift, prime, np);
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

    auto* state = new State();
    state->N = n;
    state->L = static_cast<int>(point.L);
    state->channels = channels;
    const size_t row = static_cast<size_t>(channels) * n;
    try {
        check_cuda(cudaDeviceSetLimit(cudaLimitStackSize, 16384), "stack");
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
        check_cuda(cudaMemcpy(state->d_prime, primes.data(), primes.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "primes");
        check_cuda(cudaMemcpy(state->d_nprime, nprime_table.data(), nprime_table.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "nprime");
        check_cuda(cudaMemcpy(state->d_r2, r2_table.data(), r2_table.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "r2");
        check_cuda(cudaMemcpy(state->d_mu, mu_table.data(), mu_table.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "barrett");
        check_cuda(cudaMemcpy(state->d_minv, minv.data(), minv.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "minv");
        check_cuda(cudaMemcpy(state->d_ninv, ninv.data(), ninv.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "ninv");
        check_cuda(cudaMemcpy(state->d_tw_fwd, tw_fwd.data(), row * sizeof(uint64_t), cudaMemcpyHostToDevice), "twiddles");
        check_cuda(cudaMemcpy(state->d_tw_inv, tw_inv.data(), row * sizeof(uint64_t), cudaMemcpyHostToDevice), "inverse twiddles");
        check_cuda(cudaMemcpy(state->d_psi, psi.data(), row * sizeof(uint64_t), cudaMemcpyHostToDevice), "psi");
        check_cuda(cudaMemcpy(state->d_psi_inv, psi_inv.data(), row * sizeof(uint64_t), cudaMemcpyHostToDevice), "psi inverse");
        check_cuda(cudaMemcpy(state->d_mbefore, m_before.data(), m_before.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "garner");
        check_cuda(cudaMemcpy(state->d_mb_q, mb_q.data(), mb_q.size() * sizeof(uint32_t), cudaMemcpyHostToDevice), "garner mod q");
        check_cuda(cudaMemcpy(state->d_cross, cross.data(), cross.size() * sizeof(uint64_t), cudaMemcpyHostToDevice), "cross residues");
        check_cuda(cudaMemcpy(state->d_half_m, half_m, kWide * sizeof(uint32_t), cudaMemcpyHostToDevice), "half modulus");
        check_cuda(cudaMemcpy(state->d_m_mod_q, m_mod_q, kWide * sizeof(uint32_t), cudaMemcpyHostToDevice), "modulus residue");
        check_cuda(cudaMemcpy(state->d_q, q_limbs, kWide * sizeof(uint32_t), cudaMemcpyHostToDevice), "q");
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
    check_cuda(cudaMemcpy(state->d_in, input.a.data.data(), bytes, cudaMemcpyHostToDevice), "a copy");
    to_residues_kernel<<<coeff_grid, threads>>>(
        state->d_in, state->d_a, state->d_psi, state->d_prime, state->d_nprime, state->d_r2, state->d_mu,
        state->N, state->L, state->channels);
    check_cuda(cudaMemcpy(state->d_in, input.b.data.data(), bytes, cudaMemcpyHostToDevice), "b copy");
    to_residues_kernel<<<coeff_grid, threads>>>(
        state->d_in, state->d_b, state->d_psi, state->d_prime, state->d_nprime, state->d_r2, state->d_mu,
        state->N, state->L, state->channels);
    forward_ntt(state->d_a, state->d_tw_fwd, state->d_prime, state->d_nprime, state->N, state->channels);
    forward_ntt(state->d_b, state->d_tw_fwd, state->d_prime, state->d_nprime, state->N, state->channels);
    pointwise_kernel<<<coeff_grid, threads>>>(
        state->d_a, state->d_b, state->d_prime, state->d_nprime, state->N, state->channels);
    inverse_ntt(state->d_a, state->d_tw_inv, state->d_prime, state->d_nprime, state->N, state->channels);
    untwist_kernel<<<coeff_grid, threads>>>(
        state->d_a, state->d_psi_inv, state->d_ninv, state->d_prime, state->d_nprime, state->N, state->channels);
    crt_kernel<<<(state->N + threads - 1) / threads, threads>>>(
        state->d_a, state->d_out, state->d_prime, state->d_nprime, state->d_r2, state->d_minv, state->d_mbefore,
        state->d_mb_q, state->d_cross, state->d_half_m, state->d_m_mod_q, state->d_q, state->N, state->channels,
        state->L);
    check_cuda(cudaGetLastError(), "launch");
    check_cuda(cudaDeviceSynchronize(), "sync");

    fherma::Outputs output;
    output.c.shape = {state->N, state->L};
    output.c.data.resize(count);
    check_cuda(cudaMemcpy(output.c.data.data(), state->d_out, bytes, cudaMemcpyDeviceToHost), "c copy");
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
    delete state;
}
