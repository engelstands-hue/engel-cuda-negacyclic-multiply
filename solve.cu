// Negacyclic product c = a*b in Z_q[X]/(X^N + 1).
//
// Coefficients are L little-endian u32 limbs, the layout in fherma.h.
// One thread owns one output coefficient and keeps the running total in
// registers. Setup (the Montgomery constant for q) happens in fherma_init,
// which the harness does not time.
#include "fherma.h"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace {

constexpr int kMaxLimbs = 32;

struct State {
    int N = 0;
    int L = 0;
    unsigned int n0 = 0;
    unsigned int q[kMaxLimbs]{};
    unsigned int r2[kMaxLimbs]{};
    unsigned int* d_q = nullptr;
    unsigned int* d_r2 = nullptr;
    unsigned int* d_a = nullptr;
    unsigned int* d_b = nullptr;
    unsigned int* d_c = nullptr;
};

__host__ __device__ unsigned int inv_mod_2_32(unsigned int odd) {
    unsigned int inv = odd;
    inv *= 2u - odd * inv;
    inv *= 2u - odd * inv;
    inv *= 2u - odd * inv;
    inv *= 2u - odd * inv;
    inv *= 2u - odd * inv;
    return inv;
}

__host__ __device__ void limbs_clear(unsigned int* out, int n) {
    for (int i = 0; i < n; ++i) out[i] = 0u;
}

__host__ __device__ bool limbs_geq(const unsigned int* a, const unsigned int* b, int n) {
    for (int i = n - 1; i >= 0; --i) {
        if (a[i] > b[i]) return true;
        if (a[i] < b[i]) return false;
    }
    return true;
}

__host__ __device__ void reduce_extra(unsigned int* out, unsigned int* top, const unsigned int* m, int n) {
    for (int pass = 0; pass < 2; ++pass) {
        if (*top == 0u && !limbs_geq(out, m, n)) return;
        unsigned long long borrow = 0ull;
        for (int i = 0; i < n; ++i) {
            unsigned long long cur = (unsigned long long)out[i] - m[i] - borrow;
            out[i] = (unsigned int)cur;
            borrow = (cur >> 63) & 1ull;
        }
        *top -= (unsigned int)borrow;
    }
}

__host__ __device__ void limbs_add_mod(const unsigned int* a, const unsigned int* b, const unsigned int* m, int n, unsigned int* out) {
    unsigned long long carry = 0ull;
    for (int i = 0; i < n; ++i) {
        unsigned long long cur = (unsigned long long)a[i] + b[i] + carry;
        out[i] = (unsigned int)cur;
        carry = cur >> 32;
    }
    unsigned int top = (unsigned int)carry;
    reduce_extra(out, &top, m, n);
}

__host__ __device__ void limbs_sub_mod(const unsigned int* a, const unsigned int* b, const unsigned int* m, int n, unsigned int* out) {
    unsigned long long borrow = 0ull;
    for (int i = 0; i < n; ++i) {
        unsigned long long cur = (unsigned long long)a[i] - b[i] - borrow;
        out[i] = (unsigned int)cur;
        borrow = (cur >> 63) & 1ull;
    }
    if (borrow) {
        unsigned long long carry = 0ull;
        for (int i = 0; i < n; ++i) {
            unsigned long long cur = (unsigned long long)out[i] + m[i] + carry;
            out[i] = (unsigned int)cur;
            carry = cur >> 32;
        }
    }
}

__host__ __device__ void mul_wide(const unsigned int* a, const unsigned int* b, int n, unsigned int* out) {
    for (int i = 0; i < 2 * n; ++i) out[i] = 0u;
    for (int i = 0; i < n; ++i) {
        unsigned long long carry = 0ull;
        for (int j = 0; j < n; ++j) {
            unsigned long long cur = (unsigned long long)out[i + j] + (unsigned long long)a[i] * b[j] + carry;
            out[i + j] = (unsigned int)cur;
            carry = cur >> 32;
        }
        out[i + n] = (unsigned int)carry;
    }
}

__host__ __device__ void mont_redc(unsigned int* t, const unsigned int* m, int n, unsigned int n0, unsigned int* out) {
    t[2 * n] = 0u;
    for (int i = 0; i < n; ++i) {
        unsigned int k = t[i] * n0;
        unsigned long long carry = 0ull;
        for (int j = 0; j < n; ++j) {
            unsigned long long cur = (unsigned long long)t[i + j] + (unsigned long long)k * m[j] + carry;
            t[i + j] = (unsigned int)cur;
            carry = cur >> 32;
        }
        for (int j = i + n; j <= 2 * n && carry; ++j) {
            unsigned long long cur = (unsigned long long)t[j] + carry;
            t[j] = (unsigned int)cur;
            carry = cur >> 32;
        }
    }
    unsigned int top = t[2 * n];
    for (int i = 0; i < n; ++i) out[i] = t[n + i];
    reduce_extra(out, &top, m, n);
}

__host__ __device__ void mont_mul(const unsigned int* a, const unsigned int* b, const unsigned int* m, int n, unsigned int n0, unsigned int* out) {
    unsigned int wide[kMaxLimbs * 2 + 1];
    mul_wide(a, b, n, wide);
    mont_redc(wide, m, n, n0, out);
}

void compute_r2(const unsigned int* q, int n, unsigned int* r2) {
    limbs_clear(r2, n);
    r2[0] = 1u;
    unsigned int doubled[kMaxLimbs];
    const int steps = 64 * n;
    for (int step = 0; step < steps; ++step) {
        limbs_add_mod(r2, r2, q, n, doubled);
        for (int i = 0; i < n; ++i) r2[i] = doubled[i];
    }
}

void require_known_product() {
    const unsigned int q[1] = {11u};
    const unsigned int n0 = 0u - inv_mod_2_32(q[0]);
    unsigned int r2[1];
    compute_r2(q, 1, r2);
    const unsigned int seven[1] = {7u};
    const unsigned int five[1] = {5u};
    const unsigned int one[1] = {1u};
    unsigned int a[1], b[1], mid[1], plain[1];
    mont_mul(seven, r2, q, 1, n0, a);
    mont_mul(five, r2, q, 1, n0, b);
    mont_mul(a, b, q, 1, n0, mid);
    mont_mul(mid, one, q, 1, n0, plain);
    if (plain[0] != 2u) {
        throw std::runtime_error("montgomery self-check failed");
    }
}

__global__ void to_mont_kernel(unsigned int* data, const unsigned int* r2, const unsigned int* q, int count, int limbs, unsigned int n0) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    unsigned int converted[kMaxLimbs];
    mont_mul(data + index * limbs, r2, q, limbs, n0, converted);
    for (int i = 0; i < limbs; ++i) data[index * limbs + i] = converted[i];
}

__global__ void negacyclic_kernel(const unsigned int* a, const unsigned int* b, unsigned int* c, const unsigned int* q, int n_coeffs, int limbs, unsigned int n0) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= n_coeffs) return;
    unsigned int acc[kMaxLimbs];
    unsigned int prod[kMaxLimbs];
    unsigned int one[kMaxLimbs];
    limbs_clear(acc, limbs);
    limbs_clear(one, limbs);
    one[0] = 1u;
    for (int i = 0; i < n_coeffs; ++i) {
        int j;
        bool negative;
        if (i <= k) { j = k - i; negative = false; }
        else { j = k + n_coeffs - i; negative = true; }
        mont_mul(a + i * limbs, b + j * limbs, q, limbs, n0, prod);
        if (negative) limbs_sub_mod(acc, prod, q, limbs, acc);
        else limbs_add_mod(acc, prod, q, limbs, acc);
    }
    unsigned int plain[kMaxLimbs];
    mont_mul(acc, one, q, limbs, n0, plain);
    for (int i = 0; i < limbs; ++i) c[k * limbs + i] = plain[i];
}

void check_cuda(cudaError_t err, const char* what) {
    if (err != cudaSuccess) {
        throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(err));
    }
}

}  // namespace

void* fherma_init(const fherma::Point& p) {
    require_known_product();
    if (p.N == 0 || (p.N & (p.N - 1u)) != 0u) {
        throw std::runtime_error("N must be a power of two");
    }
    if (p.L == 0 || p.L > kMaxLimbs) {
        throw std::runtime_error("L is outside 1..32");
    }
    if (p.q.data.size() != p.L || (p.q.data[0] & 1u) == 0u) {
        throw std::runtime_error("q must be an odd modulus of L limbs");
    }
    auto* state = new State();
    state->N = static_cast<int>(p.N);
    state->L = static_cast<int>(p.L);
    for (unsigned int i = 0; i < p.L; ++i) state->q[i] = p.q.data[i];
    state->n0 = 0u - inv_mod_2_32(state->q[0]);
    if (state->n0 * state->q[0] != 0xffffffffu) {
        delete state;
        throw std::runtime_error("q is not invertible mod 2^32");
    }
    compute_r2(state->q, state->L, state->r2);
    const size_t coeff_bytes = static_cast<size_t>(state->N) * state->L * sizeof(unsigned int);
    check_cuda(cudaDeviceSetLimit(cudaLimitStackSize, 8192), "stack");
    check_cuda(cudaMalloc(&state->d_q, state->L * sizeof(unsigned int)), "q");
    check_cuda(cudaMalloc(&state->d_r2, state->L * sizeof(unsigned int)), "r2");
    check_cuda(cudaMalloc(&state->d_a, coeff_bytes), "a");
    check_cuda(cudaMalloc(&state->d_b, coeff_bytes), "b");
    check_cuda(cudaMalloc(&state->d_c, coeff_bytes), "c");
    check_cuda(cudaMemcpy(state->d_q, state->q, state->L * sizeof(unsigned int), cudaMemcpyHostToDevice), "q copy");
    check_cuda(cudaMemcpy(state->d_r2, state->r2, state->L * sizeof(unsigned int), cudaMemcpyHostToDevice), "r2 copy");
    return state;
}

fherma::Outputs fherma_run(void* raw, const fherma::Inputs& in) {
    auto* state = static_cast<State*>(raw);
    const size_t count = static_cast<size_t>(state->N) * state->L;
    if (in.a.data.size() != count || in.b.data.size() != count) {
        throw std::runtime_error("a and b must contain N*L limbs");
    }
    const size_t bytes = count * sizeof(unsigned int);
    check_cuda(cudaMemcpy(state->d_a, in.a.data.data(), bytes, cudaMemcpyHostToDevice), "a copy");
    check_cuda(cudaMemcpy(state->d_b, in.b.data.data(), bytes, cudaMemcpyHostToDevice), "b copy");
    const int threads = 128;
    const int coeff_blocks = (state->N + threads - 1) / threads;
    to_mont_kernel<<<coeff_blocks, threads>>>(state->d_a, state->d_r2, state->d_q, state->N, state->L, state->n0);
    to_mont_kernel<<<coeff_blocks, threads>>>(state->d_b, state->d_r2, state->d_q, state->N, state->L, state->n0);
    negacyclic_kernel<<<coeff_blocks, threads>>>(state->d_a, state->d_b, state->d_c, state->d_q, state->N, state->L, state->n0);
    check_cuda(cudaGetLastError(), "launch");
    fherma::Outputs out;
    out.c.shape = {state->N, state->L};
    out.c.data.resize(count);
    check_cuda(cudaMemcpy(out.c.data.data(), state->d_c, bytes, cudaMemcpyDeviceToHost), "c copy");
    check_cuda(cudaDeviceSynchronize(), "sync");
    return out;
}

void fherma_free(void* raw) {
    auto* state = static_cast<State*>(raw);
    if (state == nullptr) return;
    cudaFree(state->d_q);
    cudaFree(state->d_r2);
    cudaFree(state->d_a);
    cudaFree(state->d_b);
    cudaFree(state->d_c);
    delete state;
}
