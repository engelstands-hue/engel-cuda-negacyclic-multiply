// GENERATED from polymul/negacyclic@1.0.0. Do not edit — ``--update`` rewrites it.
//
// The types your answer is written against, derived from the signature: one
// field per value parameter, per argument, per result. A tensor is typed,
// because the signature already settled what its elements are.
#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace fherma {

template <class T>
struct Tensor {
    std::vector<int64_t> shape;
    std::vector<T> data;

    int64_t count() const {
        int64_t total = 1;
        for (auto d : shape) total *= d;
        return total;
    }
};

struct Point {
    uint32_t N = 0;
    uint32_t W = 0;
    uint32_t L = 0;
    Tensor<uint32_t> q;
};

struct Inputs {
    Tensor<uint32_t> a;
    Tensor<uint32_t> b;
};

struct Outputs {
    Tensor<uint32_t> c;
};

}

void* fherma_init(const fherma::Point& p);
fherma::Outputs fherma_run(void* state, const fherma::Inputs& in);
void fherma_free(void* state);
