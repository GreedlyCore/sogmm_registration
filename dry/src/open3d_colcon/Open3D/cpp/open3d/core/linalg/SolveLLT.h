#pragma once

#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {

/// Solve AX = B with LLT decomposition. A is a square matrix.
void SolveLLT(const Tensor& A, const Tensor& B, Tensor& X);

void SolveCPULLT(void* A_data,
              void* B_data,
              int64_t n,
              int64_t k,
              Dtype dtype,
              const Device& device);

#ifdef BUILD_CUDA_MODULE
void SolveCUDALLT(void* A_data,
               void* B_data,
               int64_t n,
               int64_t k,
               Dtype dtype,
               const Device& device);
#endif

}  // namespace core
}  // namespace open3d
