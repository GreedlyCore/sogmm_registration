#pragma once

#include "open3d/core/Tensor.h"

namespace open3d {
namespace core {

/// Computes matrix multiplication Ci = Ai Bi for each Ai and Bi in
/// 3D tensors A and B. Stores the output in tensor C.
void MatmulBatched(const Tensor& A, const Tensor& B, Tensor& C);

#ifdef BUILD_CUDA_MODULE
void MatmulBatchedCUDA(void* A_data,
                       void* B_data,
                       void* C_data,
                       int64_t m,
                       int64_t k,
                       int64_t n,
                       Dtype dtype,
                       int batchCount);
void MatmulBatchedCUDA(void* A_data,
                       void* B_data,
                       void* C_data,
                       int64_t m,
                       int64_t k,
                       int64_t n,
                       Dtype dtype,
                       int64_t strideA,
                       int64_t strideB,
                       int64_t strideC,
                       int batchCount);
#endif
void MatmulBatchedCPU(void* A_data,
                      void* B_data,
                      void* C_data,
                      int64_t m,
                      int64_t k,
                      int64_t n,
                      Dtype dtype,
                      int batchCount);
}  // namespace core
}  // namespace open3d
