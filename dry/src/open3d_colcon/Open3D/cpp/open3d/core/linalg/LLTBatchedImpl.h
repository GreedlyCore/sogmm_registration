#pragma once

#include "open3d/core/linalg/LLTBatched.h"

namespace open3d {
namespace core {

void LLTBatchedCPU(void* A_data,
                   int64_t batch_size,
                   int64_t cols,
                   Dtype dtype,
                   const Device& device);

#ifdef BUILD_CUDA_MODULE
void LLTBatchedCUDA(void* A_data,
                    int64_t batch_size,
                    int64_t cols,
                    Dtype dtype,
                    const Device& device);
#endif
}  // namespace core
}  // namespace open3d
