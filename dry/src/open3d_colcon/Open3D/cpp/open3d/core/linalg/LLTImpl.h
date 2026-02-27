#pragma once

#include "open3d/core/linalg/LLT.h"

namespace open3d {
namespace core {

void LLTCPU(void* A_data,
           int64_t cols,
           Dtype dtype,
           const Device& device);

#ifdef BUILD_CUDA_MODULE
void LLTCUDA(void* A_data,
            int64_t cols,
            Dtype dtype,
            const Device& device);
#endif
}  // namespace core
}  // namespace open3d
