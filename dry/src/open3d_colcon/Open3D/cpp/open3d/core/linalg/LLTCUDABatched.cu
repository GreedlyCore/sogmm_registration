#include "open3d/core/CUDAUtils.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/linalg/LLTBatchedImpl.h"
#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"

namespace open3d {
namespace core {

void LLTBatchedCUDA(void* A_data,
                    int64_t batch_size,
                    int64_t cols,  // NOTE: this is a square matrix
                    Dtype dtype,
                    const Device& device) {
    cusolverDnHandle_t handle = CuSolverContext::GetInstance()->GetHandle();
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t* output_ptr = static_cast<scalar_t*>(A_data);
        scalar_t** A_array = static_cast<scalar_t**>(
                MemoryManager::Malloc(batch_size * sizeof(scalar_t*), device));
        ParallelFor(device, batch_size,
                    [=] OPEN3D_DEVICE(int64_t workload_idx) {
                        A_array[workload_idx] =
                                output_ptr + (workload_idx * cols * cols);
                    });

        int* info_array = static_cast<int*>(
                MemoryManager::Malloc(batch_size * sizeof(int), device));

        cusolverStatus_t potrf_batched_status = potrf_cuda_batched<scalar_t>(
                handle, cols, A_array, cols, info_array, batch_size);

        OPEN3D_CUSOLVER_CHECK_WITH_INFO_ARRAY(
                potrf_batched_status, "potrfBatched failed in LLTCUDABatched",
                info_array, batch_size, device);

        MemoryManager::Free(info_array, device);
        MemoryManager::Free(A_array, device);
    });
}

}  // namespace core
}  // namespace open3d
