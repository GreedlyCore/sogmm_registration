#include "open3d/core/Blob.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/linalg/BlasWrapper.h"
#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"
#include "open3d/core/linalg/SolveLLTBatched.h"

namespace open3d {
namespace core {

void SolveCUDALLTBatched(void* A_data,
                         void* B_data,
                         int64_t batch_size,
                         int64_t cols,
                         Dtype dtype,
                         const Device& device) {
    cusolverDnHandle_t handle = CuSolverContext::GetInstance()->GetHandle();
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        // prepare data
        scalar_t* A_ptr = static_cast<scalar_t*>(A_data);

        scalar_t** A_array = static_cast<scalar_t**>(
                MemoryManager::Malloc(batch_size * sizeof(scalar_t*), device));
        ParallelFor(device, batch_size,
                    [=] OPEN3D_DEVICE(int64_t workload_idx) {
                        A_array[workload_idx] =
                                A_ptr + (workload_idx * cols * cols);
                    });

        // start batched LLT for A_data
        int* info_array = static_cast<int*>(
                MemoryManager::Malloc(batch_size * sizeof(int), device));

        cusolverStatus_t potrf_batched_status = potrf_cuda_batched<scalar_t>(
                handle, cols, A_array, cols, info_array, batch_size);


        OPEN3D_CUSOLVER_CHECK_WITH_INFO_ARRAY(
                potrf_batched_status, "potrfBatched failed in LLTCUDABatched",
                info_array, batch_size, device);

        MemoryManager::Free(info_array, device);
        MemoryManager::Free(A_array, device);
        // end batched LLT for A_data

        Blob dinfo(sizeof(int), device);
        scalar_t* B_ptr = static_cast<scalar_t*>(B_data);

        for (int i = 0; i < batch_size; i++) {
            OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                    potrs_cuda<scalar_t>(
                            handle, cols, cols,
                            A_ptr + (i * cols * cols), cols,
                            B_ptr + (i * cols * cols), cols,
                            static_cast<int*>(dinfo.GetDataPtr())),
                    "potrs failed in SolveCUDALLTBatched",
                    static_cast<int*>(dinfo.GetDataPtr()), device);
        }
    });
}

}  // namespace core
}  // namespace open3d
