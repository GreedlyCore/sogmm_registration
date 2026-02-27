#include "open3d/core/linalg/LLTImpl.h"
#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"

namespace open3d {
namespace core {

void LLTCUDA(void* A_data,
	    int64_t cols, // NOTE: this is a square matrix
            Dtype dtype,
            const Device& device) {
    cusolverDnHandle_t handle = CuSolverContext::GetInstance()->GetHandle();
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        int len;
        OPEN3D_CUSOLVER_CHECK(
                potrf_cuda_buffersize<scalar_t>(handle, cols, cols, &len),
                "potrf_buffersize failed in LLTCUDA");

        int* dinfo =
                static_cast<int*>(MemoryManager::Malloc(sizeof(int), device));
        void* workspace = MemoryManager::Malloc(len * sizeof(scalar_t), device);

        cusolverStatus_t potrf_status = potrf_cuda<scalar_t>(
                handle, cols, static_cast<scalar_t*>(A_data), cols,
                static_cast<scalar_t*>(workspace), len,
                dinfo);

        MemoryManager::Free(workspace, device);
        MemoryManager::Free(dinfo, device);

        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(potrf_status, "potrf failed in LLTCUDA",
                                         dinfo, device);
    });
}

}  // namespace core
}  // namespace open3d
