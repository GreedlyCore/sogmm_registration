#include "open3d/core/Blob.h"
#include "open3d/core/CUDAUtils.h"
#include "open3d/core/linalg/BlasWrapper.h"
#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"
#include "open3d/core/linalg/SolveLLT.h"

namespace open3d {
namespace core {

// cuSolver's gesv will crash when A is a singular matrix.
// We implement LU decomposition-based solver (similar to Inverse) instead.
void SolveCUDALLT(void* A_data,
               void* B_data,
               int64_t n,
               int64_t k,
               Dtype dtype,
               const Device& device) {
    cusolverDnHandle_t handle = CuSolverContext::GetInstance()->GetHandle();

    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        int len;
        Blob dinfo(sizeof(int), device);

        OPEN3D_CUSOLVER_CHECK(
                potrf_cuda_buffersize<scalar_t>(handle, n, n, &len),
                "potrf_buffersize failed in SolveCUDALLT");
        Blob workspace(len * sizeof(scalar_t), device);

        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                potrf_cuda<scalar_t>(
                        handle, n, static_cast<scalar_t*>(A_data), n,
                        static_cast<scalar_t*>(workspace.GetDataPtr()),
                        len,
                        static_cast<int*>(dinfo.GetDataPtr())),
                "potrf failed in SolveCUDALLT",
                static_cast<int*>(dinfo.GetDataPtr()), device);

        OPEN3D_CUSOLVER_CHECK_WITH_DINFO(
                potrs_cuda<scalar_t>(handle, n, k,
                                     static_cast<scalar_t*>(A_data), n,
                                     static_cast<scalar_t*>(B_data), n,
                                     static_cast<int*>(dinfo.GetDataPtr())),
                "potrs failed in SolveCUDALLT",
                static_cast<int*>(dinfo.GetDataPtr()), device);
    });
}

}  // namespace core
}  // namespace open3d
