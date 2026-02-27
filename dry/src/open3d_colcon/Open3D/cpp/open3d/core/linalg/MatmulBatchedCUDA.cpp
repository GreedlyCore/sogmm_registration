#include "open3d/core/linalg/BlasWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"
#include "open3d/core/linalg/MatmulBatched.h"
#include "open3d/utility/Logging.h"

namespace open3d {
namespace core {

void MatmulBatchedCUDA(void* A_data,
                       void* B_data,
                       void* C_data,
                       int64_t m,
                       int64_t k,
                       int64_t n,
                       Dtype dtype,
                       int batchCount) {
    cublasHandle_t handle = CuBLASContext::GetInstance()->GetHandle();
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t alpha = 1, beta = 0;
        OPEN3D_CUBLAS_CHECK(
                gemm_cuda_strided_batched<scalar_t>(
                        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                        static_cast<const scalar_t*>(A_data), m, m * k,
                        static_cast<const scalar_t*>(B_data), k, k * n, &beta,
                        static_cast<scalar_t*>(C_data), m, m * n, batchCount),
                "cuda gemm strided batched failed");
    });
}

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
                       int batchCount) {
    cublasHandle_t handle = CuBLASContext::GetInstance()->GetHandle();
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        scalar_t alpha = 1, beta = 0;
        OPEN3D_CUBLAS_CHECK(
                gemm_cuda_strided_batched<scalar_t>(
                        handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
                        static_cast<const scalar_t*>(A_data), m, strideA,
                        static_cast<const scalar_t*>(B_data), k, strideB, &beta,
                        static_cast<scalar_t*>(C_data), m, strideC, batchCount),
                "cuda gemm strided batched failed");
    });
}

}  // namespace core
}  // namespace open3d
