#include "open3d/core/CUDAUtils.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"
#include "open3d/core/linalg/SolveLLTBatched.h"

namespace open3d {
namespace core {

void SolveCPULLTBatched(void* A_data,
                        void* B_data,
                        int64_t batch_size,
                        int64_t cols,
                        Dtype dtype,
                        const Device& device) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        ParallelFor(
                device, batch_size, [&] OPEN3D_DEVICE(int64_t workload_idx) {
                    scalar_t* input_ptr = static_cast<scalar_t*>(A_data) +
                                          (workload_idx * cols * cols);

                    scalar_t* output_ptr = static_cast<scalar_t*>(B_data) +
                                           (workload_idx * cols * cols);

                    OPEN3D_LAPACK_CHECK(
                            potrf_cpu<scalar_t>(LAPACK_COL_MAJOR, cols,
                                                static_cast<scalar_t*>(A_data),
                                                cols),
                            "potrf failed in LLTCPU called by SolveCPULLT");

                    OPEN3D_LAPACK_CHECK(
                            potrs_cpu<scalar_t>(LAPACK_COL_MAJOR, cols, cols,
                                                input_ptr, cols, output_ptr,
                                                cols),
                            "potrs failed in SolveCPULLTBatched");
                });
    });
}

}  // namespace core
}  // namespace open3d
