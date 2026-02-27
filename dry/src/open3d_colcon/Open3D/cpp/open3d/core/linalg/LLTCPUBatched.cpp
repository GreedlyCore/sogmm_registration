#include "open3d/core/CUDAUtils.h"
#include "open3d/core/ParallelFor.h"
#include "open3d/core/linalg/LLTBatchedImpl.h"
#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"

namespace open3d {
namespace core {

void LLTBatchedCPU(void* A_data,
                   int64_t batch_size,
                   int64_t cols,
                   Dtype dtype,
                   const Device& device) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        ParallelFor(device, batch_size,
                    [&] OPEN3D_DEVICE(int64_t workload_idx) {
                        scalar_t* output_ptr = static_cast<scalar_t*>(A_data) +
                                               (workload_idx * cols * cols);
                        OPEN3D_LAPACK_CHECK(
                                potrf_cpu<scalar_t>(LAPACK_COL_MAJOR, cols,
                                                    output_ptr, cols),
                                "potrf failed in LLTCPU");
                    });
    });
}

}  // namespace core
}  // namespace open3d
