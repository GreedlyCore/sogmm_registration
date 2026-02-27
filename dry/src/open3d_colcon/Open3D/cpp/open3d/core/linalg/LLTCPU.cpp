#include "open3d/core/linalg/LLTImpl.h"
#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"

namespace open3d {
namespace core {

void LLTCPU(void* A_data,
           int64_t cols,
           Dtype dtype,
           const Device& device) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {
        OPEN3D_LAPACK_CHECK(
    	      potrf_cpu<scalar_t>(
                        LAPACK_COL_MAJOR, cols,
                        static_cast<scalar_t*>(A_data), cols),
                "potrf failed in LLTCPU");
    });
}

}  // namespace core
}  // namespace open3d
