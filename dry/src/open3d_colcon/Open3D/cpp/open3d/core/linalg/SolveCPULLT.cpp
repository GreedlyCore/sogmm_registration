#include "open3d/core/linalg/LapackWrapper.h"
#include "open3d/core/linalg/LinalgUtils.h"
#include "open3d/core/linalg/SolveLLT.h"

namespace open3d {
namespace core {

void SolveCPULLT(void* A_data,
              void* B_data,
              int64_t n,
              int64_t k,
              Dtype dtype,
              const Device& device) {
    DISPATCH_LINALG_DTYPE_TO_TEMPLATE(dtype, [&]() {

        OPEN3D_LAPACK_CHECK(
	  potrf_cpu<scalar_t>(
	    LAPACK_COL_MAJOR, n,
	    static_cast<scalar_t*>(A_data), n),
	  "potrf failed in LLTCPU called by SolveCPULLT");

        OPEN3D_LAPACK_CHECK(
	  potrs_cpu<scalar_t>(
	    LAPACK_COL_MAJOR, n, k, static_cast<scalar_t*>(A_data),
	    n, static_cast<scalar_t*>(B_data), n),
	  "potrs failed in SolveCPULLT");

    });
}

}  // namespace core
}  // namespace open3d
