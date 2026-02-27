#include "open3d/core/linalg/LLT.h"

#include "open3d/core/linalg/LLTImpl.h"
#include "open3d/core/linalg/LinalgHeadersCPU.h"
#include "open3d/core/linalg/Tri.h"

namespace open3d {
namespace core {

// Return L matrix
void LLT(const Tensor& A, Tensor& L) {
    AssertTensorDtypes(A, {Float32, Float64});

    const Device device = A.GetDevice();
    const Dtype dtype = A.GetDtype();

    // Check dimensions.
    const SizeVector A_shape = A.GetShape();
    if (A_shape.size() != 2) {
        utility::LogError("Tensor must be 2D, but got {}D.", A_shape.size());
    }

    const int64_t rows = A_shape[0];
    const int64_t cols = A_shape[1];
    if (rows == 0 || cols == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }

    // "output" tensor is modified in-place as output.
    // Operations are COL_MAJOR.
    Tensor output = A.T().Clone();
    void* A_data = output.GetDataPtr();

    // Returns LLT decomposition in form of an output matrix,
    // with lower triangular elements as L.
    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        LLTCUDA(A_data, cols, dtype, device);
#else
        utility::LogInfo("Unimplemented device.");
#endif
    }
    else {
      LLTCPU(A_data, cols, dtype, device);
    }

    // COL_MAJOR -> ROW_MAJOR.
    output = output.T().Contiguous();
    Tril(output, L, 0);
}

}  // namespace core
}  // namespace open3d
