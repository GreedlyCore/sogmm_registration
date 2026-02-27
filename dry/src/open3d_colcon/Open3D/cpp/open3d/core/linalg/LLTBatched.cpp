#include "open3d/core/linalg/LLTBatched.h"

#include "open3d/core/linalg/LLTBatchedImpl.h"
#include "open3d/core/linalg/LinalgHeadersCPU.h"
#include "open3d/core/linalg/Tri.h"

namespace open3d {
namespace core {

// Return L matrix
void LLTBatched(const Tensor& A, Tensor& L) {
    AssertTensorDtypes(A, {Float32, Float64});

    const Device device = A.GetDevice();
    const Dtype dtype = A.GetDtype();

    // Check dimensions.
    const SizeVector A_shape = A.GetShape();
    if (A_shape.size() != 3) {
        utility::LogError("Tensor must be 3D, but got {}D.", A_shape.size());
    }

    const int64_t batch_size = A_shape[0];
    const int64_t rows = A_shape[1];
    const int64_t cols = A_shape[2];
    if (batch_size == 0 || rows == 0 || cols == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }

    // "output" tensor is modified in-place as output.
    // Operations are COL_MAJOR.
    Tensor output = A.Transpose(1, 2).Clone();
    void* A_data = output.GetDataPtr();

    // Returns LLT decomposition in form of an output matrix,
    // with lower triangular elements as L.
    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        LLTBatchedCUDA(A_data, batch_size, cols, dtype, device);
#else
        utility::LogInfo("Unimplemented device.");
#endif
    } else {
        LLTBatchedCPU(A_data, batch_size, cols, dtype, device);
    }

    // COL_MAJOR -> ROW_MAJOR.
    output = output.Transpose(1, 2).Contiguous();
    Tril3D(output, L, 0);
}

}  // namespace core
}  // namespace open3d
