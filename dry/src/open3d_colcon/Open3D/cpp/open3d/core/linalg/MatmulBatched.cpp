#include "open3d/core/linalg/MatmulBatched.h"

#include <unordered_map>

namespace open3d {
namespace core {

void MatmulBatched(const Tensor& A, const Tensor& B, Tensor& output) {
    AssertTensorDevice(B, A.GetDevice());
    AssertTensorDtype(B, A.GetDtype());

    const Device device = A.GetDevice();
    const Dtype dtype_original = A.GetDtype();
    Dtype dtype;

    if (dtype_original != core::Float32 && dtype_original != core::Float64) {
        utility::LogDebug("Converting to Float32 dtype to from {}.",
                          dtype_original.ToString());
        dtype = core::Float32;
    } else {
        dtype = dtype_original;
    }

    // Check shapes
    SizeVector A_shape = A.GetShape();
    SizeVector B_shape = B.GetShape();

    if (A_shape.size() != 3) {
        utility::LogError("Tensor A must be 3D, but got {}D.", A_shape.size());
    }
    if (B_shape.size() != 3) {
        utility::LogError("Tensor B must be 3D, but got {}D.", B_shape.size());
    }
    if (A_shape[2] != B_shape[1]) {
        utility::LogError("Tensor A columns {} mismatch with Tensor B rows {}.",
                          A_shape[2], B_shape[1]);
    }

    // Dispatch to backends
    int batch_size = std::max(A_shape[0], B_shape[0]);

    int64_t m = A_shape[1];
    int64_t k = A_shape[2];
    int64_t n = B_shape[2];

    if (m == 0 || k == 0 || n == 0) {
        utility::LogError(
                "Tensor shapes should not contain dimensions with zero.");
    }

    Tensor A_contiguous = A.Contiguous().To(dtype);
    Tensor B_contiguous = B.Contiguous().To(dtype);
#ifdef BUILD_CUDA_MODULE
    void* A_data = A_contiguous.GetDataPtr();
    void* B_data = B_contiguous.GetDataPtr();
#endif

    output = Tensor::Empty({batch_size, m, n}, dtype, device);
#ifdef BUILD_CUDA_MODULE
    void* C_data = output.GetDataPtr();
#endif

    if (device.IsCUDA()) {
#ifdef BUILD_CUDA_MODULE
        if (A_shape[0] == B_shape[0]) {
            MatmulBatchedCUDA(B_data, A_data, C_data, n, k, m, dtype,
                              batch_size);
        } else {
            if (A_shape[0] == 1) {
                MatmulBatchedCUDA(B_data, A_data, C_data, n, k, m, dtype, n * k,
                                  0, m * n, batch_size);
            } else if (B_shape[0] == 1) {
                MatmulBatchedCUDA(B_data, A_data, C_data, n, k, m, dtype, 0,
                                  k * m, m * n, batch_size);
            } else {
                utility::LogError("Unimplemented MatmulBatched call.");
            }
        }
#else
        utility::LogError("Unimplemented device.");
#endif
    } else {
        // MatmulBatchedCPU(B_data, A_data, C_data, n, k, m, dtype);
        utility::LogError("Unimplemented device.");
    }

    output = output.To(dtype_original);
};

}  // namespace core
}  // namespace open3d
