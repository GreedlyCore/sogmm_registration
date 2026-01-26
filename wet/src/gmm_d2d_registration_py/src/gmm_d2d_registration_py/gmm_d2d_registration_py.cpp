#include <Eigen/Dense>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/ndarray.h>

#include <gmm_d2d_registration/GMMD2DRegistration.h>
#include <gmm/GMM3.h>
#include <gmm/SKLearn.h>

namespace nb = nanobind;
using namespace nb::literals;
namespace fs = boost::filesystem;

std::pair<Eigen::Matrix<float, 4, 4>, float> anisotropic_registration(const Eigen::Matrix<float, 4, 4>Tin,
								      const std::string& source_file,
								      const std::string& target_file)
{
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tout;
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tinit =
    Eigen::Translation<float, 3>(Tin.block<3,1>(0,3)) * Tin.block<3,3>(0,0);

  gmm_utils::GMM3f source_gmm;
  source_gmm.load(source_file);

  gmm_utils::GMM3f target_gmm;
  target_gmm.load(target_file);

  MatcherD2D matcher;
  float score = matcher.match(source_gmm, target_gmm, Tinit, Tout);

  Eigen::Matrix<float, 4, 4> T = Tout.matrix();
  return std::pair<Eigen::Matrix<float, 4, 4>, float> (T, score);
}

std::pair<Eigen::Matrix<float, 4, 4>, float> isoplanar_registration(const Eigen::Matrix<float, 4, 4>Tin,
								    const std::string& source_file,
								    const std::string& target_file)
{
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tout;
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tinit =
    Eigen::Translation<float, 3>(Tin.block<3,1>(0,3)) * Tin.block<3,3>(0,0);

  gmm_utils::GMM3f source_gmm;
  source_gmm.load(source_file);
  source_gmm.makeCovsIsoplanar();

  gmm_utils::GMM3f target_gmm;
  target_gmm.load(target_file);
  target_gmm.makeCovsIsoplanar();

  MatcherD2D matcher;
  float score = matcher.match(source_gmm, target_gmm, Tinit, Tout);

  Eigen::Matrix<float, 4, 4> T = Tout.matrix();
  return std::pair<Eigen::Matrix<float, 4, 4>, float> (T, score);
}

NB_MODULE(gmm_d2d_registration_py, m) {
  m.def("anisotropic_registration", &anisotropic_registration);
  m.def("isoplanar_registration", &isoplanar_registration);



  // GMM3f class with sklearn-compatible API
  nb::class_<gmm_utils::GMM3f>(m, "GMM3f")
    .def(nb::init<>(), "Create a new 3D Gaussian Mixture Model")
    .def("fit", [](gmm_utils::GMM3f& self,
                   nb::ndarray<float, nb::ndim<2>, nb::c_contig> X,
                   int n_components) {
        // X is ndarray (N, 3) - direct map to Eigen matrix
        size_t N = X.shape(0);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>
            X_eigen(X.data(), N, 3);

        // Transpose to (3, N) for C++ GMM
        Eigen::MatrixXf X_transposed = X_eigen.transpose();

        // Fit using SKLearn algorithm
        self.template fit<SKLearn<float, 3, -1, -1>>(X_transposed, n_components);
    }, "X"_a, "n_components"_a,
       "Fit GMM to data. X should be (N, 3) array, n_components is number of Gaussian components")
    .def_prop_ro("weights_", [](const gmm_utils::GMM3f& self) {
        // Return as 1D array (K,)
        Eigen::VectorXf weights = self.getWeights();
        return weights;
    }, "Mixture weights (K,)")
    .def_prop_ro("means_", [](const gmm_utils::GMM3f& self) {
        // Return as (K, 3) array - transpose from (3, K)
        Eigen::MatrixXf means = self.getMeans().transpose();
        return means;
    }, "Mixture means (K, 3)")
    .def_prop_ro("covariances_", [](const gmm_utils::GMM3f& self) {
        // Return as (K, 3, 3) ndarray
        // Convert from (9, K) to (K, 3, 3)
        auto covs = self.getCovs();
        size_t K = covs.cols();

        // Allocate ndarray with capsule owner for proper memory management
        float* data = new float[K * 9];

        for (size_t k = 0; k < K; k++) {
            Eigen::Map<Eigen::Matrix<float,3,3>> cov(const_cast<float*>(covs.col(k).data()));
            for (int i = 0; i < 3; i++) {
                for (int j = 0; j < 3; j++) {
                    data[k * 9 + i * 3 + j] = cov(i, j);
                }
            }
        }

        // Create capsule that will delete the data when array is garbage collected
        nb::capsule owner(data, [](void* p) noexcept { delete[] static_cast<float*>(p); });

        size_t shape[3] = {K, 3, 3};
        return nb::ndarray<nb::numpy, float, nb::ndim<3>>(data, 3, shape, owner);
    }, "Mixture covariances (K, 3, 3)")
    .def_prop_ro("n_components", [](const gmm_utils::GMM3f& self) {
        return (int)self.getNClusters();
    }, "Number of mixture components");
}
