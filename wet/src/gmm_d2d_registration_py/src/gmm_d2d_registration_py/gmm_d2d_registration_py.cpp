#include <Eigen/Dense>
#include <limits>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>
#include <nanobind/stl/pair.h>
#include <nanobind/ndarray.h>

#include <gmm_d2d_registration/GMMD2DRegistration.h>
#include <self_organizing_gmm/GMM.h>
#include <self_organizing_gmm/KInit.h>

namespace nb = nanobind;
using namespace nb::literals;

std::pair<Eigen::Matrix<float, 4, 4>, float> anisotropic_registration(const Eigen::Matrix<float, 4, 4>Tin,
								      const std::string& source_file,
								      const std::string& target_file)
{
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tout;
  Eigen::Transform<float, 3, Eigen::Affine, Eigen::ColMajor> Tinit =
    Eigen::Translation<float, 3>(Tin.block<3,1>(0,3)) * Tin.block<3,3>(0,0);

  GMM<float, 3> source_gmm;
  source_gmm.load(source_file);

  GMM<float, 3> target_gmm;
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

  GMM<float, 3> source_gmm;
  source_gmm.load(source_file);
  source_gmm.makeCovsIsoplanar();

  GMM<float, 3> target_gmm;
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
  nb::class_<GMM<float, 3>>(m, "GMM3f")
    .def(nb::init<>(), "Create a new 3D Gaussian Mixture Model")
    .def("fit", [](GMM<float, 3>& self,
                   nb::ndarray<float, nb::ndim<2>, nb::c_contig> X,
                   int n_components) {
        // X is ndarray (N, 3) - direct map to Eigen matrix
        size_t N = X.shape(0);
        Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic, 3, Eigen::RowMajor>>
            X_eigen(X.data(), N, 3);

        // Create temporary GMM for fitting
        GMM<float, 3> temp_gmm(n_components);

        // Use KInit (k-means++) for initialization
        using KInitType = sogmm::KInit<float, 3>;
        KInitType kinit;
        typename KInitType::Matrix centers(n_components, 3);
        std::vector<int> indices;
        kinit.fit(X_eigen, n_components, centers, indices);

        // Create initial hard responsibilities based on nearest center
        Eigen::MatrixXf resp = Eigen::MatrixXf::Zero(N, n_components);
        for (size_t i = 0; i < N; ++i) {
            float min_dist = std::numeric_limits<float>::infinity();
            int nearest = 0;
            for (int k = 0; k < n_components; ++k) {
                float dist = (X_eigen.row(i) - centers.row(k)).squaredNorm();
                if (dist < min_dist) {
                    min_dist = dist;
                    nearest = k;
                }
            }
            resp(i, nearest) = 1.0f;
        }

        // Fit GMM using EM with initial responsibilities
        temp_gmm.fit(X_eigen, resp);

        // Copy results to self (only copy data members, not RNG)
        self.n_components_ = temp_gmm.n_components_;
        self.weights_ = temp_gmm.weights_;
        self.means_ = temp_gmm.means_;
        self.covariances_ = temp_gmm.covariances_;
        self.precisions_cholesky_ = temp_gmm.precisions_cholesky_;
        self.covariances_cholesky_ = temp_gmm.covariances_cholesky_;
        self.support_size_ = temp_gmm.support_size_;
        self.converged_ = temp_gmm.converged_;
    }, "X"_a, "n_components"_a,
       "Fit GMM to data. X should be (N, 3) array, n_components is number of Gaussian components")
    .def_prop_ro("weights_", [](const GMM<float, 3>& self) {
        // Return as 1D array (K,)
        Eigen::VectorXf weights = self.getWeights();
        return weights;
    }, "Mixture weights (K,)")
    .def_prop_ro("means_", [](const GMM<float, 3>& self) {
        // Return as (K, 3) array - transpose from (3, K)
        Eigen::MatrixXf means = self.getMeans().transpose();
        return means;
    }, "Mixture means (K, 3)")
    .def_prop_ro("covariances_", [](const GMM<float, 3>& self) {
        // Return as (K, 3, 3) ndarray
        // getCovs() returns (K, 9) matrix
        auto covs = self.getCovs();
        size_t K = covs.rows();

        // Allocate ndarray with capsule owner for proper memory management
        float* data = new float[K * 9];

        for (size_t k = 0; k < K; k++) {
            // Each row k is a flattened 3x3 covariance matrix
            Eigen::Map<const Eigen::Matrix<float,3,3,Eigen::RowMajor>> cov(covs.row(k).data());
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
    .def_prop_ro("n_components", [](const GMM<float, 3>& self) {
        return (int)self.getNClusters();
    }, "Number of mixture components")
    .def("load", &GMM<float, 3>::load, "filename"_a,
         "Load GMM parameters from binary file")
    .def("save", &GMM<float, 3>::save, "filename"_a,
         "Save GMM parameters to binary file")
    .def("makeCovsIsoplanar", &GMM<float, 3>::makeCovsIsoplanar,
         "Make covariances isoplanar (flatten along smallest eigenvalue)");
}
