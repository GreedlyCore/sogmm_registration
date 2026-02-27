#include <gmm/SKLearn.h>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

namespace nb = nanobind;

// SKLearn<float, 3>:
//   fit(MatrixX, n_clusters)  -- MatrixX is (3, N) col-major
//   getWeights() -> (K,)
//   getMeans()   -> (3, K)
//   getCovs()    -> (9, K)
//   getSize()    -> uint32_t

NB_MODULE(gmm_classic_py, m)
{
  using SK3f = SKLearn<float, 3>;

  nb::class_<SK3f>(m, "GMM3f")
      .def(nb::init<>())
      .def("fit",
           [](SK3f& sk, const Eigen::MatrixXf& X, int n_components) {
             // X from Python: (N, 3) row-major -> (3, N) col-major for SKLearn
             Eigen::MatrixXf Xt = X.transpose();
             sk.fit(Xt, n_components);
           },
           nb::arg("X"), nb::arg("n_components"))
      .def_prop_ro("n_components",
                   [](const SK3f& sk) { return (int)sk.getSize(); })
      .def_prop_ro("weights_",
                   [](const SK3f& sk) {
                     return Eigen::VectorXf(sk.getWeights());
                   })
      .def_prop_ro("means_",
                   [](const SK3f& sk) {
                     // (3, K) -> (K, 3) for Python
                     return Eigen::MatrixXf(sk.getMeans().transpose());
                   })
      .def_prop_ro("covariances_",
                   [](const SK3f& sk) {
                     // (9, K) -> (K, 9) for Python
                     return Eigen::MatrixXf(sk.getCovs().transpose());
                   });
}
