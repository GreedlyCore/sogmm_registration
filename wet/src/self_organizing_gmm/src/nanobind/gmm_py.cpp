#include <self_organizing_gmm/GMM.h>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;

template <typename T, uint32_t D>
void binding_generator(nb::module_& m, std::string& typestr)
{
  static constexpr uint32_t C = D * D;
  using GMMClass = GMM<T, D>;
  std::string pyclass_name = std::string("GMM") + typestr;
  nb::class_<GMMClass>(m, pyclass_name.c_str(), nb::dynamic_attr())
      .def(nb::init())
      .def(nb::init<unsigned int>())
      .def(nb::init<unsigned int, T, T, unsigned int>())
      .def(nb::init<unsigned int, bool, std::string, std::string>())
      .def(nb::init<unsigned int, T, T, unsigned int, bool, std::string, std::string>())
      .def_rw("n_components_", &GMMClass::n_components_)
      .def_prop_ro("n_components", [](const GMMClass& g) {
          return static_cast<unsigned int>(g.weights_.rows());
      })
      .def_rw("tol_", &GMMClass::tol_)
      .def_rw("reg_covar_", &GMMClass::reg_covar_)
      .def_rw("max_iter_", &GMMClass::max_iter_)
      .def_rw("support_size_", &GMMClass::support_size_)
      .def_rw("weights_", &GMMClass::weights_)
      .def_rw("means_", &GMMClass::means_)
      .def_rw("covariances_", &GMMClass::covariances_)
      .def_rw("precisions_cholesky_", &GMMClass::precisions_cholesky_)
      .def_rw("resp_", &GMMClass::resp_)
      .def("compute_log_det_cholesky", &GMMClass::computeLogDetCholesky)
      .def("compute_precision_cholesky", &GMMClass::computeCholesky)
      .def("estimate_log_gaussian_prob", &GMMClass::estimateLogGaussianProb)
      .def("estimate_log_prob", &GMMClass::estimateLogProb)
      .def("log_sum_exp_cols", &GMMClass::logSumExpCols)
      .def("estimate_weighted_log_prob", &GMMClass::estimateWeightedLogProb)
      .def("estimate_log_prob_resp", &GMMClass::estimateLogProbResp)
      .def("estimate_gaussian_parameters",
           &GMMClass::estimateGaussianParameters)
      .def("e_step", &GMMClass::eStep)
      .def("m_step", &GMMClass::mStep)
      .def("fit", nb::overload_cast<const typename GMMClass::MatrixXD&, const typename GMMClass::Matrix&>(&GMMClass::fit))
      .def("fit_mahal", nb::overload_cast<const typename GMMClass::MatrixXD&, const typename GMMClass::Matrix&, T>(&GMMClass::fit),
           nb::arg("X"), nb::arg("resp"), nb::arg("mahal_distance"))
      .def("sample", &GMMClass::sample)
      .def("score_samples", &GMMClass::scoreSamples)
      .def("score", &GMMClass::score)
      .def("merge", &GMMClass::merge)
      .def("color_conditional", &GMMClass::colorConditional)
      .def("__getstate__", [](const GMMClass& g) {
        return nb::make_tuple(
            g.n_components_, g.tol_, g.reg_covar_, g.max_iter_,
            g.support_size_, g.weights_, g.means_, g.covariances_,
            g.precisions_cholesky_, g.covariances_cholesky_);
      })
      .def("__setstate__", [](GMMClass& g, nb::tuple t) {
        new (&g) GMMClass();
        g.n_components_ = nb::cast<unsigned int>(t[0]);
        g.tol_ = nb::cast<T>(t[1]);
        g.reg_covar_ = nb::cast<T>(t[2]);
        g.max_iter_ = nb::cast<unsigned int>(t[3]);
        g.support_size_ = nb::cast<unsigned int>(t[4]);
        g.weights_ = nb::cast<Eigen::Matrix<T, Eigen::Dynamic, 1>>(t[5]);
        g.means_ = nb::cast<
            Eigen::Matrix<T, Eigen::Dynamic, D, (D == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>(t[6]);
        g.covariances_ = nb::cast<
            Eigen::Matrix<T, Eigen::Dynamic, C, (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>(t[7]);
        g.precisions_cholesky_ = nb::cast<
            Eigen::Matrix<T, Eigen::Dynamic, C, (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>(t[8]);
        g.covariances_cholesky_ = nb::cast<
            Eigen::Matrix<T, Eigen::Dynamic, C, (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>(t[9]);
      });
}

NB_MODULE(gmm_py, g)
{
  std::string t1 = "f1CPU";
  binding_generator<float, 1>(g, t1);

  std::string t2 = "f2CPU";
  binding_generator<float, 2>(g, t2);

  std::string t3 = "f3CPU";
  binding_generator<float, 3>(g, t3);

  std::string t4 = "f4CPU";
  binding_generator<float, 4>(g, t4);
}
