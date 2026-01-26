#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>

#include <sogmm_open3d/GMM.h>

namespace nb = nanobind;

template <typename T, uint32_t D>
void binding_generator(nb::module_& m, std::string& typestr)
{
  static constexpr uint32_t C = D * D;
  using GMMClass = GMM<T, D>;
  std::string pyclass_name = std::string("GMM") + typestr;
  nb::class_<GMMClass>(m, pyclass_name.c_str(), nb::dynamic_attr())
      .def(nb::init())
      .def(nb::init<unsigned int, unsigned int>())
      .def(nb::init<unsigned int, unsigned int, std::string>())
      .def(nb::init<unsigned int, unsigned int, std::string, bool>())
      .def(nb::init<unsigned int, unsigned int, std::string, bool, std::string,
                    std::string>())
      .def_rw("n_components_", &GMMClass::n_components_)
      .def_rw("n_samples_", &GMMClass::n_samples_)
      .def_rw("tol_", &GMMClass::tol_)
      .def_rw("reg_covar_", &GMMClass::reg_covar_)
      .def_rw("max_iter_", &GMMClass::max_iter_)
      .def_rw("support_size_", &GMMClass::support_size_)
      .def_rw("weights_", &GMMClass::weights_)
      .def_rw("means_", &GMMClass::means_)
      .def_rw("covariances_", &GMMClass::covariances_)
      .def_rw("covariances_cholesky_", &GMMClass::covariances_cholesky_)
      .def_rw("precisions_cholesky_", &GMMClass::precisions_cholesky_)
      .def("update_device_and_host_external",
           &GMMClass::updateDeviceAndHostExternal)
      .def("sample", &GMMClass::sample)
      .def("merge", &GMMClass::merge)
      .def("color_conditional", &GMMClass::colorConditional)
      .def("score",
           [](GMMClass& g, const typename GMMClass::MatrixXD& X) {
             // take to the GPU
             typename GMMClass::Tensor Xt = EigenMatrixToTensor(X, g.device_);

             return g.score(Xt);
           })
      .def("score_samples",
           [](GMMClass& g, const typename GMMClass::MatrixXD& X) {
             // take to the GPU
             typename GMMClass::Tensor Xt = EigenMatrixToTensor(X, g.device_);

             // compute the scores on GPU
             typename GMMClass::Tensor output;
             g.scoreSamples(Xt, output);

             // return to CPU
             return TensorToEigenMatrix<T>(output.Reshape({ X.rows(), 1 }));
           })
      .def("fit",
           [](GMMClass& g, const typename GMMClass::MatrixXD& X,
              const typename GMMClass::Matrix& resp) {
             // take to the GPU
             typename GMMClass::Tensor Xt = EigenMatrixToTensor(X, g.device_);
             typename GMMClass::Tensor Respt =
                 EigenMatrixToTensor(resp, g.device_);

             // fit
             bool success = g.fit(Xt, Respt);

             // update CPU members
             g.updateHostfromDevice();

             return success;
           })
      .def("e_step",
           [](GMMClass& g, const typename GMMClass::MatrixXD& X) {
             // take to the GPU
             typename GMMClass::Tensor Xt = EigenMatrixToTensor(X, g.device_);

             // run eStep once
             // stores Log_Resp_ internally
             g.eStep(Xt);

             // return Log_Resp_ on CPU
             return TensorToEigenMatrix<T>(
                 g.getLogResp().Reshape({ X.rows(), g.n_components_ }));
           })
      .def("m_step",
           [](GMMClass& g, const typename GMMClass::MatrixXD& X,
              const typename GMMClass::Matrix& resp) {
             // take to the GPU
             typename GMMClass::Tensor Xt = EigenMatrixToTensor(X, g.device_);
             typename GMMClass::Tensor Respt =
                 EigenMatrixToTensor(resp, g.device_);

             // run mStep once
             // stores the output within the class object
             g.mStep(Xt, Respt);

             // copy to CPU members
             g.updateHostfromDevice();
           })
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
            Eigen::Matrix<T, Eigen::Dynamic, D,
                          (D == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>(t[6]);
        g.covariances_ = nb::cast<
            Eigen::Matrix<T, Eigen::Dynamic, C,
                          (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>(t[7]);
        g.precisions_cholesky_ = nb::cast<
            Eigen::Matrix<T, Eigen::Dynamic, C,
                          (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>(t[8]);
        g.covariances_cholesky_ = nb::cast<
            Eigen::Matrix<T, Eigen::Dynamic, C,
                          (C == 1 ? Eigen::ColMajor : Eigen::RowMajor)>>(t[9]);
      });
}

NB_MODULE(gmm_open3d_py, g)
{
  std::string t1 = "f1GPU";
  binding_generator<float, 1>(g, t1);

  std::string t2 = "f2GPU";
  binding_generator<float, 2>(g, t2);

  std::string t3 = "f3GPU";
  binding_generator<float, 3>(g, t3);

  std::string t4 = "f4GPU";
  binding_generator<float, 4>(g, t4);
}