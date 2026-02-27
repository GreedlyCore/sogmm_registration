#include <self_organizing_gmm/CameraModel.h>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/pair.h>
#include <nanobind/stl/vector.h>

namespace nb = nanobind;

NB_MODULE(camera_model_py, m)
{
  nb::class_<CameraModel>(m, "CameraModel")
      .def(nb::init())
      .def(nb::init<Eigen::Matrix3f>())
      .def(nb::init<Eigen::Matrix3f, size_t, size_t>())
      .def(nb::init<float, float, float, float, size_t, size_t>())
      .def_rw("w", &CameraModel::im_w_)
      .def_rw("h", &CameraModel::im_h_)
      .def_rw("K", &CameraModel::intrinsic_matrix_)
      .def("to_3d", &CameraModel::to_3d)
      .def("to_2d", &CameraModel::to_2d)
      .def("to_2d_dim", &CameraModel::to_2d_dim);
}
