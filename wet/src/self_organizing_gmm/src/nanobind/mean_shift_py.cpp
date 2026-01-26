#include <self_organizing_gmm/MeanShift2D.h>
#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;

// Helper function to convert string to KernelType
sogmm::KernelType string_to_kernel_type(const std::string& kernel_str)
{
  if (kernel_str == "flat" || kernel_str == "uniform")
    return sogmm::KernelType::FLAT;
  else if (kernel_str == "gaussian")
    return sogmm::KernelType::GAUSSIAN;
  else if (kernel_str == "cauchy")
    return sogmm::KernelType::CAUCHY;
  else if (kernel_str == "logistic")
    return sogmm::KernelType::LOGISTIC;
  else if (kernel_str == "epanechnikov")
    return sogmm::KernelType::EPANECHNIKOV;
  else
    throw std::invalid_argument("Unknown kernel type: " + kernel_str +
                                ". Valid options: 'flat', 'gaussian', 'cauchy', 'logistic', 'epanechnikov'");
}

NB_MODULE(mean_shift_py, m)
{
  // Expose KernelType enum
  nb::enum_<sogmm::KernelType>(m, "KernelType")
      .value("FLAT", sogmm::KernelType::FLAT)
      .value("GAUSSIAN", sogmm::KernelType::GAUSSIAN)
      .value("CAUCHY", sogmm::KernelType::CAUCHY)
      .value("LOGISTIC", sogmm::KernelType::LOGISTIC)
      .value("EPANECHNIKOV", sogmm::KernelType::EPANECHNIKOV)
      .export_values();

  nb::class_<sogmm::MeanShift2D>(m, "MeanShift")
      .def(nb::init())
      .def(nb::init<float, sogmm::KernelType>(),
           "bandwidth"_a,
           "kernel_type"_a = sogmm::KernelType::FLAT)
      // Constructor with string kernel name (more Pythonic)
      .def("__init__", [](sogmm::MeanShift2D* self, float bandwidth, const std::string& kernel) {
             new (self) sogmm::MeanShift2D(bandwidth, string_to_kernel_type(kernel));
           },
           "bandwidth"_a,
           "kernel"_a = "flat")
      .def(nb::init<float, bool, std::string, std::string, sogmm::KernelType>(),
           "bandwidth"_a,
           "save_stats"_a,
           "stats_dir"_a,
           "stats_file"_a,
           "kernel_type"_a = sogmm::KernelType::FLAT)
      .def("get_num_modes", &sogmm::MeanShift2D::get_num_modes)
      .def("get_mode_centers", &sogmm::MeanShift2D::get_mode_centers)
      .def("fit", &sogmm::MeanShift2D::fit);
}
