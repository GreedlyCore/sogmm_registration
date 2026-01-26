#include <sogmm_open3d/SOGMMLearner.h>
#include <sogmm_open3d/SOGMMInference.h>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

#include <self_organizing_gmm/nanobind/SOGMM.h>
#include <self_organizing_gmm/MeanShift2D.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace sogmm
{
  namespace gpu
  {
    using sogmm::KernelType;

    // Helper function to convert string to KernelType
    inline KernelType string_to_kernel_type(const std::string& kernel_str)
    {
      if (kernel_str == "flat" || kernel_str == "uniform")
        return KernelType::FLAT;
      else if (kernel_str == "gaussian")
        return KernelType::GAUSSIAN;
      else if (kernel_str == "cauchy")
        return KernelType::CAUCHY;
      else if (kernel_str == "logistic")
        return KernelType::LOGISTIC;
      else if (kernel_str == "epanechnikov")
        return KernelType::EPANECHNIKOV;
      else
        throw std::invalid_argument("Unknown kernel type: " + kernel_str +
                                    ". Valid options: 'flat', 'gaussian', 'cauchy', 'logistic', 'epanechnikov'");
    }
    template <typename T, uint32_t D>
    void container_binding_generator(nb::module_ &m, std::string &typestr)
    {
      using Container = sogmm::gpu::SOGMM<T, D>;

      std::string pyclass_name = std::string("SOGMM") + typestr;
      nb::class_<Container>(m, pyclass_name.c_str(), "GMM parameters container on the GPU.",
                            nb::dynamic_attr())
          .def(nb::init(), "Default empty constructor.")
          .def(nb::init<const Container &>(),
               "Copy from an existing container.",
               "that"_a)
          .def(nb::init<const uint32_t &>(),
               "Initialize zero members for the given number of components.",
               "n_components"_a)
          .def_rw("n_components_", &Container::n_components_,
                         "Number of components in this GMM.")
          .def_rw("support_size_", &Container::support_size_,
                         "Number of points in the support of this GMM.")
          .def("merge", &Container::merge,
               "Merge another container into this container.")
          .def("to_host", &Container::toHost,
               "Return this container on the host machine (CPU).")
          .def("from_host", &Container::fromHost,
               "Take a host container to the device (GPU).");
    }

    template <typename T>
    void learner_binding_generator(nb::module_ &m, std::string &typestr)
    {
      using Learner = sogmm::gpu::SOGMMLearner<T>;

      std::string pyclass_name = std::string("SOGMM") + typestr;
      nb::class_<Learner>(m, pyclass_name.c_str(), "GMM parameters container on the CPU.",
                          nb::dynamic_attr())
          .def(nb::init(), "Default empty constructor.")
          .def(nb::init<const float &>(), "Initialize using the bandwidth parameter")
          .def(nb::init<const float &, const KernelType &>(),
               "Initialize using bandwidth and kernel type",
               "bandwidth"_a,
               "kernel_type"_a = KernelType::FLAT)
          // Constructor with string kernel name (more Pythonic)
          .def("__init__", [](Learner* self, float bandwidth, const std::string& kernel) {
                 new (self) Learner(bandwidth, string_to_kernel_type(kernel));
               },
               "bandwidth"_a,
               "kernel"_a = "flat")
          // Constructor with profiling support
          .def("__init__", [](Learner* self, float bandwidth, const std::string& kernel,
                          bool save_stats, const std::string& stats_dir,
                          const std::string& stats_file_prefix) {
                 new (self) Learner(bandwidth, string_to_kernel_type(kernel),
                                   save_stats, stats_dir, stats_file_prefix);
               },
               "bandwidth"_a,
               "kernel"_a = "flat",
               "save_stats"_a = false,
               "stats_dir"_a = "gpu_stats",
               "stats_file_prefix"_a = "sogmm_gpu",
               "Initialize with bandwidth, kernel type, and profiling parameters")
          .def("fit", &Learner::fit)
          .def("fit_em", &Learner::fit_em);
    }

    template <typename T>
    void inference_binding_generator(nb::module_ &m, std::string &typestr)
    {
      using Container = sogmm::gpu::SOGMM<T, 4>;
      using Inference = sogmm::gpu::SOGMMInference<T>;

      std::string pyclass_name = std::string("SOGMM") + typestr;
      nb::class_<Inference>(m, pyclass_name.c_str(), "GMM parameters container on the CPU.",
                            nb::dynamic_attr())
          .def(nb::init(), "Default empty constructor.")
          .def("reconstruct", &Inference::reconstruct)
          .def("score_3d", &Inference::score3D)
          .def("score_4d", &Inference::score4D)
          .def("generate_pcld_4d", &Inference::generatePointCloud4D)
          .def("generate_pcld_3d", &Inference::generatePointCloud3D);
    }
  }
}
