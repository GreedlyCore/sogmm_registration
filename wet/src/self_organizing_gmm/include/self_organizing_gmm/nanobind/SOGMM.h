#include <self_organizing_gmm/SOGMMLearner.h>
#include <self_organizing_gmm/SOGMMInference.h>

#include <nanobind/nanobind.h>
#include <nanobind/eigen/dense.h>
#include <nanobind/stl/string.h>

namespace nb = nanobind;
using namespace nb::literals;

namespace sogmm
{
  namespace cpu
  {
    template <typename T, uint32_t D>
    void container_binding_generator(nb::module_ &m, std::string &typestr)
    {
      using Container = sogmm::cpu::SOGMM<T, D>;

      std::string pyclass_name = std::string("SOGMM") + typestr;
      nb::class_<Container>(m, pyclass_name.c_str(), "GMM parameters container on the CPU.",
                            nb::dynamic_attr())
          .def(nb::init(), "Default empty constructor.")
          .def(nb::init<const Container &>(),
               "Copy from an existing container.",
               "that"_a)
          .def(nb::init<const uint32_t &>(),
               "Initialize zero members for the given number of components.",
               "n_components"_a)
          .def(nb::init<const typename Container::Vector &,
                        const typename Container::MatrixXD &,
                        const typename Container::MatrixXC &,
                        const uint32_t &>(),
               "weights"_a,
               "means"_a,
               "covariances"_a,
               "support_size"_a)
          .def_rw("n_components_", &Container::n_components_,
                         "Number of components in this GMM.")
          .def_rw("support_size_", &Container::support_size_,
                         "Number of points in the support of this GMM.")
          .def_rw("weights_", &Container::weights_, "All weights.")
          .def_rw("means_", &Container::means_, "All means.")
          .def_rw("covariances_", &Container::covariances_,
                         "All covariances.")
          .def_rw("precisions_cholesky_", &Container::precisions_cholesky_,
                         "All cholesky decompositions of the precision matrices.")
          .def_rw("covariances_cholesky_", &Container::covariances_cholesky_,
                         "All cholesky decompositions of the covariance matrices.")
          .def("normalize_weights", &Container::normalizeWeights,
               "Normalize the weight vector.")
          .def("update_cholesky", &Container::updateCholesky,
               "Update covariances_cholesky_ and precisions_cholesky_.")
          .def("merge", &Container::merge,
               "Merge another container into this container.")
          .def("submap_from_indices", &Container::submapFromIndices,
               "Create a sub GMM from supplied list of indices.")
          .def("__getstate__", [](const Container &g) {
            return nb::make_tuple(g.weights_, g.means_, g.covariances_, g.support_size_);
          }, "Serialization through pickling.")
          .def("__setstate__", [](Container &g, nb::tuple t) {
            new (&g) Container(nb::cast<typename Container::Vector>(t[0]),
                               nb::cast<typename Container::MatrixXD>(t[1]),
                               nb::cast<typename Container::MatrixXC>(t[2]),
                               nb::cast<uint32_t>(t[3]));
          }, "Deserialization through pickling.");
    }

    template <typename T>
    void learner_binding_generator(nb::module_ &m, std::string &typestr)
    {
      using Learner = sogmm::cpu::SOGMMLearner<T>;

      std::string pyclass_name = std::string("SOGMM") + typestr;
      nb::class_<Learner>(m, pyclass_name.c_str(), "GMM parameters container on the CPU.",
                          nb::dynamic_attr())
          .def(nb::init(), "Default empty constructor.")
          .def(nb::init<const float &>(), "Initialize using the bandwidth parameter")
          .def("fit", &Learner::fit)
          .def("fit_em", &Learner::fit_em);
    }

    template <typename T>
    void inference_binding_generator(nb::module_ &m, std::string &typestr)
    {
      using Container = sogmm::cpu::SOGMM<T, 4>;
      using Inference = sogmm::cpu::SOGMMInference<T>;

      std::string pyclass_name = std::string("SOGMM") + typestr;
      nb::class_<Inference>(m, pyclass_name.c_str(), "GMM parameters container on the CPU.",
                            nb::dynamic_attr())
          .def(nb::init(), "Default empty constructor.")
          .def("generate_pcld_4d", &Inference::generatePointCloud4D)
          .def("generate_pcld_3d", &Inference::generatePointCloud3D)
          .def("reconstruct", &Inference::reconstruct)
          .def("color_query", &Inference::colorQuery)
          .def("reconstruct_fast", &Inference::reconstructFast)
          .def("score_4d", &Inference::score4D)
          .def("score_3d", &Inference::score3D);
    }
  }
}