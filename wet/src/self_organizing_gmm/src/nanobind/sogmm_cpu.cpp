#include <nanobind/nanobind.h>

#include <self_organizing_gmm/nanobind/SOGMM.h>

namespace nb = nanobind;

NB_MODULE(sogmm_cpu, g)
{
  std::string t2 = "f2Host";
  std::string t3 = "f3Host";
  std::string t4 = "f4Host";
  sogmm::cpu::container_binding_generator<float, 2>(g, t2);
  sogmm::cpu::container_binding_generator<float, 3>(g, t3);
  sogmm::cpu::container_binding_generator<float, 4>(g, t4);

  std::string f = "Learner";
  sogmm::cpu::learner_binding_generator<float>(g, f);

  f = "Inference";
  sogmm::cpu::inference_binding_generator<float>(g, f);

  g.def("marginal_X", &sogmm::cpu::extractXpart<float>);
}