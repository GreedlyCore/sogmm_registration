#include <nanobind/nanobind.h>

#include <self_organizing_gmm/TimeProfiler.h>

namespace nb = nanobind;

NB_MODULE(time_profiler_py, m)
{
  nb::class_<TimeProfiler>(m, "TimeProfiler")
  .def(nb::init())
  .def("tic", &TimeProfiler::tic)
  .def("toc", &TimeProfiler::toc);
}
