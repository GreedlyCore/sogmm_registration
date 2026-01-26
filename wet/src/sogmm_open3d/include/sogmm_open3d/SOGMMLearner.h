#pragma once

#include <self_organizing_gmm/MeanShift2D.h>
#include <self_organizing_gmm/KInit.h>
#include <sogmm_open3d/SOGMMGPU.h>
#include <sogmm_open3d/EM.h>

namespace o3d = open3d;

namespace sogmm
{
  namespace gpu
  {
    using sogmm::KernelType;

    template <typename T>
    class SOGMMLearner
    {
    public:
      using Container = SOGMM<T, 4>;
      using HostContainer = sogmm::cpu::SOGMM<T, 4>;

      using KInitPtr = typename KInit<T, 4>::Ptr;
      using EMPtr = typename EM<T, 4>::Ptr;

      using MatrixX2 = MeanShift2D::MatrixX2;

      using Matrix = typename HostContainer::Matrix;
      using MatrixX4 = typename HostContainer::MatrixXD;

      SOGMMLearner()
      {
        ms_ = std::make_shared<MeanShift2D>();
        kinit_ = std::make_shared<KInit<T, 4>>();
        em_ = std::make_shared<EM<T, 4>>();
      }

      SOGMMLearner(const float &bandwidth)
      {
        ms_ = std::make_shared<MeanShift2D>(bandwidth);
        kinit_ = std::make_shared<KInit<T, 4>>();
        em_ = std::make_shared<EM<T, 4>>();
      }

      SOGMMLearner(const float &bandwidth, const KernelType &kernel_type)
      {
        ms_ = std::make_shared<MeanShift2D>(bandwidth, kernel_type);
        kinit_ = std::make_shared<KInit<T, 4>>();
        em_ = std::make_shared<EM<T, 4>>();
      }

      SOGMMLearner(const float &bandwidth, const KernelType &kernel_type,
                   const bool save_stats, const std::string &stats_dir,
                   const std::string &stats_file_prefix)
      {
        std::string ms_stats_file = stats_file_prefix + "_ms.csv";
        std::string em_stats_file = stats_file_prefix + "_em.csv";

        ms_ = std::make_shared<MeanShift2D>(bandwidth, save_stats, stats_dir, ms_stats_file, kernel_type);
        kinit_ = std::make_shared<KInit<T, 4>>();  // KInit doesn't support profiling
        em_ = std::make_shared<EM<T, 4>>(save_stats, stats_dir, em_stats_file);
      }

      void fit(const MatrixX2 &Y, const MatrixX4 &X, Container &sogmm)
      {
        if (Y.rows() != X.rows())
        {
          throw std::runtime_error("[SOGMMLearner] Number of samples are not the same in image and point cloud.");
        }

        // Run GBMS to estimate number of components.
        ms_->fit(Y);

        // Initialize this sogmm.
        sogmm = Container(ms_->get_num_modes());
        sogmm.support_size_ = X.rows();

        // Compute initial responsibility matrix.
        Matrix resp = Matrix::Zero(sogmm.support_size_, sogmm.n_components_);
        kinit_->getRespMat(X, resp);

        // Take stuff to the GPU
        Xt_ = EigenMatrixToTensor(X, sogmm.device_);
        Respt_ = EigenMatrixToTensor(resp, sogmm.device_);

        // Fit GMM.
        em_->fit(Xt_, Respt_, sogmm);
      }

      void fit_em(const MatrixX4 &X, const unsigned int K, Container &sogmm)
      {
        // Initialize this sogmm.
        sogmm = Container(K);
        sogmm.support_size_ = X.rows();

        // Compute initial responsibility matrix.
        Matrix resp = Matrix::Zero(sogmm.support_size_, sogmm.n_components_);
        kinit_->getRespMat(X, resp);

        // Take stuff to the GPU
        Xt_ = EigenMatrixToTensor(X, sogmm.device_);
        Respt_ = EigenMatrixToTensor(resp, sogmm.device_);

        // Fit GMM.
        em_->fit(Xt_, Respt_, sogmm);
      }

    private:
      MeanShift2D::Ptr ms_;
      KInitPtr kinit_;
      EMPtr em_;

      Tensor Xt_;
      Tensor Respt_;
    };
  }
}