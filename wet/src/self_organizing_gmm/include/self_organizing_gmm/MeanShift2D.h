#pragma once

#include <map>
#include <vector>
#include <memory>

#include <Eigen/Dense>
#include <nanoflann.hpp>
#include <unordered_map>

#include "TimeProfiler.h"

namespace sogmm
{
  /// @brief Kernel types for Mean Shift
  enum class KernelType
  {
    FLAT,      // Uniform kernel (default)
    GAUSSIAN,  // Gaussian kernel
    CAUCHY,    // Cauchy kernel
    LOGISTIC,   // Logistic kernel
    EPANECHNIKOV // Epanechnikov kernel (optimal for MSE)
  };

  /// @brief A class to run Gaussian Blurring Mean Shift (GBMS) on 2D data.
  class MeanShift2D
  {
  public:
    using Ptr = std::shared_ptr<MeanShift2D>;
    using ConstPtr = std::shared_ptr<const MeanShift2D>;

    using Vector2 = Eigen::Vector<float, 2>;
    using MatrixX2 = Eigen::Matrix<float, -1, 2, Eigen::RowMajor>;

    using EigKDTree = nanoflann::KDTreeEigenMatrixAdaptor<MatrixX2>;
    using PostProcessMap = std::multimap<unsigned int, unsigned int, std::greater<unsigned int>>;

    MeanShift2D()
    {
      bandwidth_ = 0.0f;
      num_modes_ = 0;
      kernel_type_ = KernelType::FLAT;
    }

    ~MeanShift2D()
    {
    }

    MeanShift2D(float s, KernelType kernel_type = KernelType::FLAT)
    {
      bandwidth_ = s;
      kernel_type_ = kernel_type;

      initialize(false, "ms_cpu_stats", "stats.csv");
    }

    MeanShift2D(float s, const bool save_stats, const std::string &stats_dir,
                const std::string &stats_file, KernelType kernel_type = KernelType::FLAT)
    {
      bandwidth_ = s;
      kernel_type_ = kernel_type;

      initialize(save_stats, stats_dir, stats_file);
    }

    /// @brief Initalizer.
    /// @param save_stats Boolean to control saving statistics.
    /// @param stats_dir Directory to save statistics.
    /// @param stats_file Name of the statistics file.
    void initialize(const bool save_stats, const std::string &stats_dir,
                    const std::string &stats_file)
    {
      tp_ = TimeProfiler();
      if (save_stats)
      {
        tp_.save(stats_dir, stats_file);
      }
    }

    /// @brief Perform binning over dataset and find seeds for MeanShift.
    /// @param X Input points, the same points that will be used in mean_shift.
    /// @param seeds (Output) Points used as initial kernel positions.
    inline void get_bin_seeds(const MatrixX2 &X, MatrixX2 &seeds) const
    {
      if (bandwidth_ == 0.0f)
      {
        throw std::runtime_error(
            "[MeanShift2D] error while binning data; bandwidth invalid.");
      }

      unsigned int N = X.rows();

      Vector2 min_pt = X.colwise().minCoeff();
      Vector2 max_pt = X.colwise().maxCoeff();

      // calculate the width of the sparse grid used for hash table
      // width essentially represents the bounds of the data
      unsigned int width = static_cast<unsigned int>(std::floor(
                               ((max_pt(0) - min_pt(0)) / bandwidth_) + 0.5f)) +
                           1;

      auto point_to_index = [&](Vector2 &p)
      {
        unsigned int r = static_cast<unsigned int>(
            std::floor(((p(1) - min_pt(1)) / bandwidth_) + 0.5f));
        unsigned int c = static_cast<unsigned int>(
            std::floor(((p(0) - min_pt(0)) / bandwidth_) + 0.5f));
        unsigned int idx = r * width + c;
        return idx;
      };

      auto index_to_point = [&](unsigned int &index)
      {
        unsigned int r = index / width;
        unsigned int c = index % width;
        Vector2 ret;
        ret(0) = std::fmaf(static_cast<float>(c), bandwidth_, min_pt(0));
        ret(1) = std::fmaf(static_cast<float>(r), bandwidth_, min_pt(1));
        return ret;
      };

      std::unordered_map<unsigned int, int> bin_sizes;
      for (size_t i = 0; i < N; i++)
      {
        Vector2 world_pt = X(i, Eigen::all);
        unsigned int idx = point_to_index(world_pt);
        try
        {
          bin_sizes[idx] += 1;
        }
        catch (const std::out_of_range &e)
        {
          bin_sizes[idx] = 1;
        }
      }

      // get the number of non-empty bins.
      std::vector<unsigned int> valid_bins;
      for (auto &it : bin_sizes)
      {
        if (it.second >= 1)
        {
          // if a bin contains at least 1 point, it is a valid bin.
          valid_bins.push_back(it.first);
        }
      }

      unsigned int num_bins = valid_bins.size();
      // if the number of bins are zero, we do not proceed with mean shift.
      if (num_bins == 0)
      {
        throw std::runtime_error(
            "[MeanShift2D] error while binning data; no valid bins found.");
      }
      // if the number of bins are the same as the number of points in the
      // dataset, we just return the dataset.
      if (num_bins == N)
      {
        seeds = X;
      }

      // Update the bin seeds matrix
      seeds.resize(num_bins, 2);
      for (unsigned int j = 0; j < num_bins; j++)
      {
        unsigned int bin_idx = valid_bins[j];
        Vector2 bin_pt = index_to_point(bin_idx);
        seeds(j, Eigen::all) = bin_pt;
      }
    }

    /// @brief Utility function to read nanoflann results.
    /// @param p ResultItem containing an Eigen::Index and a distance float value.
    /// @return Eigen::Index
    static Eigen::Index filter_index(const nanoflann::ResultItem<Eigen::Index, float> &p)
    {
      return p.first;
    }

    /// @brief Find neighbors in the given search radius using nanoflann.
    /// @param x Query point
    /// @param search_radius Query radius
    /// @param kdtree Input KDTree
    /// @param neighbor_indices (Output) neighbor indices
    inline void radius_neighbors(
        const MatrixX2 &x, const float &search_radius, const EigKDTree &kdtree,
        std::vector<Eigen::Index> &neighbor_indices) const
    {
      const float kdtree_search_radius = search_radius * search_radius;
      size_t dim = 2;
      float query_pt[dim];
      for (size_t d = 0; d < dim; d++)
      {
        query_pt[d] = x(d);
      }
      std::vector<nanoflann::ResultItem<Eigen::Index, float>> indices_dists;
      nanoflann::SearchParameters search_params;
      search_params.sorted = false;
      size_t n = kdtree.index_->radiusSearch(&query_pt[0], kdtree_search_radius,
                                            indices_dists, search_params);
      std::transform(indices_dists.begin(), indices_dists.end(),
                     std::back_insert_iterator(neighbor_indices), filter_index);
    }

    /// @brief Run GBMS on input dataset
    /// @param X Input dataset of shape (N, 2)
    inline void fit(const MatrixX2 &X)
    {
      tp_.tic("fit");
      unsigned int dim = 2;

      unsigned int N_data = X.rows();

      // build a kdtree over the entire dataset
      // leaf size is set to the default value as in scikit-learn implementation
      const int leaf_size = 30;
      EigKDTree kdtree(dim, std::cref(X), leaf_size);
      kdtree.index_->buildIndex();

      MatrixX2 seeds;
      get_bin_seeds(X, seeds);
      unsigned int N_seeds = seeds.rows();

      // keeping the stopping criteria (max_iter and stop_thres) the same as
      // sklearn
      float stop_thres = 1e-3f * bandwidth_;
      unsigned int max_iter = 300;
      unsigned int max_iter_actual = 0;

      // process each seed
      std::vector<unsigned int> points_within_array;
      points_within_array.resize(N_seeds);  // Pre-allocate for parallel access

      // Parallelize seed processing - each thread handles independent seeds
      // Use dynamic scheduling since convergence time varies per seed
#pragma omp parallel if(N_seeds > 50)
      {
#pragma omp for schedule(dynamic) reduction(max:max_iter_actual)
        for (size_t i = 0; i < N_seeds; i++)
        {
          // until the algorithm hits max_iter or reaches stop_thres
          unsigned int completed_iter = 0;
          while (true)
          {
            std::vector<Eigen::Index> neighbor_indices;
            radius_neighbors(seeds(i, Eigen::all), bandwidth_, kdtree,
                             neighbor_indices);
            MatrixX2 points_within = X(neighbor_indices, Eigen::all);
            if (points_within.rows() != 0)
            {
              // seed has some neighbors remaining
              // store the previous position of the seed
              MatrixX2 old_seed = seeds(i, Eigen::all);

              // shift the seed using weighted mean of the neighbors
              if (kernel_type_ == KernelType::FLAT)
              {
                // For FLAT kernel, use simple mean (original behavior)
                seeds(i, Eigen::all) = points_within.colwise().mean();
              }
              else
              {
                // For other kernels, compute weighted mean using vectorized operations
                // Compute all distances at once (vectorized)
                Eigen::VectorXf dists = (points_within.rowwise() - seeds.row(i)).rowwise().norm();

                // Apply kernel function to all distances (vectorized)
                Eigen::VectorXf weights = dists.unaryExpr([this](float d) { return this->compute_kernel_weight(d); });

                float sum_weights = weights.sum();

                // Normalize weights
                if (sum_weights > 0.0f)
                {
                  weights /= sum_weights;
                  // Compute weighted mean
                  seeds(i, Eigen::all) = (weights.transpose() * points_within);
                }
                else
                {
                  // Fallback to simple mean if all weights are zero
                  seeds(i, Eigen::all) = points_within.colwise().mean();
                }
              }

              // check if the shifting was actually substantial
              if ((seeds(i, Eigen::all) - old_seed).norm() < stop_thres ||
                  completed_iter == max_iter)
              {
                // shift was not substantial, leave the seed alone
                points_within_array[i] = points_within.rows();
                break;
              }
            }
            else
            {
              // leave the seed untouched as it has no neighbors anymore
              points_within_array[i] = points_within.rows();
              break;
            }
            completed_iter += 1;
          }

          if (completed_iter > max_iter_actual)
          {
            max_iter_actual = completed_iter;
          }
        }
      }

      // post processing: get the mode centers.

      // sort the seeds according to the number of points within
      // a the final seeds from the mean shift iterations.
      PostProcessMap c_i_map;
      for (size_t i = 0; i < N_seeds; i++)
      {
        if (points_within_array[i])
        {
          c_i_map.insert(std::make_pair(points_within_array[i], i));
        }
      }

      // get the indices from the multimap created above
      std::vector<unsigned int> sorted_idxs;
      PostProcessMap::iterator it;
      for (it = c_i_map.begin(); it != c_i_map.end(); it++)
      {
        sorted_idxs.push_back((*it).second);
      }

      // sorted seeds matrix is a subset of seeds because we use a
      // multimap above that allows repeated key values
      MatrixX2 sorted_seeds = seeds(sorted_idxs, Eigen::all);

      // create a vector of ones to track which of the sorted seeds are
      // unique. initially assume everything is unique.
      Eigen::VectorXi unique = Eigen::VectorXi::Ones(sorted_idxs.size());

      EigKDTree pst_prcss_kdtr(dim, std::cref(sorted_seeds), leaf_size);
      pst_prcss_kdtr.index_->buildIndex();

      for (size_t i = 0; i < sorted_idxs.size(); i++)
      {
        if (unique(i))
        {
          std::vector<Eigen::Index> close_nbrs;
          radius_neighbors(sorted_seeds(i, Eigen::all), bandwidth_,
                           pst_prcss_kdtr, close_nbrs);
          unique(close_nbrs) = Eigen::VectorXi::Zero(close_nbrs.size());
          unique(i) = 1;
        }
      }

      num_modes_ = (unique.array() > 0).count();
      mode_centers_.resize(num_modes_, dim);
      unsigned int k = 0;
      for (size_t i = 0; i < sorted_idxs.size(); i++)
      {
        if (unique(i))
        {
          mode_centers_(k, Eigen::all) = sorted_seeds(i, Eigen::all);
          k += 1;
        }
      }
      tp_.toc("fit");
    }

    inline unsigned int get_num_modes() const
    {
      return num_modes_;
    }

    inline MatrixX2 get_mode_centers() const
    {
      return mode_centers_;
    }

  private:
    float bandwidth_;
    unsigned int num_modes_;
    MatrixX2 mode_centers_;
    KernelType kernel_type_;

    TimeProfiler tp_;

    /// @brief Compute kernel weight based on distance
    /// @param distance Euclidean distance from kernel center
    /// @return Weight value
    inline float compute_kernel_weight(float distance) const
    {
      float normalized_dist = distance / bandwidth_;

      switch (kernel_type_)
      {
        case KernelType::FLAT:
          // Uniform kernel: weight = 1 if distance <= bandwidth, 0 otherwise
          return (distance <= bandwidth_) ? 1.0f : 0.0f;

        case KernelType::GAUSSIAN:
          // Gaussian kernel: weight = exp(-0.5 * (d/h)^2)
          return std::exp(-0.5f * normalized_dist * normalized_dist);

        case KernelType::CAUCHY:
          // Cauchy kernel: weight = 1 / (1 + (d/h)^2)
          return 1.0f / (1.0f + normalized_dist * normalized_dist);

        case KernelType::LOGISTIC:
          // Logistic kernel: weight = 1 / (exp(d/h) + 2 + exp(-d/h))
          {
            float exp_pos = std::exp(normalized_dist);
            float exp_neg = std::exp(-normalized_dist);
            return 1.0f / (exp_pos + 2.0f + exp_neg);
          }

        case KernelType::EPANECHNIKOV:
          // Epanechnikov kernel: weight = max(0, 1 - (d/h)^2)
          // This kernel is optimal in the sense of minimizing MSE
          if (distance <= bandwidth_)
          {
            float t = normalized_dist * normalized_dist;
            return std::max(0.0f, 1.0f - t);
          }
          else
          {
            return 0.0f;
          }

        default:
          return 1.0f;
      }
    }
  };
}