classdef PoseGraph < handle
  properties
    values = [];
    pose_graph = [];
    covariance = gtsam.noiseModel.Diagonal.Sigmas([0.01; 0.01; 0.01; ...
						   0.04; 0.04; 0.04]);
    key = 1;
    odometry = gtsam.Pose3;
  end

  methods

    function self = PoseGraph()
      isamParams = gtsam.ISAM2Params;
      %isamParams.setRelinearizeSkip(1);
      isamParams.setRelinearizeThreshold(0.01);
      self.pose_graph = gtsam.ISAM2(isamParams);

      new_factor = gtsam.NonlinearFactorGraph;
      new_value = gtsam.Values;

      sigma_init_x = gtsam.noiseModel.Isotropic.Precisions([0.1; 0.1; 0.1; ...
							    0.02; 0.02; 0.02]);
      new_value.insert(self.key, gtsam.Pose3);
      new_factor.add(gtsam.PriorFactorPose3(1, gtsam.Pose3, sigma_init_x));

      self.update(new_factor, new_value);
      self.increment_key()
    end

    % T \in 4 x 4
    % Consists of 3x3 rotation matrix and 3x1 translation vector
    function gtsamT = to_gtsam(self, T)
      gtsamT = gtsam.Pose3(gtsam.Rot3(T(1:3, 1:3)), ...
			  gtsam.Point3(T(1:3,4)));
    end

    function T = from_gtsam(self, gtsamT)
      T = gtsamT.matrix();
    end

    function update_odometry_incrementally(self, T)
      self.odometry = self.odometry.compose(self.to_gtsam(T))
    end

    function T = get_curr_pose(self)
      T = self.odometry.matrix()
    end

    function add_loop_constraint(self, k, delta)
      new_factor = gtsam.NonlinearFactorGraph;
      new_factor.add(gtsam.BetweenFactorPose3(k, self.key, delta, self.covariance));
      self.pose_graph.update(new_factor, gtsam.Values());
    end

    function create_between_factor(self)
      new_factor = gtsam.NonlinearFactorGraph;
      new_value = gtsam.Values;
      new_factor.add(gtsam.BetweenFactorPose3(self.key-1, self.key, ...
					      self.odometry, ...
					      self.covariance));
      self.odometry = gtsam.Pose3;

      last_pose = self.values.atPose3(self.key-1);
      new_value.insert(self.key, last_pose.compose(self.odometry));

      self.update(new_factor, new_value);
    end

    function update(self, new_factor, new_value)
      self.pose_graph.update(new_factor, new_value);
      self.values = self.pose_graph.calculateEstimate();
    end

    function increment_key(self)
      self.key = self.key + 1;
    end

    function calculate_estimate(self)
      self.values = self.pose_graph.calculateEstimate();
    end

    function plot(self)
      xyz = [];
      for i = 1:self.key-1
	T = self.values.atPose3(i).matrix();
	xyz(end+1,:) = T(1:3, 4)';
      end
      hold on;
      plot3(xyz(:,1), xyz(:,2), xyz(:,3));
    end

  end
end
