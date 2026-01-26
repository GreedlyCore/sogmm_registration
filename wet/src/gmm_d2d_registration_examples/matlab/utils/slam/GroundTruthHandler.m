classdef GroundTruthHandler < handle

  properties
    ground_truth;
    poses;
    pointclouds;
    odometry;
    Tbc;
    PCLD_DIR;
  end

  methods

    function self = GroundTruthHandler(DATA_DIR)
      self.PCLD_DIR = [DATA_DIR, 'pointclouds/'];
      self.ground_truth = load([DATA_DIR, 'odometry.mat']);
      self.pointclouds = self.ground_truth.pointclouds;
      self.Tbc = self.ground_truth.Tbc;
      self.odometry = self.ground_truth.odometry;
    end

    function Twc = get_pose_for_pcld(self, idx)
      [~, j] = self.odometry.closestTime(self.pointclouds.times(idx));
      Twb = [QuatToR(self.odometry.orientations(:, j)), ...
	     self.odometry.positions(:, j)];
      Twc = pose_compose(Twb, self.Tbc);
    end

    function display_point_cloud(self, idx)
      pcld = load([self.PCLD_DIR, num2str(idx), '.txt']);
      Twc = self.get_pose_for_pcld(idx);
      R = Twc(1:3, 1:3);
      t = Twc(1:3, 4);
      pcld = transpose(R*transpose(pcld) + repmat(t, 1, size(pcld, 1)));
      hold on; pcshow(pcld);
    end

    function plot(self, FIRST_SCAN, LAST_SCAN)

      [~, idx0] = self.odometry.closestTime(self.pointclouds.times(FIRST_SCAN));
      R0 = QuatToR(self.odometry.orientations(:, idx0));
      t0 = self.odometry.positions(:, idx0);
      T0 = [R0, t0];

      xyz = [];
      for i=FIRST_SCAN:LAST_SCAN
	[~, idx] = self.odometry.closestTime(self.pointclouds.times(i));
	T = [QuatToR(self.odometry.orientations(:, idx)) self.odometry.positions(:, idx)];
	T0i = pose_compose(pose_inverse(T0), T);
	xyz(:, end+1) = transpose(T0i(1:3,4));
      end
      xyz = transpose(xyz);
      plot3(xyz(:, 1), xyz(:, 2), xyz(:, 3));
    end

  end

 end
