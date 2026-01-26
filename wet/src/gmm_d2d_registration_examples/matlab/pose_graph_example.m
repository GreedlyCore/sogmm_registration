CPWD = pwd;
SANDBOX_NAME = 'gira3d-registration';
matches = split(CPWD, SANDBOX_NAME);
GIRA3D_REGISTRATION_SANDBOX = [matches{1}, SANDBOX_NAME];
DATA_DIR = [GIRA3D_REGISTRATION_SANDBOX, '/data/slam/'];
REGISTRATION_DIR = [GIRA3D_REGISTRATION_SANDBOX, '/data/slam/results/'];
RESULTS_DIR = [GIRA3D_REGISTRATION_SANDBOX, '/data/slam/results/'];
IMAGE_DIR = [GIRA3D_REGISTRATION_SANDBOX, '/data/slam/results/images/'];
NUM_COMPONENTS = 70;

load([DATA_DIR, 'odometry.mat']);
load([REGISTRATION_DIR, 'isoplanarhybrid_70_results.mat']);

transforms = [zeros(1,6); transforms];
ground_truths = [zeros(1,6); ground_truths];

Tgt = eye(4);
gt_poses = zeros(size(ground_truths));
for i=1:size(ground_truths,1)
  Tgt = pose_compose(Tgt, [ZYXToR(ground_truths(i,4:6)), transpose(ground_truths(i,1:3))]);
  gt_poses(i, :) = [transpose(Tgt(1:3,4)), transpose(RToZYX(Tgt(1:3, 1:3)))];
end

import gtsam.*

h = [];
pose_graph = PoseGraph();
Tm = eye(4);
count = 0;
prev_gt = zeros(1,6);
prev_gmm = zeros(1,6);

idxs = [1];
for i = 2:length(pointclouds.points)

  % for now make pretend
  T = [ZYXToR(transforms(i,4:6)) transforms(i,1:3)'];

  % GMM pose calculation
  Tm = pose_compose(Tm, T);

  % Pose graph addition
  pose_graph.update_odometry_incrementally(T);
  T = pose_graph.get_curr_pose();
  if (norm(T(1:3,4)) > 0.5)

    pose_graph.create_between_factor();
    Tm = pose_graph.values.atPose3(pose_graph.values.keys.size()).matrix();

    idxs(end+1) = i;

    curr_gt = gt_poses(i,:);
    curr_gmm = [Tm(1:3,4)' RToZYX(Tm(1:3,1:3))'];

    hold on; plot3([prev_gmm(1) curr_gmm(1)], [prev_gmm(2) curr_gmm(2)], ...
		   [prev_gmm(3) curr_gmm(3)], 'Color', [1, 0, 0], 'LineWidth', 3);
    hold on; drawPointCloud(curr_gmm, pointclouds.points{i});
    hold on; plot3(gt_poses(1:i,1), gt_poses(1:i,2), gt_poses(1:i,3), 'w', 'LineWidth', 2);
    xlim([-44 18]); ylim([-59 20]);

    prev_gt = curr_gt;
    prev_gmm = curr_gmm;

    view(-37.5,30); axis equal;
    xlabel('x (m)'); ylabel('y (m)'); zlabel('z (m)');
    drawnow();

    % check for loop closures
    idxs_potential = findPotentialLoopClosures(pose_graph.values, Tm, 10);
    idxs_in_range = find(idxs_potential < pose_graph.key-25);

    if (~isempty(idxs_in_range) && count > 1)
      fprintf('found potential loop closure within a 10m radius\n');

      closed_loop = false;
      for j = 1:length(idxs_in_range)
	k = idxs_potential(idxs_in_range(j));

	source_file = [DATA_DIR, 'gmms/', sprintf('%06d', i), '.gmm'];
	target_file = [DATA_DIR, 'gmms/', sprintf('%06d', idxs(k)), '.gmm'];

	key_target = k;

	T0t = pose_graph.values.atPose3(key_target).matrix();
	T0s = pose_graph.values.atPose3(pose_graph.key).matrix();
	Tts = pose_compose(pose_inverse(T0t), T0s);

	% align source with target
	x_opt = zeros(6,1);
	x_opt(1:3) = Tts(1:3,4)';
	aa = rotm2axang(Tts(1:3,1:3));
	x_opt(4:6) = aa(1:3) * aa(4);

	x_opt_before = x_opt;
	[x_opt, score] = isoplanar_registration(source_file, target_file, x_opt);
	if (score < -0.65)
	  fprintf(['got loop closure between %d and %d with score %d\n'], i, idxs(k), score);

	  Tts = eye(4);
	  Rotation = axang2rotm([x_opt(4:6) / norm(x_opt(4:6)); norm(x_opt(4:6))]');
	  translation = x_opt(1:3);
	  Tts(1:3, 1:3) = Rotation;
	  Tts(1:3, 4) = translation;
	  delta = pose_graph.to_gtsam([Tts(1:3,1:3), Tts(1:3,4)]);
	  pose_graph.add_loop_constraint(k, delta);
	  closed_loop = true;
	else
	  fprintf(['FAILED to get loop closure between %d and %d with score %d\n'], i, idxs(k), score);
	end
      end

      pose_graph.calculate_estimate();
      Tm = pose_graph.values.atPose3(pose_graph.values.keys.size()).matrix();
      prev_gmm = [Tm(1:3,4)' RToZYX(Tm(1:3,1:3))'];
      gmm_poses = readPoseGraphEntries(pose_graph.values);
      if (closed_loop)
	redrawMap(gmm_poses, idxs, pointclouds);
	hold on; plot3(gt_poses(1:i,1), gt_poses(1:i,2), gt_poses(1:i,3), 'w', 'LineWidth', 2);
	h = plotGTSAMTrajectory(pose_graph.values, h);
	xlim([-44 18]); ylim([-59 20]);
      end
      drawnow();
      count = 0;
    else
      count = count + 1;
    end
    view(-37.5,30); axis equal; set(gca, 'FontSize', 24);
    xlabel('x (m)'); ylabel('y (m)'); zlabel('z (m)');
    xlim([-44 18]); ylim([-59 20]);

    if ~exist(IMAGE_DIR, 'dir')
      mkdir(IMAGE_DIR)
    end
    screenshot(pose_graph.key, IMAGE_DIR);
    pose_graph.increment_key();
  end
end
