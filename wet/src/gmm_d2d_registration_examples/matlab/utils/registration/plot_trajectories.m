function [] = plot_trajectories(transforms, ground_truths, FIRST_SCAN, LAST_SCAN)
  Ts_gmm = get_poses(FIRST_SCAN, LAST_SCAN, transforms);
  Ts_gt = get_poses(FIRST_SCAN, LAST_SCAN, ground_truths);

  xyz_gmm = [];
  xyz_gt = [];
  for i = FIRST_SCAN:LAST_SCAN
    xyz_gmm(i,:) = Ts_gmm{i-FIRST_SCAN+1}(1:3, 4);
    xyz_gt(i,:) = Ts_gt{i-FIRST_SCAN+1}(1:3, 4);
  end

  figure;
  plot3(xyz_gmm(:,1), xyz_gmm(:,2), xyz_gmm(:,3));
  hold on;
  plot3(xyz_gt(:,1), xyz_gt(:,2), xyz_gt(:,3));
  legend('GMM D2D Registration', 'Ground Truth');
  axis equal;
end

function Ts = get_poses(FIRST_SCAN, LAST_SCAN, transforms)
  xyz_errors = [];
  rpy_errors = [];

  Tall = eye(4);
  Ts{1} = Tall;

  for i=FIRST_SCAN:LAST_SCAN
    pose = transforms(i-FIRST_SCAN+1,:);
    T = [ZYXToR(pose(4:6)), pose(1:3)'];
    Tall = pose_compose(Tall, T);
    Ts{end+1} = Tall;
  end
end
