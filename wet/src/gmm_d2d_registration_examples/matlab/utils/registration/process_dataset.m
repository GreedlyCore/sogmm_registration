function [transforms, ground_truths] = process_dataset(DATASET, FIRST_SCAN, LAST_SCAN, PREFIX, NUM_COMPONENTS)

  CPWD = pwd;
  SANDBOX_NAME = 'gira3d-registration';
  matches = split(CPWD, SANDBOX_NAME);
  GIRA3D_REGISTRATION_SANDBOX = [matches{1}, SANDBOX_NAME];
  DATA_DIR = [GIRA3D_REGISTRATION_SANDBOX, '/data/', DATASET, '/'];

  % load ground truth data
  load([DATA_DIR, 'odometry.mat'])

  RESULTS_DIR = [DATA_DIR, 'results/'];

  % Create results directory if it does not exist
  if ~exist(RESULTS_DIR, 'dir')
    mkdir(RESULTS_DIR)
  end

  % initialize variables
  errors = [];
  transforms = [];
  ground_truths = [];

  % append to existing results
  if (exist([RESULTS_DIR, PREFIX, 'results.mat']))
    load([output_dir, prefix, 'results.mat']);
    FIRST_SCAN = size(errors,1);
  end

  for j = FIRST_SCAN:LAST_SCAN

    fprintf(["Registering scans " + num2str(j) + " and " + num2str(j+1) + " out of " + num2str(LAST_SCAN+1) + " scans\n"]);
    target_file = [DATA_DIR, num2str(NUM_COMPONENTS), '_components/', num2str(j), '.gmm'];
    source_file = [DATA_DIR, num2str(NUM_COMPONENTS), '_components/', num2str(j+1), '.gmm'];

    x_opt = zeros(6,1);
    [x_opt, score] = isoplanar_hybrid_registration(source_file, target_file, zeros(6,1));

    R = axang2rotm([x_opt(4:6) / norm(x_opt(4:6)); norm(x_opt(4:6))]');
    zyx = RToZYX(R);
    dpose = [x_opt(1:3)', zyx'];

    % ground truth
    [~, idx] = odometry.closestTime(pointclouds.times(j));
    Twb1 = [QuatToR(odometry.orientations(:, idx)), odometry.positions(:, idx)];
    [~, idx] = odometry.closestTime(pointclouds.times(j+1));
    Twb2 = [QuatToR(odometry.orientations(:, idx)), odometry.positions(:, idx)];

    Twc1 = pose_compose(Twb1, Tbc);
    Twc2 = pose_compose(Twb2, Tbc);

    Tc1c2 = pose_compose(pose_inverse(Twc1), Twc2);
    dground_truth = [Tc1c2(1:3,4)', RToZYX(Tc1c2(1:3,1:3))'];

    transforms = [transforms; dpose];
    ground_truths = [ground_truths; dground_truth];
    errors = [errors; dpose - dground_truth];

    RESULTS_FILE = [RESULTS_DIR, PREFIX, num2str(NUM_COMPONENTS), '_results.mat'];
    save(RESULTS_FILE,'ground_truths','transforms','errors');
  end
end
