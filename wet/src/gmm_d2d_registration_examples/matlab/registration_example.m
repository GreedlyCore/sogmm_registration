CPWD = pwd;
SANDBOX_NAME = 'gira3d-registration';
matches = split(CPWD, SANDBOX_NAME);
GIRA3D_REGISTRATION_SANDBOX = [matches{1}, SANDBOX_NAME];
DATA_DIR = [GIRA3D_REGISTRATION_SANDBOX, '/data/rgbd_dataset_freiburg3_long_office_household/100_components/'];
PCLD_DIR = [GIRA3D_REGISTRATION_SANDBOX, '/data/rgbd_dataset_freiburg3_long_office_household/pointclouds/'];
RESULTS_DIR = [DATA_DIR, 'results/'];

target_file = [DATA_DIR, '1200.gmm'];
source_file = [DATA_DIR, '1260.gmm'];

target_pcld = load([PCLD_DIR, '1200.txt']);
source_pcld = load([PCLD_DIR, '1260.txt']);

if ~exist(RESULTS_DIR, 'dir')
  mkdir(RESULTS_DIR)
end

%%%%%%%%%%% Isoplanar Registration
x_opt = zeros(6,1);
[x_opt, score] = isoplanar_registration(source_file, target_file, zeros(6,1));
assert(max(abs(x_opt - [-0.7795; -0.2672; 0.5195; -0.0513; 0.5727; 0.3276])) < 0.001)

%%%%%%%%%%% Anisotropic Registration
[x_opt, score] = anisotropic_registration(source_file, target_file, x_opt);
assert(max(abs(x_opt - [-0.4217; -0.2275; 0.4124; -0.0441; 0.7002; 0.4001])) < 0.001)
Rotation = axang2rotm([x_opt(4:6) / norm(x_opt(4:6)); norm(x_opt(4:6))]');
translation = x_opt(1:3);
disp('Tests passed')

%%%%%%%%%%% Visualization
gmm_target = GMM3();
gmm_target.load(target_file);

gmm_source = GMM3();
gmm_source.load(source_file);

figure;
hold on; pcshow(target_pcld);
colormap(flipud(jet));
hold on; pcshow(source_pcld);
colormap(flipud(jet));
title('Before Registration');
axis equal;
view(0, -90);

figure;
hold on; pcshow(target_pcld);
colormap(flipud(jet));
source_pcld_transformed = Rotation * transpose(source_pcld) + repmat(translation, 1, size(source_pcld,1));
hold on; pcshow(source_pcld_transformed');
colormap(flipud(jet));
title('After Registration');
axis equal;
view(0, -90);
