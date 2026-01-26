DATASET = 'rgbd_dataset_freiburg3_long_office_household';
FIRST_SCAN = 1;
LAST_SCAN = 2508;
NUM_COMPONENTS = 100;
PREFIX = ['isoplanarhybrid_'];

[transforms, ground_truths] = process_dataset(DATASET, FIRST_SCAN, LAST_SCAN, PREFIX, NUM_COMPONENTS);
plot_trajectories(transforms, ground_truths, FIRST_SCAN, LAST_SCAN);
