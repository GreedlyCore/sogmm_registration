DATASET = 'mine_001_part3';
FIRST_SCAN = 1;
LAST_SCAN = 320;
NUM_COMPONENTS = 100;
PREFIX = ['isoplanarhybrid_'];

[transforms, ground_truths] = process_dataset(DATASET, FIRST_SCAN, LAST_SCAN, PREFIX, NUM_COMPONENTS);
plot_trajectories(transforms, ground_truths, FIRST_SCAN, LAST_SCAN);
