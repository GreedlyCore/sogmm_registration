DATASET = 'cave';
FIRST_SCAN = 500;
LAST_SCAN = 890;
NUM_COMPONENTS = 100;
PREFIX = ['isoplanarhybrid_'];

[transforms, ground_truths] = process_dataset(DATASET, FIRST_SCAN, LAST_SCAN, PREFIX, NUM_COMPONENTS);
plot_trajectories(transforms, ground_truths, FIRST_SCAN, LAST_SCAN);
