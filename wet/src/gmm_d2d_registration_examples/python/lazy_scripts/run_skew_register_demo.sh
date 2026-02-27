#!/usr/bin/env bash
cd "$(dirname "$0")/.."

# python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx 0.5 --ty 0.5 --tz 0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
# python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx 0.5 --ty 0.5 --tz 0.5 --roll 0.05 --pitch 0.02 --yaw 0.1
# python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx 0.5 --ty 0.5 --tz -0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
# python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx -0.5 --ty -0.5 --tz 0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
# python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx -0.5 --ty -0.5 --tz -0.3 --roll 0.05 --pitch 0.02 --yaw 0.1

python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx 0.1 --ty 0.1 --tz 0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx 0.1 --ty 0.1 --tz 0.1 --roll 0.05 --pitch 0.02 --yaw 0.1
python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx 0.1 --ty 0.1 --tz -0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx -0.1 --ty -0.1 --tz 0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
python skew_and_register_demo.py --gmm_dir ./100_components_24021725_gmm --tx -0.1 --ty -0.1 --tz -0.3 --roll 0.05 --pitch 0.02 --yaw 0.1
