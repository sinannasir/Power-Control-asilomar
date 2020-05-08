#!/bin/bash

echo "Only execute fig1 after train.sh is done"
echo "Test - Deployment"
python3 ./random_deployment.py --json-file "test_K10_N20_shadow10_episode5-2500_travel0_vmax2_5" &
wait
echo "Test - Policy"
#python3 ./testDDPG.py --json-file "test_K10_N20_shadow10_episode5-2500_travel0_vmax2_5" --json-files-train "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" "train_K10_N20_shadow10_episode10-5000_travel0_fd10" --json-file-policy-train "ddpg200_100_50" &
# Run parallel instead
python3 ./testDDPG.py --json-file "test_K10_N20_shadow10_episode5-2500_travel0_vmax2_5" --json-files-train "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" --json-file-policy-train "ddpg200_100_50" & 
python3 ./testDDPG.py --json-file "test_K10_N20_shadow10_episode5-2500_travel0_vmax2_5" --json-files-train "train_K10_N20_shadow10_episode10-5000_travel0_fd10" --json-file-policy-train "ddpg200_100_50" &
wait

echo "Test - Get FP WMMSE Benchmarks"
python3 ./get_benchmarks.py --json-file "test_K10_N20_shadow10_episode5-2500_travel0_vmax2_5" &
wait

echo "RESULTS:"
echo "Interactive mode, draw Fig 4"
python3 -i ./fig4.py --json-file "test_K10_N20_shadow10_episode5-2500_travel0_vmax2_5" --json-file-wmobility "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" --json-file-womobility "train_K10_N20_shadow10_episode10-5000_travel0_fd10" --json-file-policy-train "ddpg200_100_50"
