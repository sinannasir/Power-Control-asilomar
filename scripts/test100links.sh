#!/bin/bash

echo "Only execute fig1 after train.sh is done"
echo "Test - Deployment"
python3 ./random_deployment.py --json-file "test_K20_N100_shadow10_episode5-2500_travel0_vmax2_5" &
wait

echo "Test - Policy"
python3 ./testDDPG.py --json-file "test_K20_N100_shadow10_episode5-2500_travel0_vmax2_5" --json-files-train "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" --json-file-policy-train "ddpg200_100_50" &
wait

echo "Test - Get FP WMMSE Benchmarks"
python3 ./get_benchmarks.py --json-file "test_K20_N100_shadow10_episode5-2500_travel0_vmax2_5" &
wait

echo "RESULTS:"
python3 ./test_results.py --json-file "test_K20_N100_shadow10_episode5-2500_travel0_vmax2_5" --json-file-train "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" --json-file-policy-train "ddpg200_100_50"

