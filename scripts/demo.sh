#!/bin/bash

echo "Train - Random Deployment with Mobility"
python3 ./random_deployment.py --json-file "train_K5_N10_shadow10_episode2-5000_travel50000_vmax2_5" --num-sim 0 &
wait
 
echo "Train - DDPG with Mobility"
python3 ./trainDDPG.py --json-file "train_K5_N10_shadow10_episode2-5000_travel50000_vmax2_5" --num-sim 0 --json-file-policy "ddpg200_100_50" &
wait

echo "Get FP and WMMSE Benchmarks"
python3 ./get_benchmarks.py --json-file "train_K5_N10_shadow10_episode2-5000_travel50000_vmax2_5" --num-sim 0 &
wait

echo "Results - Training"
python3 -i ./train_results.py --json-file 'train_K5_N10_shadow10_episode2-5000_travel50000_vmax2_5' --num-sim 0 --json-file-policy 'ddpg200_100_50'