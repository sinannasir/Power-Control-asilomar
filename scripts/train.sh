#!/bin/bash

echo "Train - Random Deployment with Mobility"
python3 ./random_deployment.py --json-file "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" --num-sim 0 &
python3 ./random_deployment.py --json-file "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" --num-sim 1 &
python3 ./random_deployment.py --json-file "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" --num-sim 2 &
python3 ./random_deployment.py --json-file "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" --num-sim 3 &
python3 ./random_deployment.py --json-file "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" --num-sim 4 &
wait
 
echo "Train - DDPG with Mobility"
python3 ./trainDDPG.py --json-file "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" --num-sim 0 --json-file-policy "ddpg200_100_50" &
python3 ./trainDDPG.py --json-file "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" --num-sim 1 --json-file-policy "ddpg200_100_50" &
python3 ./trainDDPG.py --json-file "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" --num-sim 2 --json-file-policy "ddpg200_100_50" &
python3 ./trainDDPG.py --json-file "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" --num-sim 3 --json-file-policy "ddpg200_100_50" &
python3 ./trainDDPG.py --json-file "train_K10_N20_shadow10_episode10-5000_travel50000_vmax2_5" --num-sim 4 --json-file-policy "ddpg200_100_50" &
wait

echo "Train - Random Deployment without Mobility"
python3 ./random_deployment.py --json-file "train_K10_N20_shadow10_episode10-5000_travel0_fd10" --num-sim 0 &
python3 ./random_deployment.py --json-file "train_K10_N20_shadow10_episode10-5000_travel0_fd10" --num-sim 1 &
python3 ./random_deployment.py --json-file "train_K10_N20_shadow10_episode10-5000_travel0_fd10" --num-sim 2 &
python3 ./random_deployment.py --json-file "train_K10_N20_shadow10_episode10-5000_travel0_fd10" --num-sim 3 &
python3 ./random_deployment.py --json-file "train_K10_N20_shadow10_episode10-5000_travel0_fd10" --num-sim 4 &
wait
 
echo "Train - DDPG without Mobility"
python3 ./trainDDPG.py --json-file "train_K10_N20_shadow10_episode10-5000_travel0_fd10" --num-sim 0 --json-file-policy "ddpg200_100_50" &
python3 ./trainDDPG.py --json-file "train_K10_N20_shadow10_episode10-5000_travel0_fd10" --num-sim 1 --json-file-policy "ddpg200_100_50" &
python3 ./trainDDPG.py --json-file "train_K10_N20_shadow10_episode10-5000_travel0_fd10" --num-sim 2 --json-file-policy "ddpg200_100_50" &
python3 ./trainDDPG.py --json-file "train_K10_N20_shadow10_episode10-5000_travel0_fd10" --num-sim 3 --json-file-policy "ddpg200_100_50" &
python3 ./trainDDPG.py --json-file "train_K10_N20_shadow10_episode10-5000_travel0_fd10" --num-sim 4 --json-file-policy "ddpg200_100_50" &
wait

