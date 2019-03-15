#!/usr/bin/env bash
python3 run_kitti_experiment.py --lr 5e-3 --seq 00 --total_epochs 50
python3 run_kitti_experiment.py --lr 5e-3 --seq 02 --total_epochs 50
python3 run_kitti_experiment.py --lr 5e-3 --seq 05 --total_epochs 50

