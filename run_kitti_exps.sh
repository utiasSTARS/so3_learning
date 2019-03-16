#!/usr/bin/env bash
python3 run_kitti_experiment.py --lr 2e-3 --seq 00 --total_epochs 25
python3 run_kitti_experiment.py --lr 2e-3 --seq 02 --total_epochs 25
python3 run_kitti_experiment.py --lr 2e-3 --seq 05 --total_epochs 25

