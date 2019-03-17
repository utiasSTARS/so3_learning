#!/usr/bin/env bash
python3 run_kitti_experiment.py --lr 5e-5 --seq 00 --total_epochs 30
python3 run_kitti_experiment.py --lr 5e-5 --seq 02 --total_epochs 30
python3 run_kitti_experiment.py --lr 5e-5 --seq 05 --total_epochs 30

