#!/bin/sh

export CUDA_VISIBLE_DEVICES=1
python3 run_exp.py --conf_file_path ./config/IC_PN_BEATS.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=2
python3 run_exp.py --conf_file_path ./config/IC_PN_BEATS_2.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=3
python3 run_exp.py --conf_file_path ./config/IC_PN_BEATS_3.yaml &
sleep 3