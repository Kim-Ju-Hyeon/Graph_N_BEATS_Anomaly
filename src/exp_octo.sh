#!/bin/sh

export CUDA_VISIBLE_DEVICES=0
python3 run_exp.py --conf_file_path ./config/IC_PN_BEATS/gnn.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 run_exp.py --conf_file_path ./config/IC_PN_BEATS/none_gnn.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=2
python3 run_exp.py --conf_file_path ./config/N_BEATS_G/gnn.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=3
python3 run_exp.py --conf_file_path ./config/N_BEATS_G/none_gnn.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=4
python3 run_exp.py --conf_file_path ./config/N_BEATS_I/gnn.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=5
python3 run_exp.py --conf_file_path ./config/N_BEATS_I/none_gnn.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=6
python3 run_exp.py --conf_file_path ./config/N_HiTS/gnn.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=7
python3 run_exp.py --conf_file_path ./config/N_HiTS/none_gnn.yaml &
sleep 3

