#!/bin/sh


export CUDA_VISIBLE_DEVICES=4
python3 run_gridsearch_exp.py --conf_file_path ./config/N_BEATS.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=5
python3 run_gridsearch_exp.py --conf_file_path ./config/N_HiTS.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=6
python3 run_gridsearch_exp.py --conf_file_path ./config/IC_PN_BEATS.yaml &
sleep 3