#!/bin/sh


export CUDA_VISIBLE_DEVICES=0
python3 run_gridsearch_exp.py --conf_file_path ./config/N_BEATS.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=1
python3 run_gridsearch_exp.py --conf_file_path ./config/N_HiTS.yaml &
sleep 3

export CUDA_VISIBLE_DEVICES=2
python3 run_gridsearch_exp.py --conf_file_path ./config/IC_PN_BEATS.yaml &
sleep 3