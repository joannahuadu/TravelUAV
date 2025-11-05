#!/bin/bash

# ROOT_DIR='/path/to/your/root/eval/result/dir' # ROOT_DIR="./closeloop_eval/"
# ANALYSIS_LIST="eval dir list" # ANALYSIS_LIST="baseline baseline2"
# PATH_TYPE_LIST="full easy hard" # full easy hard

ROOT_DIR="/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV_data/TravelUAV/eval_closeloop_llava_llama_attn_bs128_all_fast"
ANALYSIS_LIST="."
PATH_TYPE_LIST="full"

# CUDA_VISIBLE_DEVICES=0 python3 ./AirVLN/utils/metric.py \
CUDA_VISIBLE_DEVICES=0 python3 /mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/utils/metric.py \
    --root_dir $ROOT_DIR \
    --analysis_list $ANALYSIS_LIST \
    --path_type_list $PATH_TYPE_LIST
