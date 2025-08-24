#!/bin/bash
# change the dataset_path to your own path

root_dir=/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV # TravelUAV directory
model_dir=$root_dir/Model/LLaMA-UAV


CUDA_VISIBLE_DEVICES=8 python -u $root_dir/src/vlnce_src/eval.py \
    --run_type eval \
    --name TravelLLM \
    --gpu_id 0 \
    --simulator_tool_port 25000 \
    --DDP_MASTER_PORT 80005 \
    --batchSize 1 \
    --always_help True \
    --use_gt True \
    --maxWaypoints 200 \
    --dataset_path /mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV_data/TravelUAV/ \
    --eval_save_path /mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV_data/TravelUAV/eval_closeloop_qwen0.5b128_1/ \
    --model_path $model_dir/work_dirs/qwen-vid-0.5b-pretrain-224-uav-full-data-lora32_bs128 \
    --model_base $model_dir/model_zoo/Qwen2-0.5B \
    --vision_tower $model_dir/model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor $model_dir/llamavid/processor/clip-patch14-224 \
    --traj_model_path $model_dir/work_dirs/train_traj_model_bs128_drop0.1_lr_5e-4 \
    --eval_json_path $root_dir/data/uav_dataset/seen_valset.json \
    --map_spawn_area_json_path $root_dir/data/meta/map_spawnarea_info.json \
    --object_name_json_path $root_dir/data/meta/object_description.json \
    --groundingdino_config $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py \
    --groundingdino_model_path $root_dir/src/model_wrapper/utils/GroundingDINO/groundingdino_swint_ogc.pth