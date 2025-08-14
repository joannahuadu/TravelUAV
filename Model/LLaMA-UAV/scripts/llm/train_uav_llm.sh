#!/bin/bash
# change the root_dir and dataset_path to your own path

root_dir=/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV # TravelUAV directory
model_dir=$root_dir/Model/LLaMA-UAV
deepspeed \
    --include localhost:0 \
    --master_port 29101 \
    $model_dir/llamavid/train/train_uav/train_uav_notice.py \
    --data_path $root_dir/data/uav_dataset/trainset.json \
    --dataset_path /mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV_data/TravelUAV \
    --output_dir $model_dir/work_dirs/llama-vid-7b-pretrain-224-uav-full-data-lora32 \
    --deepspeed $model_dir/scripts/zero2.json \
    --model_name_or_path $model_dir/model_zoo/vicuna-7b-v1.5/ \
    --version imgsp_uav \
    --is_multimodal True \
    --vision_tower $model_dir/model_zoo/LAVIS/eva_vit_g.pth \
    --image_processor $model_dir/llamavid/processor/clip-patch14-224 \
    --mm_projector_type mlp2x_gelu \
    --tune_mm_mlp_adapter True \
    --tune_waypoint_predictor True \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --video_fps 1 \
    --bert_type "qformer_pretrain_freeze" \
    --num_query 32 \
    --pretrain_qformer $model_dir/model_zoo/LAVIS/instruct_blip_vicuna7b_trimmed.pth \
    --compress_type "mean" \
    --bf16 True \
    --num_train_epochs 10 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 1000 \
    --save_total_limit 1 \
    --learning_rate 5e-4 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 2048 \
    --gradient_checkpointing True \
    --dataloader_num_workers 8 \
    --lazy_preprocess True \
    --report_to wandb \
    --lora_enable True \
