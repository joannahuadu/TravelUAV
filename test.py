import torch
import os
model_path = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/work_dirs/llama-vid-7b-pretrain-224-uav-full-data-lora32_bs128"
non_lora_weights1 = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
# non_lora_weights2 = torch.load(os.path.join(model_path, "checkpoint-2500", 'non_lora_trainables.bin'), map_location='cpu')
mm_projector_weights1 = torch.load(os.path.join("/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/work_dirs/llama-vid-7b-pretrain-224-uav-full-data-lora32_bs128/checkpoint-500/non_lora_trainables.bin"), map_location='cpu')
# mm_projector_weights2 = torch.load("/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/work_dirs/llama-vid-7b-pretrain-224-uav-full-data-lora32_bs128/mm_projector/checkpoint-2500.bin", map_location='cpu')

print(mm_projector_weights1.keys())
print(non_lora_weights1.keys())
k = 'base_model.model.lm_head.weight'
ke = 'lm_head.weight'
print(torch.sum(non_lora_weights1[k] == mm_projector_weights1[ke]))
print(non_lora_weights1[k].shape.numel())
print(non_lora_weights1[k])