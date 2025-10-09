import argparse
import multiprocessing
import torch
import os
import tqdm
import numpy as np
from transformers.models.clip import CLIPImageProcessor
from typing import Dict, Optional, Sequence, List

from PIL import Image
from llava.model import *
from llamavid.model import *

RGB_FOLDER = ['frontcamera', 'leftcamera', 'rightcamera', 'rearcamera', 'downcamera']

def arg_parse():
  parser = argparse.ArgumentParser(description="split video clip")
  parser.add_argument("--root_dir",
                      default='/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV_data/TravelUAV',
                      help='path to your dataset root dir')
  parser.add_argument("--map_list",
                        default=['NewYorkCity', 'ModernCityMap', 'NYCEnvironmentMegapa', 'TropicalIsland', 'ModularPark', 'Carla_Town01', 'Carla_Town02', 'Carla_Town03', 'Carla_Town04','Carla_Town05', 'Carla_Town06', 'Carla_Town07', 'Carla_Town10HD', 'Carla_Town15',
                                 'BattlefieldKitDesert', 'BrushifyCountryRoads', 'BrushifyForestPack', 'BrushifyUrban', 'Japanese_Street', 'London_Street', 'NordicHarbour', 'WesterTown'],
                        # default=['BrushifyForestPack'],
                      nargs="+",
                      help='processed map name')
  parser.add_argument("--workers",
                      default=16,
                      help='multiprocessing workers num')
  opt = parser.parse_args()
  return opt

clip_config = {
  "crop_size": {
    "height": 224,
    "width": 224
  },
  "do_center_crop": True,
  "do_convert_rgb": True,
  "do_normalize": True,
  "do_rescale": True,
  "do_resize": True,
  "image_mean": [
    0.48145466,
    0.4578275,
    0.40821073
  ],
  "image_std": [
    0.26862954,
    0.26130258,
    0.27577711
  ],
  "resample": 3,
  "rescale_factor": 0.00392156862745098,
  "size": {
    "shortest_edge": 224
  }
}
args = arg_parse()
# processer = CLIPImageProcessor(**clip_config)
from transformers import AutoProcessor, AutoModelForCausalLM
import transformers
import sys
sys.path.append("/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/llamavid/train/train_uav/")
from train_uav_notice import ModelArguments, DataArguments, TrainingArguments
model_name_or_path = "/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/llava-v1.5-7b"

config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

bnb_model_from_pretrained_args = dict(torch_dtype=torch.bfloat16)
model = LlavaUAVForCausalLM.from_pretrained(
    model_name_or_path,
    use_angle_and_norm_loss=True,
    config=config,
    cache_dir=None,
    **bnb_model_from_pretrained_args
)

model_args = ModelArguments(
    model_name_or_path="/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/llava-v1.5-7b/",
    version="imgsp_uav",
    freeze_backbone=False,
    tune_mm_mlp_adapter=True,
    tune_waypoint_predictor=True,
    vision_tower="/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/clip-vit-large-patch14-336",
    image_processor=None,
    mm_vision_select_layer=-2,
    pretrain_mm_mlp_adapter=None,
    mm_projector_type="mlp2x_gelu",
    mm_use_im_start_end=False,
    mm_use_im_patch_token=False,
    mm_patch_merge_type="flat",
    mm_vision_select_feature="patch",
    bert_type="qformer_pretrain_freeze",
    num_query=32,
    pretrain_qformer=None,
    compress_type="mean",
    use_angle_and_norm_loss=True,
)

model.get_model().initialize_vision_modules(
    model_args=model_args,
    fsdp=""
)
vision_tower = model.get_vision_tower()
processer = vision_tower.image_processor

if __name__ == "__main__":
  def worker(traj_dir):
    traj_camera_list = []
    for idx,camera_name in enumerate(RGB_FOLDER):
      if len(os.listdir(os.path.join(traj_dir,camera_name)))==0:
        print(f"{os.path.join(traj_dir,camera_name)} no image")        
      traj_camera_list.append(sorted([os.path.join(traj_dir, camera_name, filename) for filename in os.listdir(os.path.join(traj_dir,camera_name))]))
    assert(len(traj_camera_list[0]) == len(traj_camera_list[1]) == len(traj_camera_list[2]) == len(traj_camera_list[3]))
    traj_frames = []
    for idx in range(len(traj_camera_list[0])):
      batch = []
      for iid in range(len(RGB_FOLDER)):
          batch.append(traj_camera_list[iid][idx])
      traj_frames.append(batch)
    traj_imgs = []
    for frame_imgs in traj_frames:
      images = [Image.open(img_path).convert('RGB') for img_path in frame_imgs]
      images = np.stack(images, axis=0)
      traj_imgs.append(images)
    if len(traj_imgs)!=0:
      imgs = np.array(traj_imgs).reshape(-1, 256, 256, 3)
      imgs = processer.preprocess(imgs, return_tensors='pt')['pixel_values'].to(dtype=torch.bfloat16)
      torch.save(imgs, os.path.join(traj_dir, 'rgb_imgs_336.tensor'))
    else:
      print(f"{traj_dir} no image")
  
  for map_name in args.map_list:
    directory_path = os.path.join(args.root_dir, map_name)
    print(directory_path)
    traj_list = []
    for traj in tqdm.tqdm(os.listdir(directory_path)):
        traj_dir = os.path.join(directory_path, traj)
        output_file = os.path.join(traj_dir, 'rgb_imgs.tensor')
        if not os.path.exists(output_file):
            print(f"{output_file} 未生成")
            # traj_list.append(traj_dir)
        else:
            a = torch.load(output_file)
            if a.shape[0]==0:
                print(f"{output_file} is 0, {a.shape}")
            
    # for traj in traj_list:
    #   worker(traj)
    # # with multiprocessing.Pool(args.workers) as p:
    #   # r = list(tqdm.tqdm(p.imap_unordered(worker, traj_list), total=len(traj_list)))
    print(directory_path, 'finished.')