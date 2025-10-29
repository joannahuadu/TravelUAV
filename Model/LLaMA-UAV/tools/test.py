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
                        # default=['NewYorkCity', 'ModernCityMap', 'NYCEnvironmentMegapa', 'TropicalIsland', 'ModularPark', 'Carla_Town01', 'Carla_Town02', 'Carla_Town03', 'Carla_Town04','Carla_Town05', 'Carla_Town06', 'Carla_Town07', 'Carla_Town10HD', 'Carla_Town15',
                                #  'BattlefieldKitDesert', 'BrushifyCountryRoads', 'BrushifyForestPack', 'BrushifyUrban', 'Japanese_Street', 'London_Street', 'NordicHarbour', 'WesterTown'],
                        default = ['Carla_Town04','Carla_Town05', 'Carla_Town06', 'Carla_Town07', 'Carla_Town10HD', 'Carla_Town15',
                                 'BattlefieldKitDesert', 'BrushifyCountryRoads', 'BrushifyForestPack', 'BrushifyUrban', 'Japanese_Street', 'London_Street', 'NordicHarbour', 'WesterTown'],
                        # default=['BrushifyForestPack'],
                      nargs="+",
                      help='processed map name')
  parser.add_argument("--workers",
                      default=16,
                      help='multiprocessing workers num')
  opt = parser.parse_args()
  return opt
args = arg_parse()
if __name__ == "__main__":
  for map_name in args.map_list:
    directory_path = os.path.join(args.root_dir, map_name)
    print(directory_path)
    traj_list = []
    for traj in tqdm.tqdm(os.listdir(directory_path)):
        traj_dir = os.path.join(directory_path, traj)
        output_file = os.path.join(traj_dir, 'rgb_imgs_336.tensor')
        if not os.path.exists(output_file):
            print(f"{output_file} 未生成")
        else:
            a = torch.load(output_file)
            if a.shape[0]==0:
                print(f"{output_file} is 0, {a.shape}")
            
    print(directory_path, 'finished.')
  
# for idx,camera_name in enumerate(RGB_FOLDER):
#   if len(os.listdir(os.path.join(traj_dir,camera_name)))==0:
#         print(f"{os.path.join(traj_dir,camera_name)} no image")
#         return
#       traj_camera_list.append(sorted([os.path.join(traj_dir, camera_name, filename) for filename in os.listdir(os.path.join(traj_dir,camera_name))]))
#     assert(len(traj_camera_list[0]) == len(traj_camera_list[1]) == len(traj_camera_list[2]) == len(traj_camera_list[3]))
#     traj_frames = []
#     for idx in range(len(traj_camera_list[0])):
#       batch = []
#       for iid in range(len(RGB_FOLDER)):
#           batch.append(traj_camera_list[iid][idx])
#       traj_frames.append(batch)
#     traj_imgs = []
#     for frame_imgs in traj_frames:
#       images = [Image.open(img_path).convert('RGB') for img_path in frame_imgs]
#       images = np.stack(images, axis=0)
#       traj_imgs.append(images)
