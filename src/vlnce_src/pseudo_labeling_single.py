import os
from pathlib import Path
import sys
import time
import json
import shutil
import random

import cv2
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import tqdm

sys.path.append(str(Path(str(os.getcwd())).resolve()))
from utils.logger import logger
from utils.utils import *
from src.model_wrapper.travel_llm import TravelModelWrapper
from src.model_wrapper.base_model import BaseModelWrapper
from src.common.param import args, model_args, data_args
from env_uav import AirVLNENV
from assist import Assist
from src.vlnce_src.closeloop_util import EvalBatchState, BatchIterator, setup, CheckPort, initialize_env_eval, is_dist_avail_and_initialized
from llamavid.train.train_uav.train_uav_cot import LazySupervisedDataset, preprocess

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from torch.utils.data import DataLoader
import re

def detect_bbox_over_five_views(images: Dict[str, str], text_prompt: str, model) -> Tuple[Optional[str], Optional[List[float]]]:
    bboxes = {}
    for view in PERSPECTIVES:
        if view not in images:
            continue
        p = images[view]
        bbox = detect_bbox(p, text_prompt, model)
        if bbox is not None:
            bboxes[view] = bbox
    return bboxes

def load_groundingdino_model(device):
    cfg = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/src/model_wrapper/utils/GroundingDINO/groundingdino/config/GroundingDINO_SwinB_cfg.py"
    weight = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/src/model_wrapper/utils/GroundingDINO/groundingdino_swinb_cogcoor.pth"
    model = load_model(cfg, weight)
    model.to(device=device)
    return models


def eval(model_wrapper, dataset, eval_save_dir, batch_size=1, max_new_tokens=500, r1_gen=False):
    dino = load_groundingdino_model(device)
    os.makedirs(eval_save_dir, exist_ok=True)
    save_file = f"{eval_save_dir}/eval_results.jsonl"

    if os.path.exists(save_file):
        with open(save_file, "r", encoding="utf-8") as fr:
            done_samples = sum(1 for _ in fr)
    else:
        done_samples = 0

    print(f"➡️ Resume: Already evaluated {done_samples} samples, skipping them...")
    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    model.eval()
    
    current_idx = 0
    if r1_gen:
        max_new_tokens = 10
    with torch.no_grad(), open(save_file, "a", encoding="utf-8") as f:
        for batch in tqdm.tqdm(dataset, desc="Evaluating"):
            if current_idx < done_samples:
                current_idx += 1
                continue
            
            input_ids = batch["input_ids"].to(model.device)
            if 'attention_mask' in batch: 
                attention_mask = batch["attention_mask"].to(model.device)
            else:
                attention_mask=input_ids.ne(tokenizer.pad_token_id).to(model.device)

            if "pixel_values" in batch:
                pixel_values = batch["pixel_values"].to(model.device)
                image_sizes = batch["image_sizes"].to(model.device)
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    pixel_values=pixel_values,
                    image_sizes=image_sizes,
                    max_new_tokens=max_new_tokens,
                    cot_eval = True,
                    use_cache = False,
                )
            else:
                outputs = model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    images=batch['image'].unsqueeze(0).to(model.device),
                    prompts=[batch['prompt']],
                    max_new_tokens=max_new_tokens,
                    cot_eval = True,
                    use_cache = False,
                )
                outputs[outputs == -200] = 0
            decoded_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)
            subgoal = re.sub(r"(?i)^\s*subgoal\s*:\s*", "", decoded_text).strip()
            images = {"front": batch['image'][0], "left": batch['image'][1], "right": batch['image'][2], "rear": batch['image'][3], "down": batch['image'][4]}
            bboxes = detect_bbox_over_five_views(images, subgoal, dino)
            f.write(str(batch['view'])+ "--")
            for text in decoded_text:
                f.write(json.dumps(text, ensure_ascii=False) + "\n")
                f.flush()
                
            current_idx += 1
    
    print(f"✅ Evaluation finished! {current_idx}/{len(dataset)} samples done.")
    print(f"✅ Evaluation finished! Saved to: {save_file}")
        
if __name__ == "__main__":
    eval_save_path = args.eval_save_path
    data_args.dataset_path = args.dataset_path
    model_wrapper = TravelModelWrapper(model_args=model_args, data_args=data_args)
    if "aerial" in data_args.data_path:
        data_args.image_processor = model_wrapper.image_processor
    eval_dataset = LazySupervisedDataset(tokenizer=model_wrapper.tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args, 
                                r1=data_args.r1, 
                                r1_gen=data_args.r1_gen,
                                labeling=data_args.labeling)
    
    eval(model_wrapper=model_wrapper,
         dataset=eval_dataset,
         eval_save_dir=eval_save_path,
         r1_gen=data_args.r1_gen)
