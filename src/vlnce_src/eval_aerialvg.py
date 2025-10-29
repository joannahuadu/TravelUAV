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
from llamavid.train.train_uav.train_uav_cot import LazySupervisedDataset

from dataclasses import dataclass, field
from typing import Optional, Dict, List
from torch.utils.data import DataLoader


def eval(model_wrapper, dataset, eval_save_dir, batch_size=1, max_new_tokens=500):

    os.makedirs(eval_save_dir, exist_ok=True)
    save_file = f"{eval_save_dir}/eval_results.jsonl"

    dataloader = DataLoader(dataset, batch_size=batch_size)
    model = model_wrapper.model
    tokenizer = model_wrapper.tokenizer

    model.eval()

    with torch.no_grad(), open(save_file, "w", encoding="utf-8") as f:
        for batch in tqdm.tqdm(dataset, desc="Evaluating"):

            input_ids = batch["input_ids"].to(model.device)
            attention_mask = batch["attention_mask"].to(model.device)

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
                    max_new_tokens=max_new_tokens,
                )

            decoded_text = tokenizer.batch_decode(outputs, skip_special_tokens=True)

            for text in decoded_text:
                f.write(json.dumps(text, ensure_ascii=False) + "\n")
                f.flush()

    print(f"✅ Evaluation finished! Saved to: {save_file}")
        
if __name__ == "__main__":
    eval_save_path = args.eval_save_path
    data_args.dataset_path = args.dataset_path
    model_wrapper = TravelModelWrapper(model_args=model_args, data_args=data_args)
    
    eval_dataset = LazySupervisedDataset(tokenizer=model_wrapper.tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    
    eval(model_wrapper=model_wrapper,
         dataset=eval_dataset,
         eval_save_dir=eval_save_path)
