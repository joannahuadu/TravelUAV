# from llamavid.train.llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn
# replace_llama_attn_with_flash_attn()
# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
# ------------------------------------------------------------------------
# Modified from LLaVA (https://github.com/haotian-liu/LLaVA)
# ------------------------------------------------------------------------
import os
import copy
import random
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List, Union, Any
import pickle
import math
import time

import torch

import transformers

# import llamavid.qwen2
from llamavid.constants import IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, WAYPOINT_INPUT_TOKEN, WAYPOINT_LABEL_TOKEN, DEFAULT_WP_TOKEN, DEFAULT_HISTORY_TOKEN, WP_TOKEN_INDEX, HIS_TOKEN_INDEX
from torch.utils.data import Dataset
from llamavid.train.llava_trainer import LLaVATrainer

from llamavid import conversation as conversation_lib
from llamavid.model import *
from llava.mm_utils import tokenizer_image_token

from PIL import Image
import numpy as np
from decord import VideoReader, cpu

from transformers.utils import logging
from transformers import AutoProcessor
import re
from peft import PeftModel
from safetensors.torch import load_file
logger = logging.get_logger(__name__)

local_rank = None

def load_model(model, model_path):
    print(f"Loading LoRA weights from {model_path}")
    if isinstance(model, PeftModel):
        adapter_path = os.path.join(model_path, "adapter_model.safetensors")
        lora_state = load_file(adapter_path, device="cpu")
        model.load_state_dict(lora_state, strict=False)
    else:
        model = PeftModel.from_pretrained(model, model_path)
    non_lora_weights = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
    model.load_state_dict(non_lora_weights, strict=False)    
    mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
    model.load_state_dict(mm_projector_weights, strict=False)
    
    return model

# This is the resize function of Qwen2.5-VL
def smart_resize(
    height: int, width: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 14 * 14 * 4 * 1280
):
    """Rescales the image so that the following conditions are met:
    1. Both dimensions (height and width) are divisible by 'factor'.
    2. The total number of pixels is within the range ['min_pixels', 'max_pixels'].
    3. The aspect ratio of the image is maintained as closely as possible.
    """
    if height < factor or width < factor:
        raise ValueError(f"height:{height} or width:{width} must be larger than factor:{factor}")
    elif max(height, width) / min(height, width) > 200:
        raise ValueError(
            f"absolute aspect ratio must be smaller than 200, got {max(height, width) / min(height, width)}"
        )
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (height * width))
        h_bar = math.ceil(height * beta / factor) * factor
        w_bar = math.ceil(width * beta / factor) * factor
    return h_bar, w_bar


def convert_to_qwen25vl_format(bbox, orig_height, orig_width, factor=28, min_pixels=56*56, max_pixels=14*14*4*128*128):
    new_height, new_width = smart_resize(orig_height, orig_width, factor, min_pixels, max_pixels)
    scale_w = new_width / orig_width
    scale_h = new_height / orig_height
    
    x1, y1, x2, y2 = bbox
    x1_new = round(x1 * scale_w)
    y1_new = round(y1 * scale_h)
    x2_new = round(x2 * scale_w)
    y2_new = round(y2 * scale_h)
    
    x1_new = max(0, min(x1_new, new_width - 1))
    y1_new = max(0, min(y1_new, new_height - 1))
    x2_new = max(0, min(x2_new, new_width - 1))
    y2_new = max(0, min(y2_new, new_height - 1))
    
    return [x1_new, y1_new, x2_new, y2_new]

def rank0_print(*args):
    if local_rank == 0:
        print(*args)

def rotation_matrix_from_vector(x, y):
    v_x = np.array([x, y, 0])
    v_x = v_x / np.linalg.norm(v_x)
    v_y = np.array([-v_x[1], v_x[0], 0])
    v_y = v_y / np.linalg.norm(v_y)
    v_z = np.array([0, 0, 1])
    rotation_matrix = np.column_stack((v_x, v_y, v_z))
    return rotation_matrix

def transform_point(point, rotation_matrix):
    return np.dot(point, rotation_matrix)

def waypoint2angle(waypoints):
    angle_and_norm = []
    for waypoint in waypoints:
        norm = np.linalg.norm(waypoint)
        angle = waypoint / (norm + 1e-6)
        angle_and_norm.append([angle[0], angle[1], angle[2], norm])
    return np.array(angle_and_norm)

@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    tune_waypoint_predictor: bool = field(default=True)
    vision_tower: Optional[str] = field(default=None)
    image_processor: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")
    bert_type: Optional[str] = field(default="qformer_pretrain")
    num_query: Optional[int] = field(default=32)
    pretrain_qformer: Optional[str] = field(default=None)
    compress_type: Optional[str] = field(default=None)
    use_angle_and_norm_loss: bool = field(default=True)
    cot: bool = field(default=False)


@dataclass
class DataArguments:
    # data_path: str = field(default=None,
    #                        metadata={"help": "Path to the training data json."})
    data_path: List[str] = field(default_factory=list,
        metadata={
            "help": "Path(s) to one or multiple training data json files.",
            "nargs": "+"
        }
    )
    dataset_path: str = field(default=None,
                           metadata={"help": "Path to the raw data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    video_token: Optional[int] = field(default=2)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: Optional[bool] = field(default=True)
    dataset_state: Optional[Dict[str, List[str]]] = field(default_factory=
        lambda: {
            "traveluav": ['front', 'left', 'right', 'rear', 'down'],
            "arivln": ['current'],
            "aerialvg": ['current']
        })
    bbox_scale: bool = field(default=False)
    use_assist: bool = field(default=False)

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of `fp4` or `nf4`."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    group_by_modality_length: bool = field(default=False)
    lr_multi: Optional[str] = field(default=None)
    resume: str = field(default=None,
                           metadata={"help": "Path to the raw data."})


def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return

def find_all_linear_names(model):
    cls = torch.nn.Linear
    lora_module_names = set()
    # multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'vlm_att']
    multimodal_keywords = ['mm_projector', 'vision_tower', 'vision_resampler', 'vlm_att', 'waypoint_emb', 'waypoints_fc', 'waypoints_predictor',
                         'waypoints_output', 'history_predictor', 'history_preprocessor', 'is_help_predictor', 'visual'] # end_predictor
    
    for name, module in model.named_modules():
        if any(mm_keyword in name for mm_keyword in multimodal_keywords):
            continue
        if isinstance(module, cls):
            names = name.split('.')
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])

    if 'lm_head' in lora_module_names: # needed for 16-bit
        lora_module_names.remove('lm_head')
    return list(lora_module_names)

def find_all_exclude_names(model, keys):
    import re
    cls = torch.nn.Linear
    _EXCLUDE_KEYWORDS = (
        "visual", "vision_tower", "mm_projector",
        "vision_resampler", "vlm_att", "image", "vision"
    )

    ks = tuple(_EXCLUDE_KEYWORDS)
    ls = set(keys)
    out = []
    for name, m in model.named_modules():
        if not isinstance(m, cls):
            continue
        if re.search(rf"(?:^|[._])({'|'.join(map(re.escape, ks))})(?:[._]|$)", name):
            if name.split(".")[-1] in ls:
                out.append(name)
    return out


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        # keys_to_match = ['mm_projector']
        keys_to_match = ['mm_projector', 'vision_resampler', 'vlm_att', 'waypoint_emb', 'waypoints_fc', 'waypoints_predictor',
                         'waypoints_output', 'history_predictor', 'history_preprocessor', 'is_help_predictor', 'embed_tokens', 'lm_head', 'visual.merger', 'multi_modal_projector'] # 'end_predictor',
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg
        
def smarter_tokenizer_and_embedding_resize(
    special_tokens_list: List,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_tokens(special_tokens_list, special_tokens=True)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        model.get_input_embeddings().weight.requires_grad_(True)
        model.get_output_embeddings().weight.requires_grad_(True)
        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments,
    assist: str,
    dataset_name: str,
    stage = None,
    delta = None,
    cur = None,
    eval: bool = False,
) -> Dict:
    """
        process image token's representation
    """
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources
    
    prefix = "Navigation goal: "
    suffix = "\n"
    if dataset_name == "traveluav":
        if data_args.use_assist:
            prefix = '\n\nStage:' + stage + '\n\nPrevious displacement:' + delta  + '\n\nCurrent position:' + cur + "\n\nInstruction:These five images respectively come from five perspectives: <image>\nfront, <image>\nleft, <image>\nright, <image>\nrear, <image>\ndown. "
        else:
            prefix = "\n\nInstruction:These five images respectively come from five perspectives: <image>\nfront, <image>\nleft, <image>\nright, <image>\nrear, <image>\ndown. "
        if "Subgoal" in assist:
            suffix = "\nPlease identify useful subgoals and their bounding boxes in each image (if any). Then control the drone and find the target."
        if eval:
            suffix = "\nPlease identify useful subgoals and their bounding boxes in each image (if any). Then control the drone and find the target. ASSISTANT: Subgoal:"
    elif dataset_name == "airvln":
        prefix = "\n\nInstruction:Given one current image.<image>\n"
        if "Subgoal" in assist:
            suffix = "\nPlease identify useful subgoals and their bounding boxes (if any). Then control the drone and find the target."
    elif dataset_name == "aerialvg":
        prefix = "\n\nInstruction:Given one current image.<image>\n"
        suffix = "\nPlease identify useful subgoals and their bounding boxes (if any)."
        if eval:
            suffix = "\nPlease identify useful subgoals and their bounding boxes (if any). ASSISTANT: Subgoal:"
    else:
        raise ValueError(
                f"Unsupported conversation version: {conversation_lib.default_conversation.version}"
        )
        
    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['prompt'] = copy.deepcopy(sentence['value'])
                sentence['value'] = prefix + sentence['value'] + suffix
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            # replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            # sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)
            
            if sentence['from'] == "gpt":
                sentence["value"] = assist

    return sources

def _build_messages(item: Sequence[str], images: Union[List[str], str], system_message: Dict[str, Any]) -> List[Dict[str, Any]]:
    if isinstance(images, str):
        images = [images]

    image_pool = [
        {"type": "image", "image": img} for img in images
    ]
    
    messages = [system_message]
    for turn in item:
        role = "user" if turn["from"] == "human" else "assistant"
        text: str = turn["value"]

        if role == "user":
            content = []
            text_parts = re.split(r"(<image>|<video>)", text)

            for seg in text_parts:
                if seg == "<image>":
                    if not image_pool:
                        raise ValueError(
                            "Number of <image> placeholders exceeds the number of provided images"
                        )
                    content.append(image_pool.pop(0))
                elif seg.strip():
                    content.append({"type": "text", "text": seg.strip()})

            messages.append({"role": role, "content": content})
        else:
            messages.append({"role": role, "content": [{"type": "text", "text": text}]})

    if image_pool:
        raise ValueError(
            f"{len(image_pool)} image(s) remain unused (not consumed by placeholders)"
        )

    return messages

def preprocess_imgsp_qwen(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: List[Image.Image],
    img_token: str = '<image>',
    refine_prompt: bool = False,
) -> Dict:
    processor = AutoProcessor.from_pretrained("/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/Qwen2.5-VL-7B-Instruct")
    system_message = {
        "role": "system",
        "content": [
            {"type": "text", "text": "You are a helpful assistant. The assistant is a navigation model that output the uav waypoints according to the user's instructions and images."}
        ]
    }
    messages = _build_messages(sources[0], has_image, system_message)

    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        if input_ids_flat[pos] == 77091:
            ans_start = pos + 2
            ans_end = ans_start
            while ans_end < L and input_ids_flat[ans_end] != 151645:
                ans_end += 1
            if ans_end < L:
                labels[0, ans_start : ans_end + 2] = input_ids[
                    0, ans_start : ans_end + 2
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids

    if 'Control' in sources[0][1]['value']:
        input_ids_pad_wp = torch.zeros(input_ids.shape[0], input_ids.shape[1] + 1, dtype=torch.long)
        input_ids_pad_wp[:, :-2] = input_ids[:, :-1]
        input_ids_pad_wp[:, -2] = WAYPOINT_INPUT_TOKEN
        input_ids_pad_wp[:, -1] = input_ids[:, -1]
        full_result['input_ids'] = input_ids_pad_wp

        targets_pad_wp = torch.zeros(labels.shape[0], labels.shape[1] + 1, dtype=torch.long)
        targets_pad_wp[:, :-2] = labels[:, :-1]
        targets_pad_wp[:, -2] = WAYPOINT_LABEL_TOKEN
        targets_pad_wp[:, -1] = labels[:, -1]
        full_result['labels'] = targets_pad_wp

    full_result['prompt'] = sources
    return full_result

def preprocess_imgsp_llava(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: List[Image.Image],
    img_token: str = '<image>',
    refine_prompt: bool = False,
    eval: bool = False,
) -> Dict:
    processor = AutoProcessor.from_pretrained("/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/llava-v1.6-vicuna-7b-hf")
    system_message = {
        "role": "system",
        "content": [
            {"type": "text", "text": "A chat between a curious user and an artificial intelligence assistant. The assistant is a navigation model that output the uav waypoints or actions according to the user's instructions and images."}
        ]
    }
    messages = _build_messages(sources[0], has_image, system_message)
    if eval:
        messages = [m for m in messages if m["role"] in ("system", "user")]
    full_result = processor.apply_chat_template(
        messages, tokenize=True, return_dict=True, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length
    )

    input_ids = full_result["input_ids"]
    if isinstance(input_ids, list):
        input_ids = torch.tensor(input_ids).unsqueeze(0)

    if eval:
        full_result["labels"] = None
        full_result["input_ids"] = input_ids
        full_result['prompt'] = sources
        return full_result

    labels = torch.full_like(input_ids, IGNORE_INDEX)

    input_ids_flat = input_ids[0].tolist()
    L = len(input_ids_flat)
    pos = 0
    while pos < L:
        #ASSISANT
        if input_ids_flat[pos] == 13566 and input_ids_flat[pos-1] == 9047 and input_ids_flat[pos-2] == 1799 and input_ids_flat[pos-3] == 319:
            ans_start = pos + 2
            if input_ids_flat[ans_start] == 29871:
                ans_start = ans_start + 1
            ans_end = ans_start
            while ans_end < L and ans_end != L-1:
                ans_end += 1
            try:
                assert input_ids_flat[ans_end] == 29871
            except:
                print("input_ids_flat[ans_end] is not 29871.")
                print(ans_start, ans_end, len(input_ids_flat), tokenizer.decode(input_ids_flat[-50:]))
                # input_ids_flat[ans_end] = 29871
            if ans_end < L:
                labels[0, ans_start : ans_end + 1] = input_ids[
                    0, ans_start : ans_end + 1
                ]
                pos = ans_end
        pos += 1

    full_result["labels"] = labels
    full_result["input_ids"] = input_ids

    if 'Control' in sources[0][1]['value']:
        input_ids_pad_wp = torch.zeros(input_ids.shape[0], input_ids.shape[1] + 1, dtype=torch.long)
        input_ids_pad_wp[:, :-2] = input_ids[:, :-1]
        input_ids_pad_wp[:, -2] = WAYPOINT_INPUT_TOKEN
        input_ids_pad_wp[:, -1] = input_ids[:, -1]
        full_result['input_ids'] = input_ids_pad_wp

        targets_pad_wp = torch.zeros(labels.shape[0], labels.shape[1] + 1, dtype=torch.long)
        targets_pad_wp[:, :-2] = labels[:, :-1]
        targets_pad_wp[:, -2] = WAYPOINT_LABEL_TOKEN
        targets_pad_wp[:, -1] = labels[:, -1]
        full_result['labels'] = targets_pad_wp

    full_result['prompt'] = sources
    return full_result


def preprocess(
    sources: Sequence[str],
    images: Union[Image.Image, List[Image.Image]],
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False,
    prompt: str = None,
    refine_prompt: bool = False,
    eval: bool = False,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version.startswith("imgsp_qwen"):
        return preprocess_imgsp_qwen(sources, tokenizer, has_image=images, refine_prompt=refine_prompt)
    elif conversation_lib.default_conversation.version.startswith("imgsp_llava"):
        return preprocess_imgsp_llava(sources, tokenizer, has_image=images, refine_prompt=refine_prompt, eval=eval)
    else:
        raise ValueError(
            f"Unsupported conversation version: {conversation_lib.default_conversation.version}"
        )


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""
    RGB_FOLDER = ['frontcamera', 'leftcamera', 'rightcamera', 'rearcamera', 'downcamera']
    
    def __init__(self, data_path: Union[List[str], str],
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        if isinstance(data_path, str):
            data_path = [data_path]
        list_data_dict = []
        for path in data_path:
            with open(path, "r") as f:
                list_data_dict.extend(json.load(f))

        self.dataset_path = data_args.dataset_path
        self.dataset_state = data_args.dataset_state
        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.eval = getattr(data_args, "cot_eval", False)
        if not self.eval:
            random.shuffle(self.list_data_dict)
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if ('image' in sample) or ('video' in sample) else -cur_len
            length_list.append(cur_len)
        return length_list

    def get_stage(self, trajectory, frame_num):
        def turning_stage(p0,p1,p2):
            prev_vec = p1 - p0
            now_vec = p2 - p1
            delta_angle = np.arccos(np.dot(prev_vec, now_vec) / (np.linalg.norm(prev_vec)+ 1e-6) / (np.linalg.norm(now_vec)+ 1e-6)) * 180 / np.pi
            if delta_angle > 25 and delta_angle < 120:
                if int(np.cross(prev_vec, now_vec)) > 0:
                    return 'right'
                else:
                    return 'left'
            return 'cruise'
        assist = 0
        trajectory = np.asarray(trajectory)
        z_values = trajectory[:, 2]
        now_z = z_values[frame_num - 1]
        future_z = z_values[min(frame_num+2, len(z_values)-1)]
        stage = 'cruise'
        if now_z - future_z > 5:
            stage = 'take off'
        elif now_z - future_z < -5:
            stage = 'landing'
        prev_vec = np.array([0,0,0])
        if frame_num >= 2 and frame_num < len(trajectory):
            prev_vec =  np.array(trajectory[frame_num - 1, :3] - trajectory[frame_num - 2, :3])
            if stage == 'cruise':
                stage = turning_stage(trajectory[frame_num - 2 ,:2], trajectory[frame_num - 1, :2], trajectory[frame_num, :2])
        if frame_num >= 1 and frame_num < len(trajectory) - 1:
            future_p = trajectory[frame_num + 1, :2]
            next_p = trajectory[frame_num, :2]
            next_stage = turning_stage(trajectory[frame_num-1, :2], next_p, future_p)
            future_z = z_values[min(frame_num+3, len(z_values)-1)]
            if trajectory[frame_num, 2] - future_z < -5:
                next_stage = 'landing'
            if next_stage == 'left' or next_stage == 'right' or next_stage == 'landing':
                assist = 1
        return stage, prev_vec, assist

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        ori_sources = None
        infos = self.list_data_dict[i]
        dataset_name = infos['dataset']
        frame_num = infos['frame']
        bbox = infos['bbox']
        subgoal = infos['subgoal']
        states = self.dataset_state[dataset_name]
        if self.data_args.r1:
            states = ['front']
        stage = ''
        if dataset_name == 'traveluav':
            traj_dir = os.path.join(self.dataset_path, *infos['json'].split('/')[:-1])
            json_path = os.path.join(self.dataset_path, infos['json'])
            with open(json_path, 'r') as f:
                sources = json.load(f)

            if isinstance(i, int):
                sources = [sources]
            ori_sources = copy.deepcopy(sources)
            
            assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
            height = 256
            width = 256
            if self.data_args.use_assist:
                stage, future_delta, assist = self.get_stage(sources[0]['trajectory'], frame_num)
                cur_pos = sources[0]['trajectory'][frame_num - 1][:3]
                x, y = ori_sources[0]['trajectory'][-1][0], ori_sources[0]['trajectory'][-1][1]
                rotation_matrix = rotation_matrix_from_vector(x, y)
                future_delta =  transform_point(future_delta, rotation_matrix)
                future_delta = future_delta / (np.linalg.norm(future_delta) + 1e-8)
                future_delta_str = ','.join([str(round(x, 1)) for x in future_delta])
                
                cur_pos = transform_point(cur_pos, rotation_matrix)
                cur_pos_str = ','.join([str(round(x, 1)) for x in cur_pos])
            else:
                future_delta_str = None
                cur_pos_str = None
                
        elif dataset_name == 'aerialvg':
            image_path = infos['json']
            height = infos['height']
            width = infos['width']
            sources = [{"conversations": [{"from": "human", "value": infos['conversations']}, {"from": "gpt", "value": ""}]}]
            ori_sources = copy.deepcopy(sources)
            future_delta_str = None
            cur_pos_str = None
        else:
            raise ValueError(
                f"Unsupported dataset name: {dataset_name}"
            )
        
        assist = ""
        if subgoal != "":
            # TODO: wmq! convert_to_qwen25vl_format and llava_next_format. image rescale.
            assist += f"Subgoal: {subgoal}."
            for state in states:
                if state in bbox:
                    if state == "front":
                        if self.data_args.bbox_scale:
                            bbox_formatted = [round(bbox[state][i] * (width if i % 2 == 0 else height)) for i in range(4)]
                            bbox_formatted = convert_to_qwen25vl_format(bbox_formatted, height, width)
                        else:
                            bbox_formatted = [round(float(v), 4) for v in bbox[state]]
                        if self.data_args.r1:
                            assist += f"{{\n\"bbox_2d\": {bbox_formatted}\n}}, "
                        else:
                            assist += f"{{\n\"bbox_2d_front\": {bbox_formatted}\n}}, "
                    elif state == "left":
                        if self.data_args.bbox_scale:
                            bbox_formatted = [round(bbox[state][i] * (width if i % 2 == 0 else height)) for i in range(4)]
                            bbox_formatted = convert_to_qwen25vl_format(bbox_formatted, height, width)
                        else:
                            bbox_formatted = [round(float(v), 4) for v in bbox[state]]
                        assist += f"{{\n\"bbox_2d_left\": {bbox_formatted}\n}}, "
                    elif state == "right":
                        if self.data_args.bbox_scale:
                            bbox_formatted = [round(bbox[state][i] * (width if i % 2 == 0 else height)) for i in range(4)]
                            bbox_formatted = convert_to_qwen25vl_format(bbox_formatted, height, width)
                        else:
                            bbox_formatted = [round(float(v), 4) for v in bbox[state]]
                        assist += f"{{\n\"bbox_2d_right\": {bbox_formatted}\n}}, "
                    elif state == "rear":
                        if self.data_args.bbox_scale:
                            bbox_formatted = [round(bbox[state][i] * (width if i % 2 == 0 else height)) for i in range(4)]
                            bbox_formatted = convert_to_qwen25vl_format(bbox_formatted, height, width)
                        else:
                            bbox_formatted = [round(float(v), 4) for v in bbox[state]]
                        assist += f"{{\n\"bbox_2d_rear\": {bbox_formatted}\n}}, "
                    elif state == "down":
                        if self.data_args.bbox_scale:
                            bbox_formatted = [round(bbox[state][i] * (width if i % 2 == 0 else height)) for i in range(4)]
                            bbox_formatted = convert_to_qwen25vl_format(bbox_formatted, height, width)
                        else:
                            bbox_formatted = [round(float(v), 4) for v in bbox[state]]
                        assist += f"{{\n\"bbox_2d_down\": {bbox_formatted}\n}}, "
                    elif state == "current":
                        if self.data_args.bbox_scale:
                            bbox_formatted = [round(bbox[state][i] * (width if i % 2 == 0 else height)) for i in range(4)]
                            bbox_formatted = convert_to_qwen25vl_format(bbox_formatted, height, width)
                        else:
                            bbox_formatted = [round(float(v), 4) for v in bbox[state]]
                        assist += f"{{\n\"bbox_2d\": {bbox_formatted}\n}}, "
            assist = re.sub(r',\s*$', '.', assist.strip())
        if dataset_name != "aerialvg" and not self.data_args.r1:
            assist += "\nControl:"

        if conversation_lib.default_conversation.version.startswith("imgsp_qwen") or conversation_lib.default_conversation.version.startswith("imgsp_llava")  :
            if dataset_name == "traveluav":
                traj_camera_list = []
                for idx, camera_name in enumerate(self.RGB_FOLDER):
                    traj_camera_list.append(sorted([os.path.join(traj_dir, camera_name, filename) for filename in os.listdir(os.path.join(traj_dir,camera_name))]))
                assert(len(traj_camera_list[0]) == len(traj_camera_list[1]) == len(traj_camera_list[2]) == len(traj_camera_list[3]))
                traj_frames = []
                for idx in range(len(traj_camera_list[0])):
                    batch = []
                    for iid in range(len(self.RGB_FOLDER)):
                        batch.append(traj_camera_list[iid][idx])
                    traj_frames.append(batch)
                traj_imgs = []
                for frame_imgs in traj_frames:
                    images = [Image.open(img_path).convert('RGB') for img_path in frame_imgs]
                    traj_imgs.append(images)
                image = traj_imgs[(frame_num-1):frame_num][0]
            else:
                image = [Image.open(image_path).convert('RGB')]
        else:
            raise ValueError(
                f"Unsupported conversation version: {conversation_lib.default_conversation.version}"
            )
        sources = preprocess_multimodal(copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args, assist, dataset_name, eval=self.eval, stage=stage, delta = future_delta_str, cur = cur_pos_str)
                
        has_image = (image is not None)
        data_dict = preprocess(
            sources,
            image,
            self.tokenizer,
            has_image=has_image,
            prompt=self.data_args.input_prompt,
            refine_prompt=self.data_args.refine_prompt,
            eval=self.eval)
        
        prompt = data_dict.get('prompt', None)
        if not self.eval:
            if conversation_lib.default_conversation.version.startswith("imgsp_uav"):
                data_dict = dict(input_ids=data_dict["input_ids"][0],
                            labels=data_dict["labels"][0])
            elif conversation_lib.default_conversation.version.startswith("imgsp_qwen") or conversation_lib.default_conversation.version.startswith("imgsp_llava"):
                data_dict["input_ids"] = data_dict["input_ids"][0]
                data_dict["labels"] = data_dict["labels"][0]
        
        data_dict['image'] = image
        if dataset_name == "aerialvg":
            return data_dict
        trajectory_data = np.array(ori_sources[0]['trajectory'])
        history_waypoint = trajectory_data[0:frame_num, 0:3]
        waypoint = trajectory_data[frame_num:min(ori_sources[0]['length'], frame_num + 7), 0:3]
        if len(waypoint) == 0:
            waypoint = np.array([history_waypoint[-1] for i in range(7)])
        elif len(waypoint) < 7:
            waypoint = np.array([waypoint[i] if i < len(waypoint) else waypoint[-1] for i in range(7)])

        waypoint = waypoint - history_waypoint[-1]
        x, y = ori_sources[0]['trajectory'][-1][0], ori_sources[0]['trajectory'][-1][1]
        rotation_matrix = rotation_matrix_from_vector(x, y)
        history_waypoint = transform_point(history_waypoint, rotation_matrix)
        waypoint = transform_point(waypoint, rotation_matrix)
        
        use_angle = True
        if use_angle:
            waypoint = waypoint2angle(waypoint)
        
        data_dict['history_waypoint'] = torch.tensor(history_waypoint).view(-1)
        data_dict['waypoint'] = torch.tensor(waypoint[0]).view(-1)
        orientation = trajectory_data[frame_num-1, 3:6]
        data_dict['orientation'] = torch.tensor(orientation).view(-1)
        data_dict['is_help'] = torch.tensor(0).view(-1)
        
        # prompt exist in the data
        if prompt is not None:
            data_dict['prompt'] = prompt
        
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer
    

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        ## wmq: Qwen.
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        # attention_mask = None
        pixel_values = None
        image_grid_thw = None
        image_sizes = None
        # if 'attention_mask' in instances[0].keys():
        #     attention_mask = tuple([instance['attention_mask'][0] for instance in instances])
        #     attention_mask = torch.nn.utils.rnn.pad_sequence(attention_mask,
        #                                     batch_first=True,
        #                                     padding_value=0)
        
        if 'pixel_values' in instances[0].keys() and 'image_grid_thw' in instances[0].keys():
            pixel_values, image_grid_thw =  tuple([instance[key] for instance in instances]
                                  for key in ("pixel_values", "image_grid_thw"))
        
            pixel_values = torch.cat(pixel_values, dim=0)
            image_grid_thw = torch.cat(image_grid_thw, dim=0)
        if 'pixel_values' in instances[0].keys() and 'image_sizes' in instances[0].keys():
            pixel_values, image_sizes =  tuple([instance[key] for instance in instances]
                                  for key in ("pixel_values", "image_sizes"))
        
            pixel_values = torch.cat(pixel_values, dim=0)
            image_sizes = torch.cat(image_sizes, dim=0)
            
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            pixel_values=pixel_values,
            image_grid_thw=image_grid_thw,
            image_sizes=image_sizes,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            # images = instance['image'] for instance in instances if isinstance(instance['image'], list) else [instance['image'] for instance in instances]
            images = [
                img
                for instance in instances
                for img in (instance['image'] if isinstance(instance['image'], list) else [instance['image']])
            ]
            # TODO: maybe all list is a good thing. wmq: No! all list is not a good thing for arivln iterable dataset
            if all(isinstance(x, Image.Image) for x in images):
                batch['images'] = images
            elif all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images

        # if 'prompt' in instances[0]:
            # batch['prompts'] = [instance['prompt'] for instance in instances]
        
        # if 'waypoint' in instances[0]:
        #     batch['waypoints'] = torch.stack([instance['waypoint'] for instance in instances])
        #     batch['historys'] = [instance['history_waypoint'] for instance in instances]
        if any('waypoint' in instance for instance in instances):
            batch['waypoints'] = torch.stack([instance['waypoint'] for instance in instances if 'waypoint' in instance])
            batch['historys'] = [instance['history_waypoint'] for instance in instances if 'history_waypoint' in instance]

        # if 'orientation' in instances[0]:
        #     batch['orientations'] = torch.stack([instance['orientation'] for instance in instances])
        if any('orientation' in instance for instance in instances):
            batch['orientations'] = torch.stack([instance['orientation'] for instance in instances if 'orientation' in instance])
        
        # if 'end' in instances[0]:
        #     batch['ends'] = torch.stack([instance['end'] for instance in instances]).squeeze()
        if any('end' in instance for instance in instances):
            batch['ends'] = torch.stack([instance['end'] for instance in instances if 'end' in instance]).squeeze()
        return batch

def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                data_path=data_args.data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def train():
    global local_rank

    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
    bnb_model_from_pretrained_args = dict(
        torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32)),
    )
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=training_args.bits == 4,
            load_in_8bit=training_args.bits == 8,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=training_args.bits == 4,
                load_in_8bit=training_args.bits == 8,
                llm_int8_threshold=6.0,
                llm_int8_has_fp16_weight=False,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=training_args.double_quant,
                bnb_4bit_quant_type=training_args.quant_type # {'fp4', 'nf4'}
            )
        ))

    config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
    orig_rope_scaling = getattr(config, "rope_scaling", None)
    if orig_rope_scaling is None:
        orig_rope_scaling = {"factor": 1}

    orig_rope_scaling_factor = orig_rope_scaling["factor"] if "factor" in orig_rope_scaling.keys() else 1
    orig_ctx_len = getattr(config, "max_position_embeddings", None)
    if orig_ctx_len:
        orig_ctx_len *= orig_rope_scaling_factor
        if training_args.model_max_length > orig_ctx_len:
            scaling_factor = float(math.ceil(training_args.model_max_length / orig_ctx_len))
            config.rope_scaling = {"type": "linear", "factor": scaling_factor}
    
    if "llava-v1.6" in model_args.model_name_or_path:
        if model_args.cot:
            ModelClass = LlavaNextCOTUAVForCausalLM
        else:
            ModelClass = LlavaNextUAVForCausalLM
    elif "Qwen2.5-VL" in model_args.model_name_or_path:
        if model_args.cot:
            ModelClass = Qwen2_5_VLCOTUAVForCausalLM
        else:
            ModelClass = Qwen2_5_VLUAVForCausalLM
    else:
        raise ValueError(f"Unknown model type: {model_args.model_name_or_path}")

    model = ModelClass.from_pretrained(
        model_args.model_name_or_path,
        use_angle_and_norm_loss=model_args.use_angle_and_norm_loss,
        config=config,
        cache_dir=training_args.cache_dir,
        **bnb_model_from_pretrained_args
    )
    model.config.use_cache = False

    if model_args.freeze_backbone:
        model.model.requires_grad_(False)

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            exclude_modules=find_all_exclude_names(model, find_all_linear_names(model)),
            layers_to_transform=[i for i in range(0, config.num_hidden_layers if hasattr(config, "num_hidden_layers") else config.text_config.num_hidden_layers)], 
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        if training_args.bits == 16:
            if training_args.bf16:
                model.to(torch.bfloat16)
            if training_args.fp16:
                model.to(torch.float16)
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    if tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    else: #TODO: wmq. NOT SURE!
        tokenizer.unk_token = "<unk>"
        # tokenizer.pad_token = tokenizer.unk_token
        # tokenizer.add_special_tokens({"unk_token": "<unk>"})
        # model.resize_token_embeddings(len(tokenizer))
    if model_args.version in conversation_lib.conv_templates:
        conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
    else:
        conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is None and "Qwen2.5-VL" in model_args.model_name_or_path:
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().visual.merger.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().visual.merger.parameters():
                p.requires_grad = False
        
        if training_args.bits in [4, 8]:
            model.get_model().visual.merger.to(dtype=compute_dtype, device=training_args.device)
        
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    
    if model_args.vision_tower is None and "llava-v1.6" in model_args.model_name_or_path:
        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().multi_modal_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().multi_modal_projector.parameters():
                p.requires_grad = False
        
        if training_args.bits in [4, 8]:
            model.get_model().multi_modal_projector.to(dtype=compute_dtype, device=training_args.device)
        
        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
    
    smarter_tokenizer_and_embedding_resize(special_tokens_list=['<wp>', '<his>'], tokenizer=tokenizer, model=model)
    
    model.get_special_token_id({'<wp>': tokenizer.encode('<wp>', add_special_tokens=False)[0], '<his>': tokenizer.encode('<his>', add_special_tokens=False)[0],
                                ',': tokenizer.encode(',', add_special_tokens=False)[0], ';': tokenizer.encode(';', add_special_tokens=False)[0]})

    # smarter_tokenizer_and_embedding_resize(special_tokens_list=['<wp>', '<bbox_2d_front>', '<bbox_2d_left>', '<bbox_2d_right>', '<bbox_2d_rear>', '<bbox_2d_down>', '<bbox_2d>'], tokenizer=tokenizer, model=model)
    
    # model.get_special_token_id({'<wp>': tokenizer.encode('<wp>', add_special_tokens=False)[0], '<bbox_2d_front>': tokenizer.encode('<bbox_2d_front>', add_special_tokens=False)[0],
    #                             '<bbox_2d_left>': tokenizer.encode('<bbox_2d_left>', add_special_tokens=False)[0],'<bbox_2d_right>': tokenizer.encode('<bbox_2d_right>', add_special_tokens=False)[0],
    #                             '<bbox_2d_rear>': tokenizer.encode('<bbox_2d_rear>', add_special_tokens=False)[0], '<bbox_2d_down>': tokenizer.encode('<bbox_2d_down>', add_special_tokens=False)[0],
    #                             '<bbox_2d_current>': tokenizer.encode('<bbox_2d>', add_special_tokens=False)[0],
    #                             ',': tokenizer.encode(',', add_special_tokens=False)[0], ';': tokenizer.encode(';', add_special_tokens=False)[0]})
    
    # all the attention modules require grad
    if not ("llava" in model_args.model_name_or_path or "Qwen2.5-VL" in model_args.model_name_or_path):
        model.get_model().initialize_attention_modules(model_args)
    
    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    
    #TODO: wmq  bbox module require_grads=True.
    if model_args.tune_waypoint_predictor:
        for p in model.waypoint_emb.parameters():
            p.requires_grad = True
        for p in model.waypoints_fc.parameters():
            p.requires_grad = True
        for p in model.waypoints_output.parameters():
            p.requires_grad = True
        for p in model.history_preprocessor.parameters():
            p.requires_grad = True

    for name, param in model.named_parameters():
        if 'lora' in name:
            param.requires_grad = True
        if param.requires_grad:
            print(f"Parameter name: {name}, Parameter shape: {param.shape}")
    
    
    if training_args.resume:
        model = load_model(model, training_args.resume)
    model.print_trainable_parameters()
    trainer = LLaVATrainer(model=model,
                    tokenizer=tokenizer,
                    args=training_args,
                    **data_module)
    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        trainer.train()
    trainer.save_state()

    model.config.use_cache = True

    if training_args.lora_enable:
        print("saving model...")
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
            safe_save_model_for_hf_trainer(trainer=trainer, output_dir=training_args.output_dir)
        print("saved.")
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

        
if __name__ == "__main__":
    train()
