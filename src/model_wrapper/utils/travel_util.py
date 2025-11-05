import copy
import json
import os
import sys
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Sequence
import transformers
from scipy.spatial.transform import Rotation as R
import torch
import numpy as np
import math

from PIL import Image
from llamavid import conversation as conversation_lib

sys.path.append(str(Path(str(os.getcwd())).resolve()))
sys.path.append(str(Path(__file__).resolve().parents[3]/ 'Model' / 'LLaMA-UAV'))
from llamavid.model.builder import load_pretrained_model
from llamavid.model.vis_traj_arch import VisionTrajectoryGenerator
from peft import PeftModel
from llava.mm_utils import tokenizer_image_token, get_model_name_from_path
from llamavid.constants import (
    IGNORE_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN,
    WAYPOINT_INPUT_TOKEN, WAYPOINT_LABEL_TOKEN, DEFAULT_HISTORY_TOKEN, DEFAULT_WP_TOKEN
)
def load_model(args):
    model_path = os.path.expanduser(args.model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, _ = load_pretrained_model(model_path, args.model_base, model_name, args)
    if tokenizer.unk_token:
        tokenizer.pad_token = tokenizer.unk_token
    else: #TODO: NOT SURE!
        tokenizer.add_special_tokens({"unk_token": "<unk>"})
    smarter_tokenizer_and_embedding_resize(special_tokens_list=['<wp>', '<his>'], tokenizer=tokenizer, model=model)
    # model.get_special_token_id({'<wp>': tokenizer.encode('<wp>')[1], '<his>': tokenizer.encode('<his>')[1],
                                # ',': tokenizer.encode(',')[1], ';': tokenizer.encode(';')[1]})
    model.get_special_token_id({'<wp>': tokenizer.encode('<wp>', add_special_tokens=False)[0], '<his>': tokenizer.encode('<his>', add_special_tokens=False)[0],
                                ',': tokenizer.encode(',', add_special_tokens=False)[0], ';': tokenizer.encode(';', add_special_tokens=False)[0]})
    lora_enable = True
    if lora_enable:
        print(f"Loading LoRA weights from {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        non_lora_weights = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
        model.load_state_dict(non_lora_weights, strict=False)    
        mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
        model.load_state_dict(mm_projector_weights, strict=False)
    
    return tokenizer, model, image_processor

def load_traj_model(model_args):
    vision_config = generate_vision_tower_config(model_args.vision_tower, model_args.image_processor)
    config = transformers.AutoConfig.from_pretrained(vision_config, trust_remote_code=True)
    traj_model = VisionTrajectoryGenerator(config)
    traj_weights = torch.load(os.path.join(model_args.traj_model_path, 'model_5.pth'), map_location='cpu')
    traj_weights = {k: v.to(torch.bfloat16) for k, v in traj_weights.items()}
    traj_model.load_state_dict(traj_weights, strict=False)
    return traj_model


def generate_vision_tower_config(vision_tower, image_processor):
    default_vision_config={
    "model_type": "clip",
    "hidden_act": "silu",
    "hidden_size": 4096,
    "image_aspect_ratio": "square",
    "image_grid_pinpoints": None,
    "image_processor": "./llamavid/processor/clip-patch14-224",
    "initializer_range": 0.02,
    "intermediate_size": 11008,
    "max_position_embeddings": 4096,
    "max_token": 2048,
    "mm_hidden_size": 1408,
    "mm_projector_type": "mlp2x_gelu",
    "mm_use_im_patch_token": False,
    "mm_use_im_start_end": False,
    "mm_vision_select_feature": "patch",
    "mm_vision_select_layer": -2,
    "mm_vision_tower": "./model_zoo/LAVIS/eva_vit_g.pth",
    "torch_dtype": "float16"
    }
    default_vision_config['image_processor'] = image_processor
    default_vision_config['mm_vision_tower'] = vision_tower
    cf_path = os.path.join(os.path.split(vision_tower)[0], 'config.json')
    with open(cf_path, 'w') as f:
        json.dump(default_vision_config, f, indent=2)
    return cf_path


def prepare_data_to_traj_model(episodes, waypoints, image_processor, rot_to_targets=None):
    image_list = []
    target_list = []
    for i in range(len(episodes)):
        info = episodes[i]
        rot_to_target = None
        if rot_to_targets is not None:
            if rot_to_targets[i] is not None:
                rot_to_target = rot_to_targets[i]
        target = waypoints[i][0:3]
        rot_0 = info[0]['sensors']['imu']["rotation"]
        rot = info[-1]['sensors']['imu']["rotation"]
        if rot_to_target is not None:
            target = np.array(rot).T @ np.array(rot_0) @ np.array(rot_to_target) @ np.array(target)
        else:
            target = np.array(rot).T @ np.array(rot_0) @ np.array(target)
        image_list.append(info[-1]['rgb'][0])
        target_list.append(target)
    images = np.stack(image_list, axis=0)
    image = image_processor.preprocess(images, return_tensors='pt')['pixel_values']
    target = torch.tensor(np.array(target_list))
    
    return {'img': image, 'target': target}        

def transform_to_world(waypoints, episodes):
    waypoints_world = []
    for i in range(len(waypoints)):
        waypoint = waypoints[i]
        ep = episodes[i]
        pos = ep[-1]["sensors"]["state"]["position"]
        rot = ep[-1]["sensors"]["imu"]["rotation"]  
        waypoint_world = np.array(rot) @ np.array(waypoint).T + np.asarray(pos).reshape(3,1)
        waypoint_world = waypoint_world.T
        waypoints_world.append(waypoint_world)

    return waypoints_world

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

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def to_eularian_angles(q):
    x,y,z,w = q
    ysqr = y * y
    t0 = +2.0 * (w*x + y*z)
    t1 = +1.0 - 2.0*(x*x + ysqr)
    roll = math.atan2(t0, t1)
    t2 = +2.0 * (w*y - z*x)
    if (t2 > 1.0):
        t2 = 1
    if (t2 < -1.0):
        t2 = -1.0
    pitch = math.asin(t2)
    t3 = +2.0 * (w*z + x*y)
    t4 = +1.0 - 2.0 * (ysqr + z*z)
    yaw = math.atan2(t3, t4)
    return (pitch, roll, yaw)

def euler_to_rotation_matrix(e):
    rotation = R.from_euler('xyz', e, degrees=False)
    return rotation.as_matrix()

def project_this_state2target_state_axis(this_state, target_state):
    start_pos = target_state['position']
    start_eular = to_eularian_angles(target_state['orientation'])  # (pitch, roll, yaw)
    this_pos = this_state['position']
    this_eular = to_eularian_angles(this_state['orientation'])
    delta_pos = np.asarray(this_pos) - np.asarray(start_pos)
    delta_eular = np.asarray(this_eular) - np.asarray(start_eular)
    rot = euler_to_rotation_matrix(start_eular) 
    delta_pos = rot.T @ delta_pos
    return {'position': delta_pos.tolist(), 'orientation': delta_eular.tolist()}

@dataclass
class DataArguments:
    data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    video_folder: Optional[str] = field(default=None)
    video_fps: Optional[int] = field(default=1)
    video_token: Optional[int] = field(default=2)
    image_aspect_ratio: str = 'square'
    image_grid_pinpoints: Optional[str] = field(default=None)
    input_prompt: Optional[str] = field(default=None)
    refine_prompt: Optional[bool] = field(default=False)
    mm_use_im_start_end: bool = field(default=False)

@dataclass
class CommonArguments:
    model_path: Optional[str] = field(default="facebook/opt-350m")
    model_base: Optional[str] = field(default=None)

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


def prepare_data_to_inputs(episodes, tokenizer, image_processor, data_args, target_point, assist_notice = None):
    # TODO: wmq modify.
    from llamavid.train.train_uav.train_uav_notice import preprocess_multimodal, preprocess
    ori_sources = None
    input_prompt = data_args.input_prompt
    refine_prompt = data_args.refine_prompt
    sources = episodes
    ori_sources = copy.deepcopy(sources)
    processor = image_processor
    images = []
    for src in sources[::-1]:
        if 'rgb' in src:
            images.extend(src['rgb'])
            break
    if processor is not None:
        images = np.stack(images, axis=0)
        image = processor.preprocess(images, return_tensors='pt')['pixel_values']
    else:
        image = images
    
    conversation = [
    {
        "from": "human",
        "value": sources[-1]['instruction']},
    {
        "from": "gpt",
        "value": ""
    }]
    
    if assist_notice is not None:
        stage = assist_notice
    else:
        stage = 'cruise' if len(sources) > 20 else 'take off'
    rot = np.array(ori_sources[0]['sensors']['imu']["rotation"])
    pos = np.array(ori_sources[0]['sensors']['state']['position'])
    deltas = []
    for source in ori_sources:
        if 'rgb' not in source.keys():
            continue
        deltas.append((np.array(source['sensors']['state']['position']) - pos))
    history_waypoint = np.array([(rot.T @ delta) for delta in deltas])
    rotation_to_target = None
    
    target_point = np.array(rot.T @ (target_point - pos))
    x, y = target_point[0], target_point[1]
    rotation_to_target = rotation_matrix_from_vector(x, y)
    history_waypoint = transform_point(history_waypoint, rotation_to_target)

    if len(history_waypoint) >= 2:
        delta = history_waypoint[-1] - history_waypoint[-2]
    else:
        delta = np.array([0, 0, -4.5])
    delta = delta / (np.linalg.norm(delta) + 1e-8)
    delta = ','.join([str(round(x, 1)) for x in delta])
    cur_pos = history_waypoint[-1]
    cur_pos = ','.join([str(round(x, 1)) for x in cur_pos])
    # print('stage:', stage,'delta:', delta, 'cur_pos:', cur_pos)
    sources = preprocess_multimodal(copy.deepcopy([conversation]), data_args, stage=stage, delta=delta, cur=cur_pos)
    has_image = (image is not None)
    data_dict = preprocess(
        sources,
        image,
        tokenizer,
        has_image=has_image,
        prompt=input_prompt,
        refine_prompt=refine_prompt)
    
    prompt = data_dict.get('prompt', None)
        
    if conversation_lib.default_conversation.version.startswith("imgsp_uav"):
        data_dict = dict(input_ids=data_dict["input_ids"][0],
                    labels=data_dict["labels"][0])
    elif conversation_lib.default_conversation.version.startswith("imgsp_qwen") or conversation_lib.default_conversation.version.startswith("imgsp_llava"):
        data_dict["input_ids"] = data_dict["input_ids"][0]
        data_dict["labels"] = data_dict["labels"][0]

    data_dict['image'] = image
    data_dict['history_waypoint'] = torch.tensor(history_waypoint).view(-1)
    ori_0 = ori_sources[0]['sensors']['state']
    ori = ori_sources[-1]['sensors']['state']
    target_relative_orientation = project_this_state2target_state_axis(ori, ori_0)['orientation']
    data_dict['orientation'] =  torch.tensor(target_relative_orientation).view(-1)
    
    if prompt is not None:
        data_dict['prompt'] = prompt
        
    return data_dict, rotation_to_target

def prepare_data_to_cot_inputs(episodes, tokenizer, image_processor, data_args, target_point, assist_notice = None):
    from llamavid.train.train_uav.train_uav_cot import preprocess_multimodal, preprocess
    ori_sources = None
    input_prompt = data_args.input_prompt
    refine_prompt = data_args.refine_prompt
    sources = episodes
    ori_sources = copy.deepcopy(sources)
    processor = image_processor
    images = []
    for src in sources[::-1]:
        if 'rgb' in src:
            images.extend([Image.fromarray(rgb) for rgb in src['rgb']])
            break
    if processor is not None:
        images = np.stack(images, axis=0)
        image = processor.preprocess(images, return_tensors='pt')['pixel_values']
    else:
        image = images
    
    conversation = [
    {
        "from": "human",
        "value": sources[-1]['instruction']},
    {
        "from": "gpt",
        "value": ""
    }]
    
    if assist_notice is not None:
        stage = assist_notice
    else:
        stage = 'cruise' if len(sources) > 20 else 'take off'
    rot = np.array(ori_sources[0]['sensors']['imu']["rotation"])
    pos = np.array(ori_sources[0]['sensors']['state']['position'])
    deltas = []
    for source in ori_sources:
        if 'rgb' not in source.keys():
            continue
        deltas.append((np.array(source['sensors']['state']['position']) - pos))
    history_waypoint = np.array([(rot.T @ delta) for delta in deltas])
    rotation_to_target = None
    
    target_point = np.array(rot.T @ (target_point - pos))
    x, y = target_point[0], target_point[1]
    rotation_to_target = rotation_matrix_from_vector(x, y)
    history_waypoint = transform_point(history_waypoint, rotation_to_target)

    if len(history_waypoint) >= 2:
        delta = history_waypoint[-1] - history_waypoint[-2]
    else:
        delta = np.array([0, 0, -4.5])
    delta = delta / (np.linalg.norm(delta) + 1e-8)
    delta = ','.join([str(round(x, 1)) for x in delta])
    cur_pos = history_waypoint[-1]
    cur_pos = ','.join([str(round(x, 1)) for x in cur_pos])
    if data_args.use_assist:
        sources = preprocess_multimodal(copy.deepcopy([conversation]), data_args, assist="", dataset_name="traveluav", eval=True, stage=stage, delta=delta, cur=cur_pos)
    else:
        sources = preprocess_multimodal(copy.deepcopy([conversation]), data_args, assist="", dataset_name="traveluav", eval=True)
    has_image = (image is not None)
    data_dict = preprocess(
        sources,
        image,
        tokenizer,
        has_image=has_image,
        prompt=input_prompt,
        refine_prompt=refine_prompt,
        eval=True)

    prompt = data_dict.get('prompt', None)
    # data_dict['images'] = image
    
    if prompt is not None:
    #     data_dict['prompts'] = data_dict.pop('prompt')
        data_dict.pop('prompt')
        
    return data_dict, rotation_to_target

def inputs_to_batch(tokenizer, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
    input_ids, labels = tuple([instance[key] for instance in instances]
                                for key in ("input_ids", "labels"))
    input_ids = torch.nn.utils.rnn.pad_sequence(
        input_ids,
        batch_first=True,
        padding_value=tokenizer.pad_token_id)
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
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
    )

    if 'image' in instances[0]:
        # images = instance['image'] for instance in instances if isinstance(instance['image'], list) else [instance['image'] for instance in instances]
        images = [
            img
            for instance in instances
            for img in (instance['image'] if isinstance(instance['image'], list) else [instance['image']])
        ]
        # TODO: maybe all list is a good thing. wmq: No! all list is not a good thing for arivln iterable dataset
        if all(isinstance(x, Image.Image) for x in images) or all(isinstance(x, np.ndarray) for x in images):
            batch['images'] = images
        elif all(x is not None and x.shape == images[0].shape for x in images):
            batch['images'] = torch.stack(images)
        else:
            batch['images'] = images

    if 'prompt' in instances[0]:
        batch['prompts'] = [instance['prompt'] for instance in instances]
    if 'history_waypoint' in instances[0]:
        batch['historys'] = [instance['history_waypoint'] for instance in instances]

    if 'orientation' in instances[0]:
        batch['orientations'] = torch.stack([instance['orientation'] for instance in instances])

    return batch