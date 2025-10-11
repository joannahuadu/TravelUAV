#    Copyright 2023 Haotian Liu
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


import os
import warnings
import shutil

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig, BitsAndBytesConfig
import torch
from llamavid.model import *
from llamavid.constants import DEFAULT_IMAGE_PATCH_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from peft import PeftModel

def safe_load_state_dict(model, state_dict):
    model_dict = model.state_dict()
    new_state_dict = {}

    for k, v in state_dict.items():
        if k in model_dict:
            if model_dict[k].shape == v.shape:
                new_state_dict[k] = v
            else:
                print(f"Skip shape mismatch: {k}, "
                      f"model {tuple(model_dict[k].shape)} vs weight {tuple(v.shape)}")
        else:
            print(f"Skip unexpected key: {k}")

    model_dict.update(new_state_dict)
    model.load_state_dict(model_dict, strict=False)
    print(f"Loaded {len(new_state_dict)} compatible parameters out of {len(state_dict)}")

def load_pretrained_model(model_path, model_base, model_name, load_8bit=False, load_4bit=False, device_map="auto", device="cuda"):
    # TODO: wmq modify.
    kwargs = {"device_map": device_map}

    if load_8bit:
        kwargs['load_in_8bit'] = True
    elif load_4bit:
        kwargs['load_in_4bit'] = True
        kwargs['quantization_config'] = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type='nf4'
        )
    else:
        kwargs['torch_dtype'] = torch.bfloat16

    if 'vid' or 'uav' in model_name.lower():
        # Load LLaMA-VID model
        if model_base is not None:
            # this may be mm projector only
            from peft import PeftModel
            print('Loading Model from base model...')
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            cfg_pretrained = AutoConfig.from_pretrained(model_path)
            if "llava-v1.6" in model_base:
                ModelClass = LlavaNextUAVForCausalLM
            elif "llava" in model_base:
                ModelClass = LlavaUAVForCausalLM
            elif "Qwen2.5-VL" in model_base:
                ModelClass = Qwen2_5_VLUAVForCausalLM
                # kwargs["offload_folder"] = "/home/wmq/.cache"
                # kwargs["offload_state_dict"] = True
            elif "llama" in model_base or 'vicuna' in model_base:
                ModelClass = LlavaLlamaAttForCausalLM
            elif "Qwen" in model_base:
                ModelClass = LlavaQwenAttForCausalLM
            else:
                raise ValueError(f"Unknown model type: {model_base}")
            kwargs["ignore_mismatched_sizes"]=True
            model = ModelClass.from_pretrained(model_base, low_cpu_mem_usage=True, config=cfg_pretrained, **kwargs)
            mm_projector_weights = torch.load(os.path.join(model_path, 'mm_projector.bin'), map_location='cpu')
            mm_projector_weights_new = {
                k.replace("base_model.model.", "", 1): v
                for k, v in mm_projector_weights.items()
            }
            # model.load_state_dict(mm_projector_weights_new, strict=False)
            safe_load_state_dict(model, mm_projector_weights_new)
        else:
            tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
            model = LlavaLlamaAttForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    else:
        # Load language model
        if model_base is not None:
            # PEFT model
            
            tokenizer = AutoTokenizer.from_pretrained(model_base, use_fast=False)
            model = AutoModelForCausalLM.from_pretrained(model_base, torch_dtype=torch.float16, low_cpu_mem_usage=True, device_map="auto")
            print(f"Loading LoRA weights from {model_path}")
            model = PeftModel.from_pretrained(model, model_path)
            print(f"Merging weights")
            model = model.merge_and_unload()
            print('Convert to FP16...')
            model.to(torch.float16)
        else:
            use_fast = False
            if 'mpt' in model_name.lower():
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, trust_remote_code=True, **kwargs)
            else:
                tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=False)
                model = AutoModelForCausalLM.from_pretrained(model_path, low_cpu_mem_usage=True, **kwargs)

    image_processor = None

    if 'vid' or 'uav' in model_name.lower():
        mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
        mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
        if mm_use_im_patch_token:
            tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)
        model.resize_token_embeddings(len(tokenizer))
        image_processor = None
        model.config.model_path = model_path
        if not ("Qwen2.5-VL" in model_base or "llava-v1.6" in  model_base):
            vision_tower = model.get_vision_tower()
            if not vision_tower.is_loaded:
                vision_tower.load_model()
            vision_tower.to(device=device, dtype=torch.bfloat16)
            image_processor = vision_tower.image_processor
            model.get_model().initialize_attention_modules(model.config, for_eval=True)    

    if hasattr(model.config, "max_sequence_length"):
        context_len = model.config.max_sequence_length
    else:
        context_len = 2048

    return tokenizer, model, image_processor, context_len
