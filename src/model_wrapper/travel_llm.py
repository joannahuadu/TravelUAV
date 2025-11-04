import numpy as np
import torch
from src.model_wrapper.base_model import BaseModelWrapper
from src.model_wrapper.utils.travel_util import *
from src.vlnce_src.dino_monitor_online import DinoMonitor
from llamavid import conversation as conversation_lib
from transformers import CLIPImageProcessor
STOP_STR = "\nControl:" 
def decode_until_control(model, tokenizer, inputs,
                        max_new_tokens=256,
                        temperature=0.0,
                        top_p=1.0,
                        stop_str=STOP_STR):
    generated_text = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)
    new_text = ""
    hit_stop = False
    with torch.no_grad():
        out = model(**inputs, cot_eval=True, use_cache=True)
        past = out.past_key_values
        cur_input_ids = inputs["input_ids"][:, -1:]
        for _ in range(max_new_tokens):
            out = model(input_ids=cur_input_ids, past_key_values=past, use_cache=True, cot_eval=True)
            logits = out.logits[:, -1, :]
            past = out.past_key_values
            if temperature and temperature > 0:
                probs = F.softmax(logits / temperature, dim=-1)
                if top_p < 1.0:
                    sorted_probs, sorted_idx = torch.sort(probs, descending=True)
                    cum = torch.cumsum(sorted_probs, dim=-1)
                    mask = cum <= top_p
                    mask[..., 0] = True
                    filtered_probs = torch.where(mask, sorted_probs, torch.zeros_like(sorted_probs))
                    filtered_probs = filtered_probs / filtered_probs.sum(dim=-1, keepdim=True)
                    next_id = torch.multinomial(filtered_probs, num_samples=1)
                    next_token_id = sorted_idx.gather(-1, next_id)
                else:
                    next_token_id = torch.multinomial(probs, num_samples=1)
            else:
                next_token_id = torch.argmax(logits, dim=-1, keepdim=True)

            tok_text = tokenizer.decode(next_token_id[0], skip_special_tokens=True)

            if stop_str in generated_text:
                generated_text += tok_text
                new_text += tok_text
                break
            
            generated_text += tok_text
            new_text += tok_text
            
            if tokenizer.eos_token_id is not None and next_token_id.item() == tokenizer.eos_token_id:
                break
            
            cur_input_ids = next_token_id

    return generated_text, new_text, next_token_id, past

class TravelModelWrapper(BaseModelWrapper):
    def __init__(self, model_args, data_args):
        self.tokenizer, self.model, self.image_processor = load_model(model_args)
        self.traj_model = load_traj_model(model_args)
        self.model.to(torch.bfloat16)
        self.traj_model.to(dtype=torch.bfloat16, device=self.model.device)
        self.dino_moinitor = None
        self.model_args = model_args
        self.data_args = data_args
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    def prepare_cot_inputs(self, episodes, target_positions, assist_notices=None):
        inputs = []
        rot_to_targets = []
        for i in range(len(episodes)):
            input_item, rot_to_target = prepare_data_to_cot_inputs(
                episodes=episodes[i],
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                data_args=self.data_args,
                target_point=target_positions[i],
                assist_notice=assist_notices[i] if assist_notices is not None else None
            )
            inputs.append(input_item)
            rot_to_targets.append(rot_to_target)
        batch = inputs[0]
        inputs_device = {k: v.to(self.model.device) for k, v in batch.items() 
            if 'prompts' not in k and 'images' not in k and v is not None}
        
        return inputs_device, rot_to_targets
    
    def prepare_inputs(self, episodes, target_positions, assist_notices=None):
        inputs = []
        rot_to_targets = []
        
        for i in range(len(episodes)):
            input_item, rot_to_target = prepare_data_to_inputs(
                episodes=episodes[i],
                tokenizer=self.tokenizer,
                image_processor=self.image_processor,
                data_args=self.data_args,
                target_point=target_positions[i],
                assist_notice=assist_notices[i] if assist_notices is not None else None
            )
            inputs.append(input_item)
            rot_to_targets.append(rot_to_target)
        batch = inputs_to_batch(tokenizer=self.tokenizer, instances=inputs)
        # TODO: wmq modify.
        inputs_device = {k: v.to(self.model.device) for k, v in batch.items() 
            if 'prompts' not in k and 'images' not in k and 'historys' not in k and v is not None}
        if 'prompts' in batch and batch['prompts'] is not None:
            inputs_device['prompts'] = [item for item in batch['prompts']]
        inputs_device['images'] = [item for item in batch['images']]
        inputs_device['historys'] = [item.to(device=self.model.device, dtype=self.model.dtype) for item in batch['historys']]
        inputs_device['orientations'] = inputs_device['orientations'].to(dtype=self.model.dtype)
        inputs_device['return_waypoints'] = True
        inputs_device['use_cache'] = False
        
        return inputs_device, rot_to_targets
    
    def run_cot(self, inputs, episodes, rot_to_targets):
        # outputs = self.model.generate(
        #             **inputs,
        #             max_new_tokens=50,
        #             cot_eval = True,
        #         )
        _, new_outputs, input_ids, past = decode_until_control(self.model, self.tokenizer, inputs)
        print("CoT Text: ", new_outputs)
        input_ids_pad_wp = torch.zeros(input_ids.shape[0], input_ids.shape[1] + 1, dtype=torch.long)
        input_ids_pad_wp[:, :-2] = input_ids[:, :-1]
        input_ids_pad_wp[:, -2] = WAYPOINT_INPUT_TOKEN
        input_ids_pad_wp[:, -1] = input_ids[:, -1]

        labels = torch.full_like(input_ids, IGNORE_INDEX)
        targets_pad_wp = torch.zeros(labels.shape[0], labels.shape[1] + 1, dtype=torch.long)
        targets_pad_wp[:, :-2] = labels[:, :-1]
        targets_pad_wp[:, -2] = WAYPOINT_LABEL_TOKEN
        targets_pad_wp[:, -1] = labels[:, -1]

        waypoints_llm = self.model(input_ids=input_ids_pad_wp, labels=targets_pad_wp, past_key_values=past, use_cache=True, return_waypoints=True).cpu().to(dtype=torch.float32).numpy()
        waypoints_llm_new = []
        for waypoint in waypoints_llm:
            waypoint_new = waypoint[:3] / (1e-6 + np.linalg.norm(waypoint[:3])) * waypoint[3]
            waypoints_llm_new.append(waypoint_new)
        refined_waypoints = self.run_traj_model(episodes, np.array(waypoints_llm_new), rot_to_targets)
        return refined_waypoints
        
        
    def run_llm_model(self, inputs):
        # inputs['output_attentions'] = True
        waypoints_llm = self.model(**inputs).cpu().to(dtype=torch.float32).numpy()
        waypoints_llm_new = []
        for waypoint in waypoints_llm:
            waypoint_new = waypoint[:3] / (1e-6 + np.linalg.norm(waypoint[:3])) * waypoint[3]
            waypoints_llm_new.append(waypoint_new)
        return np.array(waypoints_llm_new)

    def run_traj_model(self, episodes, waypoints_llm_new, rot_to_targets):
        image_processor = CLIPImageProcessor.from_pretrained(self.model_args.image_processor)
        inputs = prepare_data_to_traj_model(episodes, waypoints_llm_new, image_processor, rot_to_targets)
        waypoints_traj = self.traj_model(inputs, None)
        refined_waypoints = waypoints_traj.cpu().to(dtype=torch.float32).numpy()
        refined_waypoints = transform_to_world(refined_waypoints, episodes)
        return refined_waypoints
    
    def eval(self):
        self.model.eval()
        self.traj_model.eval()
        
    def run(self, inputs, episodes, rot_to_targets):
        waypoints_llm_new = self.run_llm_model(inputs)
        refined_waypoints = self.run_traj_model(episodes, waypoints_llm_new, rot_to_targets)
        return refined_waypoints
    
    def predict_done(self, episodes, object_infos):
        prediction_dones = []
        if self.dino_moinitor is None:
            self.dino_moinitor = DinoMonitor.get_instance()
        for i in range(len(episodes)):
            prediction_done = self.dino_moinitor.get_dino_results(episodes[i], object_infos[i])
            prediction_dones.append(prediction_done)
        return prediction_dones
        

    