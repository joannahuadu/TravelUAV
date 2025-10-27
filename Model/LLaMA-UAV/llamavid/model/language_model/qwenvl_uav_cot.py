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

from typing import List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

from transformers import AutoConfig, AutoModelForCausalLM, Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration

from llamavid.model.language_model.llama_uav import CausalLMOutputWithPastUAV, CausalLMOutputWithPastUAVMulLoss
from llamavid.constants import WAYPOINT_LABEL_TOKEN, WAYPOINT_INPUT_TOKEN_QWEN

class QwenConfig(Qwen2_5_VLConfig):
    model_type = "qwenvl_uav_cot"
 
class CosineDirectionLoss(nn.Module):
    def __init__(self):
        super(CosineDirectionLoss, self).__init__()
    
    def forward(self, vec1, vec2):
        cosine_sim = F.cosine_similarity(vec1, vec2, dim=-1)
        loss = 1 - cosine_sim
        return loss.mean()

class Qwen2_5_VLCOTUAVForCausalLM(Qwen2_5_VLForConditionalGeneration):
    _checkpoint_conversion_mapping = {
        "^visual": "model.visual",
        r"^model(?!\.(language_model|visual))": "model.language_model",
    }
    _tied_weights_keys = ["lm_head.weight"]
    config_class = QwenConfig
    def __init__(self, config, **model_args):
        super().__init__(config)
        self.use_angle_and_norm_loss = model_args.get('use_angle_and_norm_loss', True)
        self.vocab_size = config.vocab_size
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.waypoint_emb = nn.Embedding(1, config.hidden_size)
        self.waypoints_fc = nn.Sequential(
            nn.Linear(config.hidden_size, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, 64),
        )
        self.waypoints_output = nn.Linear(64, 4)
        
        self.history_preprocessor = nn.Sequential(
            nn.Linear(3, config.hidden_size // 2),
            nn.ReLU(),
            nn.Linear(config.hidden_size // 2, config.hidden_size),
        )
        
        self.waypoints_loss_func = torch.nn.L1Loss()
        self.angle_loss_func = CosineDirectionLoss()
        self.waypoint_loss_scale = 1.0
        self.special_token_dict = None
        ## action
        # self.action_emb = nn.Embedding(1, config.hidden_size)
        # self.actions_fc = nn.Sequential(
        #     nn.Linear(config.hidden_size, config.hidden_size // 2),
        #     nn.ReLU(),
        #     nn.Linear(config.hidden_size // 2, 64),
        # )
        # self.actions_output = nn.Linear(64, 8)
        
        # self.actions_loss_func = torch.nn.CrossEntropyLoss()
        # self.action_loss_scale = 1.0
        ## bbox

        # Initialize weights and apply final processing
        self.post_init()
    
    def get_special_token_id(self, special_token_dict):
        self.special_token_dict = special_token_dict
        
    def get_model(self):
        return self.model
    
    def forward_waypoint(self, hidden_states):
        bs, hidden_size = hidden_states.size()
        waypoints_feature = self.waypoints_fc(hidden_states.reshape(-1, hidden_size))
        
        predicted_waypoints = self.waypoints_output(waypoints_feature)
        return predicted_waypoints
    
    # def forward_action(self, hidden_states):
    #     bs, hidden_size = hidden_states.size()
    #     actions_feature = self.actions_fc(hidden_states.reshape(-1, hidden_size))
        
    #     predicted_actions = self.actions_output(actions_feature)
    #     return predicted_actions
    
    # def forward_bbox(self, hidden_states):
    #     bs, hidden_size = hidden_states.size()
    #     ## TODO: wmq bbox regression head.
    #     predicted_bboxes = 0
    #     return predicted_bboxes

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        pixel_values: Optional[torch.Tensor] = None,
        pixel_values_videos: Optional[torch.FloatTensor] = None,
        image_grid_thw: Optional[torch.LongTensor] = None,
        video_grid_thw: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        second_per_grid_ts: Optional[torch.Tensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        waypoints: Optional[torch.FloatTensor] = None,
        bboxes: Optional[torch.FloatTensor] = None,
        actions: Optional[torch.FloatTensor] = None,
        orientations: Optional[torch.FloatTensor] = None,
        historys: Optional[torch.FloatTensor] = None,
        return_dict: Optional[bool] = None,
        return_waypoints: Optional[bool] = False,
        **kwargs,
    ) -> Union[Tuple, CausalLMOutputWithPastUAV]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if not self.training:
            if input_ids.device != self.device:
                input_ids = input_ids.to(device=self.device)
            if attention_mask.device != self.device:
                attention_mask = attention_mask.to(device=self.device)
            if labels.device != self.device:
                labels = labels.to(device=self.device)
        history_embeds = []
        
        for idx in range(len(historys)):
            history = historys[idx]
            info = history.view(-1, 3)
            history_embed = self.history_preprocessor(info)
            history_embeds.append(history_embed)

        if inputs_embeds is None:
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            if pixel_values is not None:
                image_embeds = self.model.get_image_features(pixel_values, image_grid_thw)
                n_image_tokens = (input_ids == self.config.image_token_id).sum().item()
                n_image_features = image_embeds.shape[0]
                if n_image_tokens != n_image_features:
                    raise ValueError(
                        f"Image features and image tokens do not match: tokens: {n_image_tokens}, features {n_image_features}"
                    )

                mask = input_ids == self.config.image_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                image_mask = mask_expanded.to(inputs_embeds.device)

                image_embeds = image_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(image_mask, image_embeds)

            if pixel_values_videos is not None:
                video_embeds = self.model.get_video_features(pixel_values_videos, video_grid_thw)
                n_video_tokens = (input_ids == self.config.video_token_id).sum().item()
                n_video_features = video_embeds.shape[0]
                if n_video_tokens != n_video_features:
                    raise ValueError(
                        f"Video features and video tokens do not match: tokens: {n_video_tokens}, features {n_video_features}"
                    )

                mask = input_ids == self.config.video_token_id
                mask_unsqueezed = mask.unsqueeze(-1)
                mask_expanded = mask_unsqueezed.expand_as(inputs_embeds)
                video_mask = mask_expanded.to(inputs_embeds.device)

                video_embeds = video_embeds.to(inputs_embeds.device, inputs_embeds.dtype)
                inputs_embeds = inputs_embeds.masked_scatter(video_mask, video_embeds)

            if attention_mask is not None:
                attention_mask = attention_mask.to(inputs_embeds.device)

        inputs_embeds = inputs_embeds.to(dtype=self.waypoint_emb.weight.dtype)
        inputs_embeds[labels == WAYPOINT_LABEL_TOKEN] = self.waypoint_emb.weight
        
        outputs = self.model(
            second_per_grid_ts=second_per_grid_ts,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )
        if output_attentions and  "save_attentions" in kwargs:
            torch.save(outputs.attentions, kwargs["save_attentions"])
            
        hidden_states = outputs[0]
        waypoints_feat = hidden_states[labels == WAYPOINT_LABEL_TOKEN]     
        predicted_waypoints = self.forward_waypoint(waypoints_feat)
        # if actions is not None:
        #     actions_feat = hidden_states[labels == WAYPOINT_LABEL_TOKEN]     
        #     predicted_actions = self.forward_action(actions_feat)
        # if bboxes is not None:
        #     bboxes_feat = hidden_states[labels == BBOX_LABEL_TOKEN]
        #     predicted_bboxes = self.forward_waypoint(bboxes_feat)
        
        if waypoints is None and return_waypoints:
            return predicted_waypoints
        
        loss = None
        
        logits = self.lm_head(hidden_states)
        if labels is not None:
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            # shift_labels = torch.where(
            #     shift_labels == WAYPOINT_LABEL_TOKEN,
            #     torch.tensor(-100, device=shift_labels.device, dtype=shift_labels.dtype),
            #     shift_labels
            # )
            mask = (shift_labels == WAYPOINT_LABEL_TOKEN)
            shift_labels = shift_labels.masked_fill(mask, torch.tensor(WAYPOINT_INPUT_TOKEN_QWEN, device=shift_labels.device, dtype=shift_labels.dtype))
            mask_shifted = torch.zeros_like(mask)
            mask_shifted[..., 1:] = mask[..., :-1]
            shift_labels = shift_labels.masked_fill(mask_shifted, torch.tensor(-100, device=shift_labels.device, dtype=shift_labels.dtype))
            
            # assert shift_labels.dtype == torch.long
            # Flatten the tokens
            # loss_fct = CrossEntropyLoss()
            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            shift_logits = shift_logits.view(-1, self.vocab_size)
            shift_labels = shift_labels.view(-1)
            # Enable model/pipeline parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        assert len(torch.where(labels == WAYPOINT_LABEL_TOKEN)[0]) == waypoints.shape[0]
        if waypoints is not None:
            if self.use_angle_and_norm_loss:
                waypoint_loss = self.waypoint_loss_scale * self.waypoints_loss_func(predicted_waypoints[:, 3], waypoints[:, 3])
                angle_loss = self.waypoint_loss_scale * self.angle_loss_func(predicted_waypoints[:, :3], waypoints[:, :3])
                loss += waypoint_loss + angle_loss
            else:
                loss += self.waypoint_loss_scale * self.waypoints_loss_func(predicted_waypoints, waypoints) 
        # if bboxes is not None:
        #     ## TODO: wmq bboxes regression loss
        #     pass
        # if actions is not None:
        #     loss += self.action_loss_scale * self.actions_loss_func(predicted_actions, actions) 
        
        if return_waypoints:
            return loss, predicted_waypoints
        
        if not return_dict:
            output = (waypoints_feat,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        
        return CausalLMOutputWithPastUAVMulLoss(
            loss=loss,
        )

AutoConfig.register("qwenvl_uav_cot", QwenConfig)
AutoModelForCausalLM.register(QwenConfig, Qwen2_5_VLCOTUAVForCausalLM)
