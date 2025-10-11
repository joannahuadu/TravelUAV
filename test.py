# # # # import torch
# # # # import os
# # # # model_path = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/work_dirs/llama-vid-7b-pretrain-224-uav-full-data-lora32_bs128"
# # # # non_lora_weights1 = torch.load(os.path.join(model_path, 'non_lora_trainables.bin'), map_location='cpu')
# # # # # non_lora_weights2 = torch.load(os.path.join(model_path, "checkpoint-2500", 'non_lora_trainables.bin'), map_location='cpu')
# # # # mm_projector_weights1 = torch.load(os.path.join("/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/work_dirs/llama-vid-7b-pretrain-224-uav-full-data-lora32_bs128/checkpoint-500/non_lora_trainables.bin"), map_location='cpu')
# # # # # mm_projector_weights2 = torch.load("/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/work_dirs/llama-vid-7b-pretrain-224-uav-full-data-lora32_bs128/mm_projector/checkpoint-2500.bin", map_location='cpu')

# # # # print(mm_projector_weights1.keys())
# # # # print(non_lora_weights1.keys())
# # # # k = 'base_model.model.lm_head.weight'
# # # # ke = 'lm_head.weight'
# # # # print(torch.sum(non_lora_weights1[k] == mm_projector_weights1[ke]))
# # # # print(non_lora_weights1[k].shape.numel())
# # # # print(non_lora_weights1[k])


# # # # from transformers import Qwen2_5_VLForConditionalGeneration, AutoProcessor

# # # # model = Qwen2_5_VLForConditionalGeneration.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")
# # # # processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-VL-7B-Instruct")

# # # # # 构造 conversation
# # # # conversation = [
# # # #     {
# # # #         "role": "user",
# # # #         "content": [
# # # #             {"type": "text", "text": "请描述这两张图片的区别和相似点。"},
# # # #             {"type": "image", "image": "/path/to/image1.jpg"},
# # # #             {"type": "image", "image": "/path/to/image2.jpg"},
# # # #         ]
# # # #     }
# # # # ]

# # # # # 转 prompt / inputs
# # # # prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
# # # # # processor 将会把 images 提取出来处理
# # # # inputs = processor(
# # # #     text=[prompt],
# # # #     images=[image for m in conversation for image in m["content"] if image.get("type") == "image"],
# # # #     padding=True,
# # # #     return_tensors="pt"
# # # # ).to(model.device)

# # # # # 推理
# # # # output_ids = model.generate(**inputs, max_new_tokens=64)
# # # # res = processor.batch_decode(output_ids, skip_special_tokens=True)
# # # # print(res)


# # # # import json
# # # # with open('/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV_data/TravelUAV/NewYorkCity/b096e59f-9e34-482f-ab11-1ac3507aba06/merged_data.json', 'r') as f:
# # # #     data = json.load(f)
    
# # # # print(data['image_feature_path'])


# # # import torch
# # # a = torch.load("/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV_data/TravelUAV/NewYorkCity/ffe30298-53cc-424a-9582-4811f6724d7d/rgb_imgs_336.tensor")
# # # print(a.max())
# # # print(a.min())


# # import argparse
# # import multiprocessing
# # import torch
# # import os
# # import tqdm
# # import numpy as np
# # from transformers.models.clip import CLIPImageProcessor
# # from typing import Dict, Optional, Sequence, List

# # from PIL import Image
# # from llava.model import *
# # from llamavid.model import *

# # RGB_FOLDER = ['frontcamera', 'leftcamera', 'rightcamera', 'rearcamera', 'downcamera']

# # def arg_parse():
# #   parser = argparse.ArgumentParser(description="split video clip")
# #   parser.add_argument("--root_dir",
# #                       default='/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV_data/TravelUAV',
# #                       help='path to your dataset root dir')
# #   parser.add_argument("--map_list",
# #                         default=['NewYorkCity', 'ModernCityMap', 'NYCEnvironmentMegapa', 'TropicalIsland', 'ModularPark', 'Carla_Town01', 'Carla_Town02', 'Carla_Town03', 'Carla_Town04','Carla_Town05', 'Carla_Town06', 'Carla_Town07', 'Carla_Town10HD', 'Carla_Town15',
# #                                  'BattlefieldKitDesert', 'BrushifyCountryRoads', 'BrushifyForestPack', 'BrushifyUrban', 'Japanese_Street', 'London_Street', 'NordicHarbour', 'WesterTown'],
# #                       nargs="+",
# #                       help='processed map name')
# #   parser.add_argument("--workers",
# #                       default=16,
# #                       help='multiprocessing workers num')
# #   opt = parser.parse_args()
# #   return opt

# # clip_config = {
# #   "crop_size": {
# #     "height": 224,
# #     "width": 224
# #   },
# #   "do_center_crop": True,
# #   "do_convert_rgb": True,
# #   "do_normalize": True,
# #   "do_rescale": True,
# #   "do_resize": True,
# #   "image_mean": [
# #     0.48145466,
# #     0.4578275,
# #     0.40821073
# #   ],
# #   "image_std": [
# #     0.26862954,
# #     0.26130258,
# #     0.27577711
# #   ],
# #   "resample": 3,
# #   "rescale_factor": 0.00392156862745098,
# #   "size": {
# #     "shortest_edge": 224
# #   }
# # }
# # args = arg_parse()
# # # processer = CLIPImageProcessor(**clip_config)
# # from transformers import AutoProcessor, AutoModelForCausalLM
# # import transformers
# # import sys
# # sys.path.append("/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/llamavid/train/train_uav/")
# # from train_uav_notice import ModelArguments, DataArguments, TrainingArguments
# # model_name_or_path = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/llava-v1.5-7b"

# # config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

# # bnb_model_from_pretrained_args = dict(torch_dtype=torch.bfloat16)
# # model = LlavaUAVForCausalLM.from_pretrained(
# #     model_name_or_path,
# #     use_angle_and_norm_loss=True,
# #     config=config,
# #     cache_dir=None,
# #     **bnb_model_from_pretrained_args
# # )

# # model_args = ModelArguments(
# #     model_name_or_path="/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/llava-v1.5-7b/",
# #     version="imgsp_uav",
# #     freeze_backbone=False,
# #     tune_mm_mlp_adapter=True,
# #     tune_waypoint_predictor=True,
# #     vision_tower="/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/clip-vit-large-patch14-336",
# #     image_processor=None,
# #     mm_vision_select_layer=-2,
# #     pretrain_mm_mlp_adapter=None,
# #     mm_projector_type="mlp2x_gelu",
# #     mm_use_im_start_end=False,
# #     mm_use_im_patch_token=False,
# #     mm_patch_merge_type="flat",
# #     mm_vision_select_feature="patch",
# #     bert_type="qformer_pretrain_freeze",
# #     num_query=32,
# #     pretrain_qformer=None,
# #     compress_type="mean",
# #     use_angle_and_norm_loss=True,
# # )

# # model.get_model().initialize_vision_modules(
# #     model_args=model_args,
# #     fsdp=""
# # )
# # vision_tower = model.get_vision_tower()
# # processer = vision_tower.image_processor

# # if __name__ == "__main__":
# #   def worker(traj_dir):
# #     traj_camera_list = []
# #     for idx,camera_name in enumerate(RGB_FOLDER):
# #       if len(os.listdir(os.path.join(traj_dir,camera_name)))==0:
# #         print(f"{os.path.join(traj_dir,camera_name)} no image")        
# #       traj_camera_list.append(sorted([os.path.join(traj_dir, camera_name, filename) for filename in os.listdir(os.path.join(traj_dir,camera_name))]))
# #     assert(len(traj_camera_list[0]) == len(traj_camera_list[1]) == len(traj_camera_list[2]) == len(traj_camera_list[3]))
# #     traj_frames = []
# #     for idx in range(len(traj_camera_list[0])):
# #       batch = []
# #       for iid in range(len(RGB_FOLDER)):
# #           batch.append(traj_camera_list[iid][idx])
# #       traj_frames.append(batch)
# #     traj_imgs = []
# #     for frame_imgs in traj_frames:
# #       images = [Image.open(img_path).convert('RGB') for img_path in frame_imgs]
# #       images = np.stack(images, axis=0)
# #       traj_imgs.append(images)
# #     if len(traj_imgs)!=0:
# #       imgs = np.array(traj_imgs).reshape(-1, 256, 256, 3)
# #       imgs = processer.preprocess(imgs, return_tensors='pt')['pixel_values'].to(dtype=torch.bfloat16)
# #       torch.save(imgs, os.path.join(traj_dir, 'rgb_imgs_336.tensor'))
# #     else:
# #       print(f"{traj_dir} no image")
  
# #   for map_name in args.map_list:
# #     directory_path = os.path.join(args.root_dir, map_name)
# #     print(directory_path)
# #     traj_list = []
# #     for traj in tqdm.tqdm(os.listdir(directory_path)):
# #         traj_dir = os.path.join(directory_path, traj)
# #         output_file = os.path.join(traj_dir, 'rgb_imgs_336.tensor')
# #         if not os.path.exists(output_file):
# #             print(f"{output_file} 未生成")
# #             # traj_list.append(traj_dir)
# #         else:
# #             a = torch.load(output_file)
# #             if a.shape[0]==0:
# #                 print(f"{output_file} is 0, {a.shape}")
            
# #     # for traj in traj_list:
# #     #   worker(traj)
# #     # # with multiprocessing.Pool(args.workers) as p:
# #     #   # r = list(tqdm.tqdm(p.imap_unordered(worker, traj_list), total=len(traj_list)))
# #     print(directory_path, 'finished.')


# # from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
# # from PIL import Image

# # model_path = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/Qwen2.5-VL-7B-Instruct"

# # # 1) 用 AutoProcessor（而不是只用 AutoTokenizer）
# # processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
# #     model_path, device_map="auto", trust_remote_code=True
# # ).eval()

# # # 2) 读入本地图片为 PIL.Image；不要把路径塞进 url 字段
# # image_paths = [
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_0.png",
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_1.png",
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_2.png",
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_3.png",
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_4.png",
# # ]
# # images = [Image.open(p).convert("RGB") for p in image_paths]

# # # 3) chat 模板里只放占位符 {"type": "image"}，数量与 images 对齐；
# # #    文本放最后（可加你的“五个视角”说明）
# # sentence = "请根据这些图片描述场景。"
# # user_content = [{"type": "image"} for _ in images]
# # user_content.append({
# #     "type": "text",
# #     "text": "These five images respectively come from five perspectives: "
# #             "frontcamera, leftcamera, rightcamera, rearcamera, downcamera.\n\n" + sentence
# # })

# # messages = [
# #     {
# #         "role": "system",
# #         "content": [{"type": "text", "text": "You are a helpful assistant."}]
# #     },
# #     {"role": "user", "content": user_content},
# # ]

# # # 4) 先用 processor 生成带占位的 prompt 字符串（注意 tokenize=False）
# # prompt = processor.apply_chat_template(
# #     messages,
# #     tokenize=False,
# #     add_generation_prompt=True
# # )

# # # 5) 关键：把 prompt + images 一起喂给 processor，生成模型需要的张量
# # inputs = processor(
# #     text=prompt,
# #     images=images,              # <- 真正的像素在这里传入
# #     return_tensors="pt"
# # ).to(model.device)

# # # 6) 生成与解码（只取生成的续写部分更干净）
# # generated_ids = model.generate(**inputs, max_new_tokens=512)
# # # 去掉提示词的前缀只保留新生成的内容
# # gen_only = generated_ids[:, inputs["input_ids"].shape[-1]:]
# # output_text = processor.batch_decode(gen_only, skip_special_tokens=True)[0]

# # print(output_text)



# # from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
# # from PIL import Image

# # model_path = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/Qwen2.5-VL-7B-Instruct"

# # processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
# # model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
# #     model_path, device_map="auto", trust_remote_code=True
# # ).eval()

# # # 五张图像
# # image_paths = [
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_0.png",
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_1.png",
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_2.png",
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_3.png",
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_4.png",
# # ]
# # images = [Image.open(p).convert("RGB") for p in image_paths]

# # sentence = "请根据这些图片描述场景。"
# # user_content = [{"type": "image"} for _ in images]
# # user_content.append({
# #     "type": "text",
# #     "text": "These five images respectively come from five perspectives: "
# #             "frontcamera, leftcamera, rightcamera, rearcamera, downcamera.\n\n" + sentence
# # })
# # messages = [
# #     {
# #         "role": "system",
# #         "content": [{"type": "text", "text": "You are a helpful assistant."}]
# #     },
# #     {"role": "user", "content": user_content},
# # ]

# # # 构造单条 prompt
# # prompt = processor.apply_chat_template(
# #     messages, tokenize=False, add_generation_prompt=True
# # )

# # # ===== Batch size 32 =====
# # batch_prompts = [prompt for _ in range(32)]
# # batch_images = [images for _ in range(32)]  # 每条样本对应同一组五张图

# # # 编码
# # inputs = processor(
# #     text=batch_prompts,
# #     images=batch_images,   # 这里是 list of list[Image]
# #     return_tensors="pt",
# #     padding=True
# # ).to(model.device)

# # # 推理
# # generated_ids = model.generate(**inputs, max_new_tokens=512)

# # # 解码（取续写部分）
# # gen_only = [
# #     out[input_ids.shape[-1]:] for out, input_ids in zip(generated_ids, inputs["input_ids"])
# # ]
# # output_texts = processor.batch_decode(gen_only, skip_special_tokens=True)

# # for i, txt in enumerate(output_texts):
# #     print(f"=== Sample {i} ===")
# #     print(txt)


# # import torch
# # from transformers import AutoProcessor, LlavaNextForConditionalGeneration
# # from PIL import Image

# # # ===== 1) 选择 LLaVA 多图模型（示例：LLaVA-1.6 7B）=====
# # # 也可换成： "liuhaotian/llava-v1.6-mistral-7b" / "llava-hf/llava-v1.6-34b" / "llava-hf/llava-1.6-7b-hf" 等
# # model_id = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/llava-v1.6-vicuna-7b-hf"
# # device = "cuda" if torch.cuda.is_available() else "cpu"

# # # ===== 2) 加载模型与处理器 =====
# # processor = AutoProcessor.from_pretrained(model_id)
# # model = LlavaNextForConditionalGeneration.from_pretrained(
# #     model_id,
# #     torch_dtype=torch.float16 if device == "cuda" else torch.float32,
# #     device_map="auto" if device == "cuda" else None,
# #     low_cpu_mem_usage=True,
# # ).eval()

# # # ===== 3) 准备 5 张图片 =====
# # image_paths = [
# #     # "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_0.png",
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_1.png",
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_2.png",
# #     # "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_3.png",
# #     # "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_4.png",
# # ]
# # images = [Image.open(p).convert("RGB") for p in image_paths]

# # # ===== 4) 多图对话消息（与 Qwen2.5 类似：content 里放多个 {"type":"image"}）=====
# # sentence = "请根据这些图片描述场景。"
# # user_content = [{"type": "image"} for _ in images]
# # user_content.append({
# #     "type": "text",
# #     "text": "These two images respectively come from two perspectives: "
# #             "leftcamera, rightcamera.\n\n" + sentence
# # })

# # messages = [
# #     {"role": "system", "content": [{"type": "text", "text": "You are a helpful assistant."}]},
# #     {"role": "user", "content": user_content},
# # ]

# # # ===== 5) 构造单条 prompt（LLaVA 也用 chat template）=====
# # prompt = processor.apply_chat_template(
# #     messages, tokenize=False, add_generation_prompt=True
# # )

# # # ===== 6) 批量：32 条样本，共用同一组 5 张图 =====
# # B = 4
# # batch_prompts = [prompt for _ in range(B)]
# # batch_images = [images for _ in range(B)]  # list[list[PIL.Image]]

# # # ===== 7) 编码（关键：images 传 “list of list[PIL.Image]” 与 prompt 一一对应）=====
# # inputs = processor(
# #     text=batch_prompts,
# #     images=batch_images,
# #     return_tensors="pt",
# #     padding=True
# # )
# # print(inputs['image_sizes'].shape)
# # inputs = {k: v.to(model.device) if hasattr(v, "to") else v for k, v in inputs.items()}

# # # ===== 8) 生成 =====
# # with torch.inference_mode():
# #     generated_ids = model.generate(
# #         **inputs,
# #         max_new_tokens=512,
# #         temperature=0.2,
# #         do_sample=False,
# #     )

# # # ===== 9) 只取续写部分并解码 =====
# # gen_only = [
# #     out[input_ids.shape[-1]:] for out, input_ids in zip(generated_ids, inputs["input_ids"])
# # ]
# # output_texts = processor.batch_decode(gen_only, skip_special_tokens=True)

# # for i, txt in enumerate(output_texts):
# #     print(f"=== Sample {i} ===")
# #     print(txt)


# # # Load model directly
# # from transformers import AutoProcessor, AutoModelForVision2Seq

# # processor = AutoProcessor.from_pretrained("/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/llava-v1.6-vicuna-7b-hf")
# # model = AutoModelForVision2Seq.from_pretrained("/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/llava-v1.6-vicuna-7b-hf")
# # messages = [
# #     {
# #         "role": "user",
# #         "content": [
# #             {"type": "image", "url": "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/candy.JPG"},
# #             {"type": "text", "text": "What animal is on the candy?"}
# #         ]
# #     },
# # ]
# # inputs = processor.apply_chat_template(
# # 	messages,
# # 	add_generation_prompt=True,
# # 	tokenize=True,
# # 	return_dict=True,
# # 	return_tensors="pt",
# # ).to(model.device)

# # outputs = model.generate(**inputs, max_new_tokens=40)
# # print(processor.decode(outputs[0][inputs["input_ids"].shape[-1]:]))



# # import torch
# # from transformers import AutoProcessor, LlavaNextForConditionalGeneration
# # from PIL import Image
# # from llava.model import *
# # from llamavid.model import *
# # device = "cuda" if torch.cuda.is_available() else "cpu"
# # image_paths = [
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_0.png",
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_1.png",
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_2.png",
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_3.png",
# #     "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/frame_4.png",
# # ]
# # images = [Image.open(p).convert("RGB") for p in image_paths]
# # from transformers import AutoProcessor, AutoModelForCausalLM
# # import transformers
# # import sys
# # sys.path.append("/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/llamavid/train/train_uav/")
# # from train_uav_notice import ModelArguments, DataArguments, TrainingArguments
# # import numpy as np
# # model_name_or_path = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/llava-v1.5-7b-224"

# # config = transformers.AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)

# # bnb_model_from_pretrained_args = dict(torch_dtype=torch.bfloat16)
# # model = LlavaUAVForCausalLM.from_pretrained(
# #     model_name_or_path,
# #     use_angle_and_norm_loss=True,
# #     config=config,
# #     cache_dir=None,
# #     device_map="auto" if device == "cuda" else None,
# #     **bnb_model_from_pretrained_args
# # ).eval()

# # model_args = ModelArguments(
# #     model_name_or_path="/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/llava-v1.5-7b-224/",
# #     version="imgsp_uav",
# #     freeze_backbone=False,
# #     tune_mm_mlp_adapter=True,
# #     tune_waypoint_predictor=True,
# #     vision_tower="/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/model_zoo/clip-vit-large-patch14",
# #     image_processor=None,
# #     mm_vision_select_layer=-2,
# #     pretrain_mm_mlp_adapter=None,
# #     mm_projector_type="mlp2x_gelu",
# #     mm_use_im_start_end=False,
# #     mm_use_im_patch_token=False,
# #     mm_patch_merge_type="flat",
# #     mm_vision_select_feature="patch",
# #     bert_type="qformer_pretrain_freeze",
# #     num_query=32,
# #     pretrain_qformer=None,
# #     compress_type="mean",
# #     use_angle_and_norm_loss=True,
# # )

# # model.get_model().initialize_vision_modules(
# #     model_args=model_args,
# #     fsdp=""
# # )
# # vision_tower = model.get_vision_tower()
# # processer = vision_tower.image_processor
# # pro_images = []
# # for image in images:
# #       imgs = np.array(image).reshape(-1, 256, 256, 3)
# #       imgs = processer.preprocess(imgs, return_tensors='pt')['pixel_values'].to(dtype=torch.bfloat16)
# #       pro_images.append(imgs)
# # pro_images = np.array(pro_images)
# # from transformers import AutoTokenizer
# # from llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN
# # from llava.conversation import conv_templates
# # from llava.mm_utils import tokenizer_image_token

# # tokenizer = AutoTokenizer.from_pretrained(
# #     model_name_or_path,
# #     use_fast=False,
# #     trust_remote_code=True
# # )
# # if tokenizer.pad_token is None:
# #     tokenizer.pad_token = tokenizer.eos_token  # 防止后续需要pad时报错

# # # ===== 2) 写 conversations（含 5 个 <image>）=====
# # # 选择一个合适的对话模板（llava_v1 常见；若你的权重自带自定义模板，也可替换成相应 key）
# # conv = conv_templates["llava_v1"].copy()
# # conv.system_message = "你是一名多模态无人机导航与感知助手。"

# # # 拼接 5 个 <image> 占位符（当前配置 mm_use_im_start_end=False，直接用 <image>）
# # image_tokens = "\n".join([DEFAULT_IMAGE_TOKEN] * 5)

# # # 用户输入里先放 5 张图的占位，再接具体问题/指令
# # user_prompt = (
# #     f"{image_tokens}\n"
# #     "请根据以上 5 帧图像，描述无人机的位姿变化趋势，并给出关键航点与朝向的估计。"
# # )

# # conv.append_message(conv.roles[0], user_prompt)
# # conv.append_message(conv.roles[1], None)  # 模型待回复

# # # 得到最终可喂给 tokenizer 的纯文本 prompt（其中 <image> 会按 IMAGE_TOKEN_INDEX 处理）
# # prompt = conv.get_prompt()

# # # ===== 3) 利用 llava 的 tokenizer_image_token 得到 input_ids =====
# # # 注意：这里不会处理图像张量本身；只是把文本里的 <image> 映射到 IMAGE_TOKEN_INDEX
# # input_ids = tokenizer_image_token(
# #     prompt,
# #     tokenizer,
# #     IMAGE_TOKEN_INDEX,
# #     return_tensors="pt"
# # ).to(device)

# # # （可选）做个小检查：确认确实包含了 IMAGE_TOKEN_INDEX，且一共出现 5 次
# # ids_list = input_ids[0].tolist()
# # num_image_tokens = sum(1 for t in ids_list if t == IMAGE_TOKEN_INDEX)
# # print("IMAGE_TOKEN_INDEX:", IMAGE_TOKEN_INDEX)
# # print("input_ids shape:", input_ids.shape)
# # print("num <image> tokens:", num_image_tokens)

# # # （可选）查看前若干个 token 对应的字符串，确认可读性
# # preview = tokenizer.convert_ids_to_tokens(ids_list[:60], skip_special_tokens=False)
# # print(preview)

# # data_dict = {}
# # data_dict['input_ids'] = input_ids
# # data_dict['images'] = pro_images
# # with torch.inference_mode():
# #     generated_ids = model.generate(
# #         **inputs,
# #         max_new_tokens=512,
# #         temperature=0.2,
# #         do_sample=False,
# #     )

# # # ===== 9) 只取续写部分并解码 =====
# # gen_only = [
# #     out[input_ids.shape[-1]:] for out, input_ids in zip(generated_ids, inputs["input_ids"])
# # ]
# # output_texts = processor.batch_decode(gen_only, skip_special_tokens=True)

# # for i, txt in enumerate(output_texts):
# #     print(f"=== Sample {i} ===")
# #     print(txt)


# # import json
# # with open('/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV_data/TravelUAV/NewYorkCity/b096e59f-9e34-482f-ab11-1ac3507aba06/merged_data.json', 'r') as f:
# #     data = json.load(f)
    
# # # print(data.keys)
# # print(data['conversations'])

import torch
a = torch.load("/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/work_dirs/qwen_2_5_vl-pretrain-252-uav-full-data-lora64_bs128_5e-5/mm_projector.bin")
b = torch.load("/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/work_dirs/qwen_2_5_vl-pretrain-252-uav-full-data-lora64_bs128_5e-5/checkpoint-6686/mm_projector.bin")
c = torch.load("/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/work_dirs/qwen_2_5_vl-pretrain-252-uav-full-data-lora64_bs128_5e-5/mm_projector/checkpoint-6686.bin")
d = torch.load("/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/work_dirs/qwen_2_5_vl-pretrain-252-uav-full-data-lora64_bs128_1e-4/non_lora_trainables.bin")
# d = torch.load("/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/work_dirs/qwen_2_5_vl-pretrain-252-uav-full-data-lora64_bs128/checkpoint-6686/mm_projector_1.bin")
# e = torch.load("/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/work_dirs/qwen_2_5_vl-pretrain-252-uav-full-data-lora64_bs128/checkpoint-6686/mm_projector.bin")
# f = torch.load("/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/Model/LLaMA-UAV/work_dirs/qwen_2_5_vl-pretrain-252-uav-full-data-lora64_bs128/mm_projector/checkpoint-6686.bin")
print(len(a.keys()))
print(a.keys())
print(len(b.keys()))
print(b.keys())
print(len(c.keys()))
print(c.keys())
print(len(d.keys()))
print(d.keys())
# print(len(e.keys()))
# print(e.keys())

print(torch.sum(a["base_model.model.waypoint_emb.weight"] == d["base_model.model.waypoint_emb.weight"]))
print(d["base_model.model.waypoint_emb.weight"].shape.numel())

# # print(len(c.keys()))


# import airsim

# # 默认连接 localhost:41451
# client = airsim.MultirotorClient(ip="127.0.0.1", port=25000)
# client.confirmConnection()
# print(client.call("ping"))
# from pathlib import Path
# import json
# # def get_traj_prefix(image_path: str):
# #     """提取轨迹前缀（即上级文件夹名）"""
# #     p = Path(image_path)
# #     uid = p.parts[-3] 
# #     return uid

# with open("/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/data/traj_train/train_balance.json", 'r') as f:
#     data_list = json.load(f)

# for item in data_list:
#     img_path = item['img']
#     p = Path(img_path)
#     uid = p.parts[-2] 
#     if str(uid) == "frontcamera":
#         continue
#     print(uid)