import json
import os

# 原始 JSON 路径
src_path = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/data/cot_uav_dataset/our_trainset_onlycot_1_pseudo.json"

# 读取 JSON 列表
with open(src_path, "r", encoding="utf-8") as jf:
    json_data = json.load(jf)

total = len(json_data)
parts = 16
step = total // parts

base, ext = os.path.splitext(src_path)

for idx in range(parts):

    start = idx * step
    end = start + step
    if idx == parts - 1:
        end = total   # 最后一段吃掉全部剩余数据

    sub_list = json_data[start:end]

    save_path = f"{base}_{idx}{ext}"   # 例如 data_0.json, data_1.json ...

    with open(save_path, "w", encoding="utf-8") as out_f:
        json.dump(sub_list, out_f, ensure_ascii=False, indent=2)

    print(f"Saved: {save_path}  [{start} → {end})  size={len(sub_list)}")
