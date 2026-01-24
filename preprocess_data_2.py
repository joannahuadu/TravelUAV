import json
import tqdm

# ---------- 文件路径 ----------
trainset_path = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/data/uav_dataset/trainset.json"
post1_path = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/data/cot_uav_dataset/our_trainset_onlycot_1.json"
post2_path = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/data/cot_uav_dataset/pseudo_merged.json"
output_path = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/data/cot_uav_dataset/new_trainset.json"

# ---------- 读取 ----------
with open(trainset_path, "r", encoding="utf-8") as f:
    trainset = json.load(f)

with open(post1_path, "r", encoding="utf-8") as f:
    post1 = json.load(f)

with open(post2_path, "r", encoding="utf-8") as f:
    post2 = json.load(f)

# ---------- 建立索引 ----------
def build_index(items):
    idx = {}
    for item in items:
        key = (item["json"], item["frame"])
        idx[key] = item
    return idx

post1_idx = build_index(post1)
post2_idx = build_index(post2)

# ---------- 合并 ----------
new_trainset = []

for base in tqdm.tqdm(trainset):
    key = (base["json"], base["frame"])

    merged = dict(base)  # 保留顺序 & 原始字段

    if key in post1_idx:
        for k, v in post1_idx[key].items():
            if k not in ("json", "frame"):
                merged[k] = v
        merged["pseudo"] = False

    elif key in post2_idx:
        for k, v in post2_idx[key].items():
            if k not in ("json", "frame"):
                merged[k] = v
    else:
        print(f"Missing: {key}")
    if not merged.get("dataset", False):
        print(f"Missing1: {key}")
        

    new_trainset.append(merged)

# ---------- 写出 ----------
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(new_trainset, f, ensure_ascii=False, indent=2)

print(f"完成：共 {len(new_trainset)} 条，已写入 {output_path}")
