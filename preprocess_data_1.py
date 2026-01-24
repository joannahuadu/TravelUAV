import os
import json

def extract_index(dirname: str) -> int:
    """
    pseudo_r1osall      -> 0
    pseudo_r1osall_3    -> 3
    """
    if "_" not in dirname:
        return 0
    try:
        return int(dirname.split("_")[-1])
    except ValueError:
        return 0
    
root_dir = "/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV_data/TravelUAV"
output_json = "pseudo_merged.json"

all_items = []

pseudo_dirs = [
    d for d in os.listdir(root_dir)
    if os.path.isdir(os.path.join(root_dir, d)) and "pseudo" in d
]

pseudo_dirs = sorted(pseudo_dirs, key=extract_index)

for dirnames in pseudo_dirs:
    if "pseudo" in dirnames:
        print(f"processing {dirnames}")
        jsonl_path = os.path.join(root_dir, dirnames, "eval_results.jsonl")
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                line = line.rstrip()
                if line.endswith(","):
                    line = line[:-1]
                if not line:
                    print("continue empty line")
                    continue
                try:
                    obj = json.loads(line)
                    all_items.append(obj)
                except json.JSONDecodeError as e:
                    print(f"[WARN] JSON 解析失败: {jsonl_path}:{line_num} -> {e}")
    print(f"len: {len(all_items)}")

with open(output_json, "w", encoding="utf-8") as f:
    json.dump(all_items, f, ensure_ascii=False, indent=2)

print(f"完成：共合并 {len(all_items)} 条记录，已写入 {output_json}")
