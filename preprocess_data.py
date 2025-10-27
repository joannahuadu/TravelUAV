import json
import re
from pathlib import Path
import os

# ======= 配置输入/输出路径 =======
jsonl_path = Path("/home/fit/qiuhan/WORK/wmq/Visual-CoT/jobs/1009/traveluav_cot_train.jsonl")  # 第一个：jsonl，每行一个样本
pairs_path = Path("/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/data/uav_dataset/trainset.json")   # 第二个：json 列表，每个元素含 {"json": "...", "frame": N}
out_path = Path("/home/fit/qiuhan/WORK/wmq/TravelUAV_ws/TravelUAV/data/cot_uav_dataset/traveluav_trainset.json")  # 输出：json 列表

# ======= 工具函数 =======
SUBGOAL_RE = re.compile(r"Subgoal:\s*(.+?)(?:[。\.]\s*|$)", flags=re.IGNORECASE | re.DOTALL)

def extract_subgoal(gpt_value: str) -> str:
    """
    从 conversations 里 gpt 的 value 文本中提取 `Subgoal: xxx` 的短语。
    兼容句号是中英文或省略的情况，尽量只取第一段。
    """
    if not gpt_value:
        return ""
    m = SUBGOAL_RE.search(gpt_value)
    if m:
        return m.group(1).strip()
    # 兜底：如果没有明确的 Subgoal 格式，就原样返回（或返回空串）
    return gpt_value.strip()

def image_path_has_scene(sample: dict, scene_dir: str) -> bool:
    """
    检查第一个文件中 image 各视角路径是否都包含指定的 scene 目录名。
    """
    imgs = sample.get("image", {})
    return any(scene_dir in (p or "") for p in imgs.values())

# ======= 读取数据 =======
# 第一个文件：jsonl -> 列表
jsonl_rows = []
with jsonl_path.open("r", encoding="utf-8") as f:
    for line_num, line in enumerate(f, 1):
        line = line.strip()
        if not line:
            continue
        try:
            jsonl_rows.append(json.loads(line))
        except json.JSONDecodeError as e:
            raise ValueError(f"JSONL 第 {line_num} 行解析失败: {e}") from e

# 第二个文件：json 列表
with pairs_path.open("r", encoding="utf-8") as f:
    pairs = json.load(f)
if not isinstance(pairs, list):
    raise ValueError("第二个文件应为 JSON 列表。")

# ======= 逐项对齐校验并合并 =======
merged = []
term = 0
for idx, (row, pair) in enumerate(zip(jsonl_rows, pairs[:len(jsonl_rows)])):
    # 1) frame = question_id + 1
    qid = row.get("question_id")
    frame = pair.get("frame")
    # assert isinstance(qid, int), f"第 {idx} 项 question_id 非整数：{qid}"
    # assert isinstance(frame, int), f"第 {idx} 项 frame 非整数：{frame}"
    # if frame != qid + 1:
    #     qid -= term
    # assert frame == qid + 1, f"第 {idx} 项校验失败：frame({frame}) != question_id({qid}) + 1"
    # term += 1
    # 2) 都是指定场景路径
    #    第二个文件的 'json' 字段必须等于期望相对路径
    json_rel = pair.get("json")
    parts = json_rel.split("/")
    json_rel = os.path.join(*parts[:-1])   # ✅ 正确

    #    第一个文件的 image 路径应包含指定场景目录
    assert image_path_has_scene(row, json_rel), f"第 {idx} 项 image 路径不属于指定场景目录：{EXPECTED_SCENE_DIR}"

    # 提取 subgoal
    conversations = row.get("conversations", [])
    gpt_msgs = [c for c in conversations if c.get("from") == "gpt"]
    assert gpt_msgs, f"第 {idx} 项未找到 from='gpt' 的对话。"
    subgoal_text = extract_subgoal(gpt_msgs[-1].get("value", ""))  # 取最后一条 gpt 回复更稳妥

    # 拷贝 bbox
    bbox = row.get("bbox")
    if bbox is None:
        bbox = {}
    assert isinstance(bbox, dict), f"第 {idx} 项 bbox 缺失或类型错误。"

    # 生成新字典：以第二个文件格式为基，附加 subgoal 和 bbox
    new_entry = dict(pair)  # 保留原有 'json' 和 'frame'
    new_entry["subgoal"] = subgoal_text
    new_entry["bbox"] = bbox
    new_entry["dataset"] = row.get("dataset")
    merged.append(new_entry)

# ======= 写出结果 =======
with out_path.open("w", encoding="utf-8") as f:
    json.dump(merged, f, ensure_ascii=False, indent=2)

print(f"Done. 合并得到 {len(merged)} 条记录：{out_path}")
