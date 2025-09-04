import pandas as pd
import matplotlib.pyplot as plt

# 读取两个表格（替换成你的文件路径）
file1 = "/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV_data/TravelUAV/eval_closeloop_llama7b128_all/path_len_vs.csv"
file2 = "/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV_data/TravelUAV/eval_closeloop_qwen0.5b128_all/path_len_vs.csv"


# # 读入数据
# df1 = pd.read_csv(file1).rename(columns={"path_length": "path_length_llama", "status": "status_llama"})
# df2 = pd.read_csv(file2).rename(columns={"path_length": "path_length_qwen", "status": "status_qwen"})

# # 合并
# merged = pd.merge(
#     df1[["sample", "path_length_llama", "status_llama"]],
#     df2[["sample", "path_length_qwen", "status_qwen"]],
#     on="sample",
#     how="inner"
# )


df1 = pd.read_csv(file1).rename(columns={
    "path_length": "path_length_llama",
    "status": "status_llama",
    "start_fail": "start_fail_llama"
})
df2 = pd.read_csv(file2).rename(columns={
    "path_length": "path_length_qwen",
    "status": "status_qwen",
    "start_fail": "start_fail_qwen"
})

# 合并
merged = pd.merge(
    df1[["sample", "path_length_llama", "status_llama", "start_fail_llama"]],
    df2[["sample", "path_length_qwen", "status_qwen", "start_fail_qwen"]],
    on="sample",
    how="inner"
)

# 规范化 start_fail 为 0/1（兼容数字/字符串），便于过滤
for col in ["start_fail_llama", "start_fail_qwen"]:
    merged[col] = pd.to_numeric(merged[col], errors="coerce").fillna(0).astype(int)

# ===== 关键过滤：去掉任意一方出现 (status == fail) 且 (start_fail == 1) 的样本 =====
print(((merged["status_llama"] == "fail") & (merged["start_fail_llama"] == 1)).sum())
print(((merged["status_qwen"] == "fail") & (merged["start_fail_qwen"] == 1)).sum())
mask_drop = ((merged["status_llama"] == "fail") & (merged["start_fail_llama"] == 1)) | \
            ((merged["status_qwen"] == "fail") & (merged["start_fail_qwen"] == 1))
print(mask_drop.sum())
merged = merged[~mask_drop].copy()
print(len(merged))

# 定义颜色分类规则
def assign_color(row):
    s1, s2 = row["status_llama"], row["status_qwen"]
    if s1 == "fail" and s2 == "fail":
        return "#d62728"
    elif (s1 in ["success", "oracle"]) and (s2 in ["success", "oracle"]):
        return "#2ca02c"
    elif (s1 in ["success", "oracle"]) and (s2 == "fail"):
        return "#ff7f0e"
    elif (s1 == "fail") and (s2 in ["success", "oracle"]):
        return "black"
    else:
        return "gray"  # 兜底

merged["color"] = merged.apply(assign_color, axis=1)


# 绘制散点图：green 和 orange 先画（大），red 和 black 后画（小）
plt.figure(figsize=(8,6))

# # green & orange
# subset_go = merged[merged["color"].isin(["#2ca02c", "#ff7f0e"])]
# plt.scatter(
#     subset_go["path_length_llama"], 
#     subset_go["path_length_qwen"], 
#     c=subset_go["color"], 
#     s=10,   # 点大
#     # alpha=0.7,
#     # label="Green/Orange"
# )

# # red & black
# subset_rb = merged[merged["color"].isin(["#d62728", "black"])]
# plt.scatter(
#     subset_rb["path_length_llama"], 
#     subset_rb["path_length_qwen"], 
#     c=subset_rb["color"], 
#     s=8,   # 点小
#     alpha=0.7,
#     # label="Red/Black"
# )

for color, label, size in [
    ("#2ca02c", "Both success/oracle",10),
    ("#ff7f0e", "Qwen0.5b fail & LLaMA7b success/oracle", 10),
    ("#d62728", "Both fail", 8),
    ("black", "Qwen0.5b success/oracle & LLaMA7b fail", 8)
]:
    subset = merged[merged["color"] == color]
    plt.scatter(
        subset["path_length_llama"], 
        subset["path_length_qwen"], 
        c=color, 
        s=size, 
        # alpha=0.7, 
        label=label
    )
    
# 打印 merged 中 qwen success，但是 llama fail 的 sample 名字
# mask = (merged["status_qwen"].isin(["success", "oracle"])) & (merged["status_llama"].isin(["fail"]))
# print(merged.loc[mask, "sample"].tolist())

# 计算 qwen 和 llama 的 path_length 比值
merged["path_ratio"] = merged["path_length_qwen"] / merged["path_length_llama"]

# 找出符合条件的样本，并按比值从小到大排序
mask = (merged["status_qwen"].isin(["fail"])) & (merged["status_llama"].isin(["fail"]))
sorted_samples = merged.loc[mask, ["sample", "path_ratio"]].sort_values(by="path_ratio")

# 打印按 path_length 比值从小到大排序的样本名称
# print(sorted_samples)
output_file = "/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/figures/path_len_llama7b_vs_qwen05b_nosf_bothfail.csv"
sorted_samples[["sample", "path_ratio"]].to_csv(output_file, index=False)
# # 绘制散点图
# plt.figure(figsize=(8,6))
# plt.scatter(
#     merged["path_length_llama"], 
#     merged["path_length_qwen"], 
#     c=merged["color"],
# )
plt.legend()
plt.xlabel("Path Length (LLaMA-7B)")
plt.ylabel("Path Length (Qwen-0.5B)")
plt.title("LLaMA-7B v.s. Qwen-0.5B Path Length Comparison")
plt.grid(True)
# plt.show()
out_path = "/mnt/data1/workspace/wmq/TravelUAV_ws/TravelUAV/figures/path_len_compare_llama7b_qwen05b_nosf.png"
import os
os.makedirs(os.path.dirname(out_path), exist_ok=True)
plt.savefig(out_path, dpi=200, bbox_inches="tight")
