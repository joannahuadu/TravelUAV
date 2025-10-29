#!/usr/bin/env python3
"""
Visualize attention by averaging heads for selected layers across all steps.

Input file pattern example:
/mnt/data1/.../attention/ab098a2d-0c05-41dc-b568-1af9b5a3b3e3_0.pt
- Each .pt file corresponds to one "step" (suffix number after underscore).
- Each file stores a tuple/list of length num_layers (e.g., 32).
- Each item is a Tensor of shape (batch=1, heads=32, N, N).
Goal:
- For each step, average over heads per selected layers (first, middle, last).
- Save a 1x3 heatmap (matplotlib) per step as PNG to an output directory.
"""

import os
import re
import argparse
from typing import List, Tuple, Union

import torch
import numpy as np
import matplotlib.pyplot as plt

def find_step_files(dir_path: str) -> List[str]:
    files = []
    if not os.path.isdir(dir_path):
        raise FileNotFoundError(f"Directory not found: {dir_path}")
    for fname in os.listdir(dir_path):
        if fname.endswith(".pt"):
            files.append(os.path.join(dir_path, fname))
    # sort by step number that appears as _<num>.pt at the end of filename
    def step_key(path):
        m = re.search(r'_(\d+)\.pt$', os.path.basename(path))
        return int(m.group(1)) if m else -1
    files = [f for f in files if re.search(r'_(\d+)\.pt$', os.path.basename(f))]
    files.sort(key=step_key)
    return files

def select_layers(num_layers: int) -> Tuple[int, int, int]:
    first_idx = 0
    middle_idx = num_layers // 2  # e.g., 32 -> 16
    last_idx = num_layers - 1
    return first_idx, middle_idx, last_idx

def average_over_layers_per_head(layers_attn, head_indices, to_float=True):
    """
    layers_attn: list/tuple，长度 = num_layers
        - 每个元素是 Tensor 形状 (1, num_heads, N, N)
    head_indices: 选择要展示的头列表，例如 [0, 15, 31]
    返回:
        mats_per_head: List[np.ndarray]，长度 = len(head_indices)
        其中每个是 (N, N)，等于在所有层上对该 head 的注意力矩阵求平均
    """
    # 取 N, N
    sample = layers_attn[0]
    if to_float:
        sample = sample.to(torch.float32)
    _, H, N, _ = sample.shape

    mats_per_head = []
    for h in head_indices:
        # 累加该 head 在所有层的矩阵
        acc = None
        for L, t in enumerate(layers_attn):
            if to_float:
                t = t.to(torch.float32)
            # 取该层的第 h 个头 -> (1, N, N)
            th = t[:, h, :, :]  # (1, N, N)
            if acc is None:
                acc = th.clone()
            else:
                acc += th
        # 平均并去掉 batch 维 -> (N, N)
        mean_mat = (acc / len(layers_attn)).squeeze(0).detach().cpu().numpy()
        mats_per_head.append(mean_mat)
    return mats_per_head

def plot_step_layers(step_img: List[np.ndarray], step_num: int, layer_indices: Tuple[int,int,int], out_dir: str):
    """
    step_img: list of length num_layers, each is (N, N) ndarray averaged over heads
    """
    l1, lm, ll = layer_indices
    sel = [("Layer 0", step_img[l1]), (f"Layer {lm}", step_img[lm]), (f"Layer {ll}", step_img[ll])]

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, (title, mat) in zip(axes, sel):
        im = ax.imshow(mat, aspect='equal')  # no explicit colormap to comply with constraints
        ax.set_title(title, fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"Attention (Avg over Heads) — Step {step_num}", fontsize=12)
    fig.tight_layout(rect=[0, 0.0, 1, 0.95])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"attention_step_{step_num:06d}.png")
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path

def plot_all_steps_grid(all_steps, layer_indices, step_nums, out_path,
                            dpi=300, img_start=80, img_len=85, cmap="inferno",
                            draw_vlines=True, q_lo=0.01, q_hi=0.99, remove_diag=False):
    num_steps = len(all_steps)
    heatmap_size = 2.0
    fig_w = heatmap_size * num_steps
    fig_h = heatmap_size * 3

    fig, axes = plt.subplots(num_steps, 3, figsize=(fig_h, fig_w))

    if num_steps == 1:
        axes = np.expand_dims(axes, 1)

    layer_names = [f"Head {layer_indices[0]}",
                   f"Head {layer_indices[1]}",
                   f"Head {layer_indices[2]}"]

    img_end = img_start + img_len

    # ---- 全局分位数归一化 ----
    mats = [m for step in all_steps for m in step]
    vals = np.concatenate([m.ravel() for m in mats])
    vmin = float(np.quantile(vals, 0.01))   # 下限 = 1% 分位
    vmax = float(np.quantile(vals, 0.99))   # 上限 = 99% 分位

    im = None
    for col, (step_data, step_num) in enumerate(zip(all_steps, step_nums)):
        for row, (layer_name, mat) in enumerate(zip(layer_names, step_data)):
            ax = axes[col, row]

            # ---- 去掉对角线 ----
            mat_no_diag = mat.copy()
            # np.fill_diagonal(mat_no_diag, 0.0)

            # ---- 绘制热力图 ----
            im = ax.imshow(mat_no_diag, aspect="equal",
                           vmin=vmin, vmax=vmax, cmap=cmap)

            # 标出图片 token 区间
            ax.axvline(x=img_start, color="white", linestyle="--", linewidth=0.2)
            ax.axvline(x=img_end,   color="white", linestyle="--", linewidth=0.2)
            
            ax.axvline(x=30, color="red", linestyle="--", linewidth=1.2)
            # ax.axvline(x=img_end,   color="red", linestyle="--", linewidth=1.2)

            if row == 0:
                ax.set_title(f"Step {step_num}", fontsize=8)
            if col == 0:
                ax.set_ylabel(layer_name, fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

    # ---- 全局色条 ----
    cbar = fig.colorbar(im, ax=axes.ravel().tolist(), fraction=0.015, pad=0.01)
    cbar.ax.tick_params(labelsize=6)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True,
                        help="Directory containing *_<step>.pt attention files.")
    parser.add_argument("--output_dir", type=str, default="attention_viz",
                        help="Directory to save output PNGs.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--limit", type=int, default=0,
                        help="Optional: limit number of steps to process (0 = all).")
    args = parser.parse_args()

    files = find_step_files(args.input_dir)
    if not files:
        raise FileNotFoundError(f"No step .pt files found in {args.input_dir}")

    head_indices = [0, 15, 31]

    all_steps = []  # 行：head；列：step
    step_nums = []

    for i, fpath in enumerate(files):
        fname = os.path.basename(fpath)
        traj_id = fname.split("_")[0]
        if traj_id not in ["e4fb8f38-eda1-4964-a569-2ba1eed05cf3"]:
            continue
        if args.limit and i >= args.limit:
            break
        m = re.search(r'_(\d+)\.pt$', fname)
        step_num = int(m.group(1)) if m else i

        data = torch.load(fpath, map_location=args.device)  # tuple/list of 32 layers
        # 对“层”做平均，分别得到每个指定头的 (N, N)
        mats_per_head = average_over_layers_per_head(data, head_indices, to_float=True)
        # for h_idx, M in enumerate(mats_per_head):
        all_steps.append(mats_per_head)
        step_nums.append(step_num)

    out_path = os.path.join(args.output_dir, f"{traj_id}_attn_heads.png")
    plot_all_steps_grid(
        all_steps, head_indices, step_nums, out_path,
        dpi=300, img_start=80, img_len=85, cmap="inferno",
        draw_vlines=True, q_lo=0.01, q_hi=0.99, remove_diag=False
    )
    print(f"[DONE] Saved combined plot (heads x steps): {out_path}")

if __name__ == "__main__":
    main()
