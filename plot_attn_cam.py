#!/usr/bin/env python3
import os
import re
import glob
import argparse
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

RGB_FOLDER = ['frontcamera', 'leftcamera', 'rightcamera', 'rearcamera', 'downcamera']
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".webp")

def list_images(img_dir: str):
    """
    遍历 img_dir 下的 5 个相机子文件夹，按 RGB_FOLDER 顺序组织。
    每个 step 返回 [front, left, right, rear, down] 五张图。
    返回: List[List[str]]，外层 len=steps，内层 len=5。
    """
    # 收集每个相机目录的文件列表
    cam_files = []
    for cam in RGB_FOLDER:
        subdir = os.path.join(img_dir, cam)
        if not os.path.isdir(subdir):
            raise FileNotFoundError(f"Missing subdir: {subdir}")
        files = []
        for ext in IMG_EXTS:
            files.extend(glob.glob(os.path.join(subdir, f"*{ext}")))
        files = sorted(files)
        if not files:
            raise FileNotFoundError(f"No images found in {subdir}")
        cam_files.append(files)

    # 对齐长度：保证每个相机文件数相同
    num_steps = len(cam_files[0])
    for f in cam_files[1:]:
        assert len(f) == num_steps, "Different cameras have unequal frame counts!"

    # 组装: 每个 step 是 [front, left, right, rear, down]
    steps = []
    for i in range(num_steps):
        step_imgs = [cam_files[j][i] for j in range(len(RGB_FOLDER))]
        steps.append(step_imgs)

    return steps

def find_step_files(attn_dir: str, traj_id: str) -> List[str]:
    pts = []
    for fname in os.listdir(attn_dir):
        if not fname.endswith(".pt"):
            continue
        if not fname.startswith(traj_id):
            continue
        if re.search(r'_(\d+)\.pt$', fname):
            pts.append(os.path.join(attn_dir, fname))
    def key(p):
        m = re.search(r'_(\d+)\.pt$', os.path.basename(p))
        return int(m.group(1)) if m else -1
    pts.sort(key=key)
    if not pts:
        raise FileNotFoundError(f"No step .pt files for traj_id={traj_id} in {attn_dir}")
    return pts

def filter_steps_by_range(step_files: List[str],
                          step_min: Optional[int],
                          step_max: Optional[int],
                          step_list: Optional[str]) -> List[str]:
    # If step_list provided (comma-separated), use it to filter
    if step_list:
        wanted = set(int(s.strip()) for s in step_list.split(",") if s.strip() != "")
        out = []
        for p in step_files:
            m = re.search(r'_(\d+)\.pt$', os.path.basename(p))
            if m and int(m.group(1)) in wanted:
                out.append(p)
        return out
    # else by range
    out = []
    for p in step_files:
        m = re.search(r'_(\d+)\.pt$', os.path.basename(p))
        if not m:
            continue
        s = int(m.group(1))
        if (step_min is None or s >= step_min) and (step_max is None or s <= step_max):
            out.append(p)
    return out

def average_layers_heads_to_matrix(layers_tuple) -> torch.Tensor:
    num_layers = len(layers_tuple)
    acc = None
    for layer_t in layers_tuple:
        t = layer_t.to(torch.float32)  # (1, H, N, N)
        t_mean_heads = t.mean(dim=1)   # (1, N, N)
        acc = t_mean_heads if acc is None else (acc + t_mean_heads)
    mat = (acc / num_layers).squeeze(0)  # (N, N)
    return mat

def extract_scores_row(mat: torch.Tensor, query_idx: int, start: int, length: int) -> np.ndarray:
    N = mat.shape[-1]
    q = N + query_idx if query_idx < 0 else query_idx
    assert 0 <= q < N, f"query_idx {query_idx} out of range for N={N}"
    assert 0 <= start < N and start + length <= N, f"slice {start}:{start+length} out of range N={N}"
    row = mat[q, :]
    sl = row[start:start+length]
    return sl.detach().cpu().numpy()

def make_cams_from_scores(scores_85: np.ndarray, pooled_hw=(4,4), has_ctx=True, out_hw=(224,224)) -> List[np.ndarray]:
    assert scores_85.shape[0] == 85
    per = scores_85.reshape(5, 17)  # (5,17)
    cams_up = []
    H, W = out_hw
    ph, pw = pooled_hw
    for i in range(5):
        vec17 = per[i]
        vec16 = vec17[:16] if has_ctx else vec17
        assert vec16.shape[0] == ph*pw, f"Expected {ph*pw} tokens, got {vec16.shape[0]}"
        cam_small = vec16.reshape(ph, pw)
        cam_t = torch.tensor(cam_small, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
        cam_up = F.interpolate(cam_t, size=(H, W), mode="bilinear", align_corners=False)
        cams_up.append(cam_up.squeeze().numpy())
    return cams_up

def load_image(path: str) -> np.ndarray:
    import imageio.v2 as imageio
    img = imageio.imread(path)
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    return img

def pick_base_images_for_step(img_files: List[str], mode: str, step_index: int, step_num: int) -> List[np.ndarray]:
    """
    mode:
      - 'single_mod': use img at (step_num % len) for all 5 overlays
      - 'single_by_index': use img at (step_index % len) for all overlays
      - 'window': use 5 consecutive images starting from step_index (wrap)
    """
    L = len(img_files)
    if mode == "single_mod":
        j = step_num % L
        base = load_image(img_files[j])
        return [base]*5
    elif mode == "single_by_index":
        j = step_index % L
        base = load_image(img_files[j])
        return [base]*5
    elif mode == "window":
        idxs = [ (step_index + k) % L for k in range(5) ]
        return [load_image(img_files[j]) for j in idxs]
    else:
        raise ValueError(f"Unknown image_mode: {mode}")

def build_big_figure(steps_images: List[List[np.ndarray]],  # 每步的5个CAM (H,W)
                     base_images: List[List[np.ndarray]],   # 每步的5张原图 (H,W,3)
                     step_nums: List[int],
                     out_path: str,
                     cmap: str = "inferno",
                     q_low: float = 0.01,
                     q_high: float = 0.99,
                     overlay_alpha: float = 0.5,
                     dpi: int = 200,
                     row_title_prefix: str = "Step"):
    """
    布局：每行 10 列 = [5个base] + [5个base+CAM]；共 num_steps 行。
    """
    import numpy as np
    import matplotlib.pyplot as plt
    import os

    num_steps = len(steps_images)
    assert num_steps == len(base_images) == len(step_nums)

    # 全局分位数归一化（保证不同step可比）
    all_vals = np.concatenate([cam.ravel() for cams in steps_images for cam in cams])
    vmin = float(np.quantile(all_vals, q_low))
    vmax = float(np.quantile(all_vals, q_high))
    scale = max(vmax - vmin, 1e-9)

    # 画布尺寸：10列（5 base + 5 overlay）
    cell_w = 2.2
    cell_h = 2.2
    num_cols = 10
    fig_w = cell_w * num_cols
    fig_h = cell_h * num_steps
    fig, axes = plt.subplots(num_steps, num_cols, figsize=(fig_w, fig_h))
    if num_steps == 1:
        axes = np.expand_dims(axes, 0)
    if num_cols == 1:
        axes = np.expand_dims(axes, 1)

    last_im = None
    for r in range(num_steps):
        cams = steps_images[r]      # len=5
        bases = base_images[r]      # len=5

        # 前5列：原图
        for c in range(5):
            ax = axes[r, c]
            ax.imshow(bases[c])
            if c == 0:
                ax.set_title(f"{row_title_prefix} {step_nums[r]}", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])

        # 后5列：原图 + CAM 叠加
        for c in range(5):
            ax = axes[r, 5 + c]
            ax.imshow(bases[c])
            cam_disp = np.clip((cams[c] - vmin) / scale, 0, 1)
            last_im = ax.imshow(cam_disp, cmap=cmap, alpha=overlay_alpha)
            ax.set_xticks([]); ax.set_yticks([])

    # 共享色条（基于最后一个imshow）
    if last_im is not None:
        cbar = fig.colorbar(last_im, ax=axes.ravel().tolist(), fraction=0.02, pad=0.01)
        cbar.ax.tick_params(labelsize=6)

    fig.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[OK] Saved big figure: {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--traj_id", type=str, required=True)
    parser.add_argument("--attn_dir", type=str, required=True)
    parser.add_argument("--images_dir", type=str, required=True)
    parser.add_argument("--out_path", type=str, default="cams_grid.png")

    parser.add_argument("--query_idx", type=int, default=-2)
    parser.add_argument("--slice_start", type=int, default=80)
    parser.add_argument("--slice_len", type=int, default=85)

    parser.add_argument("--pooled_h", type=int, default=4)
    parser.add_argument("--pooled_w", type=int, default=4)
    parser.add_argument("--img_h", type=int, default=256)
    parser.add_argument("--img_w", type=int, default=256)

    # Step selection
    parser.add_argument("--step_min", type=int, default=None, help="Inclusive lower bound of step number.")
    parser.add_argument("--step_max", type=int, default=None, help="Inclusive upper bound of step number.")
    parser.add_argument("--step_list", type=str, default=None, help="Comma-separated explicit step numbers, e.g., '0,3,10'.")

    # Image mapping mode
    parser.add_argument("--image_mode", type=str, default="single_mod",
                        choices=["single_mod", "single_by_index", "window"],
                        help="How to pick base images per step: single_mod (by step_num %% L), single_by_index (by step index), window (5 consecutive images).")

    parser.add_argument("--cmap", type=str, default="inferno")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--dpi", type=int, default=300)
    parser.add_argument("--q_low", type=float, default=0.01)
    parser.add_argument("--q_high", type=float, default=0.99)

    args = parser.parse_args()

    step_files_all = find_step_files(args.attn_dir, args.traj_id)
    step_files = filter_steps_by_range(step_files_all, args.step_min, args.step_max, args.step_list)
    if not step_files:
        raise FileNotFoundError("No step files after filtering; check step_min/step_max/step_list.")

    img_files = list_images(args.images_dir)

    steps_cams = []
    steps_bases = []
    step_nums = []

    for i, fpath in enumerate(step_files):
        m = re.search(r'_(\d+)\.pt$', os.path.basename(fpath))
        step_num = int(m.group(1)) if m else i

        data = torch.load(fpath, map_location="cpu")
        if not isinstance(data, (tuple, list)):
            raise TypeError(f"{fpath}: expected tuple/list of layers")

        # attn_mat = average_layers_heads_to_matrix(data)  # (N,N)
        attn_mat = data[-1][0][-1].to(torch.float32)
        # attn_mat = data[-1].mean(dim=1).squeeze(0).to(torch.float32)
        scores_85 = extract_scores_row(attn_mat, args.query_idx, args.slice_start, args.slice_len)
        
        a = scores_85.reshape(5,17)
        print(f"Step {step_num}:")
        for aa in a:
            print(np.argsort(-aa))
        cams_up = make_cams_from_scores(scores_85, pooled_hw=(args.pooled_h, args.pooled_w),
                                        has_ctx=True, out_hw=(args.img_h, args.img_w))
        steps_cams.append(cams_up)
        bases = [load_image(img_files[i][j]) for j in range(len(RGB_FOLDER))]
        steps_bases.append(bases)
        step_nums.append(step_num)

    build_big_figure(steps_cams, steps_bases, step_nums, args.out_path,
                     cmap=args.cmap, q_low=args.q_low, q_high=args.q_high,
                     overlay_alpha=args.alpha, dpi=args.dpi)

if __name__ == "__main__":
    main()
