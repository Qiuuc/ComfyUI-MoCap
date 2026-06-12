# ComfyUI-Noctyra — sidecar 脚本：GVHMR(+HaMeR) → AMASS 风格 SMPL-X npz
# GPL-3.0 (见仓库 LICENSE)
#
# 在 sidecar(py3.10/torch2.3.1)里跑。自包含，只依赖 torch/numpy/scipy。
# 取自原项目 gvhmr_to_amass_npz.py，逻辑不变。
"""
把 GVHMR 的 hmr4d_results.pt 转成 SMPL-X 55 关节的 AMASS 风格 npz：
    trans (T,3) / poses (T,165) / betas (10,) / gender / mocap_frame_rate
可选并入 HaMeR 双手(NaN 帧线性插值 + 时间平滑)。
"""
import argparse

import numpy as np
import torch
from scipy.ndimage import uniform_filter1d


def main(pt_path, out_npz, fps=30, hand_npz=None, smooth_window=5):
    # torch 2.6+ 默认 weights_only=True，GVHMR 结果是含非张量的字典，需显式关掉
    try:
        pred = torch.load(pt_path, map_location="cpu", weights_only=False)
    except TypeError:
        pred = torch.load(pt_path, map_location="cpu")

    g = pred["smpl_params_global"]
    global_orient = g["global_orient"].cpu().numpy().astype(np.float32)  # (T,3)
    body_pose = g["body_pose"].cpu().numpy().astype(np.float32)          # (T,63)
    transl = g["transl"].cpu().numpy().astype(np.float32)                # (T,3)
    betas = g["betas"].cpu().numpy().astype(np.float32).mean(axis=0)     # (10,)
    T = global_orient.shape[0]

    # SMPL-X 55 关节：pelvis + 21 body + jaw + 2 eyes + 30 hand
    poses_TJ3 = np.zeros((T, 55, 3), dtype=np.float32)
    poses_TJ3[:, 0] = global_orient
    poses_TJ3[:, 1:22] = body_pose.reshape(T, 21, 3)

    if hand_npz:
        hd = np.load(hand_npz)
        left = hd["left_pose"].astype(np.float32)
        right = hd["right_pose"].astype(np.float32)
        Th = min(left.shape[0], T)
        left, right = left[:Th].copy(), right[:Th].copy()

        def fill_and_smooth(arr):
            x = arr.reshape(Th, -1).copy()
            t = np.arange(Th)
            for c in range(x.shape[1]):
                v = x[:, c]
                mask = np.isfinite(v)
                if not mask.any():
                    x[:, c] = 0.0
                elif not mask.all():
                    x[:, c] = np.interp(t, t[mask], v[mask])
            out = x.reshape(Th, 15, 3)
            if smooth_window > 1:
                out = uniform_filter1d(out, size=smooth_window, axis=0, mode="nearest")
            return out

        poses_TJ3[:Th, 25:40] = fill_and_smooth(left)
        poses_TJ3[:Th, 40:55] = fill_and_smooth(right)
        n_lf = int(np.isfinite(hd["left_pose"][:Th, 0, 0]).sum())
        n_rf = int(np.isfinite(hd["right_pose"][:Th, 0, 0]).sum())
        print(f"  hands: left {n_lf}/{Th}  right {n_rf}/{Th}  (gaps interpolated, smooth={smooth_window})")

    poses = poses_TJ3.reshape(T, 55 * 3)
    np.savez(out_npz, trans=transl, poses=poses, betas=betas,
             gender="neutral", mocap_frame_rate=fps)
    print(f"saved {out_npz}  T={T} fps={fps}  poses{poses.shape} betas{betas.shape}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("pt")
    ap.add_argument("out")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--hand", default=None)
    ap.add_argument("--smooth", type=int, default=5)
    a = ap.parse_args()
    main(a.pt, a.out, a.fps, a.hand, smooth_window=a.smooth)
