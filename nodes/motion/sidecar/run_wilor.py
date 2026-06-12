# ComfyUI-Noctyra — sidecar 脚本：WiLoR 视频版手部推理
# GPL-3.0 (见仓库 LICENSE)
#
# 在 sidecar(py3.10/torch≤2.5)里跑。依赖 wilor_mini(端到端,自带 YOLO 手检测 +
# 自动从 HF 下载权重,无 detectron2/mmcv/pytorch3d)。
"""
逐帧输出 MANO 双手参数(轴角)到 .npz,字段与 run_hamer.py 完全一致:
    left_pose/right_pose (T,15,3) NaN where missing, fps, n_frames
所以下游合并节点不用改,WiLoR 与 HaMeR 可二选一。

WiLoR 的 hand_pose 已是 (1,15,3) 轴角且左手已内部翻转 (x,-y,-z),与 HaMeR 同约定,
直接取用即可,无需再翻。
"""
import argparse

import cv2
import numpy as np
import torch
from tqdm import tqdm

from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import WiLorHandPose3dEstimationPipeline


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="输出 .npz 路径")
    ap.add_argument("--conf", type=float, default=0.3, help="手部检测置信度阈值")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    print(f"[device] {device} {dtype}")
    print("[load] WiLoR (首次会从 HuggingFace 下载权重)")
    pipe = WiLorHandPose3dEstimationPipeline(device=device, dtype=dtype, verbose=False)

    cap = cv2.VideoCapture(args.video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[video] {W}x{H} {n_frames} frames @ {fps:.1f} fps")

    left_pose = np.full((n_frames, 15, 3), np.nan, dtype=np.float32)
    right_pose = np.full((n_frames, 15, 3), np.nan, dtype=np.float32)

    pbar = tqdm(total=n_frames, desc="WiLoR video")
    fidx = 0
    while True:
        ok, img_cv2 = cap.read()
        if not ok:
            break
        try:
            outs = pipe.predict(img_cv2, hand_conf=args.conf)
        except Exception as e:
            outs = []
            tqdm.write(f"[warn] frame {fidx}: {e}")
        for o in outs:
            wp = o.get("wilor_preds")
            if wp is None:
                continue
            hp = np.asarray(wp["hand_pose"], dtype=np.float32)[0]  # (15,3) 轴角(左手已翻)
            if float(o.get("is_right", 1)) >= 0.5:
                right_pose[fidx] = hp
            else:
                left_pose[fidx] = hp
        pbar.update(1)
        fidx += 1

    pbar.close()
    cap.release()
    np.savez_compressed(args.out, left_pose=left_pose, right_pose=right_pose, fps=fps, n_frames=n_frames)
    print(f"[saved] {args.out}")
    print(f"  detected (left/right): {np.isfinite(left_pose[:,0,0]).sum()} / {np.isfinite(right_pose[:,0,0]).sum()}")


if __name__ == "__main__":
    main()
