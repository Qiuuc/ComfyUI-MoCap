# ComfyUI-Noctyra — sidecar 脚本：HaMeR 视频版手部推理(无渲染)
# GPL-3.0 (见仓库 LICENSE)
#
# 在 sidecar(py3.10/torch2.3.1)里跑，cwd 应为 hamer 仓库目录。
# 取自原项目 hamer_video.py，唯一改动：用 --hamer-dir 显式定位 configs/_DATA，
# 不再假设脚本与 hamer/ 同级(脚本现在住在插件里)。
"""逐帧输出 MANO 双手参数(轴角)到 .npz：left_pose/right_pose/fps/n_frames。"""
import os
# Windows 上避开 pyrender 的 EGL
os.environ["PYOPENGL_PLATFORM"] = "osmesa"

import sys
from unittest.mock import MagicMock
# 在任何 hamer import 之前把 pyrender 顶层 import 桩掉，让模块图干净加载
sys.modules["pyrender"] = MagicMock()

# Pillow 10+ 删了 Image.LINEAR/CUBIC/ANTIALIAS 等大写常量,detectron2 0.6 / mmpose 0.x
# 仍在用 → 回填别名到 Resampling,避免 AttributeError(隔离环境是新 Pillow)。
import PIL.Image as _PILImage
for _old, _new in (("LINEAR", "BILINEAR"), ("CUBIC", "BICUBIC"),
                   ("NEAREST", "NEAREST"), ("ANTIALIAS", "LANCZOS")):
    if not hasattr(_PILImage, _old):
        setattr(_PILImage, _old, getattr(_PILImage.Resampling, _new))

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from tqdm import tqdm
from pytorch3d.transforms import matrix_to_axis_angle


def recursive_to(x, device):
    if isinstance(x, dict):
        return {k: recursive_to(v, device) for k, v in x.items()}
    if isinstance(x, torch.Tensor):
        return x.to(device)
    if isinstance(x, list):
        return [recursive_to(i, device) for i in x]
    return x


def build_detector(hamer_dir: Path):
    from detectron2.config import LazyConfig
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy

    cfg_path = hamer_dir / "hamer" / "configs" / "cascade_mask_rcnn_vitdet_h_75ep.py"
    detectron2_cfg = LazyConfig.load(str(cfg_path))
    detectron2_cfg.train.init_checkpoint = str(
        hamer_dir / "_DATA" / "detectron2" / "model_final_f05665.pkl"
    )
    for i in range(3):
        detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.25
    return DefaultPredictor_Lazy(detectron2_cfg)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--out", required=True, help="输出 .npz 路径")
    ap.add_argument("--hamer-dir", default=".", help="hamer 仓库目录(含 hamer 包 / _DATA)")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--batch_size", type=int, default=4)
    ap.add_argument("--rescale_factor", type=float, default=2.0)
    args = ap.parse_args()

    hamer_dir = Path(args.hamer_dir).resolve()
    sys.path.insert(0, str(hamer_dir))   # 让 vitpose_model / hamer 包可导入

    from hamer.configs import CACHE_DIR_HAMER
    from hamer.models import load_hamer, download_models, DEFAULT_CHECKPOINT
    from hamer.datasets.vitdet_dataset import ViTDetDataset

    # _DATA 已就位(权重由 install 流程预置)就别再调 download_models —— 它见 tar 缺失会
    # 重下 6GB。仅当 hamer.ckpt 真缺时才触发官方下载。
    if not (hamer_dir / "_DATA" / "hamer_ckpts" / "checkpoints" / "hamer.ckpt").exists():
        download_models(CACHE_DIR_HAMER)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")

    print("[load] HaMeR")
    model, model_cfg = load_hamer(args.checkpoint or DEFAULT_CHECKPOINT)
    model = model.to(device).eval()

    print("[load] detectron2 ViTDet")
    detector = build_detector(hamer_dir)

    print("[load] ViTPose")
    from vitpose_model import ViTPoseModel
    cpm = ViTPoseModel(device)

    cap = cv2.VideoCapture(args.video)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[video] {W}x{H} {n_frames} frames @ {fps:.1f} fps")

    left_pose = np.full((n_frames, 15, 3), np.nan, dtype=np.float32)
    right_pose = np.full((n_frames, 15, 3), np.nan, dtype=np.float32)

    pbar = tqdm(total=n_frames, desc="HaMeR video")
    fidx = 0
    while True:
        ok, img_cv2 = cap.read()
        if not ok:
            break
        img_rgb = img_cv2[:, :, ::-1]

        det_out = detector(img_cv2)
        det = det_out["instances"]
        valid = (det.pred_classes == 0) & (det.scores > 0.5)
        boxes = det.pred_boxes.tensor[valid].cpu().numpy()
        scores = det.scores[valid].cpu().numpy()
        if len(boxes) == 0:
            pbar.update(1); fidx += 1
            continue

        vit_out = cpm.predict_pose(img_rgb, [np.concatenate([boxes, scores[:, None]], axis=1)])
        hbox, is_right = [], []
        for vp in vit_out:
            lh = vp["keypoints"][-42:-21]
            rh = vp["keypoints"][-21:]
            for keyp, side in [(lh, 0), (rh, 1)]:
                ok_kp = keyp[:, 2] > 0.5
                if ok_kp.sum() > 3:
                    hbox.append([keyp[ok_kp, 0].min(), keyp[ok_kp, 1].min(),
                                 keyp[ok_kp, 0].max(), keyp[ok_kp, 1].max()])
                    is_right.append(side)
        if len(hbox) == 0:
            pbar.update(1); fidx += 1
            continue

        ds = ViTDetDataset(model_cfg, img_cv2, np.stack(hbox), np.stack(is_right),
                           rescale_factor=args.rescale_factor)
        dl = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=0)
        for batch in dl:
            batch = recursive_to(batch, device)
            with torch.no_grad():
                out = model(batch)
            hp = out["pred_mano_params"]["hand_pose"]
            hp_aa = matrix_to_axis_angle(hp.reshape(-1, 3, 3)).reshape(-1, 15, 3).cpu().numpy()
            right_flag = batch["right"].cpu().numpy().astype(bool)
            for k in range(len(hp_aa)):
                if right_flag[k]:
                    right_pose[fidx] = hp_aa[k]
                else:
                    left_pose[fidx] = hp_aa[k] * np.array([1, -1, -1], dtype=np.float32)
        pbar.update(1)
        fidx += 1

    pbar.close()
    cap.release()
    np.savez_compressed(args.out, left_pose=left_pose, right_pose=right_pose, fps=fps, n_frames=n_frames)
    print(f"[saved] {args.out}")
    print(f"  detected (left/right): {np.isfinite(left_pose[:,0,0]).sum()} / {np.isfinite(right_pose[:,0,0]).sum()}")


if __name__ == "__main__":
    main()
