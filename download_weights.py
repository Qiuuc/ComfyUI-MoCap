# ComfyUI-Mocap — 非注册权重一键下载(环境无关)
#
# 下载所有 license=false(免注册)的权重到 mocap/ 对应路径,源用本机可达的镜像:
#   GVHMR 主干/HMR2/ViTPose/YOLO/DPVO  ← hf-mirror: camenduru/GVHMR
#   HaMeR _DATA(hamer_ckpts/vitpose)   ← hf-mirror: AlenZeng/hamer_demo_data.tar.gz
#   detectron2 ViTDet 检测器            ← fbaipublicfiles(直链可达)
# license=true(SMPL/SMPL-X/MANO,MPI 注册下载)不在此列,需用户自备(见 _selfcheck)。
#
# 用法: <隔离环境或任意python> download_weights.py [--gvhmr] [--hamer]
#        不带参数 = 两者都下。幂等:已存在的跳过。
import os
import subprocess
import sys
import tarfile

try:
    from . import provision
except ImportError:
    import provision

HF = "https://hf-mirror.com"
# GVHMR(camenduru/GVHMR):仓库内路径 -> checkpoints 相对路径
GVHMR_FILES = [
    "gvhmr/gvhmr_siga24_release.ckpt",
    "hmr2/epoch=10-step=25000.ckpt",
    "vitpose/vitpose-h-multi-coco.pth",
    "yolo/yolov8x.pt",
    "dpvo/dpvo.pth",
]
HAMER_TAR = f"{HF}/AlenZeng/hamer_demo_data.tar.gz/resolve/main/hamer_demo_data.tar.gz"
DET_PKL = ("https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/"
           "cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl")


def _curl(url, dest):
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    print(f"[Mocap][weights] ↓ {os.path.basename(dest)}  <- {url[:70]}…")
    subprocess.run(
        ["curl", "-sS", "-L", "--fail", "-C", "-", "--retry", "30", "--retry-all-errors",
         "--retry-delay", "3", "--connect-timeout", "30", "-o", dest, url], check=True)


def download_gvhmr():
    ckpt = os.path.join(provision.mocap_home(), "GVHMR", "inputs", "checkpoints")
    for rel in GVHMR_FILES:
        dest = os.path.join(ckpt, rel)
        if os.path.exists(dest) and os.path.getsize(dest) > 1024:
            print(f"[Mocap][weights] 已存在: {rel}")
            continue
        _curl(f"{HF}/camenduru/GVHMR/resolve/main/{rel}", dest)
    print("[Mocap][weights] GVHMR 非注册权重就绪(SMPL/SMPL-X 需注册自备)")


def download_hamer():
    data = os.path.join(provision.mocap_home(), "hamer", "_DATA")
    if os.path.isfile(os.path.join(data, "hamer_ckpts", "checkpoints", "hamer.ckpt")):
        print("[Mocap][weights] HaMeR _DATA 已就位")
    else:
        tar = os.path.join(provision.mocap_home(), "hamer", "hamer_demo_data.tar.gz")
        _curl(HAMER_TAR, tar)
        print("[Mocap][weights] 解压 hamer_demo_data.tar.gz")
        with tarfile.open(tar) as t:
            _safe_extract(t, os.path.dirname(tar))
        os.remove(tar)
    det = os.path.join(data, "detectron2", "model_final_f05665.pkl")
    if os.path.exists(det) and os.path.getsize(det) > 1024:
        print("[Mocap][weights] detectron2 检测器已就位")
    else:
        _curl(DET_PKL, det)
    print("[Mocap][weights] HaMeR 非注册权重就绪(MANO 需注册自备)")


def _safe_extract(tar, path):
    """防 path traversal:拒绝 .. / 绝对路径成员(镜像 tar 不全可信)。"""
    base = os.path.abspath(path)
    for m in tar.getmembers():
        tgt = os.path.abspath(os.path.join(path, m.name))
        if not (tgt == base or tgt.startswith(base + os.sep)):
            raise RuntimeError(f"tar 成员越界,拒绝解压: {m.name}")
    tar.extractall(path)


if __name__ == "__main__":
    do_g = "--gvhmr" in sys.argv or len(sys.argv) == 1
    do_h = "--hamer" in sys.argv or len(sys.argv) == 1
    if do_g:
        download_gvhmr()
    if do_h:
        download_hamer()
    print("\n✅ 非注册权重下载完成。注册权重(SMPL/SMPL-X/MANO)请按 _selfcheck 提示自备。")
