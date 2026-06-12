# ComfyUI-Mocap — HaMeR(手部,detectron2 路线)一键配置(环境无关,无 docker 依赖)
#
# WiLoR 是默认手部路径(自包含、首次自动下权重);HaMeR 是进阶备选,依赖较重
# (detectron2 ViTDet + ViTPose + ~8.4G 权重),故单独用本脚本配置。
#
# 在 ComfyUI 所在环境里跑(容器/venv/裸机皆可):
#     python setup_hamer.py            # clone 仓库 + 配环境 + 下非注册权重 + 放 MANO
#     python setup_hamer.py --no-weights   # 只 clone 仓库 + 配环境
# MANO(注册权重)由环境变量 MOCAP_MANO_ZIP 指向你下载的 mano_v1_2.zip;缺则跳过并提示。
#
# 等价于在 install 时设 MOCAP_INSTALL_HAMER=1(后者不下权重/不放 MANO)。
import os
import shutil
import sys
import zipfile

try:
    from . import provision, download_weights
except ImportError:
    import provision
    import download_weights

MANO_ZIP = os.environ.get("MOCAP_MANO_ZIP", "").strip().strip('"')
# zip 内成员 -> 目标文件名
MANO_MEMBERS = [
    ("mano_v1_2/models/MANO_RIGHT.pkl", "MANO_RIGHT.pkl"),
    ("mano_v1_2/models/MANO_LEFT.pkl", "MANO_LEFT.pkl"),
]


def place_mano():
    dst_dir = os.path.join(provision.mocap_home(), "hamer", "_DATA", "data", "mano")
    if os.path.isfile(os.path.join(dst_dir, "MANO_RIGHT.pkl")):
        print("[Mocap][hamer] MANO 已就位")
        return
    if not MANO_ZIP or not os.path.isfile(MANO_ZIP):
        print("[Mocap][hamer] ! 未配置 MANO。请从 https://mano.is.tue.mpg.de 注册下载 mano_v1_2.zip,")
        print("              设环境变量 MOCAP_MANO_ZIP 指向它后重跑;或手动取出 MANO_RIGHT.pkl/")
        print(f"              MANO_LEFT.pkl 放到 {dst_dir}/")
        return
    os.makedirs(dst_dir, exist_ok=True)
    with zipfile.ZipFile(MANO_ZIP) as z:
        names = set(z.namelist())
        for member, out in MANO_MEMBERS:
            if member in names:
                with z.open(member) as fi, open(os.path.join(dst_dir, out), "wb") as fo:
                    shutil.copyfileobj(fi, fo)
                print(f"[Mocap][hamer] 放置 {out}")
    print("[Mocap][hamer] MANO 就绪")


def main(with_weights=True):
    print("== HaMeR 配置:① 源码仓库 ==")
    provision.provision_hamer_repo()
    print("== HaMeR 配置:② 隔离环境(detectron2/ViTDet/ViTPose)==")
    provision.provision_hamer_env()
    if with_weights:
        print("== HaMeR 配置:③ 非注册权重 ==")
        download_weights.download_hamer()
        print("== HaMeR 配置:④ MANO(注册权重)==")
        place_mano()
    print("\n✅ HaMeR 配置完成。节点:手部动作 HaMeR(MocapHaMeR)。")
    print("   提示:ViTDet-H 较吃显存,节点跑前会自动 unload ComfyUI 模型;独立测试需先")
    print("   POST /free 释放显存,且勿设 PYTORCH_CUDA_ALLOC_CONF=expandable_segments(易触发 assert)。")


if __name__ == "__main__":
    main(with_weights="--no-weights" not in sys.argv)
