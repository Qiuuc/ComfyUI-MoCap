# ComfyUI-Mocap — 源码仓库 + HaMeR 环境一键配置(环境无关,无 docker 依赖)
#
# 在 ComfyUI 跑 install.py 的进程里直接 subprocess 执行(容器/venv/裸机皆可):
#   - clone GVHMR / HaMeR 源码仓库到 <plugin>/mocap/
#   - HaMeR 环境侧:隔离环境装 detectron2(预编译轮子)+ ViTDet 移植(commit 锁定)+ ViTPose
#   权重不在此下载(见 download_weights.py / 注册下载),由 _selfcheck 向导提示。
#
# 设计:文件天生归当前(容器)用户,无需 chown;env 用 _env.find_env_python() 发现。
import os
import shutil
import subprocess
import sys

try:
    from . import _env
except ImportError:
    import _env

GVHMR_URL = "https://github.com/zju3dv/GVHMR.git"
HAMER_URL = "https://github.com/geopavlakos/hamer.git"
D2_URL = "https://github.com/facebookresearch/detectron2.git"
# detectron2 main 锁到与 0.6 预编译 _C.so ABI 兼容、含 ViTDet 的 commit(可复现)
D2_COMMIT = os.environ.get("MOCAP_D2_COMMIT", "02b5c4e295e990042a714712c21dc79b731e8833")
D2_WHEEL = ("https://github.com/PozzettiAndrea/cuda-wheels/releases/download/detectron2-latest/"
            "detectron2-0.6%2Bcu128torch2.11-cp310-cp310-"
            "manylinux_2_34_x86_64.manylinux_2_35_x86_64.whl")
D2_PY_DEPS = ["fvcore", "iopath", "omegaconf", "cloudpickle", "pycocotools", "tabulate"]


def plugin_dir():
    return os.path.dirname(os.path.abspath(__file__))


def mocap_home():
    return os.environ.get("MOCAP_HOME", "").strip().strip('"') or os.path.join(plugin_dir(), "mocap")


def _git(*args, cwd=None):
    subprocess.run(["git", *args], cwd=cwd, check=True)


def _clone(url, dest, recursive=False, sentinel=None):
    """idempotent git clone。dest 已有内容(或 sentinel 存在)则跳过。"""
    marker = sentinel or os.path.join(dest, ".git")
    if os.path.exists(marker):
        print(f"[Mocap][provision] 已存在,跳过 clone: {dest}")
        return False
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    args = ["clone", "--depth", "1"]
    if recursive:
        args += ["--recursive"]
    args += [url, dest]
    print(f"[Mocap][provision] clone {url} -> {dest}")
    _git(*args)
    return True


def provision_gvhmr():
    """clone GVHMR 源码仓库(身体路径必需;权重另下)。"""
    dest = os.path.join(mocap_home(), "GVHMR")
    sentinel = os.path.join(dest, "tools", "demo", "demo.py")
    return _clone(GVHMR_URL, dest, recursive=False, sentinel=sentinel)


def provision_hamer_repo():
    """clone HaMeR 源码仓库(含 ViTPose submodule;手部 HaMeR 路径用)。"""
    dest = os.path.join(mocap_home(), "hamer")
    sentinel = os.path.join(dest, "hamer", "configs", "__init__.py")
    return _clone(HAMER_URL, dest, recursive=True, sentinel=sentinel)


def provision_hamer_env(env_py=None):
    """隔离环境装 detectron2 + ViTDet 移植 + ViTPose(HaMeR 推理需要)。幂等。"""
    env_py = env_py or _env.find_env_python()
    if not env_py:
        print("[Mocap][provision] 未找到隔离环境,跳过 HaMeR 环境配置(先跑 install.py)")
        return False
    site = _env.site_packages(env_py)
    d2 = os.path.join(site, "detectron2")

    # 1) detectron2 预编译轮子 + 纯py依赖
    if not os.path.isdir(d2):
        print("[Mocap][provision] 装 detectron2 预编译轮子 + 依赖")
        subprocess.run([env_py, "-m", "pip", "install", "--no-deps", D2_WHEEL], check=True)
        subprocess.run([env_py, "-m", "pip", "install", *D2_PY_DEPS], check=True)

    # 2) ViTDet 移植:detectron2 main@commit 的 .py 覆盖安装版(保留预编译 _C.so)
    if not os.path.isfile(os.path.join(d2, "modeling", "backbone", "vit.py")):
        print(f"[Mocap][provision] ViTDet 移植(detectron2 @{D2_COMMIT[:10]})")
        import tempfile
        tmp = tempfile.mkdtemp(prefix="d2src_")
        try:
            _git("clone", "--filter=blob:none", D2_URL, tmp)
            _git("checkout", D2_COMMIT, cwd=tmp)
            # 覆盖 .py(clone 无 .so,_C.so 不受影响),同步 model_zoo common 配置
            _copytree(os.path.join(tmp, "detectron2"), d2)
            _copytree(os.path.join(tmp, "configs", "common"),
                      os.path.join(d2, "model_zoo", "configs", "common"))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        _hamer_smoke(env_py)

    # 3) ViTPose submodule 装成 mmpose
    has_mmpose = subprocess.run([env_py, "-c", "import mmpose"],
                                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).returncode == 0
    if not has_mmpose:
        vitpose = os.path.join(mocap_home(), "hamer", "third-party", "ViTPose")
        if os.path.isdir(vitpose):
            print("[Mocap][provision] 装 ViTPose(mmpose)")
            subprocess.run([env_py, "-m", "pip", "install", "--no-deps",
                            "--no-build-isolation", "-e", vitpose], check=True)
        else:
            print(f"[Mocap][provision] 缺 {vitpose},先 provision_hamer_repo()")
    print("[Mocap][provision] HaMeR 环境配置完成")
    return True


def _copytree(src, dst):
    """把 src 下内容合并进 dst(覆盖同名 .py,不删 dst 已有的 _C.so 等)。"""
    for root, _dirs, files in os.walk(src):
        rel = os.path.relpath(root, src)
        tgt = os.path.join(dst, rel) if rel != "." else dst
        os.makedirs(tgt, exist_ok=True)
        for f in files:
            shutil.copy2(os.path.join(root, f), os.path.join(tgt, f))


def _hamer_smoke(env_py):
    """覆盖后 smoke test:python 层与 0.6 _C.so 的 ABI + ViTDet 能构建,失败即报错。"""
    code = (
        "import PIL.Image as I\n"
        "for o,n in (('LINEAR','BILINEAR'),('CUBIC','BICUBIC'),('ANTIALIAS','LANCZOS')):\n"
        "    hasattr(I,o) or setattr(I,o,getattr(I.Resampling,n))\n"
        "import torch\n"
        "from detectron2 import _C\n"
        "from detectron2.modeling.backbone.vit import ViT, get_vit_lr_decay_rate\n"
        "from detectron2 import model_zoo\n"
        "model_zoo.get_config('common/models/mask_rcnn_vitdet.py').model\n"
        "print('ViTDet smoke OK')\n"
    )
    subprocess.run([env_py, "-W", "ignore", "-c", code], check=True)


def run(with_hamer=None):
    """install.py 调用入口。默认 clone GVHMR;HaMeR 由 MOCAP_INSTALL_HAMER=1 开启。"""
    if with_hamer is None:
        with_hamer = os.environ.get("MOCAP_INSTALL_HAMER", "").strip() in ("1", "true", "yes")
    try:
        provision_gvhmr()
    except Exception as e:
        print(f"[Mocap][provision] GVHMR clone 失败(可手动 git clone {GVHMR_URL}): {e}")
    if with_hamer:
        try:
            provision_hamer_repo()
            provision_hamer_env()
        except Exception as e:
            print(f"[Mocap][provision] HaMeR 配置失败: {e}")
    else:
        print("[Mocap][provision] HaMeR 未配置(设 MOCAP_INSTALL_HAMER=1 或跑 setup_hamer.py 开启)")


if __name__ == "__main__":
    run(with_hamer="--hamer" in sys.argv)
