# ComfyUI-Mocap — comfy-env(pixi)隔离环境建好后的第二段安装
#
# 为什么要分两段:
#   chumpy / mmcv / xtcocotools / cython_bbox 是老式源码包,构建期(setup.py)就要
#   import numpy/Cython。放进 comfy-env.toml 让 pixi/uv 解析时,uv 会在把 numpy 装进
#   env 之前就去构建它们 → "No module named 'numpy'" 失败。
#   故把它们挪到这里:等 pixi 把干净 env(含 numpy<2 + Cython)建好后,再用 env 自己的
#   pip --no-build-isolation 后装,此时 numpy 已就位,全部能正常 build。
#   WiLoR 走 --no-deps,避免它把 ultralytics 从 8.1.34 降级。
#
# 可独立运行:  <mocap-env-python> post_install.py
# 也由 install.py 在 comfy_env.install() 之后自动调用。
import os
import sys
import glob
import subprocess

# env 内需后装的源码包(--no-build-isolation;numpy/Cython 已在 env 内)
NO_BUILD_ISO = [
    "chumpy==0.70",
    "mmcv==1.3.9",          # lite 版,无 CUDA 编译
    "xtcocotools==1.14.3",
    "cython_bbox==0.1.5",
]
# WiLoR(手部,端到端):--no-deps 防止降级 ultralytics
WILOR = "git+https://github.com/warmshao/WiLoR-mini.git"


def _find_env_python():
    """定位 mocap 隔离环境的 python(复用共享 _env)。"""
    try:
        from . import _env
    except ImportError:
        import _env
    return _env.find_env_python()


def _pip(py, *args):
    subprocess.run([py, "-m", "pip", "install", *args], check=True)


def _patch_chumpy(py):
    """chumpy 0.70 在 numpy>=1.24 上崩:它 `from numpy import bool,int,float...`(已被删)。
    在 chumpy/__init__.py 顶部 prepend 无条件别名垫片(让后续那行 import 能成功)。
    不依赖匹配具体某一行 → 即使 chumpy 版本/写法变也稳;靠 marker 幂等;找不到包则报错。"""
    marker = "# __mocap_numpy_alias_shim__"
    shim = (
        marker + "\n"
        "import numpy as _np\n"
        "for _a, _r in (('bool','bool_'),('int','int_'),('float','float64'),"
        "('complex','complex128'),('object','object_'),('unicode','str_'),('str','str_')):\n"
        "    hasattr(_np, _a) or setattr(_np, _a, getattr(_np, _r))\n"
    )
    code = (
        "import os, chumpy\n"
        "f = os.path.join(os.path.dirname(chumpy.__file__), '__init__.py')\n"
        f"marker = {marker!r}\n"
        f"shim = {shim!r}\n"
        "s = open(f, encoding='utf-8').read()\n"
        "if marker in s:\n"
        "    print('chumpy 补丁: already')\n"
        "else:\n"
        "    open(f, 'w', encoding='utf-8').write(shim + s)\n"
        "    print('chumpy 补丁: applied (prepend shim)')\n"
        "import importlib; importlib.reload(chumpy) if False else None\n"
    )
    subprocess.run([py, "-c", code], check=True)
    # 验证补丁真生效(fail-loud,不再静默)
    subprocess.run([py, "-c", "import chumpy"], check=True)


def run():
    py = _find_env_python()
    if not py:
        print("[Mocap][post_install] 未找到 mocap 隔离环境 python,跳过(请先跑 install.py)")
        return False
    print(f"[Mocap][post_install] env python: {py}")
    print("[Mocap][post_install] 后装源码包(--no-build-isolation)…")
    _pip(py, "--no-build-isolation", *NO_BUILD_ISO)
    print("[Mocap][post_install] 后装 WiLoR(--no-deps)…")
    _pip(py, "--no-deps", WILOR)
    print("[Mocap][post_install] 打 chumpy numpy 兼容补丁…")
    _patch_chumpy(py)
    print("[Mocap][post_install] 完成 ✓")
    return True


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
