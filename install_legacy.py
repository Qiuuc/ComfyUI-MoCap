# ComfyUI-Noctyra — Mocap sidecar 一键安装
# GPL-3.0 (见仓库 LICENSE)
"""
建插件私有 sidecar 环境(py3.10/torch2.3.1+cu121),供视频转 3D 动作的 GVHMR/HaMeR
推理使用。用 uv 自动建,全程零编译(pytorch3d/detectron2 用随包 cp310 轮子,
mmcv 是 lite 版直接 pip)。

由 ComfyUI-Manager 在装节点时自动执行,也可手动:
    python_embeded\\python.exe ComfyUI\\custom_nodes\\ComfyUI-Noctyra\\install.py

模型权重不在此处理(体积大 + SMPL 系列受 license 限制),见 README。
"""
import os
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
RUNTIME = ROOT / "runtime"
WHEELS = ROOT / "wheels"
REQ = ROOT / "sidecar_requirements.txt"
MOCAP = ROOT / "mocap"

PYVER = "3.10"
TORCH_INDEX = "https://download.pytorch.org/whl/cu121"
PIP_MIRROR = os.environ.get("MOCAP_PIP_MIRROR", "https://pypi.tuna.tsinghua.edu.cn/simple")

TORCH_PINS = ["torch==2.3.1+cu121", "torchvision==0.18.1+cu121"]
BUILD_BASE = ["setuptools<70", "wheel", "Cython==3.2.4", "numpy==1.23.5"]


def log(msg):
    print(f"\033[36m[Mocap install]\033[0m {msg}", flush=True)


def run(cmd):
    log("$ " + " ".join(map(str, cmd)))
    subprocess.run([str(c) for c in cmd], check=True)


def venv_python() -> Path:
    return RUNTIME / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def find_uv() -> str:
    uv = shutil.which("uv")
    if uv:
        return uv
    log("未找到 uv,尝试装到当前环境…")
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "uv"], check=True)
    return shutil.which("uv") or "uv"


def main():
    log(f"插件目录: {ROOT}")
    uv = find_uv()

    # 1) 备好 py3.10 + 建私有 venv
    run([uv, "python", "install", PYVER])
    if not venv_python().exists():
        run([uv, "venv", "--python", PYVER, str(RUNTIME)])
    py = venv_python()
    if not py.exists():
        raise SystemExit(f"venv 创建失败: {py}")

    def pip(*args, index=None):
        base = [uv, "pip", "install", "--python", str(py)]
        if index:
            base += ["--index-url", index]
        else:
            base += ["-i", PIP_MIRROR]
        run(base + list(args))

    # 2) torch / torchvision(cu121 官方索引)
    pip(*TORCH_PINS, index=TORCH_INDEX)

    # 3) 构建底座(先种 numpy/Cython/setuptools,供后续 --no-build-isolation)
    pip(*BUILD_BASE)

    # 4) 随包 cp310 轮子:pytorch3d + detectron2
    try:
        p3d = next(WHEELS.glob("pytorch3d-*.whl"))
        d2 = next(WHEELS.glob("detectron2-*.whl"))
    except StopIteration:
        raise SystemExit(f"缺少 cp310 轮子,请确认 {WHEELS} 下有 pytorch3d/detectron2 的 .whl")
    pip(str(p3d), str(d2))

    # 5) chumpy(老式 setup,关 build 隔离)再装其余
    pip("--no-build-isolation", "chumpy==0.70")
    pip("--no-build-isolation", "-r", str(REQ))

    # 5b) WiLoR(手部备选,端到端无 detectron2/mmcv)。--no-deps 避免把 ultralytics 降级
    pip("roma", "huggingface_hub")
    pip("--no-deps", "git+https://github.com/warmshao/WiLoR-mini.git")

    # 6) 代码/权重存在性检查(不自动下,见 README)
    miss = []
    if not (MOCAP / "GVHMR" / "tools" / "demo" / "demo.py").exists():
        miss.append("mocap/GVHMR(GVHMR 代码+权重)")
    if not (MOCAP / "hamer" / "hamer").is_dir():
        miss.append("mocap/hamer(HaMeR 代码+权重)")
    smplx = MOCAP / "GVHMR" / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
    if not smplx.exists():
        miss.append("SMPL-X 模型(GLB 导出需要,license 自备)")

    # 7) 验证 sidecar
    log("验证 sidecar 导入…")
    run([py, "-c",
         "import torch,torchvision,pytorch3d,detectron2,mmcv,smplx,chumpy,scipy;"
         "print('  torch', torch.__version__, '| cuda', torch.cuda.is_available());"
         "print('  pytorch3d/detectron2/mmcv/smplx OK')"])

    log(f"\033[92m完成\033[0m  sidecar = {py}")
    if miss:
        log("仍缺以下(见 README 自行放置/下载):")
        for m in miss:
            log("  - " + m)


if __name__ == "__main__":
    main()
