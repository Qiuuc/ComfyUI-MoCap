# ComfyUI-Noctyra — 动作(视频转 3D)节点 · 固定路径
# GPL-3.0 (见仓库 LICENSE)
"""
所有路径按约定固定推导，节点上不再暴露路径配置项。

默认布局(模型/代码挪进插件，自包含)：
    ComfyUI-Noctyra/mocap/GVHMR/   (含 inputs/checkpoints 权重)
    ComfyUI-Noctyra/mocap/hamer/   (含 _DATA 权重)

可用环境变量覆盖(测试/特殊部署)：
    MOCAP_HOME             mocap 根目录(默认 插件/mocap)
    MOCAP_SIDECAR_PYTHON   sidecar 解释器(默认 插件/runtime,见 _runtime)
"""
import os
from pathlib import Path

from ._runtime import plugin_root


def mocap_home() -> Path:
    env = os.environ.get("MOCAP_HOME", "").strip().strip('"')
    return Path(env) if env else (plugin_root() / "mocap")


def gvhmr_dir() -> Path:
    return mocap_home() / "GVHMR"


def hamer_dir() -> Path:
    return mocap_home() / "hamer"


def smplx_model() -> Path:
    """SMPL-X neutral 模型(.npz),GLB 带蒙皮导出要用。"""
    return gvhmr_dir() / "inputs" / "checkpoints" / "body_models" / "smplx" / "SMPLX_NEUTRAL.npz"
