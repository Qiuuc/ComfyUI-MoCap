# ComfyUI-Noctyra — 动作(视频转 3D)节点 · 手部动作 HaMeR
# GPL-3.0 (见仓库 LICENSE)
"""
手部动作节点(可选):在 sidecar 里跑 HaMeR，逐帧得到左右手各 15 关节 MANO 姿态。
输出 MOCAP_HANDS(指向 .npz),接『合并动作』的 hands 口。
"""
import logging
import shutil
from pathlib import Path

from ._types import MOCAP_HANDS
from . import _runtime, _paths

logger = logging.getLogger("noctyra")

_HAMER_SCRIPT = Path(__file__).resolve().parent / "sidecar" / "run_hamer.py"


class MocapHaMeR:
    DESCRIPTION = (
        "HaMeR 手部动作估计(可选):逐帧检测双手并回归 MANO 姿态(每手 15 关节)。\n"
        "接『Mocap 视频输入』,输出手部动作接『合并动作』。不接=只做身体。\n"
        "逐帧跑较慢;batch_size 越大越快但越吃显存。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "接 ComfyUI 自带『加载视频』节点(建议与身体用同一段)"}),
            },
            "optional": {
                "batch_size": ("INT", {"default": 4, "min": 1, "max": 32,
                                       "tooltip": "每批手部 crop 数;越大越快越吃显存"}),
            },
        }

    RETURN_TYPES = (MOCAP_HANDS,)
    RETURN_NAMES = ("hands",)
    FUNCTION = "run"
    CATEGORY = "Noctyra/动作"

    def run(self, video, batch_size=4):
        hamer_dir = _paths.hamer_dir()
        if not (hamer_dir / "hamer").is_dir():
            raise RuntimeError(f"hamer 目录无效(缺 hamer 包):{hamer_dir}")
        py = _runtime.sidecar_python()

        src, stem = _runtime.video_to_mp4(video)
        # 暂存进 hamer 目录(脚本以 cwd=hamer_dir 跑，--video 取相对名)
        staged = hamer_dir / f"{stem}.mp4"
        _runtime.stage_file(src, staged)

        out_npz = _runtime.work_dir() / f"{stem}_hand.npz"
        try:
            if not out_npz.exists():
                _runtime.unload_main_models()
                _runtime.run(
                    [py, "-u", str(_HAMER_SCRIPT),
                     "--video", f"{stem}.mp4",
                     "--out", str(out_npz),
                     "--hamer-dir", str(hamer_dir),
                     "--batch_size", int(batch_size)],
                    cwd=hamer_dir, log_name=f"hamer_{stem}.log", label="手部动作 HaMeR",
                    # download_models guard 落空时,官方源不通 → 走国内镜像兜底
                    extra_env={"HF_ENDPOINT": "https://hf-mirror.com"},
                )
            if not out_npz.exists():
                raise RuntimeError(f"HaMeR 未产出 npz:{out_npz}")
        finally:
            # 无论成功/异常/取消,都清掉暂存视频
            staged.unlink(missing_ok=True)

        return ({"npz_path": str(out_npz)},)


NODE_CLASS_MAPPINGS = {
    "NoctyraMocapHaMeR": MocapHaMeR,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NoctyraMocapHaMeR": "Mocap 手部动作（HaMeR）",
}
