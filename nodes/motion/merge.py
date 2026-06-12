# ComfyUI-Noctyra — 动作(视频转 3D)节点 · 合并动作
# GPL-3.0 (见仓库 LICENSE)
"""
合并节点：把 GVHMR 身体结果(.pt) + 可选 HaMeR 手部(.npz)拼成 SMPL-X 55 关节的
AMASS 风格 npz(纯 CPU)。调 sidecar 里的 sidecar/run_merge.py。
输出 MOCAP_ANIM 接『导出 BVH/GLB』。
"""
import logging
from pathlib import Path

from ._types import MOCAP_MOTION, MOCAP_HANDS, MOCAP_ANIM
from . import _runtime

logger = logging.getLogger("noctyra")

_MERGE_SCRIPT = Path(__file__).resolve().parent / "sidecar" / "run_merge.py"


class MocapMerge:
    DESCRIPTION = (
        "把身体动作(GVHMR)与可选手部动作(HaMeR)合并成 SMPL-X 55 关节的动作序列(AMASS npz)。\n"
        "手部缺失帧线性插值 + 时间平滑(smooth 窗口)。纯 CPU，很快。\n"
        "输出 动作(接『导出 BVH/GLB』)。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "body": (MOCAP_MOTION, {"tooltip": "来自『身体动作 GVHMR』"}),
            },
            "optional": {
                "hands": (MOCAP_HANDS, {"tooltip": "来自『手部动作 HaMeR』(可选;不接=只身体)"}),
                "fps": ("INT", {"default": 0, "min": 0, "max": 120, "tooltip": "动作帧率(写入 npz);0=自动跟随视频真实帧率"}),
                "smooth": ("INT", {"default": 5, "min": 1, "max": 15, "tooltip": "手部平滑窗口(帧);1=不平滑"}),
            },
        }

    RETURN_TYPES = (MOCAP_ANIM,)
    RETURN_NAMES = ("anim",)
    FUNCTION = "run"
    CATEGORY = "Noctyra/动作"

    def run(self, body, hands=None, fps=0, smooth=5):
        py = _runtime.sidecar_python()
        pt_path = body["pt_path"]
        stem = body.get("stem", "anim")
        out_npz = _runtime.work_dir() / f"{stem}_amass.npz"

        # fps=0 → 跟随身体动作探到的真实帧率(GVHMR 逐帧,fps 必须=原视频否则变速)
        eff_fps = int(fps) if fps else int(round(float(body.get("fps") or 30)))
        eff_fps = max(1, eff_fps)

        cmd = [py, "-u", str(_MERGE_SCRIPT), pt_path, str(out_npz), "--fps", eff_fps]
        if hands and hands.get("npz_path"):
            cmd += ["--hand", hands["npz_path"], "--smooth", int(smooth)]

        _runtime.run(cmd, cwd=_MERGE_SCRIPT.parent, log_name=f"merge_{stem}.log", label="合并动作")
        if not out_npz.exists():
            raise RuntimeError(f"合并未产出 npz:{out_npz}")

        anim = {"npz_path": str(out_npz), "fps": eff_fps, "n_frames": body.get("n_frames", 0)}
        return (anim,)


NODE_CLASS_MAPPINGS = {
    "NoctyraMocapMerge": MocapMerge,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NoctyraMocapMerge": "Mocap 合并动作",
}
