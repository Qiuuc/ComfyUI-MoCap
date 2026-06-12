# ComfyUI-Noctyra — 动作(视频转 3D)节点 · 手部动作 WiLoR
# GPL-3.0 (见仓库 LICENSE)
"""
手部动作节点(WiLoR,HaMeR 的备选)。WiLoR 端到端、自带 YOLO 手检测、自动从 HF
下载权重,不依赖 mocap/hamer,也不用 detectron2/mmcv。

输出 MOCAP_HANDS,格式与 HaMeR 节点完全一致 → 与 HaMeR 二选一,接同一个『合并』口,
方便对比效果后取舍。
"""
import logging
from pathlib import Path

from ._types import MOCAP_HANDS
from . import _runtime

logger = logging.getLogger("noctyra")

_WILOR_SCRIPT = Path(__file__).resolve().parent / "sidecar" / "run_wilor.py"


class MocapWiLoR:
    DESCRIPTION = (
        "WiLoR 手部动作估计(HaMeR 的备选,通常更快更稳):逐帧检测双手 → MANO 姿态(每手 15 关节)。\n"
        "端到端、自带手部检测、首次自动从 HuggingFace 下载权重(无需 mocap/hamer)。\n"
        "输出与『手部动作 HaMeR』同格式,二选一接『合并动作』,可对比效果。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "接 ComfyUI 自带『加载视频』节点(建议与身体用同一段)"}),
            },
            "optional": {
                "hand_conf": ("FLOAT", {"default": 0.3, "min": 0.05, "max": 0.95, "step": 0.05,
                                        "tooltip": "手部检测置信度阈值;低=多检但可能误检,高=少检更准"}),
            },
        }

    RETURN_TYPES = (MOCAP_HANDS,)
    RETURN_NAMES = ("hands",)
    FUNCTION = "run"
    CATEGORY = "Noctyra/动作"

    def run(self, video, hand_conf=0.3):
        py = _runtime.sidecar_python()
        src, stem = _runtime.video_to_mp4(video)
        out_npz = _runtime.work_dir() / f"{stem}_wilor.npz"

        if not out_npz.exists():
            _runtime.unload_main_models()
            _runtime.run(
                [py, "-u", str(_WILOR_SCRIPT),
                 "--video", str(src),
                 "--out", str(out_npz),
                 "--conf", float(hand_conf)],
                cwd=_runtime.work_dir(), log_name=f"wilor_{stem}.log", label="手部动作 WiLoR",
                # 首次下载权重走 HF 国内镜像,不走代理
                extra_env={"HF_ENDPOINT": "https://hf-mirror.com"},
            )
        if not out_npz.exists():
            raise RuntimeError(f"WiLoR 未产出 npz:{out_npz}")

        return ({"npz_path": str(out_npz)},)


NODE_CLASS_MAPPINGS = {
    "NoctyraMocapWiLoR": MocapWiLoR,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NoctyraMocapWiLoR": "Mocap 手部动作（WiLoR）",
}
