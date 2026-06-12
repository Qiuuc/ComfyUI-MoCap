# ComfyUI-Noctyra — 动作(视频转 3D)节点 · 导出 BVH
# GPL-3.0 (见仓库 LICENSE)
"""
导出节点:把合并后的 SMPL-X 动作写成 BVH(通用动作格式,也是通往 3ds Max/BIP 的桥)。
纯 numpy/scipy,跑在 ComfyUI 主环境,不进 sidecar。
(GLB 带蒙皮导出随后补,需读 SMPL-X 网格。)
"""
import logging
from pathlib import Path

import folder_paths

from ._types import MOCAP_ANIM
from . import _bvh, _glb, _paths

logger = logging.getLogger("noctyra")


def _unique(base: Path, ext: str) -> Path:
    c = 0
    while True:
        p = base.with_name(f"{base.name}_{c:05d}{ext}")
        if not p.exists():
            return p
        c += 1


class MocapExportBVH:
    DESCRIPTION = (
        "把动作导出为 BVH(骨架 + 运动,SMPL-X 55 关节含手指)。BVH 通用,Max/MotionBuilder/"
        "Maya/Blender 都认,也是喂 3ds Max 转 BIP 的中转。文件落在 output 目录,输出其路径。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anim": (MOCAP_ANIM, {"tooltip": "来自『Mocap 合并动作』"}),
            },
            "optional": {
                "filename_prefix": ("STRING", {
                    "default": "mocap/motion",
                    "tooltip": "输出文件名前缀(可含子目录),自动追加序号与 .bvh",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("bvh_path",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Noctyra/动作"

    def run(self, anim, filename_prefix="mocap/motion"):
        npz = anim["npz_path"]
        fps = anim.get("fps") or None

        out_dir = Path(folder_paths.get_output_directory())
        base = out_dir / filename_prefix
        base.parent.mkdir(parents=True, exist_ok=True)
        out = _unique(base, ".bvh")

        # 有 SMPL-X 模型就用真实 rest 骨长(与 GLB 一致,retarget 才准);没有则回退示意骨长
        model = _paths.smplx_model()
        model_path = str(model) if model.exists() else None
        path, T, fps2 = _bvh.write_bvh(npz, str(out), fps=fps, model_path=model_path)
        logger.info(f"BVH 导出: {path}  ({T} 帧 @ {fps2}fps)")
        return {"ui": {"text": [str(path)]}, "result": (str(path),)}


class MocapExportGLB:
    DESCRIPTION = (
        "把动作导出为带蒙皮的动画 GLB(SMPL-X 网格 + 骨骼 + 动画)。Unity/UE/Blender/3ds Max "
        "都能导入,也可接 ComfyUI 的 3D 预览节点(Preview3D)在图里转着看。文件落 output 目录。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "anim": (MOCAP_ANIM, {"tooltip": "来自『Mocap 合并动作』"}),
            },
            "optional": {
                "filename_prefix": ("STRING", {
                    "default": "mocap/motion",
                    "tooltip": "输出文件名前缀(可含子目录),自动追加序号与 .glb",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("glb_path",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Noctyra/动作"

    def run(self, anim, filename_prefix="mocap/motion"):
        model = _paths.smplx_model()
        if not model.exists():
            raise RuntimeError(f"缺 SMPL-X 模型(GLB 蒙皮需要):{model}")
        npz = anim["npz_path"]
        fps = anim.get("fps") or None

        out_dir = Path(folder_paths.get_output_directory())
        base = out_dir / filename_prefix
        base.parent.mkdir(parents=True, exist_ok=True)
        out = _unique(base, ".glb")

        path, T, fps2 = _glb.write_glb(npz, str(model), str(out), fps=fps)
        logger.info(f"GLB 导出: {path}  ({T} 帧 @ {fps2}fps)")
        return {"ui": {"text": [str(path)]}, "result": (str(path),)}


NODE_CLASS_MAPPINGS = {
    "NoctyraMocapExportBVH": MocapExportBVH,
    "NoctyraMocapExportGLB": MocapExportGLB,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NoctyraMocapExportBVH": "Mocap 导出 BVH",
    "NoctyraMocapExportGLB": "Mocap 导出 GLB",
}
