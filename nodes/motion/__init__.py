# ComfyUI-Noctyra — 动作(视频转 3D)节点子包
# GPL-3.0 (见仓库 LICENSE)
"""
视频转 3D 骨骼动作管线(Mocap)：

    [ComfyUI 加载视频] ──→ [身体动作 GVHMR] ─→ [合并动作] ─→ [导出 BVH/GLB]
                          [手部动作 HaMeR](可选) ┘

视频用 ComfyUI 自带『加载视频』节点(原生 VIDEO)接入,推理节点内部落成 mp4。
重活交给私有 sidecar 环境(py3.10/torch2.3.1，install.py 用 uv 自动建)跑;
合并/导出在主环境原生跑。路径按约定固定推导(mocap/ 见 _paths.py)。
默认输出 BVH(动作) + GLB(带皮角色，可接 Preview3D)，纯 Python，无需 Blender。
"""
_MODULES = []

# 可选模块：尚未补齐的不阻断加载(_selfcheck 提供『Mocap 自检』节点)
for _name in ("gvhmr", "hamer", "wilor", "merge", "export", "_selfcheck"):
    try:
        _mod = __import__(f"{__name__}.{_name}", fromlist=[_name])
        _MODULES.append(_mod)
    except ImportError:
        pass

NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}
for _m in _MODULES:
    NODE_CLASS_MAPPINGS.update(getattr(_m, "NODE_CLASS_MAPPINGS", {}))
    NODE_DISPLAY_NAME_MAPPINGS.update(getattr(_m, "NODE_DISPLAY_NAME_MAPPINGS", {}))

# 加载时扫一遍权重清单(缺啥提示去哪下/放哪),不阻断
if NODE_CLASS_MAPPINGS:
    try:
        from . import _selfcheck
        _selfcheck.check()
    except Exception:
        pass
