# ComfyUI-Mocap — 视频转 3D 骨骼动作(Mocap)独立节点包
# 从 ComfyUI-Noctyra 拆出(原作 Qiuuc)。GPL-3.0(见 LICENSE)
#
# 说明:
#   重活由私有 sidecar 环境(py3.10/torch2.3.1)跑。使用前需:
#     1) python install.py  建 sidecar 环境(wheels/ 为 Windows 轮子,Linux 需自备)
#     2) 按 mocap_models_manifest.json 放好 GVHMR/HaMeR/SMPL-X 权重到 ./mocap/
#   未配 sidecar 时节点仍会注册显示,但运行会提示缺环境/权重。
try:
    from .nodes.motion import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
except Exception as _e:  # 缺依赖不阻断 ComfyUI 启动
    print(f"\033[34m[Mocap]\033[0m \033[91m加载失败\033[0m: {_e}")
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}

__version__ = "1.0.0"
print(f"\033[34m[Mocap]\033[0m v{__version__} \033[92m已加载\033[0m {len(NODE_CLASS_MAPPINGS)} 个动作节点")

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
