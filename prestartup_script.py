# ComfyUI-Mocap — ComfyUI 启动前挂载隔离环境(comfy-env / pixi)
# setup_env() 会在 ComfyUI 加载本节点前准备/校验 sidecar 隔离环境。
import os
import sys

# prestartup 阶段插件目录还不在 sys.path,先加上,才能 import 本插件的同级模块
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    import patch_comfy_env  # urllib→curl 兜底,保证 cuda-wheels 索引可达(见该文件)
    patch_comfy_env.apply()
except Exception as _e:
    print(f"\033[34m[Mocap]\033[0m patch_comfy_env 跳过: {_e}")

try:
    from comfy_env import setup_env
    setup_env()
except Exception as _e:
    print(f"\033[34m[Mocap]\033[0m comfy-env setup_env 跳过/失败: {_e}")
