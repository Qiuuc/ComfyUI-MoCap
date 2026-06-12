# ComfyUI-Mocap — ComfyUI 启动前挂载隔离环境(comfy-env / pixi)
# setup_env() 会在 ComfyUI 加载本节点前准备/校验 sidecar 隔离环境。
#
# 注意:这里不要往 sys.path 插入插件目录 —— 本插件有 nodes/ 子包,会顶替 ComfyUI
# 自己的 nodes 模块,导致 `module 'nodes' has no attribute 'init_extra_nodes'` 崩溃。
# patch_comfy_env(urllib→curl 兜底)只在「建环境」时需要,已由 install.py 负责;
# prestartup 仅 setup_env()(挂载已建好的 env,不拉取索引),无需 patch。
try:
    from comfy_env import setup_env
    setup_env()
except Exception as _e:
    print(f"\033[34m[Mocap]\033[0m comfy-env setup_env 跳过/失败: {_e}")
