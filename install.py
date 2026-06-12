# ComfyUI-Mocap — 一键隔离安装(comfy-env / pixi),两段式
#
# 第一段:comfy_env.install() 读 comfy-env.toml,建跨平台隔离环境
#         (匹配宿主 torch 2.11+cu128,5090 友好;pytorch3d 等 cuda 包走预编译索引,不本机编译)。
# 第二段:post_install.py 往建好的 env 里后装老式源码包(chumpy/mmcv/xtcocotools/cython_bbox)
#         + WiLoR,并打 chumpy 的 numpy 兼容补丁。为什么要第二段见 post_install.py 顶部说明。
#
# 旧的手写(Windows 轮子)方案见 install_legacy.py。
# 先给 urllib 套 curl 兜底,否则 comfy-env 拉 cuda-wheels 索引会被 CDN SSL EOF/RST
# 打断(见 patch_comfy_env.py)。必须在 install() 之前。
try:
    import patch_comfy_env
    patch_comfy_env.apply()
except Exception as _e:
    print(f"[Mocap] patch_comfy_env 跳过/失败(cuda-wheels 索引可能不稳): {_e}")

from comfy_env import install

install()

# 第二段:env 建好后后装源码包(此时 numpy<2 已在 env 内,可正常构建)
try:
    import post_install
    post_install.run()
except Exception as _e:
    print(f"[Mocap] post_install 失败(可手动跑 <mocap-env-python> post_install.py): {_e}")

# 第三段:clone 源码仓库(GVHMR 默认;HaMeR 由 MOCAP_INSTALL_HAMER=1 开启)。
# 权重不在此下(免注册的用 download_weights.py,注册的按 _selfcheck 提示自备)。
try:
    import provision
    provision.run()
except Exception as _e:
    print(f"[Mocap] provision(源码仓库)失败(可手动跑 provision.py): {_e}")
