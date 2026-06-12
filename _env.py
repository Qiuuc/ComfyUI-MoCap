# ComfyUI-Mocap — 隔离环境定位(共享,环境无关)
#
# comfy-env(pixi)把隔离环境建在 COMFY_ENV_ROOT/.pixi/envs/<name>/。本模块只负责
# 找到那个 env 的 python 与 site-packages,供 install/post_install/provision/selfcheck 复用。
# 不依赖 docker / 不写死容器名或路径:在 ComfyUI 跑 install.py 的进程里直接 subprocess。
import glob
import os
import subprocess


def _roots():
    roots = []
    er = os.environ.get("COMFY_ENV_ROOT")
    if er:
        roots.append(er)
    here = os.path.dirname(os.path.abspath(__file__))
    cur = here
    for _ in range(7):
        roots.append(os.path.join(cur, ".ce"))
        cur = os.path.dirname(cur)
    return roots


def find_env_python():
    """返回 mocap 隔离环境的 python 路径;找不到返回 None。
    优先 env 目录名含 'mocap';否则取任一装了 torch 的 env。"""
    cands = []
    for root in _roots():
        cands += glob.glob(os.path.join(root, ".pixi", "envs", "*", "bin", "python"))
        cands += glob.glob(os.path.join(root, ".pixi", "envs", "*", "Scripts", "python.exe"))
    if not cands:
        return None
    # 去重保序
    seen, uniq = set(), []
    for c in cands:
        if c not in seen:
            seen.add(c)
            uniq.append(c)
    mocap = [c for c in uniq if "mocap" in c.lower()]
    for c in (mocap + uniq):
        try:
            subprocess.run([c, "-c", "import torch"], check=True,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=120)
            return c
        except Exception:
            continue
    # 没有装 torch 的,退而求其次返回 mocap 命名的(env 可能还没装完)
    return (mocap or uniq)[0]


def site_packages(env_py):
    """该 env 的 site-packages 目录(权威:python 自报,避开 python3.1 软链歧义)。"""
    out = subprocess.run(
        [env_py, "-c", "import site;print(site.getsitepackages()[0])"],
        capture_output=True, text=True)
    return out.stdout.strip()
