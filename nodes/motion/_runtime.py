# ComfyUI-Noctyra — 动作(视频转 3D)节点 · sidecar 运行器
# GPL-3.0 (见仓库 LICENSE)
"""
把重活交给私有 sidecar 环境(py3.10/torch2.3.1)跑，节点本身只在 ComfyUI 主环境
里当壳子。这里提供：

- plugin_root() / work_dir() : 插件根目录与中间产物目录
- sidecar_python()           : 解析 sidecar 的 python.exe(配置优先，其次插件内置 runtime/)
- run()                      : 同步跑子进程，解析 stdout 的百分比驱动 ComfyUI 进度条，
                               响应中断(取消工作流即杀子进程)，非零退出抛错。

设计对齐项目原 main.py 的 run_subprocess：PYTHONUNBUFFERED + 按 \\r/\\n 分块读，
强制清空代理变量(流量有限，且推理不走外网)。
"""
import logging
import os
import re
import shutil
import signal
import subprocess
import time
import uuid
from pathlib import Path

import folder_paths

logger = logging.getLogger("noctyra")

_PCT = re.compile(rb"(\d{1,3})\s*%")
_ANSI = re.compile(rb"\x1b\[[0-9;]*[a-zA-Z]")


def plugin_root() -> Path:
    # nodes/motion/_runtime.py -> 上溯三级到插件根
    return Path(__file__).resolve().parents[2]


def work_dir() -> Path:
    """中间产物(pt/npz/log)放 ComfyUI temp 下，跟随 temp 清理策略。"""
    d = Path(folder_paths.get_temp_directory()) / "mocap"
    d.mkdir(parents=True, exist_ok=True)
    return d


def stage_file(src, dst):
    """把视频暂存到下游工具的输入目录：先试硬链接(同卷零拷贝)，失败再真拷。"""
    src, dst = Path(src), Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)
    return dst


def video_from_file(path):
    """把磁盘视频包成 ComfyUI 原生 VIDEO(惰性、保留音轨)。"""
    try:
        from comfy_api.latest import InputImpl
        return InputImpl.VideoFromFile(str(path))
    except Exception:
        from comfy_api.input_impl import VideoFromFile
        return VideoFromFile(str(path))


def video_to_mp4(video):
    """把 ComfyUI 原生 VIDEO 落成磁盘 mp4,返回 (路径, stem)。
    GVHMR/HaMeR 都需要真实存在的 mp4 文件,接 ComfyUI 自带『加载视频』节点即可。"""
    from comfy_api.util import VideoContainer
    stem = f"vid_{uuid.uuid4().hex[:8]}"
    out = work_dir() / f"{stem}.mp4"
    video.save_to(str(out), format=VideoContainer("mp4"), codec="auto", metadata=None)
    return out, stem


def _comfy_env_python():
    """定位 comfy-env(pixi)隔离环境的 python:
    优先 COMFY_ENV_ROOT,否则从插件目录向上找 .ce。env 目录名优先含 'mocap',
    其次取任一含 torch 的 env。找不到返回 None。"""
    import glob
    roots = []
    er = os.environ.get("COMFY_ENV_ROOT", "").strip()
    if er:
        roots.append(er)
    cur = plugin_root()
    for _ in range(6):
        roots.append(str(cur / ".ce"))
        cur = cur.parent
    cands = []
    for root in roots:
        cands += glob.glob(os.path.join(root, ".pixi", "envs", "*", "bin", "python"))
        cands += glob.glob(os.path.join(root, ".pixi", "envs", "*", "Scripts", "python.exe"))
    if not cands:
        return None
    # 优先 env 目录名含 mocap;否则只在"唯一候选"时回退,避免多 env 时盲选到别的插件环境
    mocap = [c for c in cands if "mocap" in c.lower()]
    if mocap:
        return mocap[0]
    if len(cands) == 1:
        return cands[0]
    logger.warning(f"找到多个 pixi 环境但无一名含 'mocap',无法确定 sidecar:{cands}")
    return None


def sidecar_python() -> str:
    """解析 sidecar 解释器:环境变量 MOCAP_SIDECAR_PYTHON 优先,其次 comfy-env(pixi)
    隔离环境,最后旧的插件内置 runtime/。"""
    p = os.environ.get("MOCAP_SIDECAR_PYTHON", "").strip().strip('"')
    if p:
        return p
    # comfy-env(pixi)隔离环境:install.py 建在 basedir/.ce/.pixi/envs/mocap
    ce = _comfy_env_python()
    if ce:
        return ce
    # 旧的手写私有环境(install_legacy.py 用 uv 建在这里)
    for cand in (
        plugin_root() / "runtime" / "Scripts" / "python.exe",   # Windows venv
        plugin_root() / "runtime" / "python.exe",
        plugin_root() / "runtime" / "bin" / "python",           # *nix venv
    ):
        if cand.exists():
            return str(cand)
    raise RuntimeError(
        "未找到 sidecar python:运行 install.py 建 comfy-env 隔离环境,"
        "或设置环境变量 MOCAP_SIDECAR_PYTHON 指向解释器。"
    )


def unload_main_models():
    """跑 sidecar 前腾显存:卸载 ComfyUI 主进程里已加载的模型(三个推理节点共用)。"""
    try:
        import comfy.model_management as mm
        mm.unload_all_models()
    except Exception as e:
        logger.debug(f"unload_all_models 跳过: {e}")


def run(cmd, cwd, log_name: str, label: str, extra_env: dict | None = None):
    """同步执行子进程并把进度映射到 ComfyUI 进度条。失败抛 RuntimeError。

    cmd      : 命令列表(元素会 str 化)
    cwd      : 工作目录(同时塞进 PYTHONPATH 头部，兼容隐式命名空间包)
    log_name : 日志文件名(落在 work_dir())
    label    : 进度/日志前缀(中文阶段名)
    """
    import comfy.utils
    import comfy.model_management as mm

    pb = comfy.utils.ProgressBar(100)

    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    # torch 2.6+ 把 torch.load 默认 weights_only=True,老式 ultralytics(8.1.34)/GVHMR/WiLoR
    # 加载权重会被拒(UnpicklingError)。这些权重均为可信官方文件,恢复旧行为(原插件设计
    # 跑在 torch 2.3.1,默认即 False)。否则 YOLO/ViTPose/GVHMR ckpt 全部加载失败。
    env["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
    # 不走代理：推理不需要外网，且用户流量有限
    env["HTTP_PROXY"] = ""
    env["HTTPS_PROXY"] = ""
    env["NO_PROXY"] = "*"
    existing_pp = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = (str(cwd) + os.pathsep + existing_pp) if existing_pp else str(cwd)
    if extra_env:
        env.update({k: str(v) for k, v in extra_env.items()})

    log_path = work_dir() / log_name
    cmd = [str(c) for c in cmd]
    logger.info(f"{label} ▶ {' '.join(cmd)}")

    # 单独进程组(POSIX)/进程组标志(Windows):sidecar 会 spawn DataLoader/ffmpeg 子进程,
    # 取消时只杀直接子进程会留下孤儿继续吃显存,故起新进程组,取消时整组 kill。
    popen_kw = {}
    if hasattr(os, "setsid"):
        popen_kw["start_new_session"] = True
    elif os.name == "nt":
        popen_kw["creationflags"] = getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0)

    proc = subprocess.Popen(
        cmd, cwd=str(cwd),
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, **popen_kw,
    )

    buf = b""
    last_pct = -1
    try:
        with open(log_path, "w", encoding="utf-8", errors="replace") as lf:
            while True:
                # 取消工作流时立刻抛出 → finally 杀子进程(组)
                mm.throw_exception_if_processing_interrupted()
                chunk = proc.stdout.read1(4096)
                if not chunk:
                    if proc.poll() is not None:
                        break
                    time.sleep(0.02)  # 无输出时让出 CPU,避免忙等待空转占核
                    continue
                buf += chunk
                while True:
                    r = buf.find(b"\r")
                    n = buf.find(b"\n")
                    if r < 0 and n < 0:
                        break
                    pos = r if n < 0 else (n if r < 0 else min(r, n))
                    raw = buf[:pos]
                    buf = buf[pos + 1:]

                    m = _PCT.search(raw)
                    if m:
                        p = max(0, min(100, int(m.group(1))))
                        if p != last_pct:
                            last_pct = p
                            pb.update_absolute(p, 100)

                    clean = _ANSI.sub(b"", raw).decode("utf-8", "replace")
                    if clean.strip():
                        lf.write(clean + "\n")
                        lf.flush()

        rc = proc.wait()
        if rc != 0:
            raise RuntimeError(f"{label} 失败 (rc={rc})，详见 {log_path}")
    finally:
        if proc.poll() is None:
            _kill_proc_tree(proc)
        try:
            if proc.stdout:
                proc.stdout.close()
        except Exception:
            pass
    return log_path


def _kill_proc_tree(proc):
    """杀掉子进程及其整个进程组(连带 DataLoader/ffmpeg 等孙进程,回收显存)。"""
    try:
        if hasattr(os, "killpg"):
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        else:
            proc.kill()
    except Exception:
        try:
            proc.kill()
        except Exception:
            pass
    try:
        proc.wait(timeout=10)
    except Exception:
        pass
