# ComfyUI-Noctyra — 动作(视频转 3D)节点 · 身体动作 GVHMR
# GPL-3.0 (见仓库 LICENSE)
"""
身体动作节点：在 sidecar 里跑 GVHMR 的 tools/demo/demo.py，得到世界坐标 SMPL 序列
(hmr4d_results.pt)。可选先把视频降到 720p/1080p(ffmpeg，有就用、没有就用原画)。

输出 MOCAP_MOTION(指向 .pt) + 身体预览 VIDEO(mesh 叠原视频)。
"""
import logging
import shutil
import subprocess
from pathlib import Path

from ._types import MOCAP_MOTION
from . import _runtime, _paths

logger = logging.getLogger("noctyra")

# 长边像素上限
_RES = {"original": None, "720": 1280, "1080": 1920}


def _probe_video(path):
    """用 ffprobe 探测真实 fps 与帧数。失败回退 (30.0, 0)。
    GVHMR 逐帧输出,导出 fps 必须 = 原视频 fps,否则播放变速(时长错误)。"""
    fp = shutil.which("ffprobe")
    if not fp:
        logger.warning("未找到 ffprobe,fps 回退 30(非 30fps 素材导出会变速)。")
        return 30.0, 0
    try:
        out = subprocess.run(
            [fp, "-v", "error", "-select_streams", "v:0", "-show_entries",
             "stream=r_frame_rate,nb_frames", "-of", "csv=p=0:s=,", str(path)],
            capture_output=True, text=True,
        ).stdout.strip()
        parts = out.split(",")
        rate = parts[0] if parts else "30/1"
        num, _, den = rate.partition("/")
        fps = float(num) / float(den) if den and float(den) else float(num or 30)
        nf = 0
        if len(parts) > 1 and parts[1].isdigit():
            nf = int(parts[1])
        return (fps if fps > 0 else 30.0), nf
    except Exception as e:
        logger.warning(f"ffprobe 解析 fps 失败({e}),回退 30。")
        return 30.0, 0


def _ffmpeg_downscale(src: Path, dst: Path, max_side: int) -> bool:
    """有 ffmpeg 才降分辨率；返回 True 表示生成了 dst，False 表示应继续用原视频。"""
    ff = shutil.which("ffmpeg")
    if not ff:
        logger.warning("未找到 ffmpeg，跳过降分辨率，用原视频。")
        return False
    try:
        w = h = None
        fp = shutil.which("ffprobe")
        if fp:
            out = subprocess.run(
                [fp, "-v", "error", "-select_streams", "v:0",
                 "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x", str(src)],
                capture_output=True, text=True,
            ).stdout.strip()
            if out and "x" in out:
                ws, _, hs = out.partition("x")
                if ws.isdigit() and hs.isdigit():
                    w, h = int(ws), int(hs)
            if w is not None and h is not None and max(w, h) <= max_side:
                return False  # 已经够小
        scale = f"{max_side}:-2" if (w is None or w >= (h or 0)) else f"-2:{max_side}"
        subprocess.run(
            [ff, "-y", "-i", str(src), "-vf", f"scale={scale}",
             "-c:v", "libx264", "-crf", "23", "-preset", "veryfast",
             "-pix_fmt", "yuv420p", "-an", str(dst)],
            check=True, capture_output=True,
        )
        return True
    except Exception as e:
        logger.warning(f"降分辨率失败({e})，用原视频。")
        return False


class MocapGVHMR:
    DESCRIPTION = (
        "GVHMR 身体动作估计:从视频提取重力对齐的世界坐标 SMPL 序列(身体 21 关节 + 全局轨迹)。\n"
        "在 sidecar 私有环境里跑,显存吃紧前会自动卸载 ComfyUI 已加载模型。\n"
        "输出 身体动作(接合并节点) + 预览视频(看动捕是否贴合)。"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video": ("VIDEO", {"tooltip": "接 ComfyUI 自带『加载视频』节点"}),
                "resolution": (["original", "720", "1080"], {
                    "default": "720",
                    "tooltip": "推理前把长边降到该档(720≈1280 / 1080≈1920),省显存提速;original=原画。需 ffmpeg",
                }),
            },
        }

    RETURN_TYPES = (MOCAP_MOTION, "VIDEO")
    RETURN_NAMES = ("motion", "preview")
    FUNCTION = "run"
    CATEGORY = "Noctyra/动作"

    def run(self, video, resolution="720"):
        gvhmr_dir = _paths.gvhmr_dir()
        if not (gvhmr_dir / "tools" / "demo" / "demo.py").exists():
            raise RuntimeError(f"GVHMR 目录无效(缺 tools/demo/demo.py):{gvhmr_dir}")
        py = _runtime.sidecar_python()

        src, stem = _runtime.video_to_mp4(video)
        # 探真实 fps(降分辨率不改帧率,探原视频即可)
        fps, n_frames = _probe_video(src)

        # 0. 可选降分辨率
        video_for_pipeline = src
        scaled = None
        max_side = _RES.get(resolution)
        if max_side is not None:
            scaled = _runtime.work_dir() / f"{stem}_scaled.mp4"
            if _ffmpeg_downscale(src, scaled, max_side):
                video_for_pipeline = scaled
            else:
                scaled = None  # 没生成,无需清理

        staged = gvhmr_dir / "inputs" / f"{stem}.mp4"
        out_dir = gvhmr_dir / "outputs" / "demo" / stem
        results = out_dir / "hmr4d_results.pt"
        try:
            # 1. 暂存进 GVHMR/inputs/{stem}.mp4
            _runtime.stage_file(video_for_pipeline, staged)
            if not results.exists():
                if out_dir.exists():
                    shutil.rmtree(out_dir, ignore_errors=True)
                # 跑前腾显存：卸载 ComfyUI 主进程里已加载的模型
                _runtime.unload_main_models()
                _runtime.run(
                    [py, "-u", "tools/demo/demo.py", f"--video=inputs/{stem}.mp4", "-s"],
                    cwd=gvhmr_dir, log_name=f"gvhmr_{stem}.log", label="身体动作 GVHMR",
                    extra_env={"HF_ENDPOINT": "https://hf-mirror.com"},
                )
            if not results.exists():
                raise RuntimeError(f"GVHMR 未产出结果:{results}")
        finally:
            # 无论成功/失败/取消都清暂存输入与降分辨率中间文件(否则残留累积)
            staged.unlink(missing_ok=True)
            if scaled is not None:
                scaled.unlink(missing_ok=True)

        # 把结果(.pt) + 预览搬到 temp,随后清掉 GVHMR 工作目录与暂存输入,
        # 避免每跑一次就在 mocap/GVHMR 里累积几百 MB(stem 每次随机)。
        wd = _runtime.work_dir()
        pt_out = wd / f"{stem}_results.pt"
        shutil.copy2(results, pt_out)
        preview_src = out_dir / f"{stem}_3_incam_global_horiz.mp4"
        has_preview = preview_src.exists()
        preview_out = wd / f"{stem}_preview.mp4"
        if has_preview:
            shutil.copy2(preview_src, preview_out)
        shutil.rmtree(out_dir, ignore_errors=True)

        motion = {"pt_path": str(pt_out), "stem": stem, "fps": fps, "n_frames": n_frames}
        vid = _runtime.video_from_file(preview_out if has_preview else src)
        return (motion, vid)


NODE_CLASS_MAPPINGS = {
    "NoctyraMocapGVHMR": MocapGVHMR,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "NoctyraMocapGVHMR": "Mocap 身体动作（GVHMR）",
}
