# ComfyUI-Mocap — comfy-env 取数兜底补丁(可复现,版本无关)
#
# 背景:comfy-env 用 urllib 拉 cuda-wheels 预编译索引(pozzettiandrea.github.io)。
# 本机/部分网络下该 CDN(Fastly)会对默认 urllib 直连回 SSL UNEXPECTED_EOF/RST,
# 且 comfy-env 的重试不把 SSL 错误判为可重试 → 首次拉索引就失败 → 隔离环境建不起来
# (pytorch3d/detectron2 等预编译轮子全靠这个索引)。
#
# 修法:不改 comfy-env 源码(各版本补丁点不同、易碎),而是运行时给
# urllib.request.urlopen 套一层兜底——urllib 失败时对相关主机改用 curl 重取。
# 版本无关、覆盖 comfy-env 所有取数点;在 comfy_env.install()/setup_env() 之前
# 调 apply() 即可。幂等。
import io
import ssl
import subprocess
import urllib.error
import urllib.request

# 仅对这些主机做 curl 兜底(cuda-wheels 索引 / GitHub Releases / HF 镜像)
_FALLBACK_HOSTS = (
    "pozzettiandrea.github.io",
    "github.com",
    "githubusercontent.com",
    "hf-mirror.com",
    "huggingface.co",
)
_orig_urlopen = urllib.request.urlopen


class _CurlResponse(io.BytesIO):
    """最小化模拟 urlopen 返回的响应对象(支持 with / read / getcode / headers)。"""
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
        return False

    def getcode(self):
        return 200

    def info(self):
        return {}

    @property
    def status(self):
        return 200


def _curl_bytes(url, timeout):
    to = int(timeout or 30)
    r = subprocess.run(
        ["curl", "-sS", "-L", "--fail", "--retry", "20", "--retry-all-errors",
         "--retry-delay", "2", "--connect-timeout", "25", "--max-time", str(max(to + 180, 240)),
         "-A", "comfy-env", url],
        capture_output=True, timeout=max(to + 240, 300))
    if r.returncode != 0:
        raise OSError("curl rc=%d for %s: %s"
                      % (r.returncode, url, r.stderr.decode("utf-8", "replace")[:200]))
    return r.stdout


def _patched_urlopen(url, *args, **kwargs):
    u = url.full_url if isinstance(url, urllib.request.Request) else url
    timeout = kwargs.get("timeout", 30)
    try:
        return _orig_urlopen(url, *args, **kwargs)
    except urllib.error.HTTPError:
        raise  # 真 4xx/5xx 不兜底
    except (ssl.SSLError, urllib.error.URLError, OSError, TimeoutError) as e:
        if isinstance(u, str) and any(h in u for h in _FALLBACK_HOSTS):
            print(f"[Mocap][patch_comfy_env] urllib 失败({type(e).__name__}),改用 curl 取: {u}")
            return _CurlResponse(_curl_bytes(u, timeout))
        raise


def apply():
    """幂等地给 urllib.request.urlopen 套 curl 兜底。"""
    if getattr(urllib.request.urlopen, "_mocap_curl_patched", False):
        return
    _patched_urlopen._mocap_curl_patched = True
    urllib.request.urlopen = _patched_urlopen
    print("[Mocap][patch_comfy_env] 已给 urllib.urlopen 套 curl 兜底(cuda-wheels 索引可达性)")


if __name__ == "__main__":
    apply()
