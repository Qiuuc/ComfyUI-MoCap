# ComfyUI-Mocap — 动作(视频转 3D)· 自检与配置向导
# GPL-3.0 (见仓库 LICENSE)
"""
检查"能不能跑"的三层:① 隔离环境(torch)② 源码仓库(GVHMR/HaMeR)③ 权重(清单)。
缺什么 → 明确告诉缺什么、用哪条命令补、注册权重去哪下。加载时打印简报(不阻断);
也提供『Mocap 自检』节点,在图里跑一下出完整报告。
"""
import glob
import json
import logging
import os

from ._runtime import plugin_root
from . import _paths

logger = logging.getLogger("noctyra")


def manifest_path():
    return plugin_root() / "mocap_models_manifest.json"


def _env_python():
    """找隔离环境 python(自包含,不跨包导入):向上找 .ce/.pixi/envs/*/bin/python。"""
    roots = []
    er = os.environ.get("COMFY_ENV_ROOT")
    if er:
        roots.append(er)
    cur = plugin_root()
    for _ in range(7):
        roots.append(str(cur / ".ce"))
        cur = cur.parent
    for root in roots:
        hits = glob.glob(os.path.join(root, ".pixi", "envs", "*", "bin", "python"))
        if hits:
            mocap = [h for h in hits if "mocap" in h.lower()]
            return (mocap or hits)[0]
    return None


def _status():
    """返回结构化状态:env / 仓库 / 权重(缺失分注册与非注册)。"""
    env_py = _env_python()
    gvhmr_ok = (_paths.gvhmr_dir() / "tools" / "demo" / "demo.py").exists()
    hamer_ok = (_paths.hamer_dir() / "hamer" / "configs" / "__init__.py").exists()

    items = []
    mf = manifest_path()
    if mf.exists():
        try:
            items = json.loads(mf.read_text(encoding="utf-8")).get("items", [])
        except Exception as e:
            logger.warning(f"Mocap 清单读取失败: {e}")
    home = _paths.mocap_home()
    missing = [it for it in items if it.get("path") and not (home / it["path"]).exists()]
    return {
        "env_py": env_py,
        "gvhmr_repo": gvhmr_ok,
        "hamer_repo": hamer_ok,
        "missing_licensed": [m for m in missing if m.get("license")],
        "missing_free": [m for m in missing if not m.get("license")],
        "home": home,
    }


def check(verbose=True):
    """加载时简报(只在控制台,不阻断)。返回缺失权重列表(向后兼容)。"""
    s = _status()
    if verbose:
        env = "✓" if s["env_py"] else "✗ 未建(跑 install.py)"
        gv = "✓" if s["gvhmr_repo"] else "✗ 未 clone(install.py 自动 / git clone)"
        hm = "✓" if s["hamer_repo"] else "—(可选,setup_hamer.py)"
        nfree, nlic = len(s["missing_free"]), len(s["missing_licensed"])
        logger.info(f"Mocap 自检: 环境{env} | GVHMR仓库{gv} | HaMeR仓库{hm} | "
                    f"缺权重 非注册{nfree}/注册{nlic}(『Mocap 自检』节点看详情与补法)")
    return s["missing_licensed"] + s["missing_free"]


def full_report():
    """完整报告 + 可操作补法(给自检节点输出)。"""
    s = _status()
    L = ["===== ComfyUI-Mocap 自检 =====", f"mocap 根目录: {s['home']}", ""]

    L.append("【环境】" + ("✓ 隔离环境就绪" if s["env_py"] else "✗ 未建 → 跑:python install.py"))
    L.append("【GVHMR 身体】仓库 " + ("✓" if s["gvhmr_repo"]
             else "✗ → install.py 会自动 clone,或 git clone https://github.com/zju3dv/GVHMR.git mocap/GVHMR"))
    L.append("【HaMeR 手部(可选)】仓库 " + ("✓" if s["hamer_repo"]
             else "—(默认用 WiLoR;要 HaMeR 跑:python setup_hamer.py)"))
    L.append("【WiLoR 手部】环境就绪即用,首次运行自动从 hf-mirror 下权重(无需手动)")
    L.append("")

    if s["missing_free"]:
        L.append(f"【缺·非注册权重 {len(s['missing_free'])} 项】一键下:python download_weights.py")
        for m in s["missing_free"]:
            L.append(f"    ✗ {m['path']}  ({m.get('size','?')})")
    else:
        L.append("【非注册权重】✓ 全部就位")

    if s["missing_licensed"]:
        L.append(f"【缺·注册权重 {len(s['missing_licensed'])} 项】须本人注册下载,放到 mocap 对应路径:")
        for m in s["missing_licensed"]:
            L.append(f"    ✗ {m['path']}")
            L.append(f"        来源: {m.get('source', '')}")
        L.append("    MANO 注册后:设 MOCAP_MANO_ZIP 指向 mano_v1_2.zip 再跑 setup_hamer.py")
    else:
        L.append("【注册权重】✓ 全部就位")

    L.append("")
    L.append("结论: " + ("✅ GVHMR+WiLoR 可跑" if (s["env_py"] and s["gvhmr_repo"])
             else "⚠ 还差环境/仓库,见上"))
    return "\n".join(L)


class MocapSelfCheck:
    DESCRIPTION = "Mocap 自检:看隔离环境/源码仓库/权重是否就位,缺啥给出补齐命令与下载链接。"

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("report",)
    FUNCTION = "run"
    OUTPUT_NODE = True
    CATEGORY = "Noctyra/动作"

    def run(self):
        rep = full_report()
        print("\n" + rep + "\n")
        return {"ui": {"text": [rep]}, "result": (rep,)}


NODE_CLASS_MAPPINGS = {"NoctyraMocapSelfCheck": MocapSelfCheck}
NODE_DISPLAY_NAME_MAPPINGS = {"NoctyraMocapSelfCheck": "Mocap 自检"}
