# ComfyUI-Noctyra — 动作(视频转 3D)· SMPL-X → BVH 写出
# GPL-3.0 (见仓库 LICENSE)
"""
把合并后的 SMPL-X 动作(AMASS 风格 npz)写成 BVH。纯 numpy/scipy，跑在 ComfyUI
主环境即可,不进 sidecar。

输入 npz 字段(由合并节点产出)：
    poses (T, 165)  —— 55 关节 × 轴角(轴角=旋转向量)
    trans (T, 3)    —— 根位移
    mocap_frame_rate—— 帧率

骨架取 _skeleton 的 55 关节。轴角→ZXY 欧拉(BVH 通道顺序),根/各关节一致地留在
SMPL-X 原生坐标系(不做 Y/Z 互换),保证旋转与位移同一参考系、动作不串。

骨长(OFFSET):优先用 SMPL-X 真实 rest 关节(J_regressor@v_shaped,含 betas 体型),
与 GLB 导出骨架一致 → retarget(3ds Max/Maya/BIP)时 IK/脚步贴地才准。模型缺失时
回退到 _skeleton 的示意骨长(仅靠旋转驱动,纯 FK 播放仍正确,但比例失真)。
"""
import logging

import numpy as np
from scipy.spatial.transform import Rotation as R

try:
    from ._skeleton import SMPLX_NAMES, SMPLX_PARENTS, SMPLX_OFFSETS
except ImportError:  # 允许独立导入(测试用)
    from _skeleton import SMPLX_NAMES, SMPLX_PARENTS, SMPLX_OFFSETS

logger = logging.getLogger("noctyra")

_OFFSET_SCALE = 10.0    # 示意骨长缩放(仅回退路径用)
_TRANS_SCALE = 100.0    # 米 → 厘米


def _children(parent_idx):
    return [i for i, p in enumerate(SMPLX_PARENTS) if p == parent_idx]


def _rest_offsets(model_path, betas):
    """用 SMPL-X 真实 rest 关节算各关节相对父的局部骨向(米)。失败返回 None(回退示意)。
    与 _glb 同源:Jp = J_regressor @ (v_template + shapedirs·betas)。"""
    try:
        from ._glb import _load_model
    except ImportError:
        from _glb import _load_model
    try:
        m = _load_model(model_path)
        betas = np.asarray(betas, np.float64).reshape(-1)
        nb = min(m["shapedirs"].shape[2], betas.shape[0]) if betas.size else 0
        v = m["v_template"].copy()
        if nb:
            v = v + np.einsum("vij,j->vi", m["shapedirs"][:, :, :nb], betas[:nb])
        Jp = m["J_regressor"] @ v                       # (55,3) rest 关节(米)
        off = np.zeros((55, 3), np.float64)
        for c in range(55):
            p = SMPLX_PARENTS[c]
            if 0 <= p < 55:
                off[c] = Jp[c] - Jp[p]
        return off
    except Exception as e:
        logger.warning(f"BVH 取真实 rest 关节失败({e}),回退示意骨长。")
        return None


def _aa_to_zxy_deg(aa):
    """轴角 → ZXY 欧拉角(度),返回 [z, x, y] 对应 BVH 的 Zrot Xrot Yrot。"""
    if not np.any(aa):
        return (0.0, 0.0, 0.0)
    z, x, y = R.from_rotvec(aa).as_euler("ZXY", degrees=True)
    return (z, x, y)


def _offset_cm(idx, offsets):
    """关节 idx 相对父的 OFFSET(厘米)。offsets 为真实骨向(米,55×3)则用之,
    否则回退 _skeleton 示意骨长。"""
    if offsets is not None:
        o = offsets[idx] * _TRANS_SCALE
        return [float(o[0]), float(o[1]), float(o[2])]
    return [v * _OFFSET_SCALE for v in SMPLX_OFFSETS[idx]]


def _build_hierarchy(offsets=None):
    """递归生成 BVH HIERARCHY 段。根 6 通道,其余 3 通道。
    offsets:真实 rest 骨向(米,55×3);None=用示意骨长。"""
    lines = ["HIERARCHY", "ROOT " + SMPLX_NAMES[0], "{",
             "\tOFFSET 0.000000 0.000000 0.000000",
             "\tCHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation"]

    def recurse(idx, depth):
        indent = "\t" * depth
        kids = _children(idx)
        if kids:
            for c in kids:
                o = _offset_cm(c, offsets)
                lines.append(f"{indent}JOINT {SMPLX_NAMES[c]}")
                lines.append(f"{indent}{{")
                lines.append(f"{indent}\tOFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
                lines.append(f"{indent}\tCHANNELS 3 Zrotation Xrotation Yrotation")
                recurse(c, depth + 1)
                lines.append(f"{indent}}}")
        else:
            # 叶子用一个短 End Site 收尾(沿入骨方向延伸一截)
            o = [v * 0.4 for v in _offset_cm(idx, offsets)]
            lines.append(f"{indent}End Site")
            lines.append(f"{indent}{{")
            lines.append(f"{indent}\tOFFSET {o[0]:.6f} {o[1]:.6f} {o[2]:.6f}")
            lines.append(f"{indent}}}")

    recurse(0, 1)
    lines.append("}")
    return lines


def write_bvh(npz_path, out_path, fps=None, model_path=None):
    data = np.load(npz_path)
    poses = np.asarray(data["poses"], dtype=np.float64)      # (T, 165)
    T = poses.shape[0]
    if T == 0:
        raise RuntimeError("空动作(0 帧),无可导出帧")
    poses = poses.reshape(T, -1, 3)                          # (T, J, 3)
    J = min(poses.shape[1], 55)
    trans = np.asarray(data["trans"], dtype=np.float64) if "trans" in data else np.zeros((T, 3))
    betas = np.asarray(data["betas"], np.float64).reshape(-1) if "betas" in data else np.zeros(10)
    if fps is None:
        fps = int(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else 30
    if not fps or fps <= 0:   # 防 npz 里 fps=0 导致 1/fps 除零
        fps = 30

    # 真实 rest 骨长(与 GLB 一致;含 betas 体型)。model_path 缺/失败则回退示意骨长。
    offsets = _rest_offsets(model_path, betas) if model_path else None

    # HIERARCHY + 关节写出顺序(深度优先,与 hierarchy 递归一致)
    order = []

    def collect(idx):
        order.append(idx)
        for c in _children(idx):
            collect(c)
    collect(0)

    lines = _build_hierarchy(offsets)
    lines += ["MOTION", f"Frames: {T}", f"Frame Time: {1.0 / fps:.6f}"]

    for f in range(T):
        vals = []
        # 根:位移(原生坐标系,m→cm) + 旋转
        t = trans[f] * _TRANS_SCALE
        vals += [t[0], t[1], t[2]]
        for j in order:
            aa = poses[f, j] if j < J else np.zeros(3)
            vals += list(_aa_to_zxy_deg(aa))
        lines.append(" ".join(f"{v:.6f}" for v in vals))

    text = "\n".join(lines) + "\n"
    with open(out_path, "w", encoding="utf-8") as fh:
        fh.write(text)
    return out_path, T, fps
