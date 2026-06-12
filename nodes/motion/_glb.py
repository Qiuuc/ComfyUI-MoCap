# ComfyUI-Noctyra — 动作(视频转 3D)· SMPL-X → 带蒙皮 GLB 写出
# GPL-3.0 (见仓库 LICENSE)
"""
把 SMPL-X 动作写成带蒙皮的动画 GLB(rest 网格 + 骨骼 + 每帧关节旋转,蒙皮由查看器
端 LBS 完成)。纯 numpy/scipy 手写 glTF 二进制,跑在 ComfyUI 主环境,不进 sidecar。

SMPL-X 与 glTF 同为 Y-up,无需换轴。pose 校正 blendshape(posedirs)略去——查看器
本就不算,影响轻微。
"""
import json
import struct

import numpy as np
from scipy.spatial.transform import Rotation as R

_model_cache = {}


def _load_model(model_path):
    key = str(model_path)
    if key in _model_cache:
        return _model_cache[key]
    d = np.load(model_path, allow_pickle=True)
    parents = np.asarray(d["kintree_table"])[0].astype(np.int64).copy()
    parents[(parents < 0) | (parents > 100)] = -1   # 根
    m = {
        "v_template": np.asarray(d["v_template"], np.float64),       # (V,3)
        "faces": np.asarray(d["f"], np.uint32),                      # (F,3)
        "shapedirs": np.asarray(d["shapedirs"], np.float64),         # (V,3,400)
        "J_regressor": np.asarray(d["J_regressor"], np.float64),     # (55,V)
        "weights": np.asarray(d["weights"], np.float64),             # (V,55)
        "parents": parents,                                          # (55,)
    }
    _model_cache[key] = m
    return m


def _vertex_normals(verts, faces):
    v0, v1, v2 = verts[faces[:, 0]], verts[faces[:, 1]], verts[faces[:, 2]]
    fn = np.cross(v1 - v0, v2 - v0)
    n = np.zeros_like(verts)
    for c in range(3):
        np.add.at(n, faces[:, c], fn)
    ln = np.linalg.norm(n, axis=1, keepdims=True)
    return (n / np.maximum(ln, 1e-8)).astype(np.float32)


def _topk_skin(weights, k=4):
    """每顶点取权重最大的 k 个关节,归一化。返回 (idx uint16, w float32)。"""
    order = np.argsort(-weights, axis=1)[:, :k]
    idx = order.astype(np.uint16)
    w = np.take_along_axis(weights, order, axis=1)
    s = w.sum(axis=1, keepdims=True)
    w = np.where(s > 0, w / np.maximum(s, 1e-8), 0.0).astype(np.float32)
    return idx, w


def write_glb(npz_path, model_path, out_path, fps=None):
    data = np.load(npz_path)
    poses = np.asarray(data["poses"], np.float64).reshape(len(data["poses"]), -1, 3)  # (T,J,3)
    T = poses.shape[0]
    if T == 0:
        raise RuntimeError("空动作(0 帧),无可导出帧")
    J = min(poses.shape[1], 55)
    trans = np.asarray(data["trans"], np.float64) if "trans" in data else np.zeros((T, 3))
    betas = np.asarray(data["betas"], np.float64).reshape(-1) if "betas" in data else np.zeros(10)
    if fps is None:
        fps = int(data["mocap_frame_rate"]) if "mocap_frame_rate" in data else 30
    if not fps or fps <= 0:
        fps = 30

    m = _load_model(model_path)
    nb = min(10, betas.shape[0])
    v_shaped = (m["v_template"] + np.einsum("vij,j->vi", m["shapedirs"][:, :, :nb], betas[:nb])).astype(np.float32)
    faces = m["faces"]
    parents = m["parents"]
    Jp = (m["J_regressor"] @ v_shaped.astype(np.float64)).astype(np.float32)   # (55,3) rest 关节
    normals = _vertex_normals(v_shaped, faces)
    j_idx, j_w = _topk_skin(m["weights"])

    # 每关节局部 rest 平移 = J[i]-J[parent](根=J[0])
    local_t = Jp.copy()
    for i in range(55):
        p = parents[i]
        if p >= 0:
            local_t[i] = Jp[i] - Jp[p]

    # 逆绑定矩阵 = translate(-J[i]),列主序
    ibm = np.tile(np.eye(4, dtype=np.float32), (55, 1, 1))
    ibm[:, :3, 3] = -Jp
    ibm_cm = np.transpose(ibm, (0, 2, 1)).reshape(55, 16)   # 列主序

    # 动画:每关节四元数 (T,4)[x,y,z,w];根另加平移 = J[0]+trans
    quats = np.zeros((55, T, 4), np.float32)
    quats[:, :, 3] = 1.0
    for i in range(J):
        quats[i] = R.from_rotvec(poses[:, i, :]).as_quat().astype(np.float32)
    root_trans = (Jp[0][None, :] + trans).astype(np.float32)   # (T,3)
    times = (np.arange(T, dtype=np.float32) / float(fps))

    # ---- 组装二进制缓冲 ----
    bin_buf = bytearray()
    bufferViews, accessors = [], []

    def add_view(b: bytes, target=None):
        while len(bin_buf) % 4:
            bin_buf.append(0)
        off = len(bin_buf)
        bin_buf.extend(b)
        bv = {"buffer": 0, "byteOffset": off, "byteLength": len(b)}
        if target:
            bv["target"] = target
        bufferViews.append(bv)
        return len(bufferViews) - 1

    def add_acc(bv, comp, count, typ, **extra):
        a = {"bufferView": bv, "componentType": comp, "count": count, "type": typ}
        a.update(extra)
        accessors.append(a)
        return len(accessors) - 1

    F, UI, US, UB = 5126, 5125, 5123, 5121  # FLOAT, UINT, USHORT, UBYTE

    a_idx = add_acc(add_view(faces.reshape(-1).astype(np.uint32).tobytes(), 34963),
                    UI, faces.size, "SCALAR")
    a_pos = add_acc(add_view(v_shaped.tobytes(), 34962), F, len(v_shaped), "VEC3",
                    min=v_shaped.min(0).tolist(), max=v_shaped.max(0).tolist())
    a_nrm = add_acc(add_view(normals.tobytes(), 34962), F, len(normals), "VEC3")
    a_jnt = add_acc(add_view(j_idx.tobytes(), 34962), US, len(j_idx), "VEC4")
    a_wgt = add_acc(add_view(j_w.tobytes(), 34962), F, len(j_w), "VEC4")
    a_ibm = add_acc(add_view(ibm_cm.astype(np.float32).tobytes()), F, 55, "MAT4")
    a_time = add_acc(add_view(times.tobytes()), F, T, "SCALAR",
                     min=[float(times.min())], max=[float(times.max())])
    a_rot = [add_acc(add_view(quats[i].tobytes()), F, T, "VEC4") for i in range(55)]
    a_rt = add_acc(add_view(root_trans.tobytes()), F, T, "VEC3")

    # ---- 节点 / 蒙皮 / 动画 JSON ----
    try:
        from ._skeleton import SMPLX_NAMES
    except ImportError:
        from _skeleton import SMPLX_NAMES

    nodes = []
    for i in range(55):
        node = {"name": SMPLX_NAMES[i], "translation": local_t[i].tolist()}
        kids = [j for j in range(55) if parents[j] == i]
        if kids:
            node["children"] = kids
        nodes.append(node)
    mesh_node = len(nodes)
    nodes.append({"name": "SMPLX_Mesh", "mesh": 0, "skin": 0})

    samplers, channels = [], []
    for i in range(55):
        samplers.append({"input": a_time, "output": a_rot[i], "interpolation": "LINEAR"})
        channels.append({"sampler": len(samplers) - 1, "target": {"node": i, "path": "rotation"}})
    samplers.append({"input": a_time, "output": a_rt, "interpolation": "LINEAR"})
    channels.append({"sampler": len(samplers) - 1, "target": {"node": 0, "path": "translation"}})

    gltf = {
        "asset": {"version": "2.0", "generator": "Noctyra-Mocap"},
        "scene": 0,
        "scenes": [{"nodes": [mesh_node, 0]}],
        "nodes": nodes,
        "meshes": [{"primitives": [{
            "attributes": {"POSITION": a_pos, "NORMAL": a_nrm, "JOINTS_0": a_jnt, "WEIGHTS_0": a_wgt},
            "indices": a_idx,
        }]}],
        "skins": [{"joints": list(range(55)), "inverseBindMatrices": a_ibm, "skeleton": 0}],
        "animations": [{"samplers": samplers, "channels": channels}],
        "buffers": [{"byteLength": len(bin_buf)}],
        "bufferViews": bufferViews,
        "accessors": accessors,
    }

    # ---- 打包 GLB ----
    json_bytes = json.dumps(gltf, separators=(",", ":")).encode("utf-8")
    json_bytes += b" " * ((4 - len(json_bytes) % 4) % 4)
    bin_bytes = bytes(bin_buf)
    bin_bytes += b"\x00" * ((4 - len(bin_bytes) % 4) % 4)
    total = 12 + 8 + len(json_bytes) + 8 + len(bin_bytes)

    with open(out_path, "wb") as f:
        f.write(struct.pack("<III", 0x46546C67, 2, total))           # 'glTF', ver2, length
        f.write(struct.pack("<II", len(json_bytes), 0x4E4F534A))     # JSON chunk
        f.write(json_bytes)
        f.write(struct.pack("<II", len(bin_bytes), 0x004E4942))      # BIN chunk
        f.write(bin_bytes)
    return out_path, T, fps
