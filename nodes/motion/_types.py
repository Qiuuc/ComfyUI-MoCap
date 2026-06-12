# ComfyUI-Noctyra — 动作(视频转 3D)节点 · 连线类型
# GPL-3.0 (见仓库 LICENSE)
"""
动作子包的自定义连线类型名。

V1 节点 API 下，自定义 socket 类型就是一个字符串，靠名字相等来判断能否连接。
这里集中定义，避免各节点里手写字符串拼错。
"""

# GVHMR 身体结果：{"pt_path": str, "stem": str, "fps": int, "n_frames": int}
MOCAP_MOTION = "MOCAP_MOTION"
# HaMeR 手部结果：{"npz_path": str}
MOCAP_HANDS = "MOCAP_HANDS"
# 合并后的 SMPL-X 动作：{"npz_path": str(amass npz), "fps": int, "n_frames": int}
MOCAP_ANIM = "MOCAP_ANIM"
