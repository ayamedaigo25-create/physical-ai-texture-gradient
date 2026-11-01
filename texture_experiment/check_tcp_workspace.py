import json
from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

root = Path.home() / "pai_ws/texture_experiment"
urdf = root.parent / "eye_in_hand/crane_eye_in_hand.urdf"

robot = ET.parse(urdf).getroot()
joints = {j.get("name"): j for j in robot.findall("joint")}
names = [f"crane_plus_joint{i}" for i in range(1, 5)]

# 第4リンク座標系における仮TCP
TCP = np.array([0.0, -0.020, 0.110])

visible = np.array(json.loads(
    (root / "visible_positions.json").read_text()
)["positions"])

targets = [
    ("center", 0.190, -0.010),
    ("near", 0.160, 0.000),
    ("far", 0.230, 0.000),
    ("left", 0.190, 0.090),
    ("right", 0.190, -0.110),
]

def rotation(axis, q):
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(q), np.sin(q)
    C = 1 - c
    R = np.empty((len(q), 3, 3))
    R[:, 0, 0] = c + x*x*C
    R[:, 0, 1] = x*y*C - z*s
    R[:, 0, 2] = x*z*C + y*s
    R[:, 1, 0] = y*x*C + z*s
    R[:, 1, 1] = c + y*y*C
    R[:, 1, 2] = y*z*C - x*s
    R[:, 2, 0] = z*x*C - y*s
    R[:, 2, 1] = z*y*C + x*s
    R[:, 2, 2] = c + z*z*C
    return R

N = 120000
rng = np.random.default_rng(2026)
q = np.column_stack([
    rng.uniform(
        float(joints[n].find("limit").get("lower")),
        float(joints[n].find("limit").get("upper")),
        N
    )
    for n in names
])

R = np.broadcast_to(np.eye(3), (N, 3, 3)).copy()
p = np.zeros((N, 3))

for i, name in enumerate(names):
    joint = joints[name]
    origin = joint.find("origin")
    xyz = np.array([
        float(v) for v in origin.get("xyz").split()
    ])
    rpy = [
        float(v) for v in origin.get("rpy", "0 0 0").split()
    ]
    assert np.allclose(rpy, 0), name

    p += np.einsum("nij,j->ni", R, xyz)

    axis = [
        float(v) for v in joint.find("axis").get("xyz").split()
    ]
    R = R @ rotation(axis, q[:, i])

tcp_positions = p + np.einsum("nij,j->ni", R, TCP)

target_z = 0.025
mask = np.abs(tcp_positions[:, 2] - target_z) <= 0.010
near_height = tcp_positions[mask]

print("Joint samples:", N)
print("Samples near cube center height:", len(near_height))

print("\n=== Five representative positions ===")
for name, x, y in targets:
    errors = np.linalg.norm(
        tcp_positions - [x, y, target_z], axis=1
    )
    best = np.argmin(errors)
    point = tcp_positions[best]
    print(
        f"{name}: 3D error={errors[best]*1000:.1f} mm, "
        f"TCP=({point[0]:.3f}, {point[1]:.3f}, "
        f"{point[2]:.3f})"
    )

reachable = []
for x, y in visible:
    if len(near_height) == 0:
        break

    errors = np.linalg.norm(
        near_height[:, :2] - [x, y], axis=1
    )
    if errors.min() <= 0.010:
        reachable.append([float(x), float(y)])

output = root / "tcp_reachable_candidates.json"
output.write_text(json.dumps({
    "tcp_link": "crane_plus_link4",
    "tcp_offset_m": TCP.tolist(),
    "tcp_status": "provisional",
    "samples": N,
    "target_z_base_m": target_z,
    "collision_checked": False,
    "grasp_checked": False,
    "positions": reachable
}, indent=2))

print("\n=== Visible and TCP-reachable candidates ===")
print("Visible:", len(visible))
print("TCP candidates:", len(reachable))

if reachable:
    a = np.array(reachable)
    print("X:", a[:, 0].min(), a[:, 0].max())
    print("Y:", a[:, 1].min(), a[:, 1].max())

print("Saved:", output)
