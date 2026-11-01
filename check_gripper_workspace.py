from pathlib import Path
import xml.etree.ElementTree as ET
import numpy as np

urdf = Path.home() / "pai_ws/eye_in_hand/crane_eye_in_hand.urdf"
robot = ET.parse(urdf).getroot()
joints = {j.get("name"): j for j in robot.findall("joint")}

names = [
    "crane_plus_joint1",
    "crane_plus_joint2",
    "crane_plus_joint3",
    "crane_plus_joint4",
    "crane_plus_joint_hand",
]

def rotation(axis, angle):
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    x, y, z = axis
    c, s = np.cos(angle), np.sin(angle)
    C = 1 - c
    return np.array([
        [c+x*x*C, x*y*C-z*s, x*z*C+y*s],
        [y*x*C+z*s, c+y*y*C, y*z*C-x*s],
        [z*x*C-y*s, z*y*C+x*s, c+z*z*C]
    ])

def fk(q):
    position = np.zeros(3)
    R = np.eye(3)

    for name, angle in zip(names, list(q) + [0.0]):
        joint = joints[name]
        origin = joint.find("origin")
        offset = np.array(
            [float(v) for v in origin.get("xyz").split()]
        )

        # 現在のURDFでは対象関節のorigin rpyはすべて0
        assert all(
            abs(float(v)) < 1e-9
            for v in origin.get("rpy", "0 0 0").split()
        )

        position += R @ offset

        axis = [
            float(v)
            for v in joint.find("axis").get("xyz").split()
        ]
        R = R @ rotation(axis, angle)

    return position

limits = []
for name in names[:4]:
    lim = joints[name].find("limit")
    limits.append((
        float(lim.get("lower")),
        float(lim.get("upper"))
    ))

targets = [
    ("中央", 0.190, -0.010),
    ("近距離側", 0.160, 0.000),
    ("遠距離側", 0.230, 0.000),
    ("左側", 0.190, 0.090),
    ("右側", 0.190, -0.110),
]

rng = np.random.default_rng(2026)
samples = np.column_stack([
    rng.uniform(lo, hi, 50000)
    for lo, hi in limits
])

positions = np.array([fk(q) for q in samples])

print("Gripper joint origin FK samples:", len(positions))

# ロボット基準座標で高さを評価
for z in (0.025, 0.075, 0.125):
    mask = np.abs(positions[:, 2] - z) <= 0.015
    subset = positions[mask]

    print(f"\n=== Height z={z:.3f} m ===")
    print("Height-matched samples:", len(subset))

    if len(subset) == 0:
        continue

    for name, x, y in targets:
        distances = np.linalg.norm(
            subset[:, :2] - [x, y], axis=1
        )
        index = int(np.argmin(distances))
        point = subset[index]

        print(
            f"{name}: 最短水平距離={distances[index]:.3f} m, "
            f"FK位置=({point[0]:.3f}, "
            f"{point[1]:.3f}, {point[2]:.3f})"
        )
