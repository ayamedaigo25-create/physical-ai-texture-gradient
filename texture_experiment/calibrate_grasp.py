from pathlib import Path
import runpy
import numpy as np
from scipy.optimize import least_squares

root = Path.home() / "pai_ws/texture_experiment"

# 前回のFK計算を再利用
data = runpy.run_path(str(root / "check_tcp_workspace.py"))

q_samples = data["q"]
tcp_positions = data["tcp_positions"]
joints = data["joints"]
names = data["names"]
rotation = data["rotation"]
TCP = data["TCP"]

# キューブ中心（base_link基準）
target = np.array([0.190, -0.010, 0.025])

bounds = np.array([
    [
        float(joints[n].find("limit").get("lower")),
        float(joints[n].find("limit").get("upper"))
    ]
    for n in names
])

def fk(angles):
    R = np.eye(3)
    p = np.zeros(3)

    for name, angle in zip(names, angles):
        joint = joints[name]
        origin = joint.find("origin")

        xyz = np.array([
            float(v) for v in origin.get("xyz").split()
        ])

        p += R @ xyz

        axis = [
            float(v)
            for v in joint.find("axis").get("xyz").split()
        ]

        R = R @ rotation(axis, np.array([angle]))[0]

    return p + R @ TCP

errors = np.linalg.norm(tcp_positions - target, axis=1)
indices = np.argsort(errors)[:20]

solutions = []

for i in indices:
    result = least_squares(
        lambda angles: fk(angles) - target,
        q_samples[i],
        bounds=(bounds[:, 0], bounds[:, 1]),
        max_nfev=200
    )
    error = np.linalg.norm(fk(result.x) - target)
    solutions.append((error, result.x))

solutions.sort(key=lambda s: s[0])
error, angles = solutions[0]

print("\n=== Central grasp pose candidate ===")
for name, angle in zip(names, angles):
    print(f"{name}: {angle:.5f} rad")

print("TCP position:", fk(angles))
print(f"Position error: {error*1000:.2f} mm")

output = root / "central_grasp_candidate.json"

import json
output.write_text(json.dumps({
    "target_xyz_base_m": target.tolist(),
    "joint_names": names,
    "joint_positions_rad": angles.tolist(),
    "tcp_position_base_m": fk(angles).tolist(),
    "position_error_mm": error * 1000,
    "tcp_calibrated": False,
    "collision_checked": False,
    "grasp_verified": False
}, indent=2))

print("Saved:", output)
