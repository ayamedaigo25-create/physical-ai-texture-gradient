import json
import math
import random
from pathlib import Path

root = Path.home() / "pai_ws/texture_experiment"
visible = json.loads(
    (root / "visible_positions.json").read_text()
)["positions"]

visible_set = {tuple(p) for p in visible}
rng = random.Random(2026)

# URDFで確認した各関節の可動範囲
limits = [
    (-2.61799, 2.61799),
    (-1.82474, 1.78372),
    (-2.55656, 2.45428),
    (-1.84010, 1.76837),
]

# テーブル上のキューブ中心に対応する高さ
target_z = 0.025

reachable = set()
height_matches = 0

for _ in range(300000):
    q1, q2, q3, q4 = [
        rng.uniform(lo, hi)
        for lo, hi in limits
    ]

    a2 = q2
    a3 = q2 - q3
    a4 = q2 - q3 - q4

    # ハンド関節原点の順運動学（URDF基準）
    r = (
        0.083 * math.sin(a2)
        + 0.0935 * math.sin(a3)
        + 0.045 * math.sin(a4)
    )

    z = (
        0.0454 + 0.026
        + 0.083 * math.cos(a2)
        + 0.0935 * math.cos(a3)
        + 0.045 * math.cos(a4)
    )

    if abs(z - target_z) > 0.015:
        continue

    height_matches += 1

    x = r * math.cos(q1) + 0.0145 * math.sin(q1)
    y = r * math.sin(q1) - 0.0145 * math.cos(q1)

    # 1cm格子上の近傍候補
    ix = round(x * 100)
    iy = round(y * 100)

    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            p = (round((ix + dx) / 100, 2),
                 round((iy + dy) / 100, 2))

            if p in visible_set:
                reachable.add(p)

result = sorted(reachable)
output = root / "fk_reachable_candidates.json"

output.write_text(json.dumps({
    "method": "random joint sampling + forward kinematics",
    "samples": 300000,
    "target": "hand joint origin, approximate cube center height",
    "height_tolerance_m": 0.015,
    "xy_neighborhood_m": 0.02,
    "collision_checked": False,
    "grasp_checked": False,
    "positions": result,
}, indent=2))

print("Visible points:", len(visible))
print("Samples near target height:", height_matches)
print("Potentially reachable visible points:", len(result))

if result:
    print("X:", min(x for x, y in result),
                max(x for x, y in result))
    print("Y:", min(y for x, y in result),
                max(y for x, y in result))

print("Saved:", output)
