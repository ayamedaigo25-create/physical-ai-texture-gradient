import json
import math
from pathlib import Path

ROOT = Path.home() / "pai_ws/texture_experiment"

# 初期観測姿勢で取得したカメラのTF
CX, CY, CZ = -0.002, -0.010, 0.227
THETA = 0.679

# CameraInfo
FX = FY = 554.3827056884766
U0, V0 = 320.0, 240.0
WIDTH, HEIGHT = 640, 480

# Gazeboモデル原点をbase_linkの原点と仮定
BASE_Z = 1.015
TABLE_Z = 1.015
CUBE_SIZE = 0.05
MARGIN = 20
STEP = 0.01

c, s = math.cos(THETA), math.sin(THETA)

def project(x, y, z):
    # Gazebo座標 -> ロボット基準座標
    z -= BASE_Z

    dx, dy, dz = x - CX, y - CY, z - CZ

    # カメラ座標へ逆回転
    forward = c * dx - s * dz
    left = dy
    up = s * dx + c * dz

    if forward <= 0:
        return None

    u = U0 - FX * left / forward
    v = V0 - FY * up / forward
    return u, v

def cube_visible(x, y):
    half = CUBE_SIZE / 2
    for dx in (-half, half):
        for dy in (-half, half):
            for dz in (0, CUBE_SIZE):
                uv = project(
                    x + dx,
                    y + dy,
                    TABLE_Z + dz
                )
                if uv is None:
                    return False

                u, v = uv
                if not (
                    MARGIN <= u <= WIDTH - MARGIN
                    and MARGIN <= v <= HEIGHT - MARGIN
                ):
                    return False
    return True

# テーブル範囲内で1cm刻みの候補を抽出
valid = []
for ix in range(-11, 70):
    for iy in range(-75, 76):
        x, y = ix * STEP, iy * STEP

        # キューブ全体がテーブルに載る範囲
        if not (-0.085 <= x <= 0.665):
            continue
        if not (-0.725 <= y <= 0.725):
            continue

        if cube_visible(x, y):
            valid.append([round(x, 3), round(y, 3)])

assert valid, "No visible positions found"

result = {
    "camera_pose_base": [CX, CY, CZ],
    "camera_pitch": THETA,
    "image_size": [WIDTH, HEIGHT],
    "camera_focal_length_px": FX,
    "cube_size_m": CUBE_SIZE,
    "image_margin_px": MARGIN,
    "grid_step_m": STEP,
    "criterion": "all 8 cube vertices within image margins",
    "coordinate_frame": "Gazebo world; base alignment assumed",
    "positions": valid
}

output = ROOT / "visible_positions.json"
output.write_text(json.dumps(result, indent=2))

print("Visible positions:", len(valid))
print("X range:", min(p[0] for p in valid),
      max(p[0] for p in valid))
print("Y range:", min(p[1] for p in valid),
      max(p[1] for p in valid))
print("Saved:", output)
