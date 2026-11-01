import subprocess
import time

positions = [
    ("中央", 0.300, 0.000),
    ("近距離", 0.170, 0.000),
    ("遠距離", 0.530, 0.000),
    ("左側", 0.320, 0.150),
    ("右側", 0.320, -0.170),
]

for name, x, y in positions:
    input(f"\n{name}: ({x:.3f}, {y:.3f}) Enterで移動 → ")

    request = (
        f'name: "experiment_cube" '
        f'position: {{x: {x}, y: {y}, z: 1.09}} '
        'orientation: {w: 1}'
    )

    result = subprocess.run([
        "ign", "service",
        "-s", "/world/default/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "5000",
        "--req", request,
    ], capture_output=True, text=True)

    if result.returncode != 0 or "data: true" not in result.stdout.lower():
        print("移動失敗:", result.stdout, result.stderr)
        break

    time.sleep(2)
    print(f"{name}: カメラ映像を確認してください")

print("\n検証終了")
