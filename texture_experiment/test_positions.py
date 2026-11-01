import subprocess

positions = {
    "base":  (0.20, -0.10),
    "A":     (0.30, -0.10),
    "B":     (0.15, -0.10),
    "C":     (0.20,  0.00),
    "D":     (0.20, -0.20),
}

for name, (x, y) in positions.items():
    input(f"\n{name}: x={x}, y={y} → Enterで移動")

    request = (
        f'name: "experiment_cube" '
        f'position: {{x: {x}, y: {y}, z: 1.09}} '
        'orientation: {w: 1}'
    )

    subprocess.run([
        "ign", "service",
        "-s", "/world/default/set_pose",
        "--reqtype", "ignition.msgs.Pose",
        "--reptype", "ignition.msgs.Boolean",
        "--timeout", "5000",
        "--req", request,
    ], check=True)

print("\n確認終了")
