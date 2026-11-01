import argparse
import json
import random
import secrets
import subprocess
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path.home() / "pai_ws/texture_experiment"

parser = argparse.ArgumentParser()
parser.add_argument("--split", choices=["train", "eval"], default="train")
parser.add_argument("--episode", type=int, required=True)
parser.add_argument("--seed", type=int, default=None)
args = parser.parse_args()

seed = args.seed if args.seed is not None else secrets.randbits(64)
rng = random.Random(seed)

x = rng.uniform(-0.06, 0.06)
y = rng.uniform(-0.17, -0.04)
z = 1.06

request = (
    f'name: "experiment_cube" '
    f'position: {{x: {x}, y: {y}, z: {z}}} '
    'orientation: {x: 0, y: 0, z: 0, w: 1}'
)

cmd = [
    "ign", "service",
    "-s", "/world/default/set_pose",
    "--reqtype", "ignition.msgs.Pose",
    "--reptype", "ignition.msgs.Boolean",
    "--timeout", "5000",
    "--req", request,
]

result = subprocess.run(cmd, capture_output=True, text=True, timeout=12)

print(result.stdout)
if result.returncode != 0 or "data: true" not in result.stdout.lower():
    raise RuntimeError(
        "Gazebo position update failed:\n"
        + result.stdout + "\n" + result.stderr
    )

record_dir = ROOT / "placements" / args.split
record_dir.mkdir(parents=True, exist_ok=True)
record = record_dir / f"episode_{args.episode:06d}.json"

if record.exists():
    raise FileExistsError(f"Record already exists: {record}")

record.write_text(json.dumps({
    "split": args.split,
    "episode": args.episode,
    "seed": seed,
    "model": "experiment_cube",
    "commanded_position": {"x": x, "y": y, "z": z},
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "validation": "visibility and grasp reachability not yet checked"
}, indent=2))

print(f"Position: x={x:.4f}, y={y:.4f}, z={z:.4f}")
print(f"Seed: {seed}")
print(f"Saved: {record}")
