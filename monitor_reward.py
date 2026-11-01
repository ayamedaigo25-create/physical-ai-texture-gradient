import json
import math
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rosgraph_msgs.msg import Clock
from sensor_msgs.msg import JointState

ROOT = Path.home() / "pai_ws/texture_experiment"
LIFT_HEIGHT = 0.04
HOLD_SECONDS = 0.5
MAX_SECONDS = 30.0
LAMBDA = 0.05

# テーブル上に静止した5cmキューブの中心高さ
BASELINE_Z = 1.040

def cube_height():
    proc = subprocess.run(
        ["ign", "model", "-m", "experiment_cube", "-p"],
        capture_output=True, text=True, timeout=5
    )
    if proc.returncode != 0:
        raise RuntimeError(proc.stderr)

    lines = proc.stdout.splitlines()
    for i, line in enumerate(lines):
        if "Pose [ XYZ" in line:
            for value_line in lines[i+1:i+4]:
                match = re.search(
                    r"\[\s*([-+0-9.eE]+)\s+"
                    r"([-+0-9.eE]+)\s+"
                    r"([-+0-9.eE]+)\s*\]",
                    value_line
                )
                if match:
                    return float(match.group(3))

    raise RuntimeError("Cube Z could not be parsed")

class Monitor(Node):
    def __init__(self):
        super().__init__("experiment_reward_monitor")
        self.sim_time = None
        self.gripper_angle = None

        self.create_subscription(
            Clock, "/clock", self.on_clock,
            qos_profile_sensor_data
        )
        self.create_subscription(
            JointState, "/joint_states", self.on_joints,
            qos_profile_sensor_data
        )

    def on_clock(self, msg):
        self.sim_time = (
            msg.clock.sec + msg.clock.nanosec * 1e-9
        )

    def on_joints(self, msg):
        if "crane_plus_joint_hand" in msg.name:
            i = msg.name.index("crane_plus_joint_hand")
            self.gripper_angle = msg.position[i]

rclpy.init()
node = Monitor()

print("Waiting for Gazebo clock and joint states...")

for _ in range(100):
    rclpy.spin_once(node, timeout_sec=0.1)
    if node.sim_time is not None and node.gripper_angle is not None:
        break

if node.sim_time is None or node.gripper_angle is None:
    raise RuntimeError("Clock or joint states unavailable")

initial_z = cube_height()

# 実験開始時はキューブがテーブル上にあること
if abs(initial_z - BASELINE_Z) > 0.015:
    raise RuntimeError(
        f"Cube is not at initial height: z={initial_z:.3f}"
    )

start_time = node.sim_time
hold_start = None
success = False
success_time = None

print(f"START sim_time={start_time:.3f}")
print(f"Initial cube Z={initial_z:.3f}")
print("Monitoring lift and grip...")

while rclpy.ok():
    rclpy.spin_once(node, timeout_sec=0.2)

    elapsed = node.sim_time - start_time
    if elapsed < 0:
        raise RuntimeError("Simulation clock was reset")

    if elapsed >= MAX_SECONDS:
        break

    z = cube_height()
    lifted = z >= initial_z + LIFT_HEIGHT
    gripped = node.gripper_angle >= 0.10

    if lifted and gripped:
        if hold_start is None:
            hold_start = node.sim_time
        if node.sim_time - hold_start >= HOLD_SECONDS:
            success = True
            success_time = elapsed
            break
    else:
        hold_start = None

    print(
        f"T={elapsed:5.2f}s "
        f"Z={z:.3f}m "
        f"gripper={node.gripper_angle:.2f}rad",
        end="\r", flush=True
    )

reward = math.exp(-LAMBDA * success_time) if success else 0.0

result = {
    "success": success,
    "time_seconds": success_time if success else MAX_SECONDS,
    "reward": reward,
    "lambda": LAMBDA,
    "max_seconds": MAX_SECONDS,
    "lift_height_m": LIFT_HEIGHT,
    "hold_seconds": HOLD_SECONDS,
    "initial_cube_z": initial_z,
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "status": "preliminary_lift_proxy_not_verified_grasp"
}

output_dir = ROOT / "results"
output_dir.mkdir(parents=True, exist_ok=True)
output = output_dir / (
    "reward_test_" +
    datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S_%f") +
    ".json"
)
output.write_text(json.dumps(result, indent=2))

print("\n\n=== RESULT ===")
print("Success:", success)
print("Time:", success_time)
print("Reward:", round(reward, 5))
print("Saved:", output)

node.destroy_node()
rclpy.shutdown()
