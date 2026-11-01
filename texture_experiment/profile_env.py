import time
import numpy as np
import rclpy
from texture_env import TextureEnv

rclpy.init()
env = TextureEnv()
records = {}

def measure(name, func):
    def wrapped(*args, **kwargs):
        start = time.perf_counter()
        try:
            return func(*args, **kwargs)
        finally:
            records.setdefault(name, []).append(
                time.perf_counter() - start
            )
    return wrapped

original_command = env.command

def timed_command(client, names, positions, seconds=0.6):
    label = "gripper" if "crane_plus_joint_hand" in names else "arm"
    return measure(label, original_command)(
        client, names, positions, seconds
    )

env.command = timed_command
env.cube_xyz = measure("cube_xyz", env.cube_xyz)

try:
    start = time.perf_counter()
    env.reset()
    print("Reset:", round(time.perf_counter() - start, 3), "s")

    for i in range(3):
        start = time.perf_counter()
        env.step([0, 0.01, 0, 0, -1])
        print(
            f"Step {i+1}:",
            round(time.perf_counter() - start, 3), "s"
        )

    print("\n=== TIME BREAKDOWN ===")
    for name, values in records.items():
        print(
            name,
            "calls:", len(values),
            "total:", round(sum(values), 3), "s",
            "average:", round(np.mean(values), 3), "s"
        )
finally:
    env.destroy_node()
    rclpy.shutdown()
