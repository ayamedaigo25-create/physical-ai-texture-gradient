import math
import re
import subprocess
import numpy as np
import rclpy

from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image, JointState
from rosgraph_msgs.msg import Clock
from control_msgs.action import FollowJointTrajectory
from trajectory_msgs.msg import JointTrajectoryPoint
from tf2_ros import Buffer, TransformListener
from rclpy.time import Time

ARM = [
    "crane_plus_joint1",
    "crane_plus_joint2",
    "crane_plus_joint3",
    "crane_plus_joint4",
]
HOME = np.array([0., -1.16, -2.01, -1.40])

class TextureEnv(Node):
    def __init__(self):
        super().__init__("texture_rl_env")
        self.image = None
        self.joints = None
        self.clock = None
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(
            self.tf_buffer, self)


        self.create_subscription(
            Image, "/eye_in_hand/image_raw",
            self.on_image, qos_profile_sensor_data)
        self.create_subscription(
            JointState, "/joint_states",
            self.on_joints, qos_profile_sensor_data)
        self.create_subscription(
            Clock, "/clock",
            self.on_clock, qos_profile_sensor_data)

        self.arm = ActionClient(
            self, FollowJointTrajectory,
            "/crane_plus_arm_controller/follow_joint_trajectory")
        self.hand = ActionClient(
            self, FollowJointTrajectory,
            "/crane_plus_gripper_controller/follow_joint_trajectory")

    def on_image(self, msg):
        a = np.frombuffer(msg.data, dtype=np.uint8)
        if msg.encoding.lower() not in ("rgb8", "bgr8"):
            return
        if msg.step < msg.width * 3:
            return
        self.image = a.reshape(msg.height, msg.step)[:, :msg.width*3].reshape(
            msg.height, msg.width, 3).copy()
        if msg.encoding.lower() == "bgr8":
            self.image = self.image[:, :, ::-1].copy()

    def on_joints(self, msg):
        self.joints = dict(zip(msg.name, msg.position))

    def on_clock(self, msg):
        self.clock = msg.clock.sec + msg.clock.nanosec * 1e-9

    def spin_until(self, condition, limit=15):
        import time
        deadline = time.monotonic() + limit
        while not condition():
            if time.monotonic() > deadline:
                raise RuntimeError("ROS data timeout")
            rclpy.spin_once(self, timeout_sec=0.1)

    def command(self, client, names, positions, seconds=0.6):
        if not client.wait_for_server(timeout_sec=5):
            raise RuntimeError("Controller unavailable")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = names
        p = JointTrajectoryPoint()
        p.positions = list(map(float, positions))
        p.time_from_start.nanosec = int(seconds * 1e9)
        goal.trajectory.points = [p]

        f = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(self, f, timeout_sec=10)
        if not f.done() or not f.result().accepted:
            raise RuntimeError("Controller rejected action")
        h = f.result()
        result = h.get_result_async()
        rclpy.spin_until_future_complete(self, result, timeout_sec=15)
        if not result.done() or result.result().status != 4:
            raise RuntimeError("Controller action failed")
        if result.result().result.error_code != 0:
            raise RuntimeError("Controller reported trajectory error")

    def cube_xyz(self):
        import time

        for attempt in range(3):
            try:
                proc = subprocess.run(
                    ["ign", "model",
                     "-m", "experiment_cube", "-p"],
                    capture_output=True,
                    text=True,
                    timeout=12
                )

                lines = proc.stdout.splitlines()

                for i, line in enumerate(lines):
                    if "Pose [ XYZ (m) ] [ RPY (rad) ]:" in line:
                        if i + 1 >= len(lines):
                            break

                        coords = lines[i + 1].strip()
                        coords = coords.strip("[]").split()

                        if len(coords) == 3:
                            return np.array(
                                [float(v) for v in coords],
                                dtype=float
                            )

            except (subprocess.TimeoutExpired, ValueError):
                pass

            time.sleep(0.5)

        raise RuntimeError(
            "Gazebo cube position unavailable after 3 attempts"
        )

    def distance_to_cube(self, cube_xyz):
        self.spin_until(
            lambda: self.tf_buffer.can_transform(
                "base_link",
                "crane_plus_link_endtip",
                Time()
            ),
            limit=10
        )

        tf = self.tf_buffer.lookup_transform(
            "base_link",
            "crane_plus_link_endtip",
            Time()
        )

        t = tf.transform.translation
        endtip = np.array([t.x, t.y, t.z])

        # Gazebo world -> robot base_link
        cube_base = np.asarray(cube_xyz) - np.array(
            [0.0, 0.0, 1.015]
        )

        return float(np.linalg.norm(endtip - cube_base))

    def observe(self):
        self.spin_until(
            lambda: self.image is not None
            and self.joints is not None
            and self.clock is not None)
        return {
            "image": self.image.copy(),
            "joints": np.array(
                [self.joints[n] for n in ARM], dtype=np.float32),
            "gripper": float(self.joints["crane_plus_joint_hand"]),
        }

    def reset(self, rng=None):
        if rng is None:
            rng = np.random.default_rng()
        self.spin_until(lambda: self.joints is not None)
        self.command(self.hand, ["crane_plus_joint_hand"], [-0.60], 1.0)
        self.last_gripper_target = -0.60
        self.command(self.arm, ARM, HOME, 3.0)

        # 中央周辺のランダム初期位置
        x = float(rng.uniform(0.18, 0.22))
        y = float(rng.uniform(-0.04, 0.02))

        req = (
            'name: "experiment_cube" '
            f'position: {{x: {x}, y: {y}, z: 1.090}} '
            'orientation: {w: 1.0}'
        )
        p = subprocess.run([
            "ign", "service", "-s", "/world/default/set_pose",
            "--reqtype", "ignition.msgs.Pose",
            "--reptype", "ignition.msgs.Boolean",
            "--timeout", "5000", "--req", req
        ], capture_output=True, text=True, timeout=8)
        if p.returncode != 0 or not re.search(
            r"\btrue\b", p.stdout, re.I
        ):
            raise RuntimeError("Cube reset failed: " + p.stdout + p.stderr)

        self.spin_until(lambda: self.clock is not None)
        t = self.clock
        self.spin_until(lambda: self.clock >= t + 1.0)
        xyz0 = self.cube_xyz()
        self.z0 = xyz0[2]
        self.prev_distance = self.distance_to_cube(xyz0)
        self.t0 = self.clock
        self.hold_since = None
        return self.observe()

    def step(self, action):
        # action = [delta_j1, delta_j2, delta_j3, delta_j4, gripper]
        action = np.asarray(action, dtype=float)
        if action.shape != (5,):
            raise ValueError("Action must contain 5 numbers")

        obs = self.observe()
        target = obs["joints"] + np.clip(action[:4], -0.05, 0.05)
        self.command(self.arm, ARM, target, 0.6)

        gripper_target = -0.60 if action[4] < 0 else 0.30
        if gripper_target != self.last_gripper_target:
            self.command(
                self.hand, ["crane_plus_joint_hand"],
                [gripper_target], 0.4)
            self.last_gripper_target = gripper_target

        obs = self.observe()
        elapsed = self.clock - self.t0

        xyz = self.cube_xyz()
        distance = self.distance_to_cube(xyz)

        # 学習用：接近した分を報酬にする
        approach_reward = 2.0 * (
            self.prev_distance - distance
        )
        self.prev_distance = distance

        lifted = xyz[2] - self.z0 >= 0.04
        closed = obs["gripper"] > 0.02

        if lifted and closed:
            if self.hold_since is None:
                self.hold_since = self.clock
        else:
            self.hold_since = None

        success = (
            self.hold_since is not None
            and self.clock - self.hold_since >= 0.5
        )
        timeout = elapsed >= 30.0

        success_reward = (
            math.exp(-0.05 * elapsed) if success else 0.0
        )

        reward = approach_reward + success_reward

        info = {
            "cube_xyz": xyz.tolist() if xyz is not None else None,
            "elapsed": round(elapsed, 3),
            "success": success,
            "distance_m": round(distance, 4),
            "approach_reward": round(approach_reward, 5),
            "success_reward": round(success_reward, 5),
        }
        return obs, reward, success, timeout, info


if __name__ == "__main__":
    rclpy.init()
    env = TextureEnv()
    try:
        print("Resetting environment...")
        obs = env.reset()
        print("Observation:", obs["image"].shape, obs["joints"])

        # 実際に1ステップ操作する
        obs, reward, done, truncated, info = env.step(
            [0, 0.04, 0, 0, -1]
        )
        print("After step:", obs["joints"])
        print("Reward:", reward)
        print("Done:", done)
        print("Info:", info)
        print("ENVIRONMENT STEP COMPLETE")
    finally:
        env.destroy_node()
        rclpy.shutdown()
