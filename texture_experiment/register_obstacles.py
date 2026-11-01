import json
import math
from pathlib import Path

import rclpy
from rclpy.node import Node
from moveit_msgs.msg import PlanningScene, CollisionObject
from moveit_msgs.srv import ApplyPlanningScene, GetStateValidity
from shape_msgs.msg import SolidPrimitive
from geometry_msgs.msg import Pose

root = Path.home() / "pai_ws/texture_experiment"
candidate = json.loads(
    (root / "central_grasp_candidate.json").read_text()
)

def make_box(name, xyz, size, yaw=0.0):
    obj = CollisionObject()
    obj.header.frame_id = "base_link"
    obj.id = name
    obj.operation = CollisionObject.ADD

    primitive = SolidPrimitive()
    primitive.type = SolidPrimitive.BOX
    primitive.dimensions = list(size)

    pose = Pose()
    pose.position.x, pose.position.y, pose.position.z = xyz
    pose.orientation.z = math.sin(yaw / 2)
    pose.orientation.w = math.cos(yaw / 2)

    obj.primitives.append(primitive)
    obj.primitive_poses.append(pose)
    return obj

rclpy.init()
node = Node("register_experiment_obstacles")

apply_client = node.create_client(
    ApplyPlanningScene, "/apply_planning_scene"
)
valid_client = node.create_client(
    GetStateValidity, "/check_state_validity"
)

for client in (apply_client, valid_client):
    if not client.wait_for_service(timeout_sec=10):
        raise RuntimeError("MoveIt service unavailable")

# Gazeboのモデル原点とROS base_linkの高さ差1.015mを使用
# テーブル：元モデルと同じ天板形状・配置
table = make_box(
    "experiment_table_top",
    (0.29, 0.0, -0.015),
    (1.5, 0.8, 0.03),
    yaw=1.5708
)

# キューブ：中央位置、テーブル上面から中心25mm
cube = make_box(
    "experiment_cube",
    (0.190, -0.010, 0.025),
    (0.05, 0.05, 0.05)
)

req = ApplyPlanningScene.Request()
req.scene = PlanningScene()
req.scene.is_diff = True
req.scene.robot_state.is_diff = True
req.scene.world.collision_objects = [table, cube]

future = apply_client.call_async(req)
rclpy.spin_until_future_complete(node, future, timeout_sec=10)

if not future.done() or future.result() is None:
    raise RuntimeError("Planning Scene update timed out")

print("Planning Scene applied:", future.result().success)
if not future.result().success:
    raise RuntimeError("Planning Scene update failed")

# 初期姿勢と把持候補姿勢を比較
states = [
    ("initial_observation", [0.0, -1.16, -2.01, -1.40]),
    ("grasp_candidate", candidate["joint_positions_rad"])
]

for name, angles in states:
    check = GetStateValidity.Request()
    check.group_name = "arm"
    check.robot_state.joint_state.name = (
        candidate["joint_names"] + ["crane_plus_joint_hand"]
    )
    check.robot_state.joint_state.position = list(angles) + [-0.60]

    future = valid_client.call_async(check)
    rclpy.spin_until_future_complete(
        node, future, timeout_sec=10
    )

    if not future.done() or future.result() is None:
        print(name, "TIMEOUT")
        continue

    result = future.result()
    print(f"\n{name}: valid={result.valid}")
    for contact in result.contacts[:15]:
        print(
            "  Collision:",
            contact.contact_body_1,
            "<->",
            contact.contact_body_2
        )

node.destroy_node()
rclpy.shutdown()
