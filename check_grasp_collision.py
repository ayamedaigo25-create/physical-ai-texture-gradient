import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetStateValidity

root = Path.home() / "pai_ws/texture_experiment"
data = json.loads(
    (root / "central_grasp_candidate.json").read_text()
)

rclpy.init()
node = Node("check_grasp_collision")
client = node.create_client(
    GetStateValidity, "/check_state_validity"
)

if not client.wait_for_service(timeout_sec=10):
    raise RuntimeError("Collision service unavailable")

req = GetStateValidity.Request()
req.group_name = "arm"

req.robot_state.joint_state.name = (
    data["joint_names"] + ["crane_plus_joint_hand"]
)
req.robot_state.joint_state.position = (
    data["joint_positions_rad"] + [-0.60]
)

future = client.call_async(req)
rclpy.spin_until_future_complete(
    node, future, timeout_sec=10
)

if not future.done() or future.result() is None:
    raise RuntimeError("Collision check timed out")

result = future.result()

print("State valid:", result.valid)

for contact in result.contacts:
    print(
        "Collision:",
        contact.contact_body_1,
        "<->",
        contact.contact_body_2
    )

node.destroy_node()
rclpy.shutdown()
