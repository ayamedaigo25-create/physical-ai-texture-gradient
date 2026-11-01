import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK

rclpy.init()
node = Node("test_ik_initial_pose")

client = node.create_client(GetPositionIK, "/compute_ik")
assert client.wait_for_service(timeout_sec=10.0), "IK service unavailable"

req = GetPositionIK.Request()
ik = req.ik_request

ik.group_name = "arm"
ik.ik_link_name = "crane_plus_link_endtip"
ik.pose_stamped.header.frame_id = "base_link"

# 初期観測姿勢で取得した手先のTF
p = ik.pose_stamped.pose
p.position.x = -0.006
p.position.y = 0.0
p.position.z = 0.166

p.orientation.x = 0.0
p.orientation.y = 0.902
p.orientation.z = 0.0
p.orientation.w = 0.431

ik.robot_state.joint_state.name = [
    "crane_plus_joint1",
    "crane_plus_joint2",
    "crane_plus_joint3",
    "crane_plus_joint4",
]
ik.robot_state.joint_state.position = [
    0.0, -1.16, -2.01, -1.40
]

ik.avoid_collisions = False
ik.timeout.sec = 3

future = client.call_async(req)
rclpy.spin_until_future_complete(node, future, timeout_sec=6)

if future.done() and future.result():
    result = future.result()
    print("IK error code:", result.error_code.val)

    if result.error_code.val == 1:
        for name, position in zip(
            result.solution.joint_state.name,
            result.solution.joint_state.position
        ):
            if name.startswith("crane_plus_joint"):
                print(name, round(position, 4))
else:
    print("IK request timed out")

node.destroy_node()
rclpy.shutdown()
