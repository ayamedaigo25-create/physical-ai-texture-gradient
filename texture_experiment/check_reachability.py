import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetPositionIK

POSITIONS = [
    ("center", 0.300,  0.000),
    ("near",   0.170,  0.000),
    ("far",    0.530,  0.000),
    ("left",   0.320,  0.150),
    ("right",  0.320, -0.170),
]

JOINTS = [
    "crane_plus_joint1",
    "crane_plus_joint2",
    "crane_plus_joint3",
    "crane_plus_joint4",
    "crane_plus_joint_hand",
]

INITIAL = [0.0, -1.16, -2.01, -1.40, 0.0]

rclpy.init()
node = Node("experiment_ik_checker")
client = node.create_client(GetPositionIK, "/compute_ik")

if not client.wait_for_service(timeout_sec=10.0):
    raise RuntimeError("/compute_ik is unavailable")

for name, x, y in POSITIONS:
    print(f"\n=== {name}: ({x:.3f}, {y:.3f}) ===")

    for height in (0.12, 0.17):
        req = GetPositionIK.Request()
        ik = req.ik_request

        ik.group_name = "arm"
        ik.ik_link_name = "crane_plus_link_endtip"
        ik.robot_state.joint_state.name = JOINTS
        ik.robot_state.joint_state.position = INITIAL

        ik.pose_stamped.header.frame_id = "base_link"
        pose = ik.pose_stamped.pose

        pose.position.x = x
        pose.position.y = y
        pose.position.z = height

        # 初期観測姿勢で取得した手先方向を暫定使用
        pose.orientation.x = 0.0
        pose.orientation.y = 0.902
        pose.orientation.z = 0.0
        pose.orientation.w = 0.431

        ik.timeout.sec = 1
        ik.avoid_collisions = False

        future = client.call_async(req)
        rclpy.spin_until_future_complete(
            node, future, timeout_sec=4.0
        )

        if not future.done() or future.result() is None:
            print(f"  z={height:.2f}: TIMEOUT")
            continue

        result = future.result()
        code = result.error_code.val

        if code == 1:
            state = result.solution.joint_state
            angles = dict(zip(state.name, state.position))
            values = [
                round(angles.get(j, float("nan")), 3)
                for j in JOINTS[:4]
            ]
            print(f"  z={height:.2f}: IK SUCCESS {values}")
        else:
            print(f"  z={height:.2f}: IK FAILED (code={code})")

node.destroy_node()
rclpy.shutdown()
