import json
from pathlib import Path

import rclpy
from rclpy.node import Node
from moveit_msgs.srv import GetMotionPlan
from moveit_msgs.msg import Constraints, JointConstraint

root = Path.home() / "pai_ws/texture_experiment"
candidate = json.loads(
    (root / "central_grasp_candidate.json").read_text()
)

names = candidate["joint_names"]
initial = [0.0, -1.16, -2.01, -1.40]
target = candidate["joint_positions_rad"]

rclpy.init()
node = Node("plan_experiment_grasp")
client = node.create_client(
    GetMotionPlan, "/plan_kinematic_path"
)

if not client.wait_for_service(timeout_sec=10):
    raise RuntimeError("Planning service unavailable")

req = GetMotionPlan.Request()
plan = req.motion_plan_request

plan.group_name = "arm"
plan.num_planning_attempts = 5
plan.allowed_planning_time = 10.0
plan.max_velocity_scaling_factor = 0.2
plan.max_acceleration_scaling_factor = 0.2

plan.start_state.joint_state.name = (
    names + ["crane_plus_joint_hand"]
)
plan.start_state.joint_state.position = (
    initial + [-0.60]
)

goal = Constraints()

for name, angle in zip(names, target):
    constraint = JointConstraint()
    constraint.joint_name = name
    constraint.position = float(angle)
    constraint.tolerance_above = 0.01
    constraint.tolerance_below = 0.01
    constraint.weight = 1.0
    goal.joint_constraints.append(constraint)

plan.goal_constraints.append(goal)

future = client.call_async(req)
rclpy.spin_until_future_complete(
    node, future, timeout_sec=20
)

if not future.done() or future.result() is None:
    raise RuntimeError("Planning timed out")

response = future.result().motion_plan_response

print("Planning error code:", response.error_code.val)
print("Planning time:", response.planning_time)

if response.error_code.val == 1:
    trajectory = response.trajectory.joint_trajectory
    print("Trajectory points:", len(trajectory.points))
    print("Joint names:", list(trajectory.joint_names))
    print("Start:", list(trajectory.points[0].positions))
    print("Goal:", list(trajectory.points[-1].positions))
    print("PLAN SUCCESS - NOT EXECUTED")

    from rclpy.action import ActionClient
    from control_msgs.action import FollowJointTrajectory
    from moveit_msgs.action import ExecuteTrajectory
    from trajectory_msgs.msg import JointTrajectoryPoint
    from sensor_msgs.msg import JointState

    def execute_action(client, goal, label):
        assert client.wait_for_server(timeout_sec=10), label
        future = client.send_goal_async(goal)
        rclpy.spin_until_future_complete(node, future, timeout_sec=15)
        handle = future.result()
        if handle is None or not handle.accepted:
            raise RuntimeError(f"{label}: goal rejected")

        result_future = handle.get_result_async()
        rclpy.spin_until_future_complete(
            node, result_future, timeout_sec=120
        )
        if not result_future.done():
            raise RuntimeError(f"{label}: timeout")

        result = result_future.result()
        print(label, "status:", result.status)
        if result.status != 4:
            raise RuntimeError(f"{label}: execution failed")

        if hasattr(result.result, "error_code"):
            error = result.result.error_code
            value = error.val if hasattr(error, "val") else error
            expected = 1 if label == "ARM" else 0
            if value != expected:
                raise RuntimeError(f"{label}: error_code={value}")

    gripper = ActionClient(
        node, FollowJointTrajectory,
        "/crane_plus_gripper_controller/follow_joint_trajectory"
    )

    def set_gripper(angle):
        goal = FollowJointTrajectory.Goal()
        goal.trajectory.joint_names = ["crane_plus_joint_hand"]
        point = JointTrajectoryPoint()
        point.positions = [angle]
        point.time_from_start.sec = 2
        goal.trajectory.points = [point]
        execute_action(gripper, goal, "GRIPPER")

    # 現在姿勢が計画開始姿勢に近いことを確認
    states = []
    sub = node.create_subscription(
        JointState, "/joint_states",
        lambda msg: states.append(msg), 10
    )
    for _ in range(30):
        if states:
            break
        rclpy.spin_once(node, timeout_sec=0.2)
    if not states:
        raise RuntimeError("Joint states unavailable")

    actual = dict(zip(states[-1].name, states[-1].position))
    for name, expected in zip(names, initial):
        if abs(actual[name] - expected) > 0.05:
            raise RuntimeError(
                f"Return to initial observation pose first: {name}"
            )

    print("1/3 OPEN")
    set_gripper(-0.60)

    print("2/3 APPROACH")
    executor = ActionClient(
        node, ExecuteTrajectory, "/execute_trajectory"
    )
    goal = ExecuteTrajectory.Goal()
    goal.trajectory = response.trajectory
    execute_action(executor, goal, "ARM")

    print("3/3 CLOSE")
    set_gripper(0.30)

    print("GRASP TEST COMPLETE")


    from moveit_msgs.msg import DisplayTrajectory
    from rclpy.qos import QoSProfile, DurabilityPolicy
    import time

    qos = QoSProfile(depth=1)
    qos.durability = DurabilityPolicy.TRANSIENT_LOCAL

    publisher = node.create_publisher(
        DisplayTrajectory,
        "/display_planned_path",
        qos
    )

    display = DisplayTrajectory()
    display.model_id = "crane_plus"
    display.trajectory_start = response.trajectory_start
    display.trajectory.append(response.trajectory)

    time.sleep(1)
    publisher.publish(display)
    time.sleep(2)
    print("Trajectory published to RViz")
else:
    print("PLAN FAILED - NOT EXECUTED")

node.destroy_node()
rclpy.shutdown()
