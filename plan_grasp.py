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
