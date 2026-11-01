import copy
import re
import subprocess

import rclpy
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.duration import Duration
from rclpy.time import Time

from sensor_msgs.msg import JointState
from moveit_msgs.msg import RobotState
from moveit_msgs.srv import GetCartesianPath
from moveit_msgs.action import ExecuteTrajectory
from tf2_ros import Buffer, TransformListener

LIFT = 0.05

def cube_z():
    out = subprocess.check_output(
        ["ign", "model", "-m", "experiment_cube", "-p"],
        text=True, timeout=8
    )
    m = re.search(
        r"Pose \[ XYZ \(m\) \] \[ RPY \(rad\) \]:\s*"
        r"\[\s*([-+\d.eE]+)\s+([-+\d.eE]+)\s+([-+\d.eE]+)",
        out
    )
    if not m:
        raise RuntimeError("Cannot read cube position")
    return float(m.group(3))

rclpy.init()
node = Node("lift_test")
tf_buffer = Buffer()
tf_listener = TransformListener(tf_buffer, node)

states = []
sub = node.create_subscription(
    JointState, "/joint_states",
    lambda msg: states.append(msg), 10
)

for _ in range(100):
    rclpy.spin_once(node, timeout_sec=0.1)
    if states and tf_buffer.can_transform(
        "base_link", "crane_plus_link_endtip", Time()
    ):
        break

if not states:
    raise RuntimeError("Joint states unavailable")

if not tf_buffer.can_transform(
    "base_link", "crane_plus_link_endtip", Time()
):
    raise RuntimeError("End-effector TF unavailable")

state = states[-1]
actual = dict(zip(state.name, state.position))

if actual.get("crane_plus_joint_hand", -1) < 0.0:
    raise RuntimeError("Gripper is not closed")

tf = tf_buffer.lookup_transform(
    "base_link", "crane_plus_link_endtip", Time()
)

from geometry_msgs.msg import Pose

target = Pose()
target.position.x = tf.transform.translation.x
target.position.y = tf.transform.translation.y
target.position.z = tf.transform.translation.z + LIFT
target.orientation = copy.deepcopy(tf.transform.rotation)

print("Current endtip Z:", round(tf.transform.translation.z, 4))
print("Target endtip Z: ", round(target.position.z, 4))

z_before = cube_z()
print("Cube Z before:", z_before)

client = node.create_client(
    GetCartesianPath, "/compute_cartesian_path"
)

if not client.wait_for_service(timeout_sec=10):
    raise RuntimeError("Cartesian path service unavailable")

request = GetCartesianPath.Request()
request.header.frame_id = "base_link"
request.group_name = "arm"
request.link_name = "crane_plus_link_endtip"

request.start_state = RobotState()
request.start_state.joint_state = state

request.waypoints = [target]
request.max_step = 0.005
request.jump_threshold = 0.0
request.avoid_collisions = True

future = client.call_async(request)
rclpy.spin_until_future_complete(node, future, timeout_sec=30)

if not future.done():
    raise RuntimeError("Cartesian path planning timed out")

response = future.result()
print("Planning error:", response.error_code.val)
print("Cartesian fraction:", response.fraction)

if response.error_code.val != 1 or response.fraction < 0.999:
    raise RuntimeError("Full 5cm lift path unavailable; not executing")

trajectory = response.solution.joint_trajectory
if not trajectory.points:
    raise RuntimeError("Empty trajectory")

print("Trajectory points:", len(trajectory.points))

executor = ActionClient(
    node, ExecuteTrajectory, "/execute_trajectory"
)
if not executor.wait_for_server(timeout_sec=10):
    raise RuntimeError("Execution action unavailable")

goal = ExecuteTrajectory.Goal()
goal.trajectory = response.solution

send = executor.send_goal_async(goal)
rclpy.spin_until_future_complete(node, send, timeout_sec=15)

handle = send.result()
if handle is None or not handle.accepted:
    raise RuntimeError("Lift goal rejected")

result_future = handle.get_result_async()
rclpy.spin_until_future_complete(
    node, result_future, timeout_sec=120
)

if not result_future.done():
    raise RuntimeError("Lift execution timeout")

result = result_future.result()
print("Execution status:", result.status)
print("Execution error:", result.result.error_code.val)

z_after = cube_z()
dz = z_after - z_before

print("\n=== LIFT TEST ===")
print("Cube Z before:", round(z_before, 5))
print("Cube Z after: ", round(z_after, 5))
print("Cube delta Z: ", round(dz, 5), "m")
print("Lift detected:", dz >= 0.04)

node.destroy_node()
rclpy.shutdown()
