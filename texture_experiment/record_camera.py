from pathlib import Path
import subprocess
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

OUT = Path.home() / "pai_ws/texture_experiment/results/dense_final/inference"
OUT.mkdir(parents=True, exist_ok=True)

class CameraRecorder(Node):
    def __init__(self):
        super().__init__("eye_camera_recorder")
        self.proc = None
        self.count = 0

        self.create_subscription(
            Image,
            "/eye_in_hand/image_raw",
            self.callback,
            qos_profile_sensor_data
        )

    def callback(self, msg):
        if msg.encoding not in ("rgb8", "bgr8"):
            self.get_logger().warning(
                f"Unsupported encoding: {msg.encoding}"
            )
            return

        if self.proc is None:
            self.proc = subprocess.Popen([
                "ffmpeg", "-y",
                "-loglevel", "error",
                "-f", "rawvideo",
                "-pixel_format", "rgb24",
                "-video_size", f"{msg.width}x{msg.height}",
                "-framerate", "15",
                "-i", "pipe:0",
                "-c:v", "libx264",
                "-preset", "ultrafast",
                "-crf", "18",
                "-pix_fmt", "yuv420p",
                str(OUT / "eye_in_hand.mp4")
            ], stdin=subprocess.PIPE)

            print("=== CAMERA RECORDING START ===", flush=True)

        frame = np.frombuffer(
            msg.data, dtype=np.uint8
        ).reshape(msg.height, msg.step)

        frame = frame[:, :msg.width * 3]
        frame = frame.reshape(msg.height, msg.width, 3)

        if msg.encoding == "bgr8":
            frame = frame[:, :, ::-1]

        try:
            self.proc.stdin.write(
                np.ascontiguousarray(frame).tobytes()
            )
            self.count += 1
        except BrokenPipeError:
            raise RuntimeError("FFmpeg recording failed")

    def finish(self):
        if self.proc:
            self.proc.stdin.close()
            self.proc.wait()
        print("Recorded frames:", self.count)
        print("Camera video:", OUT / "eye_in_hand.mp4")

rclpy.init()
node = CameraRecorder()

try:
    print("Waiting for Eye-in-Hand images...", flush=True)
    rclpy.spin(node)
except KeyboardInterrupt:
    pass
finally:
    node.finish()
    node.destroy_node()
    rclpy.shutdown()
