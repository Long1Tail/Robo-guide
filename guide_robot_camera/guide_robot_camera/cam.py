"""Snapshot-by-trigger node for the Guide robot camera.

Subscribes to the raw camera stream, caches the latest frame, and
re-publishes it on a separate topic only when a Bool trigger arrives.
This lets downstream consumers (VLM, telemetry) grab a single frame
on demand without processing the full video stream.
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, QoSReliabilityPolicy, QoSHistoryPolicy
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


# Best-effort QoS matching most camera drivers (usb_cam, gazebo_ros_camera).
_SENSOR_QOS = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=1,
)


class SnapshotTriggerNode(Node):
    """Cache the latest camera frame and publish it on trigger."""

    def __init__(self):
        super().__init__('snapshot_trigger_node')

        # -- Parameters --
        self.declare_parameter('trigger_topic', 'camera/trigger')
        self.declare_parameter('image_in_topic', 'camera/image_raw')
        self.declare_parameter('image_out_topic', 'camera/image')
        self.declare_parameter('max_age_sec', 5.0)

        trigger_topic = str(self.get_parameter('trigger_topic').value)
        image_in_topic = str(self.get_parameter('image_in_topic').value)
        image_out_topic = str(self.get_parameter('image_out_topic').value)
        self._max_age_sec: float = float(
            self.get_parameter('max_age_sec').value,
        )

        # -- Subscriptions --
        self.trigger_sub = self.create_subscription(
            Bool, trigger_topic, self._trigger_callback, 10)

        self.image_sub = self.create_subscription(
            Image, image_in_topic, self._image_callback, _SENSOR_QOS)

        # -- Publisher --
        self.image_pub = self.create_publisher(Image, image_out_topic, 10)

        self.bridge = CvBridge()
        self._latest_frame: Image | None = None
        self._frame_stamp_sec: float = 0.0

        self.get_logger().info(
            f'Snapshot node started. '
            f'Listening: {image_in_topic}, trigger: {trigger_topic}, '
            f'output: {image_out_topic}, max_age: {self._max_age_sec}s'
        )

    def _image_callback(self, msg: Image) -> None:
        """Cache the most recent frame and its timestamp."""
        self._latest_frame = msg
        self._frame_stamp_sec = self.get_clock().now().nanoseconds * 1e-9

    def _trigger_callback(self, msg: Bool) -> None:
        """On trigger, publish the cached frame if it is fresh enough."""
        if not msg.data:
            return

        if self._latest_frame is None:
            self.get_logger().warn('Trigger received but no frame cached yet.')
            return

        age = self.get_clock().now().nanoseconds * 1e-9 - self._frame_stamp_sec
        if age > self._max_age_sec:
            self.get_logger().warn(
                f'Frame is stale ({age:.1f}s old, limit {self._max_age_sec}s). '
                'Skipping publish.'
            )
            return

        self.image_pub.publish(self._latest_frame)
        self.get_logger().info(f'Snapshot published (age {age:.2f}s).')


def main(args=None):
    rclpy.init(args=args)
    node = SnapshotTriggerNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()