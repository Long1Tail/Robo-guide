#!/usr/bin/env python3
"""Имитатор камеры: бесконечно публикует статичное фото с актуальным timestamp."""

import sys
from PIL import Image as PILImage
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from sensor_msgs.msg import Image

class FakeCameraPublisher(Node):
    def __init__(self, image_path: str, rate_hz: float = 5.0):
        super().__init__("fake_camera_publisher")
        self.publisher_ = self.create_publisher(Image, "/camera/color/image_raw", qos_profile_sensor_data)
        
        # Загружаем фото и переводим в RGB
        img = PILImage.open(image_path).convert("RGB")
        self.width, self.height = img.size
        self.data = img.tobytes()
        self.step = self.width * 3

        self.timer = self.create_timer(1.0 / rate_hz, self.publish_frame)
        self.get_logger().info(f"Публикация {image_path} ({self.width}x{self.height}) на {rate_hz} Гц")

    def publish_frame(self):
        msg = Image()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "camera_optical_link"
        msg.height = self.height
        msg.width = self.width
        msg.encoding = "rgb8"
        msg.is_bigendian = False
        msg.step = self.step
        msg.data = self.data
        self.publisher_.publish(msg)

def main():
    image_path = "guide_robot_vlm\test\test.jpeg"
    rclpy.init()
    node = FakeCameraPublisher(image_path)
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == "__main__":
    main()
