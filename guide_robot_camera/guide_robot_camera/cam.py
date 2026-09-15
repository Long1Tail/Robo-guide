import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

class SnapshotTriggerNode(Node):
    def __init__(self):
        super().__init__('snapshot_trigger_node')
        self.trigger_sub = self.create_subscription(
            Bool, 'camera/trigger', self.trigger_callback, 10)
        
        # 2. Подписка на поток с камеры (usb_cam публикует сюда)
        self.image_sub = self.create_subscription(
            Image, 'camera/image_raw', self.image_callback, 10)
        
        # 3. Публикация снимка (Канал Б)
        self.image_pub = self.create_publisher(
            Image, 'camera/image', 10)
        
        self.bridge = CvBridge()
        self.latest_frame = None
        
        self.get_logger().info('Snapshot node started. Waiting...')

    def image_callback(self, msg):
        self.latest_frame = msg

    def trigger_callback(self, msg):
        if msg.data and self.latest_frame is not None:
            self.image_pub.publish(self.latest_frame)
            self.get_logger().info('Snapshot published!')
        elif self.latest_frame is None:
            self.get_logger().warn('No frame yet!')

def main(args=None):
    rclpy.init(args=args)
    node = SnapshotTriggerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()