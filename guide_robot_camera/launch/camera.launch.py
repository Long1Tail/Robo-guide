"""Launch the USB camera driver (usb_cam) for the Guide robot."""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    # -- Arguments ----------------------------------------------------------
    device_arg = DeclareLaunchArgument(
        'video_device', default_value='/dev/video0',
        description='V4L2 video device path',
    )
    width_arg = DeclareLaunchArgument(
        'image_width', default_value='640',
        description='Capture width in pixels',
    )
    height_arg = DeclareLaunchArgument(
        'image_height', default_value='480',
        description='Capture height in pixels',
    )
    fps_arg = DeclareLaunchArgument(
        'framerate', default_value='10.0',
        description='Capture framerate',
    )
    sim_time_arg = DeclareLaunchArgument(
        'use_sim_time', default_value='false',
        description='Use simulation (Gazebo) clock',
    )

    # -- usb_cam node -------------------------------------------------------
    usb_cam_node = Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam',
        parameters=[{
            'video_device': LaunchConfiguration('video_device'),
            'image_width': LaunchConfiguration('image_width'),
            'image_height': LaunchConfiguration('image_height'),
            'framerate': LaunchConfiguration('framerate'),
            'pixel_format': 'yuyv2rgb',
            'camera_name': 'logitech_c270',
            'camera_frame_id': 'camera_optical_frame',
            'use_sim_time': LaunchConfiguration('use_sim_time'),
        }],
        remappings=[
            ('image_raw', '/camera/image_raw'),
        ],
    )

    return LaunchDescription([
        device_arg,
        width_arg,
        height_arg,
        fps_arg,
        sim_time_arg,
        usb_cam_node,
    ])