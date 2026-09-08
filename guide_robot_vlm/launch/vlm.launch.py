"""Запуск ноды VLM-оркестратора `vlm_agent_node`."""

from __future__ import annotations

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterFile
from launch_ros.substitutions import FindPackageShare


def generate_launch_description() -> LaunchDescription:
    """Собрать описание запуска."""
    params = LaunchConfiguration("params_file")
    autostart = LaunchConfiguration("autostart")
    log_level = LaunchConfiguration("log_level")

    arguments = [
        DeclareLaunchArgument(
            "params_file",
            default_value=PathJoinSubstitution(
                [FindPackageShare("guide_robot_vlm"), "config", "vlm.yaml"]
            ),
        ),
        DeclareLaunchArgument("autostart", default_value="false"),
        DeclareLaunchArgument("log_level", default_value="info"),
    ]

    vlm_node = Node(
        package="guide_robot_vlm",
        executable="vlm_agent_node",
        name="vlm_agent_node",
        output="screen",
        parameters=[ParameterFile(params, allow_substs=True)],
        arguments=["--ros-args", "--log-level", log_level],
    )

    manager = Node(
        package="nav2_lifecycle_manager",
        executable="lifecycle_manager",
        name="lifecycle_manager_vlm",
        output="screen",
        condition=IfCondition(autostart),
        parameters=[
            {
                "autostart": True,
                "node_names": ["vlm_agent_node"],
                "bond_timeout": 0.0,
            }
        ],
    )

    return LaunchDescription([*arguments, vlm_node, manager])
