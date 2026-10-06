import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        Node(
            package='nmpc_sim_nodes',
            executable='current_awareness__advantage',
            name='current_awareness__advantage',
            output='screen',
        )
    ])
