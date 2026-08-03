import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('nmpc_sim_nodes')
    default_params = os.path.join(pkg_share, 'params', 'sim_params.yaml')

    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Path to the shared map/nmpc/mmg parameters YAML (params/sim_params.yaml)')
    params_file = LaunchConfiguration('params_file')

    return LaunchDescription([
        params_file_arg,
        Node(
            package='nmpc_sim_nodes', executable='map_node', name='map_node',
            output='screen', parameters=[params_file],
        ),
        Node(
            package='nmpc_sim_nodes', executable='nmpc_node', name='nmpc_node',
            output='screen', parameters=[params_file],
        ),
        Node(
            package='nmpc_sim_nodes', executable='mmg_node', name='mmg_node',
            output='screen', parameters=[params_file],
        ),
        Node(
            package='nmpc_sim_nodes', executable='sensor_node', name='sensor_node',
            output='screen', parameters=[params_file],
        ),
    ])
