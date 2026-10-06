import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('nmpc_sim_nodes')
    default_params = os.path.join(pkg_share, 'params', 'sim_params.yaml')
    default_scenario = os.path.join(pkg_share, 'params', 'scenario.json')

    params_file_arg = DeclareLaunchArgument(
        'params_file', default_value=default_params,
        description='Path to the shared map/nmpc/mmg parameters YAML (params/sim_params.yaml)')
    scenario_file_arg = DeclareLaunchArgument(
        'scenario_file', default_value=default_scenario,
        description='Path to scenario.json (defaults to installed share scenario.json)')

    params_file = LaunchConfiguration('params_file')
    scenario_file = LaunchConfiguration('scenario_file')

    return LaunchDescription([
        params_file_arg,
        scenario_file_arg,
        Node(
            package='nmpc_sim_nodes', executable='map_node', name='map_node',
            output='screen', parameters=[params_file, {'scenario_path': scenario_file}],
        ),
        Node(
            package='nmpc_sim_nodes', executable='nmpc_node', name='nmpc_node',
            output='screen', parameters=[params_file],
        ),
        Node(
            package='nmpc_sim_nodes', executable='ukf_node', name='ukf_node',
            output='screen', parameters=[params_file],
        ),
        Node(
            package='nmpc_sim_nodes', executable='mmg_node', name='mmg_node',
            output='screen', parameters=[params_file],
        ),
        Node(
            package='nmpc_sim_nodes', executable='logger_node', name='logger_node',
            output='screen', parameters=[params_file, {'scenario_path': scenario_file}],
        ),
        # Safe to always launch: idles (never ticks) if the loaded scenario has
        # no enabled obstacle ship. Its keyboard controller (obstacle_ship_teleop_node)
        # is deliberately NOT launched here -- it needs an interactive TTY and is
        # started/stopped manually (`ros2 run nmpc_sim_nodes obstacle_ship_teleop_node`).
        Node(
            package='nmpc_sim_nodes', executable='obstacle_ship_node', name='obstacle_ship_node',
            output='screen', parameters=[params_file],
        ),
    ])
