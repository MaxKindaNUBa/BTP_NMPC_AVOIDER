from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    num_runs_arg = DeclareLaunchArgument(
        'num_runs', default_value='1',
        description='Number of repetition rounds (default: 1)')

    return LaunchDescription([
        num_runs_arg,
        Node(
            package='nmpc_sim_nodes',
            executable='nmpc_ablation_runs',
            name='nmpc_ablation_runs',
            output='screen',
            arguments=['--num-runs', LaunchConfiguration('num_runs')],
        )
    ])
