import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description():
    pkg_share = get_package_share_directory('nmpc_sim_nodes')
    rviz_config = os.path.join(pkg_share, 'rviz', 'sim_view.rviz')

    # VS Code's snap sets GTK_PATH to its own bundled GTK module dir, which breaks
    # rviz2's library loading (see README's "unset GTK_PATH" note) -- clearing it
    # here means this launch file works regardless of which terminal it's run from.
    os.environ.pop('GTK_PATH', None)

    return LaunchDescription([
        Node(
            package='nmpc_sim_nodes', executable='rviz_node', name='rviz_node',
            output='screen',
        ),
        Node(
            package='rviz2', executable='rviz2', name='rviz2',
            output='screen', arguments=['-d', rviz_config],
        ),
        Node(
            package='nmpc_sim_nodes', executable='hud_node', name='hud_node',
            output='screen',
        ),
    ])
