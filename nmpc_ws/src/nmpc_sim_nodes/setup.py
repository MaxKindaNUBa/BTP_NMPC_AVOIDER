import os
from glob import glob

from setuptools import find_packages, setup

package_name = 'nmpc_sim_nodes'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'params'), glob('params/*.yaml') + glob('params/*.json')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='intern',
    maintainer_email='intern@umagine.co.in',
    description='ROS2 nodes for the NMPC ship obstacle-avoidance simulator',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'map_node = nmpc_sim_nodes.nodes.map_node:main',
            'nmpc_node = nmpc_sim_nodes.nodes.nmpc_node:main',
            'mmg_node = nmpc_sim_nodes.nodes.mmg_node:main',
            'ukf_node = nmpc_sim_nodes.nodes.ukf_node:main',
            'viz_node = nmpc_sim_nodes.nodes.viz_node:main',
            'rviz_node = nmpc_sim_nodes.nodes.rviz_node:main',
            'hud_node = nmpc_sim_nodes.nodes.hud_node:main',
            'logger_node = nmpc_sim_nodes.nodes.logger_node:main',
            'run_demo = nmpc_sim_nodes.run_demo:main',
            # command names unchanged -- only the module path moved, into tests/
            'test_nmpc = nmpc_sim_nodes.tests.test_nmpc:main',
            'test_sensor_model = nmpc_sim_nodes.sensor_model.test_sensor_model:main',
            'test_closed_loop_noise = nmpc_sim_nodes.tests.test_closed_loop_noise:main',
            'test_env_model = nmpc_sim_nodes.env_model.test_env_model:main',
            'test_closed_loop_env = nmpc_sim_nodes.tests.test_closed_loop_env:main',
            'test_ukf = nmpc_sim_nodes.tests.test_ukf:main',
            'tune_ukf = nmpc_sim_nodes.tests.tune_ukf:main',
        ],
    },
)
