from setuptools import find_packages, setup

package_name = 'scenario_maker'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='intern',
    maintainer_email='intern@umagine.co.in',
    description='Standalone GUI tool for authoring NMPC scenario JSON files',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'scenario_editor = scenario_maker.scenario_editor:main',
        ],
    },
)
