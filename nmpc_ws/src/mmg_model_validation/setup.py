from setuptools import find_packages, setup

package_name = 'mmg_model_validation'

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
    description='Standalone NumPy-vs-CasADi MMG dynamics cross-validation tool',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'validate_casadi = mmg_model_validation.validate_casadi:main',
        ],
    },
)
