from setuptools import find_packages, setup

package_name = 'control'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        ('share/' + package_name + '/launch', ['launch/simulation.launch.py']),
        ('share/' + package_name + '/simulator/worlds', ['simulator/worlds/surveillance_building.world']),
        ('share/' + package_name + '/simulator/models', ['simulator/models/model.sdf']),
        ('share/' + package_name + '/simulator/models', ['simulator/models/model.config']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Eetu Silvennoinen',
    maintainer_email='eetu.silvennoinen@gmail.com',
    description='ROS package to hold the control node and related files.',
    license='MIT',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
        ],
    },
)
