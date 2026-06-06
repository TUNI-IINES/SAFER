"""
simulation.launch.py
════════════════════
Launches the full Gazebo simulation stack:
  1. Gazebo Classic  +  surveillance_building.world
  2. Anafi 4K model   (spawned 3.5 m south of building, at window height)
  3. Static TF        world → odom

Usage
─────
  ros2 launch control simulation.launch.py

Optional arguments
──────────────────
  world:=<path>      Override world file path
  x:=0.0             Drone spawn X  (default: 0.0)
  y:=-3.5            Drone spawn Y  (default: -3.5, south of building)
  z:=1.6             Drone spawn Z  (default: 1.6, window mid-height)
  yaw:=1.5708        Drone spawn yaw (default: π/2, facing north)
  verbose:=false     Gazebo verbosity

Move drone during a running simulation
───────────────────────────────────────
  ros2 service call /gazebo/set_entity_state \\
    gazebo_msgs/srv/SetEntityState \\
    "{state: {name: anafi4k,
              pose: {position: {x: 0.0, y: -3.5, z: 1.6},
                     orientation: {w: 1.0}}}}"

View /anafi/image in rqt
──────────────────────────
  ros2 run rqt_image_view rqt_image_view /anafi/image

Echo odometry
──────────────
  ros2 topic echo /anafi/odometry
"""

import os

from launch import LaunchDescription
from launch.actions import (
    IncludeLaunchDescription,
    SetEnvironmentVariable,
    TimerAction,
    LogInfo,
    DeclareLaunchArgument,
)
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import (
    LaunchConfiguration,
    EnvironmentVariable,
    PathJoinSubstitution,
)
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory


def generate_launch_description():

    # ── Package paths ─────────────────────────────────────────────────
    pkg_this       = get_package_share_directory('control')
    pkg_gazebo_ros = get_package_share_directory('gazebo_ros')

    world_file_default = os.path.join(pkg_this, 'simulator', 'worlds', 'surveillance_building.world')
    models_dir         = os.path.join(pkg_this, 'simulator', 'models')

    # ── Declare launch arguments ──────────────────────────────────────
    args = [
        DeclareLaunchArgument('world',   default_value=world_file_default,
                              description='Path to .world file'),
        DeclareLaunchArgument('x',       default_value='0.0',
                              description='Drone spawn X position'),
        DeclareLaunchArgument('y',       default_value='-3.5',
                              description='Drone spawn Y position (south of building)'),
        DeclareLaunchArgument('z',       default_value='1.6',
                              description='Drone spawn Z position (window mid-height)'),
        DeclareLaunchArgument('yaw',     default_value='1.5708',
                              description='Drone spawn yaw in radians (π/2 = facing north)'),
        DeclareLaunchArgument('verbose', default_value='false',
                              description='Gazebo verbose output'),
    ]

    # ── 1. Prepend our models/ dir to Gazebo model search path ───────
    #    Gazebo searches GAZEBO_MODEL_PATH to resolve '-database anafi4k'.
    set_model_path = SetEnvironmentVariable(
        name='GAZEBO_MODEL_PATH',
        value=[
            models_dir,
            ':',
            EnvironmentVariable('GAZEBO_MODEL_PATH', default_value=''),
        ],
    )

    # ── 2. Launch Gazebo with the surveillance world ──────────────────
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_gazebo_ros, 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={
            'world':   LaunchConfiguration('world'),
            'verbose': LaunchConfiguration('verbose'),
            'pause':   'false',
        }.items(),
    )

    # ── 3. Spawn Anafi 4K ─────────────────────────────────────────────
    #    '-database anafi4k'  →  looks up anafi4k/ in GAZEBO_MODEL_PATH.
    #    '-Y'                 →  yaw angle (radians).
    #    Delay 3 s to let Gazebo fully initialise before spawning.
    spawn_drone = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        name='spawn_anafi4k',
        arguments=[
            '-entity',   'anafi4k',
            '-database', 'anafi4k',
            '-x', LaunchConfiguration('x'),
            '-y', LaunchConfiguration('y'),
            '-z', LaunchConfiguration('z'),
            '-Y', LaunchConfiguration('yaw'),
        ],
        output='screen',
    )

    delayed_spawn = TimerAction(period=3.0, actions=[spawn_drone])

    # ── 4. Static TF: world → odom ────────────────────────────────────
    #    The P3D odometry plugin publishes in 'world' frame.
    #    Publishing world→odom as identity keeps nav stack happy.
    static_tf = Node(
        package='tf2_ros',
        executable='static_transform_publisher',
        name='world_to_odom_tf',
        arguments=['0', '0', '0',   # x y z
                   '0', '0', '0',   # roll pitch yaw
                   'world', 'odom'],
        output='screen',
    )

    # ── 5. Log useful info ────────────────────────────────────────────
    info = [
        LogInfo(msg='─────────────────────────────────────────────'),
        LogInfo(msg='  World  : ' + world_file_default),
        LogInfo(msg='  Models : ' + models_dir),
        LogInfo(msg='  Topics : /anafi/image  /anafi/odometry'),
        LogInfo(msg='─────────────────────────────────────────────'),
    ]

    return LaunchDescription(
        args
        + [set_model_path]
        + info
        + [gazebo, delayed_spawn, static_tf]
    )
