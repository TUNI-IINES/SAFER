#!/usr/bin/env python3

import math

import rclpy
from geometry_msgs.msg import Point, Twist, Vector3
from nav_msgs.msg import Odometry
from rclpy.node import Node
from ros_gz_interfaces.msg import EntityFactory
from visualization_msgs.msg import Marker, MarkerArray


def clamp(value, min_value, max_value):
    return max(min_value, min(max_value, value))


def yaw_from_quaternion(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def wrap_angle(angle):
    return math.atan2(math.sin(angle), math.cos(angle))


class ForwardErgodicDemoNode(Node):
    """
    Visualization-focused Gazebo node.

    Mission:
      1. Mark home from first odometry.
      2. Take off with XY velocity forced to zero.
      3. Fly straight 24 m to the inspection ellipse center.
      4. Keep the original takeoff heading during the whole mission.
      5. Run ergodic-looking coverage inside a 2 x 1 m ellipse for 20 s.
      6. Return straight back to home.
      7. Land at home and stop.
    """

    def __init__(self):
        super().__init__("forward_ergodic_demo_node")

        self.declare_parameter("odom_topic", "/anafi/odometry")
        self.declare_parameter("cmd_vel_topic", "/anafi/cmd_vel")
        self.declare_parameter("marker_topic", "/forward_ergodic_demo/markers")
        self.declare_parameter("gazebo_create_topic", "/world/surveillance_building/create")
        self.declare_parameter("world_frame", "world")

        # Odom guard: gazebo_pose_to_odom can occasionally publish another model/link
        # from /world/.../pose/info if the PoseArray index changes. These limits reject
        # obvious building-link poses such as (-1.62, 2.40, 1.50).
        self.declare_parameter("reject_bad_odom", True)
        self.declare_parameter("expected_start_x", 0.0)
        self.declare_parameter("expected_start_y", -27.0)
        self.declare_parameter("initial_odom_radius", 3.0)
        self.declare_parameter("max_odom_jump", 2.5)
        self.declare_parameter("max_abs_x", 8.0)
        self.declare_parameter("min_y", -32.0)
        self.declare_parameter("max_y", 1.0)

        # IMPORTANT:
        # If the Gazebo plugin interprets /cmd_vel as world-frame velocity, set this to "world".
        # If it interprets /cmd_vel as body-frame velocity, set this to "body".
        self.declare_parameter("velocity_command_frame", "body")

        self.declare_parameter("enable_gazebo_trajectory", False)

        # Start at (0, -27), fly to (0, -3): 24 m travel.
        self.declare_parameter("goal_x", 0.0)
        self.declare_parameter("goal_y", -3.0)
        self.declare_parameter("takeoff_altitude", 2.5)
        self.declare_parameter("flight_altitude", 2.5)

        # 2 x 1 m ellipse.
        self.declare_parameter("area_width", 2.0)
        self.declare_parameter("area_depth", 1.0)
        self.declare_parameter("search_duration", 20.0)

        self.declare_parameter("max_speed", 0.8)
        self.declare_parameter("max_vertical_speed", 0.45)
        self.declare_parameter("kp_goal", 0.35)
        self.declare_parameter("kp_altitude", 0.6)
        self.declare_parameter("yaw_gain", 2.0)
        self.declare_parameter("max_yaw_rate", 1.4)
        self.declare_parameter("position_tolerance", 0.25)
        self.declare_parameter("altitude_tolerance", 0.15)
        self.declare_parameter("landing_altitude_tolerance", 0.10)
        self.declare_parameter("return_altitude", 2.5)
        self.declare_parameter("control_rate_hz", 20.0)

        # Lissajous-style ergodic coverage.
        self.declare_parameter("ergodic_wx", 1.7)
        self.declare_parameter("ergodic_wy", 2.9)
        self.declare_parameter("ergodic_phase_y", 1.5708)

        self.declare_parameter("gazebo_trajectory_spacing", 0.35)
        self.declare_parameter("gazebo_trajectory_z", 0.035)

        self.odom_topic = self.get_parameter("odom_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.gazebo_create_topic = self.get_parameter("gazebo_create_topic").value
        self.world_frame = self.get_parameter("world_frame").value
        self.reject_bad_odom = bool(self.get_parameter("reject_bad_odom").value)
        self.expected_start_x = float(self.get_parameter("expected_start_x").value)
        self.expected_start_y = float(self.get_parameter("expected_start_y").value)
        self.initial_odom_radius = float(self.get_parameter("initial_odom_radius").value)
        self.max_odom_jump = float(self.get_parameter("max_odom_jump").value)
        self.max_abs_x = float(self.get_parameter("max_abs_x").value)
        self.min_y = float(self.get_parameter("min_y").value)
        self.max_y = float(self.get_parameter("max_y").value)
        self.velocity_command_frame = self.get_parameter("velocity_command_frame").value
        self.enable_gazebo_trajectory = bool(self.get_parameter("enable_gazebo_trajectory").value)

        self.goal_x = float(self.get_parameter("goal_x").value)
        self.goal_y = float(self.get_parameter("goal_y").value)
        self.takeoff_altitude = float(self.get_parameter("takeoff_altitude").value)
        self.flight_altitude = float(self.get_parameter("flight_altitude").value)

        self.area_width = float(self.get_parameter("area_width").value)
        self.area_depth = float(self.get_parameter("area_depth").value)
        self.search_duration = float(self.get_parameter("search_duration").value)

        self.max_speed = float(self.get_parameter("max_speed").value)
        self.max_vertical_speed = float(self.get_parameter("max_vertical_speed").value)
        self.kp_goal = float(self.get_parameter("kp_goal").value)
        self.kp_altitude = float(self.get_parameter("kp_altitude").value)
        self.yaw_gain = float(self.get_parameter("yaw_gain").value)
        self.max_yaw_rate = float(self.get_parameter("max_yaw_rate").value)
        self.position_tolerance = float(self.get_parameter("position_tolerance").value)
        self.altitude_tolerance = float(self.get_parameter("altitude_tolerance").value)
        self.landing_altitude_tolerance = float(self.get_parameter("landing_altitude_tolerance").value)
        self.return_altitude = float(self.get_parameter("return_altitude").value)

        self.ergodic_wx = float(self.get_parameter("ergodic_wx").value)
        self.ergodic_wy = float(self.get_parameter("ergodic_wy").value)
        self.ergodic_phase_y = float(self.get_parameter("ergodic_phase_y").value)

        self.gazebo_trajectory_spacing = float(self.get_parameter("gazebo_trajectory_spacing").value)
        self.gazebo_trajectory_z = float(self.get_parameter("gazebo_trajectory_z").value)
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.have_odom = False
        self.last_good_x = None
        self.last_good_y = None
        self.last_good_z = None

        self.state = "WAIT_FOR_HOME"
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_z = 0.0
        self.home_yaw = 0.0
        self.home_set = False

        self.search_start_time = None

        self.target = (self.goal_x, self.goal_y)
        self.applied_vec = (0.0, 0.0)
        self.path = []

        self.last_gazebo_trail_point = None
        self.gazebo_trail_count = 0

        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)

        self.gazebo_create_pub = None
        if self.enable_gazebo_trajectory:
            self.gazebo_create_pub = self.create_publisher(EntityFactory, self.gazebo_create_topic, 10)

        self.timer = self.create_timer(1.0 / max(1.0, control_rate_hz), self.control_loop)

        self.get_logger().info(
            "Forward ergodic demo node started. "
            f"Goal center=({self.goal_x:.2f}, {self.goal_y:.2f}), "
            f"ellipse={self.area_width:.1f} x {self.area_depth:.1f} m, "
            f"duration={self.search_duration:.1f} s, frame={self.velocity_command_frame}, odom_guard={self.reject_bad_odom}."
        )

    def odom_callback(self, msg):
        x = msg.pose.pose.position.x
        y = msg.pose.pose.position.y
        z = msg.pose.pose.position.z
        yaw = yaw_from_quaternion(msg.pose.pose.orientation)

        if self.reject_bad_odom:
            # Before accepting the first pose, require it to be near the expected drone start.
            if not self.have_odom:
                d0 = math.hypot(x - self.expected_start_x, y - self.expected_start_y)
                if d0 > self.initial_odom_radius:
                    self.get_logger().warn(
                        f"Rejected initial odom pose ({x:.2f}, {y:.2f}, {z:.2f}); "
                        f"not near expected start ({self.expected_start_x:.2f}, {self.expected_start_y:.2f}).",
                        throttle_duration_sec=1.0,
                    )
                    return
            else:
                jump = math.hypot(x - self.last_good_x, y - self.last_good_y)
                if jump > self.max_odom_jump:
                    self.get_logger().warn(
                        f"Rejected odom jump to ({x:.2f}, {y:.2f}, {z:.2f}); "
                        f"last good was ({self.last_good_x:.2f}, {self.last_good_y:.2f}, {self.last_good_z:.2f}).",
                        throttle_duration_sec=1.0,
                    )
                    return

            # Mission sanity bounds. The drone should remain south of the building
            # while approaching the south-window inspection ellipse.
            if abs(x) > self.max_abs_x or y < self.min_y or y > self.max_y:
                self.get_logger().warn(
                    f"Rejected out-of-mission odom pose ({x:.2f}, {y:.2f}, {z:.2f}).",
                    throttle_duration_sec=1.0,
                )
                return

        self.x = x
        self.y = y
        self.z = z
        self.yaw = yaw
        self.have_odom = True
        self.last_good_x = x
        self.last_good_y = y
        self.last_good_z = z

    def control_loop(self):
        if not self.have_odom:
            self.get_logger().warn("Waiting for /anafi/odometry...", throttle_duration_sec=2.0)
            self.publish_markers()
            return

        if self.state == "DONE":
            self.cmd_pub.publish(Twist())
            self.publish_markers()
            return

        tx, ty = self.mission_target()
        self.target = (tx, ty)

        vx = 0.0
        vy = 0.0

        # Hard rule: no XY motion until takeoff is complete.
        if self.state not in ("TAKEOFF", "WAIT_FOR_HOME", "LAND"):
            vx = self.kp_goal * (tx - self.x)
            vy = self.kp_goal * (ty - self.y)
            vx, vy = self.limit_xy(vx, vy, self.max_speed)

        vz = clamp(
            self.kp_altitude * (self.current_target_altitude() - self.z),
            -self.max_vertical_speed,
            self.max_vertical_speed,
        )

        yaw_rate = self.yaw_hold_command()

        cmd_vx, cmd_vy = self.world_velocity_to_command_frame(vx, vy)

        cmd = Twist()
        cmd.linear.x = float(cmd_vx)
        cmd.linear.y = float(cmd_vy)
        cmd.linear.z = float(vz)
        cmd.angular.z = float(yaw_rate)
        self.cmd_pub.publish(cmd)

        self.applied_vec = (vx, vy)
        self.append_path_point()
        self.maybe_spawn_gazebo_trail_point()
        self.publish_markers()

        self.get_logger().info(
            f"state={self.state} pos=({self.x:.2f},{self.y:.2f},{self.z:.2f}) "
            f"target=({tx:.2f},{ty:.2f}) world_v=({vx:.2f},{vy:.2f}) "
            f"cmd_v=({cmd_vx:.2f},{cmd_vy:.2f}) yaw={self.yaw:.2f}",
            throttle_duration_sec=2.0,
        )

    def mission_target(self):
        now = self.now_seconds()

        if self.state == "WAIT_FOR_HOME":
            self.home_x = self.x
            self.home_y = self.y
            self.home_z = self.z
            self.home_yaw = self.yaw
            self.home_set = True
            self.state = "TAKEOFF"
            self.get_logger().info(
                f"Home marked at ({self.home_x:.2f}, {self.home_y:.2f}, {self.home_z:.2f}), "
                f"yaw={math.degrees(self.home_yaw):.1f} deg. "
                f"Taking off to {self.takeoff_altitude:.1f} m."
            )

        if self.state == "TAKEOFF":
            if abs(self.takeoff_altitude - self.z) < self.altitude_tolerance:
                self.state = "TRANSIT"
                self.get_logger().info(
                    f"Takeoff complete. Flying straight to inspection center "
                    f"({self.goal_x:.2f}, {self.goal_y:.2f}) while holding heading."
                )
            return self.home_x, self.home_y

        if self.state == "TRANSIT":
            if self.distance_xy(self.goal_x, self.goal_y) < self.position_tolerance:
                self.state = "INSPECT"
                self.search_start_time = now
                self.get_logger().info(
                    f"Reached ellipse center. Starting ergodic coverage for {self.search_duration:.1f} s."
                )
                return self.ergodic_target(now)
            return self.goal_x, self.goal_y

        if self.state == "INSPECT":
            elapsed = now - self.search_start_time
            if elapsed >= self.search_duration:
                self.state = "RETURN_HOME"
                self.get_logger().info("Ergodic coverage complete. Returning home.")
                return self.home_x, self.home_y
            return self.ergodic_target(now)

        if self.state == "RETURN_HOME":
            if self.distance_xy(self.home_x, self.home_y) < self.position_tolerance:
                self.state = "LAND"
                self.get_logger().info("Home reached. Landing.")
            return self.home_x, self.home_y

        if self.state == "LAND":
            if abs(self.z - self.home_z) < self.landing_altitude_tolerance:
                self.state = "DONE"
                self.get_logger().info("Landing complete. Holding position.")
            return self.home_x, self.home_y

        return self.x, self.y

    def ergodic_target(self, now):
        if self.search_start_time is None:
            t = 0.0
        else:
            t = max(0.0, now - self.search_start_time)

        a = 0.5 * self.area_width
        b = 0.5 * self.area_depth

        margin = 0.90
        lx = margin * a * math.sin(self.ergodic_wx * t)
        ly = margin * b * math.sin(self.ergodic_wy * t + self.ergodic_phase_y)

        return self.goal_x + lx, self.goal_y + ly

    def current_target_altitude(self):
        if self.state == "RETURN_HOME":
            return self.return_altitude
        if self.state in ("LAND", "DONE"):
            return self.home_z
        return self.flight_altitude

    def yaw_hold_command(self):
        yaw_error = wrap_angle(self.home_yaw - self.yaw)
        return clamp(self.yaw_gain * yaw_error, -self.max_yaw_rate, self.max_yaw_rate)

    def world_velocity_to_command_frame(self, vx, vy):
        if self.velocity_command_frame == "world":
            return vx, vy

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        body_x = cos_yaw * vx + sin_yaw * vy
        body_y = -sin_yaw * vx + cos_yaw * vy
        return body_x, body_y

    def distance_xy(self, tx, ty):
        return math.hypot(tx - self.x, ty - self.y)

    def limit_xy(self, vx, vy, max_speed):
        norm = math.hypot(vx, vy)
        if norm > max_speed and norm > 1e-9:
            scale = max_speed / norm
            return vx * scale, vy * scale
        return vx, vy

    def append_path_point(self):
        self.path.append(Point(x=float(self.x), y=float(self.y), z=float(self.z)))
        if len(self.path) > 900:
            self.path.pop(0)

    def maybe_spawn_gazebo_trail_point(self):
        if self.gazebo_create_pub is None:
            return

        current = (float(self.x), float(self.y))
        if self.last_gazebo_trail_point is not None:
            dx = current[0] - self.last_gazebo_trail_point[0]
            dy = current[1] - self.last_gazebo_trail_point[1]
            if math.hypot(dx, dy) < self.gazebo_trajectory_spacing:
                return

        self.last_gazebo_trail_point = current
        self.gazebo_trail_count += 1

        msg = EntityFactory()
        msg.name = f"forward_ergodic_trail_{self.gazebo_trail_count:04d}"
        msg.allow_renaming = True
        msg.pose.position.x = current[0]
        msg.pose.position.y = current[1]
        msg.pose.position.z = float(self.gazebo_trajectory_z)
        msg.pose.orientation.w = 1.0
        msg.relative_to = self.world_frame
        msg.sdf = self.gazebo_trail_sdf(msg.name)
        self.gazebo_create_pub.publish(msg)

    def gazebo_trail_sdf(self, name):
        return f"""
<sdf version="1.9">
  <model name="{name}">
    <static>true</static>
    <link name="link">
      <visual name="visual">
        <geometry>
          <cylinder>
            <radius>0.075</radius>
            <length>0.018</length>
          </cylinder>
        </geometry>
        <material>
          <ambient>0.02 0.80 0.24 1</ambient>
          <diffuse>0.02 0.95 0.30 1</diffuse>
          <emissive>0.00 0.25 0.07 1</emissive>
        </material>
      </visual>
    </link>
  </model>
</sdf>
"""

    def publish_markers(self):
        stamp = self.get_clock().now().to_msg()
        markers = MarkerArray()

        markers.markers.append(self.ellipse_marker(stamp))
        markers.markers.append(self.goal_marker(stamp))
        markers.markers.append(self.target_marker(stamp))
        markers.markers.append(self.path_marker(stamp))
        markers.markers.append(self.home_marker(stamp))
        markers.markers.append(
            self.arrow_marker(stamp, 30, "applied_velocity", self.x, self.y, self.z + 0.35, self.applied_vec)
        )

        self.marker_pub.publish(markers)

    def ellipse_marker(self, stamp):
        marker = self.base_marker(stamp, 0, "inspection_ellipse", Marker.LINE_STRIP)
        marker.scale.x = 0.05
        marker.color.r = 0.05
        marker.color.g = 0.55
        marker.color.b = 1.0
        marker.color.a = 0.95

        a = 0.5 * self.area_width
        b = 0.5 * self.area_depth
        for i in range(145):
            th = 2.0 * math.pi * i / 144.0
            marker.points.append(
                Point(
                    x=float(self.goal_x + a * math.cos(th)),
                    y=float(self.goal_y + b * math.sin(th)),
                    z=float(self.flight_altitude),
                )
            )
        return marker

    def goal_marker(self, stamp):
        marker = self.base_marker(stamp, 1, "inspection_center", Marker.SPHERE)
        marker.pose.position.x = float(self.goal_x)
        marker.pose.position.y = float(self.goal_y)
        marker.pose.position.z = float(self.flight_altitude)
        marker.scale = Vector3(x=0.28, y=0.28, z=0.28)
        marker.color.r = 1.0
        marker.color.g = 0.25
        marker.color.b = 0.05
        marker.color.a = 0.95
        return marker

    def target_marker(self, stamp):
        marker = self.base_marker(stamp, 2, "current_ergodic_target", Marker.SPHERE)
        marker.pose.position.x = float(self.target[0])
        marker.pose.position.y = float(self.target[1])
        marker.pose.position.z = float(self.flight_altitude)
        marker.scale = Vector3(x=0.22, y=0.22, z=0.22)
        marker.color.r = 0.05
        marker.color.g = 0.85
        marker.color.b = 1.0
        marker.color.a = 0.95
        return marker

    def path_marker(self, stamp):
        marker = self.base_marker(stamp, 3, "covered_trajectory", Marker.LINE_STRIP)
        marker.scale.x = 0.035
        marker.color.r = 0.05
        marker.color.g = 0.9
        marker.color.b = 0.35
        marker.color.a = 0.95
        marker.points = list(self.path)
        return marker

    def home_marker(self, stamp):
        marker = self.base_marker(stamp, 4, "home_start_x", Marker.LINE_LIST)
        marker.scale.x = 0.08
        marker.color.r = 0.0
        marker.color.g = 0.0
        marker.color.b = 0.0
        marker.color.a = 1.0

        x = self.home_x if self.home_set else self.x
        y = self.home_y if self.home_set else self.y
        z = 0.08
        size = 0.45

        marker.points = [
            Point(x=float(x - size), y=float(y - size), z=float(z)),
            Point(x=float(x + size), y=float(y + size), z=float(z)),
            Point(x=float(x - size), y=float(y + size), z=float(z)),
            Point(x=float(x + size), y=float(y - size), z=float(z)),
        ]
        return marker

    def arrow_marker(self, stamp, marker_id, namespace, x, y, z, vec):
        marker = self.base_marker(stamp, marker_id, namespace, Marker.ARROW)
        marker.scale = Vector3(x=0.08, y=0.16, z=0.22)
        marker.color.r = 0.05
        marker.color.g = 0.9
        marker.color.b = 0.25
        marker.color.a = 0.95

        vx, vy = vec
        marker.points = [
            Point(x=float(x), y=float(y), z=float(z)),
            Point(x=float(x + vx), y=float(y + vy), z=float(z)),
        ]
        return marker

    def base_marker(self, stamp, marker_id, namespace, marker_type):
        marker = Marker()
        marker.header.frame_id = self.world_frame
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = marker_type
        marker.action = Marker.ADD
        marker.pose.orientation.w = 1.0
        return marker

    def now_seconds(self):
        return self.get_clock().now().nanoseconds * 1e-9


def main(args=None):
    rclpy.init(args=args)
    node = ForwardErgodicDemoNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.cmd_pub.publish(Twist())
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()