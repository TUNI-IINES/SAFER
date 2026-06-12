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


class ErgodicCbfDemoNode(Node):
    """Visualization-focused ergodic coverage + CBF repulsion controller."""

    def __init__(self):
        super().__init__("ergodic_cbf_demo_node")

        self.declare_parameter("odom_topic", "/anafi/odometry")
        self.declare_parameter("cmd_vel_topic", "/anafi/cmd_vel")
        self.declare_parameter("marker_topic", "/ergodic_cbf_demo/markers")
        self.declare_parameter("gazebo_create_topic", "/world/surveillance_building/create")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("velocity_command_frame", "body")
        self.declare_parameter("enable_gazebo_trajectory", True)

        self.declare_parameter("building_center_x", 0.0)
        self.declare_parameter("building_center_y", 0.0)
        self.declare_parameter("cbf_radius", 4.8)
        self.declare_parameter("cbf_influence_radius", 6.0)
        self.declare_parameter("ergodic_radius_x", 6.8)
        self.declare_parameter("ergodic_radius_y", 6.8)
        self.declare_parameter("obstacle_x", 0.0)
        self.declare_parameter("obstacle_y", 6.6)
        self.declare_parameter("obstacle_radius", 0.55)
        self.declare_parameter("obstacle_influence_radius", 1.25)
        self.declare_parameter("obstacle_cbf_gain", 1.4)
        self.declare_parameter("obstacle_tangent_gain", 0.0)
        self.declare_parameter("obstacle_bypass_margin", 1.2)

        self.declare_parameter("takeoff_altitude", 2.5)
        self.declare_parameter("flight_altitude", 2.5)
        self.declare_parameter("max_speed", 1.35)
        self.declare_parameter("max_vertical_speed", 1.2)
        self.declare_parameter("kp_goal", 0.85)
        self.declare_parameter("kp_altitude", 1.6)
        self.declare_parameter("cbf_gain", 4.2)
        self.declare_parameter("cbf_buffer", 0.55)
        self.declare_parameter("cbf_min_outward_speed", 0.35)
        self.declare_parameter("yaw_gain", 3.0)
        self.declare_parameter("max_yaw_rate", 2.8)
        self.declare_parameter("transit_radius", 6.5)
        self.declare_parameter("inspection_radius", 6.5)
        self.declare_parameter("inspection_angular_rate", 0.42)
        self.declare_parameter("window_view_angle", -1.5708)
        self.declare_parameter("window_dwell_duration", 4.0)
        self.declare_parameter("detection_angle_tolerance", 0.75)
        self.declare_parameter("detection_distance_tolerance", 2.2)
        self.declare_parameter("detection_min_inspection_time", 1.5)
        self.declare_parameter("position_tolerance", 0.25)
        self.declare_parameter("control_rate_hz", 20.0)
        self.declare_parameter("landing_altitude_tolerance", 0.10)
        self.declare_parameter("gazebo_trajectory_spacing", 0.35)
        self.declare_parameter("gazebo_trajectory_z", 0.035)

        self.odom_topic = self.get_parameter("odom_topic").value
        self.cmd_vel_topic = self.get_parameter("cmd_vel_topic").value
        self.marker_topic = self.get_parameter("marker_topic").value
        self.gazebo_create_topic = self.get_parameter("gazebo_create_topic").value
        self.world_frame = self.get_parameter("world_frame").value
        self.velocity_command_frame = self.get_parameter("velocity_command_frame").value
        self.enable_gazebo_trajectory = bool(self.get_parameter("enable_gazebo_trajectory").value)

        self.cx = float(self.get_parameter("building_center_x").value)
        self.cy = float(self.get_parameter("building_center_y").value)
        self.cbf_radius = float(self.get_parameter("cbf_radius").value)
        self.cbf_influence_radius = float(self.get_parameter("cbf_influence_radius").value)
        self.ergodic_radius_x = float(self.get_parameter("ergodic_radius_x").value)
        self.ergodic_radius_y = float(self.get_parameter("ergodic_radius_y").value)
        self.obstacle_x = float(self.get_parameter("obstacle_x").value)
        self.obstacle_y = float(self.get_parameter("obstacle_y").value)
        self.obstacle_radius = float(self.get_parameter("obstacle_radius").value)
        self.obstacle_influence_radius = float(self.get_parameter("obstacle_influence_radius").value)
        self.obstacle_cbf_gain = float(self.get_parameter("obstacle_cbf_gain").value)
        self.obstacle_tangent_gain = float(self.get_parameter("obstacle_tangent_gain").value)
        self.obstacle_bypass_margin = float(self.get_parameter("obstacle_bypass_margin").value)

        self.takeoff_altitude = float(self.get_parameter("takeoff_altitude").value)
        self.flight_altitude = float(self.get_parameter("flight_altitude").value)
        self.max_speed = float(self.get_parameter("max_speed").value)
        self.max_vertical_speed = float(self.get_parameter("max_vertical_speed").value)
        self.kp_goal = float(self.get_parameter("kp_goal").value)
        self.kp_altitude = float(self.get_parameter("kp_altitude").value)
        self.cbf_gain = float(self.get_parameter("cbf_gain").value)
        self.cbf_buffer = float(self.get_parameter("cbf_buffer").value)
        self.cbf_min_outward_speed = float(self.get_parameter("cbf_min_outward_speed").value)
        self.yaw_gain = float(self.get_parameter("yaw_gain").value)
        self.max_yaw_rate = float(self.get_parameter("max_yaw_rate").value)
        self.transit_radius = float(self.get_parameter("transit_radius").value)
        self.inspection_radius = float(self.get_parameter("inspection_radius").value)
        self.inspection_angular_rate = float(self.get_parameter("inspection_angular_rate").value)
        self.window_view_angle = float(self.get_parameter("window_view_angle").value)
        self.window_dwell_duration = float(self.get_parameter("window_dwell_duration").value)
        self.detection_angle_tolerance = float(self.get_parameter("detection_angle_tolerance").value)
        self.detection_distance_tolerance = float(self.get_parameter("detection_distance_tolerance").value)
        self.detection_min_inspection_time = float(self.get_parameter("detection_min_inspection_time").value)
        self.position_tolerance = float(self.get_parameter("position_tolerance").value)
        self.landing_altitude_tolerance = float(self.get_parameter("landing_altitude_tolerance").value)
        self.gazebo_trajectory_spacing = float(self.get_parameter("gazebo_trajectory_spacing").value)
        self.gazebo_trajectory_z = float(self.get_parameter("gazebo_trajectory_z").value)
        control_rate_hz = float(self.get_parameter("control_rate_hz").value)

        self.x = 0.0
        self.y = 0.0
        self.z = 0.0
        self.yaw = 0.0
        self.have_odom = False

        self.nominal_vec = (0.0, 0.0)
        self.cbf_vec = (0.0, 0.0)
        self.applied_vec = (0.0, 0.0)
        self.target = (self.cx, self.cy)
        self.path = []
        self.last_gazebo_trail_point = None
        self.gazebo_trail_count = 0

        self.state = "WAIT_FOR_HOME"
        self.home_x = 0.0
        self.home_y = 0.0
        self.home_z = 0.0
        self.home_set = False
        self.transit_waypoints = []
        self.transit_waypoint_index = 0
        self.inspection_start_time = None
        self.inspection_start_angle = 0.0
        self.return_start_time = None
        self.return_start_angle = 0.0
        self.home_angle = 0.0
        self.dwell_start_time = None
        self.detection_announced = False

        self.create_subscription(Odometry, self.odom_topic, self.odom_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.gazebo_create_pub = None
        if self.enable_gazebo_trajectory:
            self.gazebo_create_pub = self.create_publisher(EntityFactory, self.gazebo_create_topic, 10)

        timer_period = 1.0 / max(1.0, control_rate_hz)
        self.timer = self.create_timer(timer_period, self.control_loop)

        self.get_logger().info(
            "Ergodic + CBF demo node started. "
            f"Publishing {self.cmd_vel_topic} and {self.marker_topic}."
        )

    def odom_callback(self, msg):
        self.x = msg.pose.pose.position.x
        self.y = msg.pose.pose.position.y
        self.z = msg.pose.pose.position.z
        self.yaw = yaw_from_quaternion(msg.pose.pose.orientation)
        self.have_odom = True

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

        nominal_x = self.kp_goal * (tx - self.x)
        nominal_y = self.kp_goal * (ty - self.y)
        nominal_x, nominal_y = self.limit_xy(nominal_x, nominal_y, self.max_speed)

        vx, vy, building_cbf_x, building_cbf_y = self.apply_building_cbf(nominal_x, nominal_y)
        vx, vy, obstacle_cbf_x, obstacle_cbf_y = self.apply_obstacle_cbf(vx, vy)
        vx, vy = self.limit_xy(vx, vy, self.max_speed)


        if self.state in ("RETURN_CLIMB", "RETURN_HOME", "LAND"):
            vx, vy = nominal_x, nominal_y
            building_cbf_x = building_cbf_y = 0.0
            obstacle_cbf_x = obstacle_cbf_y = 0.0
        else:
            vx, vy, building_cbf_x, building_cbf_y = self.apply_building_cbf(nominal_x, nominal_y)
            vx, vy, obstacle_cbf_x, obstacle_cbf_y = self.apply_obstacle_cbf(vx, vy)

        vz = clamp(
            self.kp_altitude * (self.current_target_altitude() - self.z),
            -self.max_vertical_speed,
            self.max_vertical_speed,
        )

        yaw_rate = self.yaw_rate_command()

        cmd_vx, cmd_vy = self.world_velocity_to_command_frame(vx, vy)

        cmd = Twist()
        cmd.linear.x = float(cmd_vx)
        cmd.linear.y = float(cmd_vy)
        cmd.linear.z = float(vz)
        cmd.angular.z = float(yaw_rate)
        self.cmd_pub.publish(cmd)

        self.nominal_vec = (nominal_x, nominal_y)
        self.cbf_vec = (building_cbf_x + obstacle_cbf_x, building_cbf_y + obstacle_cbf_y)
        self.applied_vec = (vx, vy)
        self.append_path_point()
        self.maybe_spawn_gazebo_trail_point()
        self.publish_markers()

        self.get_logger().info(
            f"state={self.state} world_v=({vx:.2f},{vy:.2f}) "
            f"cmd_v=({cmd_vx:.2f},{cmd_vy:.2f}) yaw={self.yaw:.2f}",
            throttle_duration_sec=2.0,
        )

    def world_velocity_to_command_frame(self, vx, vy):
        if self.velocity_command_frame == "world":
            return vx, vy

        cos_yaw = math.cos(self.yaw)
        sin_yaw = math.sin(self.yaw)
        body_x = cos_yaw * vx + sin_yaw * vy
        body_y = -sin_yaw * vx + cos_yaw * vy
        return body_x, body_y

    def yaw_rate_command(self):
        desired_yaw = math.atan2(self.cy - self.y, self.cx - self.x)
        yaw_error = wrap_angle(desired_yaw - self.yaw)
        return clamp(self.yaw_gain * yaw_error, -self.max_yaw_rate, self.max_yaw_rate)

    def mission_target(self):
        now = self.now_seconds()

        if self.state == "WAIT_FOR_HOME":
            self.home_x = self.x
            self.home_y = self.y
            self.home_z = self.z
            self.home_set = True
            self.state = "TAKEOFF"
            self.get_logger().info(
                f"Home marked at ({self.home_x:.2f}, {self.home_y:.2f}, {self.home_z:.2f}). "
                f"Taking off to {self.takeoff_altitude:.1f} m."
            )

        if self.state == "TAKEOFF":
            if abs(self.takeoff_altitude - self.z) < 0.15:
                self.state = "TRANSIT"
                self.transit_waypoints = self.make_transit_waypoints()
                self.transit_waypoint_index = 0
                self.get_logger().info("Takeoff complete. Flying around the obstacle toward the object.")
            return self.home_x, self.home_y

        if self.state == "TRANSIT":
            tx, ty = self.transit_waypoints[self.transit_waypoint_index]
            if self.distance_xy(tx, ty) < 0.35:
                self.transit_waypoint_index += 1
                if self.transit_waypoint_index >= len(self.transit_waypoints):
                    self.state = "INSPECT"
                    self.inspection_start_time = now
                    self.inspection_start_angle = math.atan2(self.y - self.cy, self.x - self.cx)
                    self.get_logger().info("Reached CBF-safe standoff. Starting donut-shaped ergodic inspection.")
                    return self.donut_target(now)[0:2]

                tx, ty = self.transit_waypoints[self.transit_waypoint_index]
                self.get_logger().info(
                    f"Transit waypoint {self.transit_waypoint_index + 1}/{len(self.transit_waypoints)}."
                )

            return tx, ty

        if self.state == "INSPECT":
            tx, ty, angle = self.donut_target(now)
            if self.window_reached(now):
                self.state = "DWELL_AT_WINDOW"
                self.dwell_start_time = now
                self.detection_announced = True
                self.get_logger().info("Fire/smoke/human spotted through the window. Holding briefly, then returning home.")
            return tx, ty

        if self.state == "DWELL_AT_WINDOW":
            if now - self.dwell_start_time >= self.window_dwell_duration:
                self.state = "RETURN_CLIMB"
                self.get_logger().info("Dwell complete. Climbing to 4 m, then flying straight home.")
            return self.window_view_point()
        
        if self.state == "RETURN_CLIMB":
            if abs(4.0 - self.z) < 0.20:
                self.state = "RETURN_HOME"
                self.get_logger().info("Return altitude reached. Flying straight home.")
            return self.x, self.y

        if self.state == "RETURN_AROUND_HOME":
            tx, ty, angle = self.return_arc_target(now)
            if self.home_side_reached():
                self.state = "RETURN_HOME"
                self.get_logger().info("Home side reached. Flying straight to home.")
            return tx, ty

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

        return self.home_x, self.home_y

    def make_transit_waypoints(self):
        side = 1.0 if self.home_x <= self.obstacle_x else -1.0
        side_x = self.obstacle_x + side * (self.obstacle_influence_radius + self.obstacle_bypass_margin)
        entry_x, entry_y = self.transit_entry_point(side_x)
        return [
            (side_x, self.home_y),
            (side_x, self.obstacle_y),
            (entry_x, entry_y),
        ]

    def current_target_altitude(self):
        if self.state == "RETURN_CLIMB":
            return 4.0
        if self.state == "RETURN_HOME":
            return 4.0
        if self.state in ("LAND", "DONE"):
            return self.home_z
        return self.flight_altitude
    def transit_entry_point(self, preferred_x=None):
        if preferred_x is None:
            side = 1.0 if self.home_x <= self.obstacle_x else -1.0
            preferred_x = side * (self.obstacle_influence_radius + self.obstacle_bypass_margin)

        offset_x = clamp(preferred_x - self.cx, -self.transit_radius + 0.25, self.transit_radius - 0.25)
        offset_y = math.sqrt(max(0.0, self.transit_radius * self.transit_radius - offset_x * offset_x))
        return self.cx + offset_x, self.cy + offset_y

    def obstacle_bypass_point(self):
        side = 1.0 if self.home_x <= self.obstacle_x else -1.0
        return (
            self.obstacle_x + side * (self.obstacle_influence_radius + self.obstacle_bypass_margin),
            self.obstacle_y,
        )

    def donut_target(self, now):
        elapsed = max(0.0, now - self.inspection_start_time)
        angle = self.inspection_start_angle - self.inspection_angular_rate * elapsed
        radius = max(self.inspection_radius, self.cbf_influence_radius + 0.3)
        tx = self.cx + radius * math.cos(angle)
        ty = self.cy + radius * math.sin(angle)
        return tx, ty, angle

    def window_view_point(self):
        radius = max(self.inspection_radius, self.cbf_influence_radius + 0.3)
        return (
            self.cx + radius * math.cos(self.window_view_angle),
            self.cy + radius * math.sin(self.window_view_angle),
        )

    def return_arc_target(self, now):
        elapsed = max(0.0, now - self.return_start_time)
        angle_delta = wrap_angle(self.home_angle - self.return_start_angle)
        direction = 1.0 if angle_delta >= 0.0 else -1.0
        angle = self.return_start_angle + direction * self.inspection_angular_rate * elapsed

        if direction * wrap_angle(self.home_angle - angle) <= 0.0:
            angle = self.home_angle

        radius = max(self.inspection_radius, self.cbf_influence_radius + 0.3)
        tx = self.cx + radius * math.cos(angle)
        ty = self.cy + radius * math.sin(angle)
        return tx, ty, angle

    def window_reached(self, now):
        if self.inspection_start_time is None:
            return False

        if now - self.inspection_start_time < self.detection_min_inspection_time:
            return False

        wx, wy = self.window_view_point()
        actual_angle = self.current_building_angle()
        angle_error = abs(wrap_angle(actual_angle - self.window_view_angle))
        drone_close = self.distance_xy(wx, wy) < self.detection_distance_tolerance
        angle_close = angle_error < self.detection_angle_tolerance

        return angle_close and drone_close

    def current_building_angle(self):
        return math.atan2(self.y - self.cy, self.x - self.cx)

    def home_side_reached(self):
        radius = max(self.inspection_radius, self.cbf_influence_radius + 0.3)
        hx = self.cx + radius * math.cos(self.home_angle)
        hy = self.cy + radius * math.sin(self.home_angle)
        actual_angle_error = abs(wrap_angle(self.current_building_angle() - self.home_angle))
        return actual_angle_error < 0.22 and self.distance_xy(hx, hy) < 0.9

    def distance_xy(self, tx, ty):
        return math.hypot(tx - self.x, ty - self.y)

    def apply_building_cbf(self, nominal_x, nominal_y):
        dx = self.x - self.cx
        dy = self.y - self.cy
        raw_dist = math.hypot(dx, dy)
        if raw_dist < 1e-6:
            dx = 1.0
            dy = 0.0
        dist = max(1e-6, math.hypot(dx, dy))

        radial_x = dx / dist
        radial_y = dy / dist

        if dist < self.cbf_radius:
            outward_speed = self.max_speed
            return outward_speed * radial_x, outward_speed * radial_y, outward_speed * radial_x - nominal_x, outward_speed * radial_y - nominal_y

        if dist >= self.cbf_influence_radius:
            return nominal_x, nominal_y, 0.0, 0.0

        radial_velocity = nominal_x * radial_x + nominal_y * radial_y
        filtered_x = nominal_x
        filtered_y = nominal_y

        if radial_velocity < 0.0:
            # Remove inward motion first: this is the visual CBF "magnet wall".
            filtered_x -= radial_velocity * radial_x
            filtered_y -= radial_velocity * radial_y

        band = max(1e-6, self.cbf_influence_radius - self.cbf_radius)
        h = dist - self.cbf_radius
        proximity = clamp(1.0 - h / band, 0.0, 1.0)
        outward_speed = self.cbf_gain * proximity * proximity

        buffer_h = dist - (self.cbf_radius + self.cbf_buffer)
        if buffer_h < 0.0:
            outward_speed += self.cbf_min_outward_speed + self.cbf_gain * abs(buffer_h)

        cbf_x = filtered_x - nominal_x + outward_speed * radial_x
        cbf_y = filtered_y - nominal_y + outward_speed * radial_y

        return filtered_x + outward_speed * radial_x, filtered_y + outward_speed * radial_y, cbf_x, cbf_y

    def apply_obstacle_cbf(self, nominal_x, nominal_y):
        dx = self.x - self.obstacle_x
        dy = self.y - self.obstacle_y
        raw_dist = math.hypot(dx, dy)
        if raw_dist < 1e-6:
            dx = 1.0
            dy = 0.0
        dist = max(1e-6, math.hypot(dx, dy))

        if dist >= self.obstacle_influence_radius:
            return nominal_x, nominal_y, 0.0, 0.0

        radial_x = dx / dist
        radial_y = dy / dist
        radial_velocity = nominal_x * radial_x + nominal_y * radial_y

        if dist < self.obstacle_radius:
            outward_speed = self.max_speed
            return outward_speed * radial_x, outward_speed * radial_y, outward_speed * radial_x - nominal_x, outward_speed * radial_y - nominal_y

        filtered_x = nominal_x
        filtered_y = nominal_y
        if radial_velocity < 0.0:
            filtered_x -= radial_velocity * radial_x
            filtered_y -= radial_velocity * radial_y

        band = max(1e-6, self.obstacle_influence_radius - self.obstacle_radius)
        h = dist - self.obstacle_radius
        proximity = clamp(1.0 - h / band, 0.0, 1.0)
        outward_speed = self.obstacle_cbf_gain * proximity * proximity

        # Optional small bias; the mission bypass point does most of the routing.
        tangent_x = radial_y
        tangent_y = -radial_x
        tangent_speed = self.obstacle_tangent_gain * proximity

        cbf_x = filtered_x - nominal_x + outward_speed * radial_x + tangent_speed * tangent_x
        cbf_y = filtered_y - nominal_y + outward_speed * radial_y + tangent_speed * tangent_y

        return (
            filtered_x + outward_speed * radial_x + tangent_speed * tangent_x,
            filtered_y + outward_speed * radial_y + tangent_speed * tangent_y,
            cbf_x,
            cbf_y,
        )

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
        msg.name = f"ergodic_trail_{self.gazebo_trail_count:04d}"
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

        markers.markers.append(self.ellipse_marker(stamp, 0, "ergodic_area", self.ergodic_radius_x, self.ergodic_radius_y))
        markers.markers.append(self.circle_marker(stamp, 1, "cbf_boundary", self.cbf_radius, 1.52, 1.0, 0.08, 0.02, 0.95))
        markers.markers.append(self.circle_marker(stamp, 2, "cbf_influence", self.cbf_influence_radius, 1.54, 1.0, 0.30, 0.02, 0.35))
        markers.markers.append(self.obstacle_marker(stamp))
        markers.markers.append(self.obstacle_bypass_marker(stamp))
        markers.markers.append(
            self.circle_at_marker(
                stamp,
                4,
                "obstacle_cbf_boundary",
                self.obstacle_x,
                self.obstacle_y,
                self.obstacle_influence_radius,
                0.08,
                1.0,
                0.45,
                0.0,
                0.9,
            )
        )
        markers.markers.extend(self.repulsion_arrows(stamp))

        markers.markers.append(
            self.arrow_marker(stamp, 30, "nominal_ergodic_velocity", self.x, self.y, self.z + 0.25, self.nominal_vec, 0.95, 0.85, 0.05, 0.90)
        )
        markers.markers.append(
            self.arrow_marker(stamp, 31, "cbf_repulsion_velocity", self.x, self.y, self.z + 0.45, self.cbf_vec, 1.0, 0.05, 0.02, 0.95)
        )
        markers.markers.append(
            self.arrow_marker(stamp, 32, "applied_velocity", self.x, self.y, self.z + 0.65, self.applied_vec, 0.05, 0.9, 0.25, 0.95)
        )
        markers.markers.append(self.target_marker(stamp))
        markers.markers.append(self.path_marker(stamp))
        markers.markers.append(self.home_marker(stamp))
        markers.markers.append(self.window_marker(stamp))

        self.marker_pub.publish(markers)

    def ellipse_marker(self, stamp, marker_id, namespace, rx, ry):
        marker = self.base_marker(stamp, marker_id, namespace, Marker.LINE_STRIP)
        marker.scale.x = 0.05
        marker.color.r = 0.05
        marker.color.g = 0.55
        marker.color.b = 1.0
        marker.color.a = 0.95

        for i in range(145):
            th = 2.0 * math.pi * i / 144.0
            marker.points.append(
                Point(
                    x=float(self.cx + rx * math.cos(th)),
                    y=float(self.cy + ry * math.sin(th)),
                    z=float(self.flight_altitude),
                )
            )
        return marker

    def circle_marker(self, stamp, marker_id, namespace, radius, z, r, g, b, alpha):
        return self.circle_at_marker(stamp, marker_id, namespace, self.cx, self.cy, radius, z, r, g, b, alpha)

    def circle_at_marker(self, stamp, marker_id, namespace, cx, cy, radius, z, r, g, b, alpha):
        marker = self.base_marker(stamp, marker_id, namespace, Marker.LINE_STRIP)
        marker.scale.x = 0.055
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = alpha

        for i in range(121):
            th = 2.0 * math.pi * i / 120.0
            marker.points.append(
                Point(
                    x=float(cx + radius * math.cos(th)),
                    y=float(cy + radius * math.sin(th)),
                    z=float(z),
                )
            )
        return marker

    def obstacle_marker(self, stamp):
        marker = self.base_marker(stamp, 3, "path_obstacle", Marker.CYLINDER)
        marker.pose.position.x = float(self.obstacle_x)
        marker.pose.position.y = float(self.obstacle_y)
        marker.pose.position.z = 0.30
        marker.scale = Vector3(x=0.55, y=0.55, z=0.60)
        marker.color.r = 0.08
        marker.color.g = 0.08
        marker.color.b = 0.08
        marker.color.a = 1.0
        return marker

    def obstacle_bypass_marker(self, stamp):
        marker = self.base_marker(stamp, 5, "obstacle_bypass_target", Marker.SPHERE)
        bx, by = self.obstacle_bypass_point()
        marker.pose.position.x = float(bx)
        marker.pose.position.y = float(by)
        marker.pose.position.z = float(self.flight_altitude)
        marker.scale = Vector3(x=0.20, y=0.20, z=0.20)
        marker.color.r = 1.0
        marker.color.g = 0.75
        marker.color.b = 0.0
        marker.color.a = 0.95
        return marker

    def repulsion_arrows(self, stamp):
        arrows = []
        for i in range(16):
            th = 2.0 * math.pi * i / 16.0
            sx = self.cx + self.cbf_radius * math.cos(th)
            sy = self.cy + self.cbf_radius * math.sin(th)
            vec = (0.75 * math.cos(th), 0.75 * math.sin(th))
            arrows.append(
                self.arrow_marker(stamp, 100 + i, "cbf_opposing_magnet", sx, sy, 1.7, vec, 1.0, 0.05, 0.02, 0.80)
            )
        return arrows

    def arrow_marker(self, stamp, marker_id, namespace, x, y, z, vec, r, g, b, alpha):
        marker = self.base_marker(stamp, marker_id, namespace, Marker.ARROW)
        marker.scale = Vector3(x=0.08, y=0.16, z=0.22)
        marker.color.r = r
        marker.color.g = g
        marker.color.b = b
        marker.color.a = alpha

        vx, vy = vec
        marker.points = [
            Point(x=float(x), y=float(y), z=float(z)),
            Point(x=float(x + vx), y=float(y + vy), z=float(z)),
        ]
        return marker

    def target_marker(self, stamp):
        marker = self.base_marker(stamp, 40, "ergodic_target", Marker.SPHERE)
        marker.pose.position.x = float(self.target[0])
        marker.pose.position.y = float(self.target[1])
        marker.pose.position.z = float(self.flight_altitude)
        marker.scale = Vector3(x=0.28, y=0.28, z=0.28)
        marker.color.r = 0.05
        marker.color.g = 0.65
        marker.color.b = 1.0
        marker.color.a = 0.90
        return marker

    def path_marker(self, stamp):
        marker = self.base_marker(stamp, 50, "covered_trajectory", Marker.LINE_STRIP)
        marker.scale.x = 0.035
        marker.color.r = 0.05
        marker.color.g = 0.9
        marker.color.b = 0.35
        marker.color.a = 0.95
        marker.points = list(self.path)
        return marker

    def home_marker(self, stamp):
        marker = self.base_marker(stamp, 60, "home_start_x", Marker.LINE_LIST)
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

    def window_marker(self, stamp):
        marker = self.base_marker(stamp, 61, "detection_window_viewpoint", Marker.SPHERE)
        wx, wy = self.window_view_point()
        marker.pose.position.x = float(wx)
        marker.pose.position.y = float(wy)
        marker.pose.position.z = float(self.flight_altitude)
        marker.scale = Vector3(x=0.22, y=0.22, z=0.22)
        marker.color.r = 1.0
        marker.color.g = 0.35 if self.detection_announced else 0.9
        marker.color.b = 0.02
        marker.color.a = 0.95
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
    node = ErgodicCbfDemoNode()
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
