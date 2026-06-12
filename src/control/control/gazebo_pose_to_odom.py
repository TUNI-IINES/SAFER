#!/usr/bin/env python3

import rclpy
from rclpy.node import Node

from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseArray, TransformStamped
from tf2_ros import TransformBroadcaster


class GazeboPoseToOdom(Node):

    def __init__(self):
        super().__init__('gazebo_pose_to_odom')

        # params
        self.declare_parameter('model_name', 'anafi4k')
        self.declare_parameter('world_frame', 'world')
        self.declare_parameter('child_frame', 'anafi4k/base_link')
        self.declare_parameter('input_topic', '/gz/pose_info')
        self.declare_parameter('output_topic', '/anafi/odometry')
        self.declare_parameter('pose_index', -999)
        self.declare_parameter('expected_start_x', 0.0)
        self.declare_parameter('expected_start_y', 8.0)
        self.declare_parameter('expected_start_z', 0.5)

        self.model_name = self.get_parameter('model_name').value
        self.world_frame = self.get_parameter('world_frame').value
        self.child_frame = self.get_parameter('child_frame').value
        self.pose_index = int(self.get_parameter('pose_index').value)
        self.expected_start_x = float(self.get_parameter('expected_start_x').value)
        self.expected_start_y = float(self.get_parameter('expected_start_y').value)
        self.expected_start_z = float(self.get_parameter('expected_start_z').value)
        self.locked_pose_index = None

        input_topic = self.get_parameter('input_topic').value
        output_topic = self.get_parameter('output_topic').value

        self.odom_pub = self.create_publisher(Odometry, output_topic, 10)
        self.tf_broadcaster = TransformBroadcaster(self)

        self.create_subscription(
            PoseArray,
            input_topic,
            self.callback,
            10
        )

        self.get_logger().info("gazebo_pose_to_odom started")

    def callback(self, msg):

        if not msg.poses:
            return

        # The PoseArray bridge drops Gazebo entity names. Auto mode locks onto
        # the pose nearest the configured spawn point, then keeps that index.
        index = self.resolve_pose_index(msg)

        if index < 0 or index >= len(msg.poses):
            self.get_logger().warn(
                f"pose_index {self.pose_index} out of range for {len(msg.poses)} poses",
                throttle_duration_sec=2.0,
            )
            return

        pose = msg.poses[index]

        now = self.get_clock().now().to_msg()

        # -----------------------------
        # Odometry
        # -----------------------------
        odom = Odometry()
        odom.header.stamp = now
        odom.header.frame_id = self.world_frame
        odom.child_frame_id = self.child_frame

        odom.pose.pose = pose

        self.odom_pub.publish(odom)

        # -----------------------------
        # TF: world → base_link
        # -----------------------------
        t = TransformStamped()
        t.header.stamp = now
        t.header.frame_id = self.world_frame
        t.child_frame_id = self.child_frame

        t.transform.translation.x = pose.position.x
        t.transform.translation.y = pose.position.y
        t.transform.translation.z = pose.position.z
        t.transform.rotation = pose.orientation

        self.tf_broadcaster.sendTransform(t)

    def resolve_pose_index(self, msg):
        if self.pose_index != -999:
            index = self.pose_index
            if index < 0:
                index = len(msg.poses) + index
            return index

        if self.locked_pose_index is not None:
            return self.locked_pose_index

        best_index = 0
        best_dist_sq = float('inf')

        for i, pose in enumerate(msg.poses):
            dx = pose.position.x - self.expected_start_x
            dy = pose.position.y - self.expected_start_y
            dz = pose.position.z - self.expected_start_z
            dist_sq = dx * dx + dy * dy + dz * dz
            if dist_sq < best_dist_sq:
                best_dist_sq = dist_sq
                best_index = i

        self.locked_pose_index = best_index
        self.get_logger().info(
            f"Locked odometry to PoseArray index {best_index} near expected "
            f"spawn ({self.expected_start_x:.2f}, {self.expected_start_y:.2f}, {self.expected_start_z:.2f})."
        )
        return best_index


def main():
    rclpy.init()
    node = GazeboPoseToOdom()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
