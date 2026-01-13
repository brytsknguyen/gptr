#!/usr/bin/env python3

import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker
from nav_msgs.msg import Odometry

# For quaternion multiplication
from tf_transformations import quaternion_multiply as quatmult
from tf_transformations import quaternion_inverse as quatinv

class STLPublisher(Node):
    def __init__(self):
        
        super().__init__('stl_publisher_node')
        
        # Publisher to the visualization_marker topic
        self.marker_publisher = self.create_publisher(Marker, '/cartinbot_marker', 10)
        
        # Subscriber to the input topic
        self.subscription = self.create_subscription(Odometry, '/lidar_0/odom', self.odom_callback, 10)
       
        self.get_logger().info("Odom to marker node has started.")

        # Initialize the marker
        self.marker = Marker()
        self.init_marker()

    def init_marker(self):
        # Set the frame of reference (usually "map" or "odom")
        self.marker.header.frame_id = "world"

        # Type of marker (MESH_RESOURCE for STL)
        self.marker.type = Marker.MESH_RESOURCE
        self.marker.action = Marker.ADD

        # Marker ID and lifetime (leave 0 for never expire)
        self.marker.id = 0
        # self.marker.lifetime = rclpy.Duration()

        # Set the mesh resource (path to the STL file)
        self.marker.mesh_resource = "package://gptr/scripts/cartinbot.stl"

        # Scale (adjust depending on your STL file size)
        self.marker.scale.x = 1.0
        self.marker.scale.y = 1.0
        self.marker.scale.z = 1.0

        # Color of the model
        self.marker.color.r = 0.0
        self.marker.color.g = 1.0
        self.marker.color.b = 0.0
        self.marker.color.a = 1.0

    def odom_callback(self, msg: Odometry):
        
        # Extract the position and orientation from the odometry message
        orientation = msg.pose.pose.orientation
        q_W_L = [orientation.x, orientation.y, orientation.z, orientation.w]
        q_B_L = [0, 0.38268343236, 0, 0.92387953251] # 45-degree pitch
        q_W_B = quatmult(q_W_L, quatinv(q_B_L))
        orientation.x = q_W_B[0]
        orientation.y = q_W_B[1]
        orientation.z = q_W_B[2]
        orientation.w = q_W_B[3]

        position = msg.pose.pose.position
        position.x = position.x
        position.y = position.y
        position.z = position.z

        # Rotate the vehicle
        # Set the marker's pose to the robot's current position and orientation
        self.marker.pose.position = position
        self.marker.pose.orientation = orientation

        # Update the timestamp for RViz
        self.marker.header.stamp = self.get_clock().now().to_msg()

        # Publish the marker to RViz
        self.marker_publisher.publish(self.marker)

def main(args=None):
    rclpy.init(args=args)
    node = STLPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
