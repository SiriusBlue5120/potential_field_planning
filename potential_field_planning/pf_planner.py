import rclpy
from rclpy.node import Node
# importa el "tipo de mensaje" 
# 1)ros2 interface list |grep String;ros2 interface show std_msgs/msg/String
from geometry_msgs.msg import Twist, PoseStamped, TransformStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
import tf2_ros
import tf_transformations as tf
from tf_transformations import euler_from_quaternion, quaternion_from_euler
import tf2_geometry_msgs
import numpy as np


class PotentialFieldPlanner(Node):
    def __init__(self, userInput=False, usePlan=True):        
        super().__init__(node_name="pf_planner")

        # Logging
        self.verbose = True

        # Setting up buffer and transform listener
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Set frames
        self.robot_frame = 'base_link'
        self.world_frame = 'odom'

        # Robot position
        self.robot_pose = self.set_posestamped(
                        PoseStamped(),
                        [0.0, 0.0, 0.0], 
                        [0.0, 0.0, 0.0],
                        self.robot_frame
                        )

        # Twist command
        self.vel_command = Twist()

        # Linear velocity limits
        self.max_angular_vel = 1
        self.max_linear_vel = 1
        # self.creep_velocity = 0.25

        # Angular velocity limits
        self.angle_threshold = 0.05
        self.distance_threshold = 0.25

        # Set input goal wrt world_frame (odom)
        if userInput:
            pose_robile = [float(x) for x in \
                           input('Enter a pose as (x, y, theta):').split(',')]
        else:
            pose_robile = [4.0, 10.0, -1.0]

        assert len(pose_robile) == 3
        self.goal = self.set_posestamped(
                        PoseStamped(),
                        [pose_robile[0], pose_robile[1],            0.0], 
                        [           0.0,            0.0, pose_robile[2]],
                        self.world_frame
                        )
        
        # Creacion de publisher
        self.vel_publisher = self.create_publisher(
                Twist, "/cmd_vel", 
                10)
        self.control_step = 0.1
        self.timer = self.create_timer(self.control_step, self.control_loop)

        # Creacion de subscriber
        self.scan_subscriber = self.create_subscription(
                LaserScan, "/scan",
                self.process_scan,
                10)
        self.lidar_init = False
        self.laser_info= {}
        self.laser_cartesian_info = {}

        # Path subscriber
        self.usePlan = usePlan
        if self.usePlan:
            self.plan_subscriber = self.create_subscription(
                Path,
                "/plan",
                self.plan_callback,
                10
            )
        self.plan: np.ndarray

        ### TODO: Define states ###
        self.IDLE = 0
        self.INIT = 1
        self.TRAVEL_TO_GOAL = 2
        self.ALIGN_TO_GOAL = 3

        # State
        self.state = self.INIT

        # Constants for velocity field planning
        self.k_a = 1.0
        self.k_r = 0.1
        self.repulsor_threshold = 8.0


    def plan_callback(self, msg: Path):
        plan_length = len(msg.poses)

        # Each pose is x, y, theta
        # Theta may be ignored here though
        self.plan = np.zeros((plan_length, 3))

        for index, pose in enumerate(msg.poses):
            # pose: PoseStamped
            self.plan[index, 0] = pose.pose.position.x
            self.plan[index, 1] = pose.pose.position.y
            self.plan[index, 2] = 0.0

        return


    def process_scan(self, msg:LaserScan):
        '''
        Recieves scan data
        '''
        # If robot is starting up, get lidar sensor info
        if not self.lidar_init:
            self.laser_info = {}
            self.laser_info.update(
                {
                "frame_id": msg.header.frame_id,
                "range_min": msg.range_min,
                "range_max": msg.range_max,
                "angle_min": msg.angle_min,
                "angle_max": msg.angle_max,
                "angle_increment": msg.angle_increment,
                }
            )
            self.lidar_init = True

        self.scan_ranges = np.array(msg.ranges)
        
        self.scan_cartesian = self.convert_scan_to_cartesian(self.scan_ranges)
        self.scan_cartesian = self.transform_coordinates('base_link', \
                                            'base_laser_front_link', \
                                            self.scan_cartesian)


    def convert_scan_to_cartesian(self, scan_ranges:LaserScan.ranges):
        '''
        Converts scan point to the cartesian coordinate system
        '''
        scanPoints = []
        for index, range in enumerate(scan_ranges):
            theta = self.laser_info["angle_min"] + (index * self.laser_info["angle_increment"])
            point = [range * np.cos(theta), range * np.sin(theta), 0.0]
            scanPoints.append(point)

        return np.array(scanPoints)


    def get_homogenous_transformation(self, transform:TransformStamped):
        '''
        Return the equivalent homogenous transform of a TransformStamped object
        '''
        transformation_matrix = tf.quaternion_matrix([
                                    transform.transform.rotation.x,
                                    transform.transform.rotation.y,
                                    transform.transform.rotation.z,
                                    transform.transform.rotation.w,
                                    ])
        transformation_matrix[0,-1] = transform.transform.translation.x
        transformation_matrix[1,-1] = transform.transform.translation.y
        transformation_matrix[2,-1] = transform.transform.translation.z

        return transformation_matrix

    
    def transform_coordinates(self, target_frame, source_frame, points):
        '''
        Transforms points from source frame to target frame
        '''
        transform = self.calculate_transform(target_frame, source_frame)
        transformation_matrix = self.get_homogenous_transformation(transform)

        transformed_points = []
        for point in points:
            point = np.array([*point, 1.0])
            transformed_points.append(np.matmul(transformation_matrix, point)[0:3])

        return np.array(transformed_points)
    
    
    def transform_posestamped(self, pose_object:PoseStamped, frame_id):
        '''
        Transforms pose_object to frame_id
        '''
        transformed_pose = self.tf_buffer.transform(pose_object, frame_id, \
                                                    rclpy.duration.Duration(seconds=0.05))

        return transformed_pose
        
           
    def calculate_transform(self, target_frame, source_frame):
        '''
        Calculates the transform from the source frame to the target frame
        '''
        transform = self.tf_buffer.lookup_transform(target_frame, source_frame, \
                                                    rclpy.time.Time())
        
        if self.verbose:
            self.get_logger().info(f"transform btw target {target_frame} and " + \
                                   f"source {source_frame}: {transform}")

        return transform
    

    def set_posestamped(self, pose:PoseStamped, position, orientation_euler, frame_id):
        '''
        Sets the fields of a PoseStamped object
        '''
        pose.header.frame_id = frame_id
        
        pose.pose.position.x = position[0]
        pose.pose.position.y = position[1]
        pose.pose.position.z = position[2]

        orientation_quat = quaternion_from_euler(*orientation_euler)

        pose.pose.orientation.x = orientation_quat[0]
        pose.pose.orientation.y = orientation_quat[1]
        pose.pose.orientation.z = orientation_quat[2]
        pose.pose.orientation.w = orientation_quat[3]

        if self.verbose:
            self.get_logger().info(f"pose set: {pose}")

        return pose
    

    def get_position_from_posestamped(self, pose:PoseStamped):
        '''
        Indexes out position (x, y, z) from a PoseStamped object
        '''
        position = np.array([
            pose.pose.position.x, 
            pose.pose.position.y,
            pose.pose.position.z
        ])

        return position


    def attraction_vel_vect(self, robot_position, goal_position):
        v_att = - self.k_a * (robot_position - goal_position / \
                              np.linalg.norm(robot_position - goal_position))
        return v_att
    

    def repulsive_vel_vect(self, robot_position, repulsor_positions):
        total_repulsive_vel = np.zeros(3)

        repulsor_positions = np.array(repulsor_positions)
        if repulsor_positions.size == 3:
            repulsor_positions = repulsor_positions[np.newaxis]

        for repulsor_position in repulsor_positions:
            repulsor_vector = robot_position - repulsor_position
            norm_repulsor_vector = np.linalg.norm(repulsor_vector)

            repulsive_vel:np.ndarray = self.k_r * ((1 / norm_repulsor_vector) - \
                                        (1 / self.repulsor_threshold)) * \
                                        (1 / norm_repulsor_vector ** 2) * \
                                        repulsor_vector / norm_repulsor_vector
                        
            total_repulsive_vel += repulsive_vel.flatten()

        return total_repulsive_vel
    

    def control_loop(self):
        '''
        Runs the main control loop for the planner
        '''
        goal_transformed = self.transform_posestamped(self.goal, self.robot_frame)
        goal_position = self.get_position_from_posestamped(goal_transformed)
        robot_position = self.get_position_from_posestamped(self.robot_pose)
        distance_to_goal = np.linalg.norm(goal_position - robot_position)

        if self.verbose:
            self.get_logger().info(f"current state: {self.state}")
            self.get_logger().info(f"goal transformed: {goal_transformed}")
            self.get_logger().info(f"distance to goal: {distance_to_goal}")
            

        # TODO: Initialize robot / planner here somehow if anything needs to be 
        # done and switch to a different state
        if self.state == self.INIT:

            # Seems like there's nothing to do here
            self.state = self.TRAVEL_TO_GOAL


        if self.state == self.TRAVEL_TO_GOAL:
            if distance_to_goal > self.distance_threshold:
                detected_repulsor_mask = self.scan_ranges < self.repulsor_threshold
                repulsor_positions = self.scan_cartesian[np.where(detected_repulsor_mask)]

                attractor_vel = self.attraction_vel_vect(robot_position, 
                                                     goal_position)
                repulsor_vel = self.repulsive_vel_vect(robot_position,
                                                   repulsor_positions)
            
                effective_vel = attractor_vel + repulsor_vel

                velocity_heading = np.arctan2(effective_vel[1], effective_vel[0])
            
                self.vel_command.linear.x = effective_vel[0] \
                            if abs(effective_vel[0]) < self.max_linear_vel else \
                            self.max_linear_vel * np.sign(effective_vel[0])
                self.vel_command.angular.z = velocity_heading \
                            if abs(velocity_heading) < self.max_angular_vel else \
                            self.max_angular_vel * np.sign(velocity_heading)
                
                # # Creep forward if 'stuck'
                # if np.abs(self.vel_command.linear.x) < self.creep_velocity and \
                #                                 np.sign(self.vel_command.linear.x) == 1:
                #     self.get_logger().info(f"creeping in the direction of travel")
                #     self.vel_command.linear.x = self.creep_velocity * np.sign(self.vel_command.linear.x)
                
            else:
                self.vel_command.linear.x = 0.0
                self.vel_command.angular.z = 0.0
                self.state = self.ALIGN_TO_GOAL        
        

        if self.state == self.ALIGN_TO_GOAL:
            goal_heading = euler_from_quaternion([
                goal_transformed.pose.orientation.x,
                goal_transformed.pose.orientation.y,
                goal_transformed.pose.orientation.z,
                goal_transformed.pose.orientation.w,
            ])[2]

            if self.verbose:
                self.get_logger().info(f"goal heading error: {goal_heading}")

            if np.abs(goal_heading) > self.angle_threshold:
                self.vel_command.angular.z = self.max_angular_vel * \
                                        np.sign(goal_heading)
            else:
                self.vel_command.angular.z = 0.0

                self.state = self.IDLE


        if self.state == self.IDLE:
            if distance_to_goal > self.distance_threshold:

                self.state = self.TRAVEL_TO_GOAL  


        # Publish vel_command
        if self.verbose:
            self.get_logger().info(f"vel_command: {self.vel_command}")

        self.vel_publisher.publish(self.vel_command)

    
    def send_zero_velocity(self):
        self.vel_command.linear.x = 0.0
        self.vel_command.linear.y = 0.0
        self.vel_command.linear.z = 0.0

        self.vel_command.angular.x = 0.0
        self.vel_command.angular.y = 0.0
        self.vel_command.angular.z = 0.0

        self.vel_publisher.publish(self.vel_command)



def main(args=None)-> None:
    rclpy.init(args=args)

    node = PotentialFieldPlanner(userInput=False)

    try:
        while rclpy.ok():
            rclpy.spin_once(node)

    except KeyboardInterrupt:
        node.send_zero_velocity()

    finally:
        rclpy.shutdown()

    rclpy.spin(node)
    rclpy.shutdown()