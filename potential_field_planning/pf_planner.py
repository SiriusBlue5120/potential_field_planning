import time

import numpy as np
import rclpy
import tf2_ros
import tf_transformations as tf
# importa el "tipo de mensaje"
# 1)ros2 interface list |grep String;ros2 interface show std_msgs/msg/String
from geometry_msgs.msg import (PoseStamped, PoseWithCovarianceStamped,
                               TransformStamped, Twist)
from nav_msgs.msg import Path, OccupancyGrid
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class PotentialFieldPlanner(Node):
    def __init__(self, pose_topic="/pose", userInput=False, usePlan=True, behavior=False):
        super().__init__(node_name="pf_planner")

        # Path source
        self.usePlan = usePlan
        self.behavior = behavior

        # self.get_logger().info(f"Parameters: usePlan: {self.usePlan}"\
                            #    +f" behavior: {self.behavior}")

        # Logging
        self.verbose = False

        # Set frames
        self.robot_frame = 'base_link'
        self.world_frame = 'map' if usePlan else 'odom'
        self.laser_frame = 'base_laser_front_link'
        self.robot_offset = np.zeros(3)

        # Robot position
        self.robot_pose = self.set_posestamped(
                        PoseStamped(),
                        [0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0],
                        self.world_frame
                        )

        # Twist command
        self.vel_command = Twist()

        # Velocity limits
        self.max_linear_vel = 0.25
        # self.max_linear_vel = 0.5
        self.max_angular_vel = 0.4
        # self.max_angular_vel = 1.0
        # self.creep_velocity = 0.25

        # Constants for velocity field planning
        self.k_a = 30
        self.k_r = 0.1
        self.repulsor_threshold = 8.0

        # Acceleration limits
        self.max_linear_acc = 0.25
        
        self.SLOW_DOWN_DISTANCE = 0.3
        self.LOOKAHEAD_DISTANCE = 0.5

        self.angle_threshold = 0.2
        self.distance_threshold = 0.5

        self.stuck = False
        self.stuck_check_time = 5.0
        self.stuck_distance_threshold = self.max_linear_vel * self.stuck_check_time / 8
        self.previous_robot_wrt_world_position = np.zeros(3)
        self.previous_stuck_check = time.time()

        self.POSE_TOPIC = pose_topic

        # Set input goal wrt world_frame (odom)
        if userInput:
            goal_pose_world = [float(x) for x in \
                           input('Enter a pose as (x, y, theta):').split(',')]
        else:
            goal_pose_world = [4.0, 10.0, -1.0]

        assert len(goal_pose_world) == 3
        self.goal = self.set_posestamped(
                        PoseStamped(),
                        [goal_pose_world[0], goal_pose_world[1],            0.0],
                        [           0.0,            0.0, goal_pose_world[2]],
                        self.world_frame
                        )

        if not self.behavior:
            # Setting up buffer and transform listener
            self.tf_buffer = tf2_ros.Buffer()
            self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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
            
            self.map_subscriber = self.create_subscription(
                OccupancyGrid,
                "/map",
                self.map_callback,
                qos_profile=rclpy.qos.qos_profile_sensor_data,
            ) if self.usePlan else None

            self.pose_subscriber = self.create_subscription(
                PoseWithCovarianceStamped,
                self.POSE_TOPIC,
                self.pose_callback,
                10
            ) if self.usePlan else None

            # Path subscriber
            self.plan_subscriber = self.create_subscription(
                Path,
                "/plan",
                self.plan_callback,
                10
            ) if self.usePlan else None
        else:
            self.static_laser_transform: TransformStamped = None

        self.lidar_init = False
        self.laser_info= {}

        self.plan: np.ndarray = None
        self.lookahead_waypoint = None

        self.previous_loop_time = None
        self.previous_vel_command = None

        ### TODO: Define states ###
        self.IDLE = 0
        self.INIT = 1
        self.TRAVEL_TO_GOAL = 2
        self.ALIGN_TO_GOAL = 3

        # State
        self.state = self.INIT


    def map_callback(self, msg: OccupancyGrid):

        pass

        return


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

        if self.verbose:
            self.get_logger().info(f"plan of length {self.plan.shape[0]} received")

        return


    def pose_callback(self, msg:PoseWithCovarianceStamped):
        self.robot_pose.header.frame_id = msg.header.frame_id
        self.robot_pose.header.stamp = msg.header.stamp

        self.robot_pose.pose.position.x = msg.pose.pose.position.x
        self.robot_pose.pose.position.y = msg.pose.pose.position.y
        self.robot_pose.pose.position.z = msg.pose.pose.position.z

        self.robot_pose.pose.orientation.x = msg.pose.pose.orientation.x
        self.robot_pose.pose.orientation.y = msg.pose.pose.orientation.y
        self.robot_pose.pose.orientation.z = msg.pose.pose.orientation.z
        self.robot_pose.pose.orientation.w = msg.pose.pose.orientation.w

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

            if self.verbose:
                self.get_logger().info(f"Laser info obtained: \n{self.laser_info}")

        self.scan_ranges = np.array(msg.ranges)

        self.scan_cartesian = self.convert_scan_to_cartesian(self.scan_ranges)

        if not self.behavior:
            self.scan_cartesian = self.transform_coordinates('base_link', \
                                            'base_laser_front_link', \
                                            self.scan_cartesian)

        else:
            laser_to_base_transformation = \
                self.get_homogeneous_transformation_from_transform(self.static_laser_transform)
            self.scan_cartesian = \
                self.transform_coordinates_with_homogeneous_matrix(laser_to_base_transformation, \
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


    def get_homogeneous_transformation_from_transform(self, transform:TransformStamped):
        '''
        Return the equivalent homogeneous transform of a TransformStamped object
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


    def get_homogeneous_transform_from_posestamped(self, pose:PoseStamped):
        transformation_matrix = tf.quaternion_matrix([
                                        pose.pose.orientation.x,
                                        pose.pose.orientation.y,
                                        pose.pose.orientation.z,
                                        pose.pose.orientation.w,
                                    ])
        transformation_matrix[0,-1] = pose.pose.position.x
        transformation_matrix[1,-1] = pose.pose.position.y
        transformation_matrix[2,-1] = pose.pose.position.z

        return transformation_matrix


    def transform_coordinates(self, target_frame, source_frame, points):
        '''
        Transforms points from source frame to target frame
        '''
        transform = self.calculate_transform(target_frame, source_frame)
        transformation_matrix = self.get_homogeneous_transformation_from_transform(transform)

        transformed_points = []
        for point in points:
            point = np.array([*point, 1.0])
            transformed_points.append(np.matmul(transformation_matrix, point)[0:3])

        return np.array(transformed_points)


    def transform_coordinates_with_homogeneous_matrix(self, transformation_matrix, points):
        '''
        Transforms points from source frame to target frame
        '''
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
                                                    rclpy.time.Time(), timeout=rclpy.time.Duration(seconds=2.0))

        if self.verbose and False:
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

        orientation_quat = tf.quaternion_from_euler(*orientation_euler)

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


    def get_homogeneous_transform(self, position, orientation_euler):
        transformation_matrix = tf.euler_matrix(*orientation_euler)
        transformation_matrix[:-1,-1] = position

        return transformation_matrix


    def get_position_from_homogeneous_matrix(self, homogeneous_matrix: np.ndarray):
        position = homogeneous_matrix[:-1, -1]

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


    def get_lookahead_waypoint(self, robot_position, plan):
        plan_positions = plan[:, :-1]
        robot_position_2D = robot_position[:-1]

        diffs = plan_positions - robot_position_2D
        distances = np.linalg.norm(diffs, axis=1)
        # self.get_logger().info(f"distances: {distances}")

        current_index = np.argmin(distances)
        # self.get_logger().info(f"current_index: {current_index}")

        distance_mask = distances[current_index:] > self.LOOKAHEAD_DISTANCE
        # self.get_logger().info(f"distance_mask: {distance_mask}")

        if not np.sum(distance_mask, dtype=np.int64):
            lookahead_waypoint = plan[-1, :]

        else:
            # self.get_logger().info(f"condition: {np.argwhere(distances[current_index:] > self.LOOKAHEAD_DISTANCE)}")
            lookahead_index = np.argwhere(distances[current_index:] > self.LOOKAHEAD_DISTANCE)[0]

            lookahead_waypoint: np.ndarray = plan[current_index + lookahead_index]

        # self.get_logger().info(f"lookahead_waypoint: {lookahead_waypoint}")
        self.lookahead_waypoint = lookahead_waypoint.squeeze()

        return lookahead_waypoint


    def control_loop(self):
        '''
        Runs the main control loop for the planner
        '''
        if not self.lidar_init:
            return

        if self.usePlan:
            if self.plan is None or not self.plan.shape[0]:
                return

            robot_wrt_world_position = self.get_position_from_posestamped(self.robot_pose)
            robot_wrt_world_homogeneous = \
                self.get_homogeneous_transform_from_posestamped(self.robot_pose)
            world_wrt_robot_homogeneous = np.linalg.pinv(robot_wrt_world_homogeneous)

            goal_wrt_world_position = self.get_lookahead_waypoint(robot_wrt_world_position, self.plan)
            goal_wrt_world_homogeneous = self.get_homogeneous_transform(goal_wrt_world_position, np.zeros(3))
            goal_wrt_robot_homogeneous = world_wrt_robot_homogeneous @ \
                goal_wrt_world_homogeneous

            robot_position = self.robot_offset
            goal_position = \
                self.get_position_from_homogeneous_matrix(goal_wrt_robot_homogeneous)
            distance_to_goal = np.linalg.norm(goal_position - robot_position)

            # self.get_logger().info(f"distance_to_goal: {distance_to_goal}")

            # Check if stuck
            if (time.time() - self.previous_stuck_check) > self.stuck_check_time:
                if np.linalg.norm(robot_wrt_world_position - self.previous_robot_wrt_world_position) \
                    < self.stuck_distance_threshold:
                    self.stuck = True

                self.previous_stuck_check = time.time()
                self.previous_robot_wrt_world_position = robot_wrt_world_position


        else:
            goal_transformed = self.transform_posestamped(self.goal, self.robot_frame)
            goal_position = self.get_position_from_posestamped(goal_transformed)
            robot_position = self.get_position_from_posestamped(self.robot_pose)
            distance_to_goal = np.linalg.norm(goal_position - robot_position)

        if self.verbose:
            self.get_logger().info(f"current state: {self.state}")
            self.get_logger().info(f"goal transformed: {goal_position if self.usePlan else goal_transformed}")
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

                if distance_to_goal < self.SLOW_DOWN_DISTANCE:
                    self.vel_command.linear.x *= (distance_to_goal / self.SLOW_DOWN_DISTANCE)

            else:
                self.vel_command.linear.x = 0.0
                self.vel_command.angular.z = 0.0
                self.state = self.ALIGN_TO_GOAL


        if self.state == self.ALIGN_TO_GOAL:
            if self.usePlan:
                self.state = self.IDLE

                return

            goal_heading = tf.euler_from_quaternion([
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
            self.zero_velocity()

            if distance_to_goal > self.distance_threshold:

                self.state = self.TRAVEL_TO_GOAL

        # Limiting acceleration
        if self.previous_loop_time is None:
            self.previous_loop_time = time.time()
            self.previous_linear_vel_command = float(self.vel_command.linear.x)

        else:
            current_time = time.time()

            # self.get_logger().info(f"Velocity: linear x = {self.vel_command.linear.x}")

            diff = self.vel_command.linear.x - self.previous_linear_vel_command

            max_linear_vel_diff = self.max_linear_acc * (current_time - self.previous_loop_time)

            # self.get_logger().info(f"max_linear_vel_diff: linear x = {max_linear_vel_diff}")

            if np.abs(diff) > max_linear_vel_diff:
                self.vel_command.linear.x = self.previous_linear_vel_command + \
                    (np.sign(diff) * max_linear_vel_diff)
                
                # self.get_logger().info(f"Limited velocity: linear x = {self.vel_command.linear.x}")
            
            self.previous_loop_time = current_time
            self.previous_linear_vel_command = float(self.vel_command.linear.x)
        # Acceleration limited

        # Publish vel_command
        if self.verbose:
            self.get_logger().info(f"vel_command: {self.vel_command}")

        if not self.behavior:
            self.vel_publisher.publish(self.vel_command)


    def zero_velocity(self):
        self.vel_command.linear.x = 0.0
        self.vel_command.linear.y = 0.0
        self.vel_command.linear.z = 0.0

        self.vel_command.angular.x = 0.0
        self.vel_command.angular.y = 0.0
        self.vel_command.angular.z = 0.0


    def send_zero_velocity(self):
        self.zero_velocity()

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