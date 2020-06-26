#!/usr/bin/python3
""" ROS interface to GTSAM FUSION.
If bag_file_path points to valid rosbag file, data is read from it as fast as possible,
otherwise subscribes to GPS, IMU, and 6DOF base pose topics.
Publishes PoseWithCovarianceStamped in topic base_pose_in_map_100hz.
"""

import rospy
import rosbag
from sensor_msgs.msg import NavSatFix, Imu
from geometry_msgs.msg import PoseWithCovarianceStamped
from tf import transformations
import tf2_ros

import numpy as np
import utm
import gtsam_fusion_core
import plots
import os

class GtsamFusionRos():
    """ROS interface to GTSAM FUSION"""

    def __init__(self):
        """Constructor"""
        rospy.init_node('gtsam_fusion_ros', anonymous=True)
        # ROS parameters
        self.pose_topic = rospy.get_param('~pose_topic', "/base_pose_in_map") # 6DOF pose messages
        self.gps_topic = rospy.get_param('~gps_topic', "/gps/fix") # GPS messages
        self.imu_topic = rospy.get_param('~imu_topic', "/roveo_mid_imu/sensor_imu") # IMU messages (only raw data is used, not the orientation)
        self.imu_frame = rospy.get_param('~imu_frame', 'imu_gyro') # moving frame indicating the IMU placement, z points at IMU z axis etc.
        self.base_frame = rospy.get_param('~base_frame', 'base_frame') # moving frame of the robot base that is tracked
        self.gps_frame = rospy.get_param('~gps_frame', 'gps') # moving frame indicating the GPS antenna placement
        self.world_frame = rospy.get_param('~world_frame', 'enu') # GPS measurements are in this global planar frame
        self.map_frame = rospy.get_param('~map_frame', 'map') # 6DOF pose measurements are in this local map frame
        self.bag_file_path = rospy.get_param('~bag_file_path', "") # if this path is set, data is read from the bag file via rosbag API as fast as possible
        self.gps_interval = rospy.get_param('~gps_interval', 0) # set zero to use all available GPS messages
        self.bag_secs_to_skip = rospy.get_param('~bag_secs_to_skip', 0.0) # skip data from the start when reading from bag
        self.origo = rospy.get_param('~origo', None) # UTM coordinates for the origo. If not set, the first GPS location is the origo
        self.fixed_dt = rospy.get_param('~fixed_dt', None) # use this to set fixed dt for IMU samples (instead dt calculated from the time stamps)
        self.use_gps = rospy.get_param('~use_gps', False) # enable / disable the use of GPS messages
        self.use_pose = rospy.get_param('~use_pose', False) # enable / disable the use of 6DOF pose messages
        self.plot_results = rospy.get_param('~plot_results', False) # plot results after the bag file is processed
        self.save_dir = rospy.get_param('~save_dir', '/tmp') # directory where the result images are saved
        # GTSAM FUSION parameters
        params = {}
        # minimum IMU sample count to accept measurement (GPS / 6DOF pose) for optimization
        params['opt_meas_buffer_time'] = rospy.get_param('~opt_meas_buffer_time', 0.3) # Buffer size in [s] for GPS / 6DOF pose measurements
        # optimization
        params['relinearize_th'] = rospy.get_param('~relinearize_th', 0.01) 
        params['relinearize_skip'] = rospy.get_param('~relinearize_skip', 1)
        params['factorization'] = rospy.get_param('~relinearize_skip', 'CHOLESKY')
        # IMU preintegration
        params['g'] = rospy.get_param('~g', [0, 0, -9.81])
        params['sigma_accelerometer'] = rospy.get_param('~sigma_accelerometer', [0.05, 0.05, 0.05]) # from calibration
        params['sigma_gyroscope'] = rospy.get_param('~sigma_gyroscope', [0.005, 0.005, 0.005]) # from calibration
        params['sigma_integration'] = rospy.get_param('~sigma_integration', 0) # error committed in integrating position from velocities
        params['use_2nd_order_coriolis'] = rospy.get_param('~use_2nd_order_coriolis', False)
        params['omega_coriolis'] = rospy.get_param('~omega_coriolis', [0, 0, 0])
        # initial state (default values assume that the robot is statioary at origo)
        params['init_pos'] = rospy.get_param('~init_pos', [0, 0, 0])
        params['init_ori'] = rospy.get_param('~init_ori', [0, 0, 0, 1])
        params['init_vel'] = rospy.get_param('~init_vel', [0, 0, 0])
        params['init_acc_bias'] = rospy.get_param('~init_acc_bias', [0, 0, 0])
        params['init_gyr_bias'] = rospy.get_param('~init_gyr_bias', [0, 0, 0])
        # uncertainty of the initial state
        params['sigma_init_pos'] = rospy.get_param('~sigma_init_pos', 10.0)
        params['sigma_init_ori'] = rospy.get_param('~sigma_init_ori', 2*np.pi)
        params['sigma_init_vel'] = rospy.get_param('~sigma_init_vel', 10.0)
        params['sigma_acc_init_bias'] = rospy.get_param('~sigma_acc_init_bias', 1.0)
        params['sigma_gyr_init_bias'] = rospy.get_param('~sigma_gyr_init_bias', 0.1)
        # measurement noise
        params['sigma_pose_pos'] = rospy.get_param('~sigma_pose_pos', [0.5, 0.5, 0.5]) # [m] error in 6DOF pose position
        params['sigma_pose_rot'] = rospy.get_param('~sigma_pose_rot', [np.inf, np.inf, 5.0/180.0*np.pi]) # rpy [rad] error in 6DOF pose rotation
        params['sigma_gps'] = rospy.get_param('~sigma_gps', [1, 1, 1]) # error in gps position
        params['sigma_acc_bias_evol'] = rospy.get_param('~sigma_acc_bias_evol', [1e-3, 1e-3, 1e-3]) 
        params['sigma_gyr_bias_evol'] = rospy.get_param('~sigma_gyr_bias_evol', [1e-5, 1e-5, 1e-5])
        # IMU body to sensor transform
        params['b2s_pos'] = rospy.get_param('~b2s_pos', [0, 0, 0])
        params['b2s_ori'] = rospy.get_param('~b2s_ori', [-4.59837371e-02, -1.34857154e-03, -6.20780829e-05,  9.98941276e-01])
        # variables
        self.is_initialized = False
        self.__last_imu_t = self.__last_gps_t = None
        if self.use_gps:
            self.__output_frame = self.world_frame
        else:
            self.__output_frame = self.map_frame
        self.__latest_base_pose = None
        if self.plot_results:
            if self.use_gps and self.use_pose:
                self.__fusion_name = "GPS + LIDAR + IMU"
            elif not self.use_gps:
                self.__fusion_name = "LIDAR + IMU"
            else:
                self.__fusion_name = "GPS + IMU"
            self.__results = {'IMU': [], 'GPS': [], 'LIDAR': [], self.__fusion_name: []}
        # get the static transforms
        self.__base_T_imu = self.__get_transform(self.imu_frame, self.base_frame)
        self.__gps_T_imu = self.__get_transform(self.imu_frame, self.gps_frame)
        if self.__base_T_imu.size == 0 or self.__gps_T_imu.size == 0:
            return
        self.__imu_T_base = transformations.inverse_matrix(self.__base_T_imu)
        # publishers
        self.__base_pose_in_map_100hz_pub = rospy.Publisher(
            "base_pose_in_map_100hz", PoseWithCovarianceStamped, queue_size=10)
        # Start fusion core
        self.__fusion_core = gtsam_fusion_core.GtsamFusionCore(params)
        self.is_initialized = True

    def run(self):
        """Either process bag file via rosbag API or subscribe to topics"""
        if self.bag_file_path: # use rosbag API
            rospy.loginfo("Processing file using rosbag API: {}. Please wait..".format(self.bag_file_path))
            if self.bag_secs_to_skip > 0:
                rospy.loginfo("Skipping {} seconds from the start of the bag file.".format(self.bag_secs_to_skip))
            bag = rosbag.Bag(self.bag_file_path)
            total_time_secs = int(bag.get_end_time() - bag.get_start_time())
            init_t = None
            last_info_time_secs = int(bag.get_start_time())
            for topic, msg, t in bag.read_messages(topics=[self.pose_topic, self.gps_topic, self.imu_topic]):
                if not init_t:
                    init_t = t
                    continue
                if (t - init_t).to_sec() < self.bag_secs_to_skip:
                    continue
                if rospy.is_shutdown():
                    break
                elapsed_time_secs = int(t.to_sec() - bag.get_start_time())
                if elapsed_time_secs % 100 == 0 and elapsed_time_secs != last_info_time_secs:
                    last_info_time_secs = elapsed_time_secs
                    rospy.loginfo("Elapsed time: {}/{} [s]".format(elapsed_time_secs, total_time_secs))
                if topic == self.gps_topic:
                    self.__gps_callback(msg)
                elif topic == self.pose_topic:
                    self.__pose_callback(msg)
                elif topic == self.imu_topic:
                    self.__imu_callback(msg)
            bag.close()
            rospy.loginfo("Bag processed.")
            if self.plot_results:
                rospy.loginfo("Preparing plots. Please wait..")
                if not os.path.exists(self.save_dir):
                    os.makedirs(self.save_dir)
                plots.plot_results(self.__results, self.__fusion_name, self.save_dir, self.use_gps, self.use_pose)
        else: # subscribe to topics
            self.gps_sub = rospy.Subscriber(
                self.gps_topic, NavSatFix, self.__gps_callback, queue_size=100)
            self.pose_sub = rospy.Subscriber(
                self.pose_topic, PoseWithCovarianceStamped, self.__pose_callback, queue_size=100)
            self.imu_sub = rospy.Subscriber(
                self.imu_topic, Imu, self.__imu_callback, queue_size=1000)
            rospy.spin()

    def __imu_callback(self, msg):
        """Handle IMU message"""
        if self.__last_imu_t:
            # Convert to numpy
            lin_acc = np.array([
                msg.linear_acceleration.x,
                msg.linear_acceleration.y,
                msg.linear_acceleration.z
            ])
            ang_vel = np.array([
                msg.angular_velocity.x,
                msg.angular_velocity.y,
                msg.angular_velocity.z
            ])
            dt = msg.header.stamp.to_sec() - self.__last_imu_t
            if self.fixed_dt:
                dt = self.fixed_dt
            # IMU update
            imu_pos, imu_ori, vel, acc_bias, gyr_bias = self.__fusion_core.add_imu_measurement(
                msg.header.stamp.to_sec(), lin_acc, ang_vel, dt)
            # convert pose from IMU frame to base frame
            base_pos, base_ori = self.__imu_frame_to_base_frame(imu_pos, imu_ori)
            # store internally
            self.__latest_base_pose = (base_pos, base_ori)
            # publish pose
            self.__publish_pose(
                msg.header.stamp,
                self.__output_frame,
                base_pos,
                base_ori
            )
            # data for plots
            if self.plot_results:
                # store input
                self.__results['IMU'].append(
                    np.concatenate((np.array([msg.header.stamp.to_sec(), dt]), lin_acc, ang_vel), axis=0))
                # store output
                euler_imu_ori = np.asarray(transformations.euler_from_quaternion(imu_ori)) / np.pi * 180.
                self.__results[self.__fusion_name].append(
                    np.concatenate((np.array([msg.header.stamp.to_sec()]), imu_pos, euler_imu_ori, vel, acc_bias, gyr_bias), axis=0))
        self.__last_imu_t = msg.header.stamp.to_sec()

    def __pose_callback(self, msg):
        """Handle 6DOF pose message"""
        base_pos = np.array([
            msg.pose.pose.position.x, 
            msg.pose.pose.position.y, 
            msg.pose.pose.position.z
        ])
        base_ori = np.array([
            msg.pose.pose.orientation.x, 
            msg.pose.pose.orientation.y, 
            msg.pose.pose.orientation.z,
            msg.pose.pose.orientation.w
        ])
        imu_pos, imu_ori = self.__base_frame_to_imu_frame(base_pos, base_ori)
        if self.plot_results:
            euler_imu_ori = np.asarray(transformations.euler_from_quaternion(imu_ori)) / np.pi * 180.
            self.__results['LIDAR'].append(
                np.concatenate((np.array([msg.header.stamp.to_sec()]), imu_pos, euler_imu_ori), axis=0))
        if not self.use_pose:
            return
        # use as relative pose update if GPS is also in use
        if self.use_gps:
            self.__fusion_core.add_relative_pose_measurement(msg.header.stamp.to_sec(), imu_pos, imu_ori)
        else: # use as absolute if GPS is not in use
            self.__fusion_core.add_absolute_pose_measurement(msg.header.stamp.to_sec(), imu_pos, imu_ori)

    def __gps_callback(self, msg):
        """Handle GPS message"""
        if (not self.__last_gps_t or 
            msg.header.stamp.to_sec() - self.__last_gps_t > self.gps_interval):
            gps_xyz = self.__gps_msg_to_enu_frame(msg)
            self.__last_gps_t = msg.header.stamp.to_sec()
            # TODO: GPS location in ENU frame is measured from gps frame
            # It should be converted to imu_gyro frame
            # Is it ok to use current base orientation to convert gps position to imu position?
            if not self.__latest_base_pose:
                return
            base_pos, base_ori =  self.__latest_base_pose
            (gps_imu_pos, gps_imu_ori) = self.__gps_frame_to_imu_vehicle_frame(gps_xyz, base_ori)
            if self.plot_results:
                self.__results['GPS'].append(
                    np.concatenate((np.array([msg.header.stamp.to_sec()]), gps_imu_pos, # gps_xyz)
                        np.array([msg.status.status, msg.position_covariance[0], msg.position_covariance[4], msg.position_covariance[8]])), axis=0))
            if self.use_gps:
                # perform GPS update
                self.__fusion_core.add_gps_measurement(self.__last_gps_t, gps_imu_pos) # gps_xyz)

    def __gps_msg_to_enu_frame(self, gps_msg):
        """Convert NavSatFix message to planar, metric coordinates.
        The first GPS point is used as origo if it is not defined."""
        # convert GPS fix to metric
        (x, y, zone_number, zone_letter) = utm.from_latlon(
            gps_msg.latitude,
            gps_msg.longitude)
        # set origo of not set
        if not self.origo:
            self.origo = [x, y, gps_msg.altitude]
        return np.array([
            x - self.origo[0], 
            y - self.origo[1], 
            gps_msg.altitude - self.origo[2]])

    def __base_frame_to_imu_frame(self, base_pos, base_ori):
        """Convert position and orientation from base frame to IMU vehicle frame"""
        map_T_base = self.__pose_to_rotation_matrix(base_pos, base_ori)
        map_T_imu = map_T_base.dot(self.__base_T_imu)
        return self.__pose_from_rotation_matrix(map_T_imu)

    def __imu_frame_to_base_frame(self, imu_pos, imu_ori):
        """Convert position and orientation from IMU vehicle frame to base frame"""
        map_T_imu = self.__pose_to_rotation_matrix(imu_pos, imu_ori)
        map_T_base = map_T_imu.dot(self.__imu_T_base)
        return self.__pose_from_rotation_matrix(map_T_base)

    def __gps_frame_to_imu_vehicle_frame(self, gps_pos, gps_ori):
        """Convert position from GPS frame to IMU vehicle frame"""
        map_T_gps = self.__pose_to_rotation_matrix(gps_pos, gps_ori)
        map_T_imu = map_T_gps.dot(self.__gps_T_imu)
        return self.__pose_from_rotation_matrix(map_T_imu)

    def __pose_to_rotation_matrix(self, pos, ori):
        """Form 4x4 rotation matrix from position and orientation (quaternion)"""
        R = transformations.quaternion_matrix(ori)
        R[0:3, 3] = pos
        return R
    
    def __pose_from_rotation_matrix(self, rotation_matrix):
        """Extract position and orientation from 4x4 rotation matrix"""
        return (transformations.translation_from_matrix(rotation_matrix), 
            transformations.quaternion_from_matrix(rotation_matrix))

    def __get_transform(self, source_frame, target_frame):
        """Use TF2 to get the transform between source frame and target frame"""
        # TF listener
        tfBuffer = tf2_ros.Buffer()
        listener = tf2_ros.TransformListener(tfBuffer)
        try:
            trans = tfBuffer.lookup_transform(target_frame, source_frame, rospy.Time().now(), rospy.Duration(2.0))
        except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
            rospy.logerr("Unable to lookup transform from {} to {}.".format(source_frame, target_frame))
            rospy.signal_shutdown("Required TF missing")
            return np.array([])
        # Convert to rotation matrix
        target_T_source = transformations.quaternion_matrix(np.array([
            trans.transform.rotation.x,
            trans.transform.rotation.y,
            trans.transform.rotation.z,
            trans.transform.rotation.w
        ]))
        target_T_source[0, 3] = trans.transform.translation.x
        target_T_source[1, 3] = trans.transform.translation.y
        target_T_source[2, 3] = trans.transform.translation.z
        return target_T_source

    def __publish_pose(self, stamp, frame_id, position, orientation):
        """Publish PoseWithCovarianceStamped"""
        msg = PoseWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = frame_id
        msg.pose.pose.position.x = position[0]
        msg.pose.pose.position.y = position[1]
        msg.pose.pose.position.z = position[2]
        msg.pose.pose.orientation.x = orientation[0]
        msg.pose.pose.orientation.y = orientation[1]
        msg.pose.pose.orientation.z = orientation[2]
        msg.pose.pose.orientation.w = orientation[3]
        self.__base_pose_in_map_100hz_pub.publish(msg)

def main():
    """Main"""
    node = GtsamFusionRos()
    if node.is_initialized:
        node.run()
    rospy.loginfo("Exiting..")

if __name__ == '__main__':
    main()