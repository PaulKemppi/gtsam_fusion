#!/usr/bin/python3
""" FUSION core.
Provides funtionality to fuse IMU, GPS and 6DOF (absolute / relative) pose data using GTSAM's 
incremental smoothing and mapping based on the Bayes tree (ISAM2).
"""

from gtsam import gtsam
import numpy as np
from collections import deque
import heapq
import time

def gtsam_pose_to_numpy(gtsam_pose):
    """Convert GTSAM pose to numpy arrays 
    (position, orientation)"""
    position = np.array([
        gtsam_pose.x(),
        gtsam_pose.y(),
        gtsam_pose.z()])
    quat = gtsam_pose.rotation().quaternion()
    orientation = np.array([quat[1], quat[2], quat[3], quat[0]]) # xyzw
    return position, orientation

def numpy_pose_to_gtsam(position, orientation):
    """Convert numpy arrays (position, orientation)
    to GTSAM pose"""
    return gtsam.Pose3(
        gtsam.Rot3.Quaternion(
            orientation[3],
            orientation[0],
            orientation[1],
            orientation[2]),
        gtsam.Point3(
            position[0],
            position[1],
            position[2])
    )

class GtsamFusionCore():
    """Core functions for ISAM2 fusion."""

    def __init__(self, params):
        """Initialize ISAM2, IMU preintegration, and set prior factors"""
        self.__imu_measurements_predict = [] # IMU measurements for real-time pose prediction
        self.__imu_measurements_optimize = [] # IMU measurements for pose prediction between measurement updates
        self.__imu_samples = []
        self.__opt_measurements = [] # GPS + 6DOF measurement updates
        self.__opt_meas_buffer_time = params['opt_meas_buffer_time']
        # ISAM2 initialization
        isam2_params = gtsam.ISAM2Params()
        isam2_params.setRelinearizeThreshold(params['relinearize_th'])
        isam2_params.setRelinearizeSkip(params['relinearize_skip'])
        isam2_params.setFactorization(params['factorization'])
        self.__isam2 = gtsam.ISAM2(isam2_params)
        self.__graph = gtsam.NonlinearFactorGraph()
        self.__initial_estimate = gtsam.Values()
        self.__prev_relative_pose = None
        self.__prev_pose_key = None
        self.__min_imu_sample_count_for_integration = 2
        # ISAM2 keys
        self.__pose_key = gtsam.symbol(ord('x'), 0)
        self.__vel_key = gtsam.symbol(ord('v'), 0)
        self.__bias_key = gtsam.symbol(ord('b'), 0)
        # IMU preintegration
        self.__pre_integration_params = gtsam.PreintegrationParams(np.asarray(params['g']))
        self.__pre_integration_params.setAccelerometerCovariance(np.eye(3) * np.power(params['sigma_accelerometer'], 2))
        self.__pre_integration_params.setGyroscopeCovariance(np.eye(3) * np.power(params['sigma_gyroscope'], 2))
        self.__pre_integration_params.setIntegrationCovariance(np.eye(3) * params['sigma_integration']**2)
        self.__pre_integration_params.setUse2ndOrderCoriolis(params['use_2nd_order_coriolis'])
        self.__pre_integration_params.setOmegaCoriolis(np.array(params['omega_coriolis']))
        self.__pre_integration_params.setBodyPSensor(numpy_pose_to_gtsam(params['b2s_pos'], params['b2s_ori']))
        # initial state
        self.__current_time = 0
        self.__current_pose = numpy_pose_to_gtsam(
            params['init_pos'], params['init_ori']
        ) 
        self.__predicted_pose = self.__current_pose
        self.__current_vel = np.asarray(params['init_vel'])
        self.__current_bias = gtsam.imuBias_ConstantBias(
            np.asarray(params['init_acc_bias']),
            np.asarray(params['init_gyr_bias'])
        )
        # store for predict
        self.__last_opt_time = self.__current_time
        self.__last_opt_pose = self.__current_pose
        self.__last_opt_vel = self.__current_vel
        self.__last_opt_bias = self.__current_bias
        self.__imu_accum = gtsam.PreintegratedImuMeasurements(self.__pre_integration_params)
        # uncertainty of the initial state
        self.__sigma_init_pose = gtsam.noiseModel_Isotropic.Sigmas(np.array([
            params['sigma_init_ori'], params['sigma_init_ori'], params['sigma_init_ori'],
            params['sigma_init_pos'], params['sigma_init_pos'], params['sigma_init_pos']
        ]))
        self.__sigma_init_vel = gtsam.noiseModel_Isotropic.Sigmas(np.array([
            params['sigma_init_vel'], params['sigma_init_vel'], params['sigma_init_vel']
        ]))
        self.__sigma_init_bias = gtsam.noiseModel_Isotropic.Sigmas(np.array([
            params['sigma_acc_init_bias'], params['sigma_acc_init_bias'], params['sigma_acc_init_bias'],
            params['sigma_gyr_init_bias'], params['sigma_gyr_init_bias'], params['sigma_gyr_init_bias']
        ]))
        # measurement noise
        self.__pose_noise = gtsam.noiseModel_Isotropic.Sigmas(np.concatenate((
            params['sigma_pose_rot'],
            params['sigma_pose_pos']
        )))
        self.__gps_noise = gtsam.noiseModel_Isotropic.Sigmas(np.asarray(params['sigma_gps']))
        self.__bias_noise = gtsam.noiseModel_Isotropic.Sigmas(np.concatenate((
            params['sigma_acc_bias_evol'],
            params['sigma_gyr_bias_evol']
        )))
        # prior factors
        prior_pose_factor = gtsam.PriorFactorPose3(
            self.__pose_key, 
            self.__current_pose, 
            self.__sigma_init_pose)
        self.__graph.add(prior_pose_factor)
        prior_vel_factor = gtsam.PriorFactorVector(
            self.__vel_key,
            self.__current_vel,
            self.__sigma_init_vel)
        self.__graph.add(prior_vel_factor)
        prior_bias_factor = gtsam.PriorFactorConstantBias(
            self.__bias_key, 
            self.__current_bias,
            self.__sigma_init_bias)
        self.__graph.add(prior_bias_factor)
        # initial estimates
        self.__initial_estimate.insert(self.__pose_key, self.__current_pose)
        self.__initial_estimate.insert(self.__vel_key, self.__current_vel)
        self.__initial_estimate.insert(self.__bias_key, self.__current_bias)

    def add_relative_pose_measurement(self, time, position, orientation):
        """Add relative 6DOF pose measurement i.e.
        a between constraint is created from the previous relative pose to 
        this current pose 
        Input:
            position: np.array([x, y, z])
            orientation: np.array([rx, ry, rz, rw])
        Output:
            -
        """
        heapq.heappush(self.__opt_measurements, (time, 'relative_pose', position, orientation))
        buffer_time = heapq.nlargest(1, self.__opt_measurements)[0][0] - heapq.nsmallest(1, self.__opt_measurements)[0][0]
        if buffer_time >= self.__opt_meas_buffer_time:
            self.__trigger_update(heapq.heappop(self.__opt_measurements))

    def add_absolute_pose_measurement(self, time, position, orientation):
        """Add absolute 6DOF pose measurement
        Input:
            position: np.array([x, y, z])
            orientation: np.array([rx, ry, rz, rw])
        Output:
            -
        """
        heapq.heappush(self.__opt_measurements, (time, 'absolute_pose', position, orientation))
        buffer_time = heapq.nlargest(1, self.__opt_measurements)[0][0] - heapq.nsmallest(1, self.__opt_measurements)[0][0]
        if buffer_time >= self.__opt_meas_buffer_time:
            self.__trigger_update(heapq.heappop(self.__opt_measurements))

    def add_gps_measurement(self, time, position):
        """Add GPS (metric, planar) measurement
        Input:
            time: float [s]
            position: np.array([x, y, z])
        Output:
            -
        """
        heapq.heappush(self.__opt_measurements, (time, 'gps', position))
        buffer_time = heapq.nlargest(1, self.__opt_measurements)[0][0] - heapq.nsmallest(1, self.__opt_measurements)[0][0]
        if buffer_time >= self.__opt_meas_buffer_time:
            self.__trigger_update(heapq.heappop(self.__opt_measurements))

    def add_imu_measurement(self, time, linear_acceleration, angular_velocity, dt):
        """Add IMU measurement
        Input:
            linear_acceleration: np.array([x, y, z])
            angular_velocity: np.array([x, y, z])
        Output:
            position, orientation, velocity, acc_bias, gyr_bias: np.array([x, y, z]), np.array([rx, ry, rz, rw]), 
                   np.array([vx, vy, vz]), np.array([abx, aby, abz]), np.array([gbx, gby, gbz])
        """
        # Add measurement
        heapq.heappush(self.__imu_measurements_predict, (time, linear_acceleration, angular_velocity, dt))
        heapq.heappush(self.__imu_measurements_optimize, (time, linear_acceleration, angular_velocity, dt))
        # Process oldest sample
        return self.__imu_predict()

    def __trigger_update(self, measurement):
        """Trigger update based on the measurement type"""
        meas_time = measurement[0]
        meas_type = measurement[1]
        imu_samples = []
        while True:
            if not self.__imu_measurements_optimize:
                break
            imu_sample = heapq.heappop(self.__imu_measurements_optimize)
            if imu_sample[0] < meas_time:
                imu_samples.append(imu_sample)
            else:
                break
        if len(imu_samples) < self.__min_imu_sample_count_for_integration: 
            # Must have (at least 2) new IMU measurements since last meas update 
            # If not, put samples back and ignore this measurement
            for imu_sample in imu_samples:
                heapq.heappush(self.__imu_measurements_optimize, imu_sample)
            print("Ignoring measurement at: {}".format(meas_time))
            return 
        # else:
        #     print("IMU samples before measurement time: {}".format(len(imu_samples)))
        if meas_type == "relative_pose" and not self.__prev_relative_pose:
            pose = numpy_pose_to_gtsam(measurement[2], measurement[3])
            self.__prev_relative_pose = pose
            self.__prev_pose_key = self.__pose_key
        else:
            # new pose & velocity estimate
            self.__pose_key += 1
            self.__vel_key += 1
            self.__bias_key += 1
            if meas_type == "gps":
                position = measurement[2]
                # create new point3
                point = gtsam.Point3(
                    position[0],
                    position[1],
                    position[2])
                # add gps factor
                gps_factor = gtsam.GPSFactor(
                    self.__pose_key, 
                    point,
                    self.__gps_noise)
                self.__graph.add(gps_factor)
            elif meas_type == 'relative_pose':
                pose = numpy_pose_to_gtsam(measurement[2], measurement[3])
                # add between pose factor
                between_pose_factor = gtsam.BetweenFactorPose3(
                    self.__prev_pose_key, 
                    self.__pose_key, 
                    self.__prev_relative_pose.between(pose), 
                    self.__pose_noise)
                self.__graph.add(between_pose_factor)
                # store for next update
                self.__prev_relative_pose = pose
                self.__prev_pose_key = self.__pose_key
            elif meas_type == 'absolute_pose':
                pose = numpy_pose_to_gtsam(measurement[2], measurement[3])
                # add pose factor
                pose_factor = gtsam.PriorFactorPose3(
                    self.__pose_key,
                    pose,
                    self.__pose_noise
                )
                self.__graph.add(pose_factor)
            # optimize
            self.__optimize(meas_time, imu_samples)

    def __imu_predict(self):
        """Predict with IMU"""
        if self.__current_time > self.__last_opt_time: # when new optimized pose is available
            # store state
            self.__last_opt_time = self.__current_time
            self.__last_opt_pose = self.__current_pose
            self.__last_opt_vel = self.__current_vel
            self.__last_opt_bias = self.__current_bias
            # start new integration onwards from optimization time
            self.__imu_accum.resetIntegration()
            #print("Oldest IMU, newest IMU, opt: {}, {}, {}".format(imu_samples[0][0], imu_samples[-1][0], last_opt_time))
            new_imu_samples = []
            for sample in self.__imu_samples:
                if sample[0] > self.__last_opt_time:
                    self.__imu_accum.integrateMeasurement(sample[1], sample[2], sample[3])
                    new_imu_samples.append(sample)
            self.__imu_samples = new_imu_samples
        # get new sample from the queue
        (time, linear_acceleration, angular_velocity, dt) = heapq.heappop(self.__imu_measurements_predict)
        # store sample for re-integration after new measurement is available
        self.__imu_samples.append((time, linear_acceleration, angular_velocity, dt))
        # integrate
        self.__imu_accum.integrateMeasurement(linear_acceleration, angular_velocity, dt)
        # predict pose
        predicted_nav_state = self.__imu_accum.predict(
            gtsam.NavState(self.__last_opt_pose, self.__last_opt_vel), self.__last_opt_bias)
        pos, ori = gtsam_pose_to_numpy(predicted_nav_state.pose())
        return (
            pos,
            ori,
            self.__last_opt_vel,
            self.__last_opt_bias.accelerometer(),
            self.__last_opt_bias.gyroscope()
        )

    def __imu_update(self, imu_samples):
        """Create new IMU factor and perform bias evolution"""
        # reset integration done for the prediction
        imu_accum = gtsam.PreintegratedImuMeasurements(self.__pre_integration_params)
        # preintegrate IMU measurements up to meas_time
        for imu_sample in imu_samples:
            imu_accum.integrateMeasurement(imu_sample[1], imu_sample[2], imu_sample[3])
        # predict pose at meas_time for the optimization
        last_opt_pose = self.__current_pose
        last_opt_vel = self.__current_vel
        last_opt_bias = self.__current_bias
        # predict the pose using the last optimized state and current bias estimate
        predicted_nav_state = imu_accum.predict(
            gtsam.NavState(last_opt_pose, last_opt_vel), last_opt_bias)
        # add IMU factor
        imu_factor = gtsam.ImuFactor(
            self.__pose_key - 1, self.__vel_key - 1, 
            self.__pose_key, self.__vel_key, 
            self.__bias_key, imu_accum)
        self.__graph.add(imu_factor)
        return predicted_nav_state
        
    def __optimize(self, meas_time, imu_samples):
        """Perform optimization"""
        # perform IMU preintegration until meas_time
        predicted_nav_state = self.__imu_update(imu_samples)
        # add current pose to initial estimates
        self.__initial_estimate.insert(self.__pose_key, predicted_nav_state.pose())
        self.__initial_estimate.insert(self.__vel_key, predicted_nav_state.velocity())
        self.__initial_estimate.insert(self.__bias_key, self.__current_bias)
        # add factor for bias evolution
        bias_factor = gtsam.BetweenFactorConstantBias(
            self.__bias_key - 1, self.__bias_key, gtsam.imuBias_ConstantBias(), self.__bias_noise)
        self.__graph.add(bias_factor)
        result = self.__isam2_update()
        if result:
            # update current state
            self.__current_time = meas_time
            self.__current_pose = result.atPose3(self.__pose_key)
            self.__current_vel = result.atVector(self.__vel_key)
            self.__current_bias = result.atimuBias_ConstantBias(self.__bias_key)

    def __isam2_update(self):
        """ISAM2 update and pose estimation""" 
        result = None
        try:
            # update ISAM2
            self.__isam2.update(self.__graph, self.__initial_estimate)
            result = self.__isam2.calculateEstimate()
        except RuntimeError as e:
            print("Runtime error in optimization: {}".format(e))
        except IndexError as e:
            print("Index error in optimization: {}".format(e))
        except TypeError as e:
            print("Type error in optimization: {}".format(e))
        # reset
        self.__graph.resize(0)
        self.__initial_estimate.clear()
        return result