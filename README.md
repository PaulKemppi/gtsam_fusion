# gtsam_fusion
Estimates pose, velocity, and accelerometer / gyroscope biases by fusing GPS position and/or 6DOF pose with IMU data. The fusion is done using GTSAM's sparse nonlinear incremental optimization (ISAM2). The ROS (rospy) node is implemented using GTSAM's python3 inteface.
