#!/usr/bin/env python3  
"""Plot input and output of the GTSAM FUSION"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib as mpl
from matplotlib.colors import ListedColormap
import os

def plot_results(data, fusion_name, save_dir, use_gps, use_pose):
    """Input is a dict containing the input and output in IMU frame"""
    # convert list to np array for easier operation
    data_np = {}
    for meas_type, samples in data.items():
        if samples:
            data_np[meas_type] = np.asarray(samples)
    if not data_np:
        print("Nothing to plot..")
        return 
    # find min time
    min_time = None
    for meas_type, sample_array in data_np.items():
        if not min_time or sample_array[0, 0] < min_time:
            min_time = sample_array[0, 0]
    # roll
    plt.figure(figsize=(8, 8))
    legends = []
    for meas_type, sample_array in data_np.items():
        if meas_type == fusion_name:
            plt.plot(sample_array[:, 0] - min_time, sample_array[:, 4], '-', markersize=1)
            legends.append(meas_type)
        elif meas_type == "LIDAR" and use_pose:
            plt.plot(sample_array[:, 0] - min_time, sample_array[:, 4], '-', markersize=1)
            legends.append(meas_type)
        elif meas_type == "IMU":
            roll_vehicle = 180 * np.arctan2(sample_array[:, 3], np.sqrt(np.power(sample_array[:, 2], 2) + np.power(sample_array[:, 4], 2))) / np.pi
            plt.plot(sample_array[:, 0] - min_time, roll_vehicle, '-', markersize=1)
            legends.append("IMU")
    plt.xlabel('time [s]')
    plt.ylabel('roll [deg]')
    plt.title('Roll (IMU frame)')
    plt.legend(legends)
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'roll.png'))
    # pitch
    plt.figure(figsize=(8, 8))
    legends = []
    for meas_type, sample_array in data_np.items():
        if meas_type == fusion_name:
            plt.plot(sample_array[:, 0] - min_time, sample_array[:, 5], '-', markersize=1)
            legends.append(meas_type)
        elif meas_type == "LIDAR" and use_pose:
            plt.plot(sample_array[:, 0] - min_time, sample_array[:, 5], '-', markersize=1)
            legends.append(meas_type)
        elif meas_type == "IMU":
            pitch_vehicle = 180 * np.arctan2(-sample_array[:, 2], np.sqrt(np.power(sample_array[:, 3], 2) + np.power(sample_array[:, 4], 2))) / np.pi
            plt.plot(sample_array[:, 0] - min_time, pitch_vehicle, '-', markersize=1)
            legends.append("IMU")
    plt.xlabel('time [s]')
    plt.ylabel('pitch [deg]')
    plt.title('Pitch (IMU frame)')
    plt.legend(legends)
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'pitch.png'))
    # heading
    plt.figure(figsize=(8, 8))
    legends = []
    for meas_type, sample_array in data_np.items():
        if meas_type == fusion_name:
            plt.plot(sample_array[:, 0] - min_time, sample_array[:, 6] + 180.0, '-', markersize=1)
            legends.append(meas_type)
        elif meas_type == "LIDAR" and use_pose:
            plt.plot(sample_array[:, 0] - min_time, sample_array[:, 6] + 180.0, '-', markersize=1)
            legends.append(meas_type)
    plt.xlabel('time [s]')
    plt.ylabel('heading [deg]')
    plt.title('Heading (IMU frame)')
    plt.legend(legends)
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'heading.png'))
    # gps status
    if 'GPS' in data_np and use_gps:
        col_dict={-1: "black", 0: "blue", 1: "red", 2: 'green'}
        cm = ListedColormap([col_dict[x] for x in col_dict.keys()])
        labels = np.array(['NO FIX', 'FIX', 'SBAS FIX', 'GBAS FIX'])
        len_lab = len(labels)
        norm_bins = np.sort([*col_dict.keys()]) + 0.5
        norm_bins = np.insert(norm_bins, 0, np.min(norm_bins) - 1.0)
        norm = mpl.colors.BoundaryNorm(norm_bins, len_lab, clip=True)
        fmt = mpl.ticker.FuncFormatter(lambda x, pos: labels[norm(x)])
        diff = norm_bins[1:] - norm_bins[:-1]
        tickz = norm_bins[:-1] + diff / 2
        plt.figure(figsize=(8, 8))
        legends = ['NO FIX', 'FIX', 'SBAS FIX', 'GBAS FIX']
        for meas_type, sample_array in data_np.items():
            if meas_type == "GPS":
                plt.scatter(sample_array[:, 1], sample_array[:, 2], s=np.sqrt(sample_array[:, 5])*100, 
                    c=sample_array[:, 4].astype(int), cmap=cm, norm=norm, alpha=1.0)
        plt.colorbar(format=fmt, ticks=tickz)
        plt.xlabel('x [m]')
        plt.ylabel('y [m]')
        plt.title('GPS Status')
        plt.grid(True)
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'gps_status.png'))
    # xy
    plt.figure(figsize=(8, 8))
    legends = []
    for meas_type, sample_array in data_np.items():
        if meas_type == fusion_name:
            plt.plot(sample_array[:, 1], sample_array[:, 2], '.', markersize=1)
            legends.append(meas_type)
        elif (meas_type == "GPS" and use_gps) or (meas_type == "LIDAR" and use_pose):
            plt.plot(sample_array[:, 1], sample_array[:, 2], 'o', markersize=3)
            legends.append(meas_type)
    plt.xlabel('x [m]')
    plt.ylabel('y [m]')
    plt.title('Horizontal location')
    plt.legend(legends)
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'xy.png'))
    # z
    plt.figure(figsize=(8, 8))
    legends = []
    for meas_type, sample_array in data_np.items():
        if meas_type == fusion_name:
            plt.plot(sample_array[:, 0] - min_time, sample_array[:, 3], '.', markersize=1)
            legends.append(meas_type)
        elif (meas_type == "GPS" and use_gps) or (meas_type == "LIDAR" and use_pose):
            plt.plot(sample_array[:, 0] - min_time, sample_array[:, 3], '.', markersize=1)
            legends.append(meas_type)
    plt.xlabel('time [s]')
    plt.ylabel('z [m]')
    plt.title('Vertical location')
    plt.legend(legends)
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'z.png'))
     # acc bias
    plt.figure(figsize=(8, 8))
    plt.plot(data_np[fusion_name][:, 0] - min_time, data_np[fusion_name][:, 10:13], '.', markersize=1)
    plt.xlabel('time [s]')
    plt.ylabel('bias [g]')
    plt.title('Accelerometer Bias')
    plt.legend(['x', 'y', 'z'])
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'acc_bias.png'))
    # gyr bias
    plt.figure(figsize=(8, 8))
    plt.plot(data_np[fusion_name][:, 0] - min_time, data_np[fusion_name][:, 13:16], '.', markersize=1)
    plt.xlabel('time [s]')
    plt.ylabel('bias [rad/s]')
    plt.title('Gyroscope Bias')
    plt.legend(['x', 'y', 'z'])
    plt.grid(True)
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'gyr_bias.png'))
    # show plots
    plt.show()
