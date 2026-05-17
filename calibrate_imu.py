import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
import json

from imu_calib import imu_calib


def plot_imu_data_and_standstill(imu_data, standstill_flags, times):
    fig, ax = plt.subplots(2, 1, figsize=(16, 9))
    ax[0].plot(times, imu_data[:, 0:3])
    ax[0].legend(["ax", "ay", "az"])

    ax[1].plot(times, imu_data[:, 3], label = "wx")
    ax[1].plot(times, imu_data[:, 4], label = "wx")
    ax[1].plot(times, imu_data[:, 5], label = "wx")

    standstill_changes = np.hstack((0, np.diff(standstill_flags)))
    motion_starts = np.where(standstill_changes == -1)
    motion_ends = np.where(standstill_changes == 1)
    motion_start_end_idxs = np.array([[s, e] for s, e in zip(*motion_starts, *motion_ends)], dtype=int)

    for xmin, xmax in motion_start_end_idxs:
        ax[0].axvspan(times[xmin], times[xmax], color='green', alpha=0.2)
        ax[1].axvspan(times[xmin], times[xmax], color='green', alpha=0.2)

    ax[1].legend()

    plt.show()

def plot_accelerations_before_and_after(accs, accs_calibrated, times):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(times, np.linalg.norm(accs, axis=1), alpha = 0.5)
    ax.plot(times, np.linalg.norm(accs_calibrated, axis=1), alpha = 0.5)
    ax.legend(["Uncalibrated norm", "Calibrated norm"])
    ax.set(xlabel='$time, s$', ylabel='$m/s^2$', ylim = [8.81, 10.81])
    plt.show()

def plot_gyro_before_and_after(accs, accs_calibrated, gyrs, gyrs_calibrated, dt, times, compare=None):
    p, q, v = imu_calib.evaluate_states(accs, gyrs, dt)
    p_, q_, v_ = imu_calib.evaluate_states(accs_calibrated, gyrs_calibrated, dt)
    p0, q0, v0 = imu_calib.evaluate_states(compare[:, 0:3], compare[:, 3:6], dt) if compare is not None else (None, None, None)

    fig, ax = plt.subplots(3, 2, figsize=(16, 9))
    ax[0, 0].plot(times, q[:, 1], label = 'uncalibrated orientation')
    ax[0, 0].plot(times, q_[:, 1], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated orientation')
    if q0 is not None:
        ax[0, 0].plot(times, q0[:, 1], marker = 'o', markersize = 2, linestyle = 'none', label = 'true orientation')
    ax[0, 0].legend()
    ax[0, 0].set(xlabel = "time", ylim = [-1.0, 1.0], title = "Calibration results for orientation q.x")

    ax[1, 0].plot(times, q[:, 2], label = 'uncalibrated orientation')
    ax[1, 0].plot(times, q_[:, 2], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated orientation')
    if q0 is not None:
        ax[1, 0].plot(times, q0[:, 2], marker = 'o', markersize = 2, linestyle = 'none', label = 'true orientation')
    ax[1, 0].legend()
    ax[1, 0].set(xlabel = "time", ylim = [-1.0, 1.0], title = "Calibration results for orientation q.y")

    ax[2, 0].plot(times, q[:, 3], label = 'uncalibrated orientation')
    ax[2, 0].plot(times, q_[:, 3], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated orientation')
    if q0 is not None:
        ax[2, 0].plot(times, q0[:, 3], marker = 'o', markersize = 2, linestyle = 'none', label = 'true orientation')
    ax[2, 0].legend()
    ax[2, 0].set(xlabel = "time", ylim = [-1.0, 1.0], title = "Calibration results for orientation q.z")

    ax[0, 1].plot(times, v[:, 0], label = 'uncalibrated velocity')
    ax[0, 1].plot(times, v_[:, 0], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated velocity')
    if v0 is not None:
        ax[0, 1].plot(times, v0[:, 0], marker = 'o', markersize = 2, linestyle = 'none', label = 'true velocity')
    ax[0, 1].legend()
    ax[0, 1].set(xlabel = "time", ylim = [-10.0, 10.0], title = "Calibration results for velocity v.x")

    ax[1, 1].plot(times, v[:, 1], label = 'uncalibrated velocity')
    ax[1, 1].plot(times, v_[:, 1], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated velocity')
    if v0 is not None:
        ax[1, 1].plot(times, v0[:, 1], marker = 'o', markersize = 2, linestyle = 'none', label = 'true velocity')
    ax[1, 1].legend()
    ax[1, 1].set(xlabel = "time", ylim = [-10.0, 10.0], title = "Calibration results for velocity v.y")

    ax[2, 1].plot(times, v[:, 2], label = 'uncalibrated velocity')
    ax[2, 1].plot(times, v_[:, 2], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated velocity')
    if v0 is not None:
        ax[2, 1].plot(times, v0[:, 2], marker = 'o', markersize = 2, linestyle = 'none', label = 'true velocity')
    ax[2, 1].legend()
    ax[2, 1].set(xlabel = "time", ylim = [-10.0, 10.0], title = "Calibration results for velocity v.z")

    plt.tight_layout()
    plt.show()

def main():
    parser = argparse.ArgumentParser(description = 'Run calibration on real data from IMU.')
    parser.add_argument('-i', '--imu', help = 'Path to file with data from IMU (accs gyrs times).',
        required = False, default = "data/imu_sim.log", type = str)
    parser.add_argument('-f', '--sampling-frequency', help = 'Sampling frequency for logfile, use if t is not given.', 
        required = False, type = float)
    parser.add_argument('-t', '--imu-true', help = 'Path to file with data of true values.',
        required = False, type = str)
    parser.add_argument('-o', '--calib', help = 'Path to output json file for calibration paramters.',
        required = False, default = "data/calib.json", type = str)
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'Plot report')
    args = parser.parse_args()

    np.set_printoptions(edgeitems=30, linewidth=1000, formatter={'float': '{: 0.4f}'.format})
    standstill_gyr_threshold = 0.13

    # read file with ax, ay, az, wx, wy, wz [, dt] measurements from IMU
    imu_data = np.genfromtxt(args.imu, delimiter=' ')
    standstill_flag = imu_calib.generate_standstill_flags(imu_data, standstill_gyr_threshold)
    accs, gyrs = imu_data[:,0:3], imu_data[:,3:6]
    dt = np.pad(np.diff(imu_data[:,6]), (1, 0), mode="edge") if imu_data.shape[1] == 7 else 1 / args.sampling_frequency
    times = np.arange(accs.shape[0]) * dt if np.isscalar(dt) else np.cumsum(dt)

    if args.verbose:
        plot_imu_data_and_standstill(imu_data, standstill_flag, times)

    # find accelerometer calibration parameters and calibrate accel measurements
    time_start = time.time()
    theta_acc = imu_calib.calib_acc(accs, standstill_flag)
    time_end = time.time()

    print("ACC calibration done in: ", time_end - time_start, "seconds")
    print("[ S_X     S_Y     S_Z     NO_X    NO_Y    NO_Z    B_X     B_Y     B_Z   ]")
    print(theta_acc)
    accs_calibrated = imu_calib.correct_acc(accs, theta_acc)
    if args.verbose:
        plot_accelerations_before_and_after(accs, accs_calibrated, times)

    # find gyroscope calibration parameters
    time_start = time.time()
    theta_gyr = imu_calib.calib_gyr(gyrs, accs_calibrated, dt, standstill_flag)
    time_end = time.time()

    print("GYR calibration done in: ", time_end - time_start, "seconds")
    print("[ S_X     S_Y     S_Z     NO_X    NO_Y    NO_Z    B_X     B_Y     B_Z     E_X     E_Y     E_Z  ]")
    print(theta_gyr)
    gyrs_calibrated = imu_calib.correct_gyr(gyrs, theta_gyr)

    if args.verbose:
        true_data = None
        if args.imu_true:
            true_data = np.genfromtxt(args.imu_true, delimiter=' ')
        plot_gyro_before_and_after(accs, accs_calibrated, gyrs, gyrs_calibrated, dt, times, compare=true_data)

    print("Write to file: ", args.calib)
    theta_json = {
        "theta_acc": {
            "scale": [float(theta_acc[0]), float(theta_acc[1]), float(theta_acc[2])],
            "nonorth": [float(theta_acc[3]), float(theta_acc[4]), float(theta_acc[5])],
            "bias": [float(theta_acc[6]), float(theta_acc[7]), float(theta_acc[8])],
        },
        "theta_gyr": {
            "scale": [float(theta_gyr[0]), float(theta_gyr[1]), float(theta_gyr[2])],
            "nonorth": [float(theta_gyr[3]), float(theta_gyr[4]), float(theta_gyr[5])],
            "bias": [float(theta_gyr[6]), float(theta_gyr[7]), float(theta_gyr[8])],
            "misalign": [float(theta_gyr[9]), float(theta_gyr[10]), float(theta_gyr[11])],
        },
    }
    with open(args.calib, "w") as output_file:
        json.dump(theta_json, output_file, indent=2)

if __name__ == '__main__':
    main()
