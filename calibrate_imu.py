import numpy as np
import argparse
import time
import json

from imu_calib import imu_calib, plot_utils


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
        plot_utils.plot_imu_data_and_standstill(imu_data, standstill_flag, times)

    # find accelerometer calibration parameters and calibrate accel measurements
    time_start = time.time()
    theta_acc = imu_calib.calib_acc(accs, standstill_flag)
    time_end = time.time()

    print("ACC calibration done in: ", time_end - time_start, "seconds")
    print("[ S_X     S_Y     S_Z     NO_X    NO_Y    NO_Z    B_X     B_Y     B_Z   ]")
    print(theta_acc)
    accs_calibrated = imu_calib.correct_acc(accs, theta_acc)
    if args.verbose:
        plot_utils.plot_accelerations_before_and_after(accs, accs_calibrated, times)

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
        plot_utils.plot_gyro_before_and_after(accs, accs_calibrated, gyrs, gyrs_calibrated, dt, times, compare=true_data)

        plot_utils.plot_theta(theta_acc, theta_gyr)

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
