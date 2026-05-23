import numpy as np
import argparse
import time
import json
from pathlib import Path

from imu_calib import imu_calib, plot_utils

g = 9.81
motion_margn_sec = 0.5
standstill_gyr_threshold = 0.13

def main():
    parser = argparse.ArgumentParser(description = 'Run calibration on real data from IMU.')
    parser.add_argument('-i', '--imu', help = 'Path to file with data from IMU (accs gyrs times).',
        required = True, type = str)
    parser.add_argument('-f', '--sampling-frequency', help = 'Sampling frequency for logfile, use if t is not given.', 
        required = False, type = float)
    parser.add_argument('-o', '--calib', help = 'Path to output json file for calibration paramters.',
        required = False, default = None, type = str)
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'Plot report')
    args = parser.parse_args()

    # read file with ax, ay, az, wx, wy, wz [, dt] measurements from IMU
    imu_data = np.genfromtxt(args.imu, delimiter=' ')
    accs, gyrs = imu_data[:,0:3], imu_data[:,3:6]
    if imu_data.shape[1] == 7:
        dt = np.pad(np.diff(imu_data[:,6]), (0, 1), mode="edge")
    else:
        dt = np.repeat(1 / args.sampling_frequency, (imu_data.shape[0],))
    times = np.hstack((0.0, np.cumsum(dt)[:-1]))

    motion_margn_frame = int(motion_margn_sec / np.mean(dt))
    standstill_indices = imu_calib.generate_standstill_indices(imu_data, motion_margn_frame, standstill_gyr_threshold)

    if args.verbose:
        plot_utils.plot_imu_data_and_standstill(imu_data, standstill_indices, times)

    print(f"calibrate ACC...")

    # find accelerometer calibration parameters and calibrate accel measurements
    time_start = time.time()
    theta_acc, cov_theta_acc = imu_calib.calib_acc(accs, standstill_indices, g)
    time_end = time.time()

    print(f"ACC calibration done in: {time_end - time_start:.1f} s")
    print(imu_calib.explain(theta_acc, "ACC"))
    accs_calibrated = imu_calib.correct_acc(accs, theta_acc)
    if args.verbose:
        plot_utils.plot_accelerations_before_and_after(accs, accs_calibrated, times, g)

    print(f"calibrate GYR...")

    # find gyroscope calibration parameters
    time_start = time.time()
    theta_gyr, cov_theta_gyr = imu_calib.calib_gyr(gyrs, accs_calibrated, dt, standstill_indices)
    time_end = time.time()

    print(f"GYR calibration done in: {time_end - time_start:.1f} s")
    print(imu_calib.explain(theta_gyr, "GYR"))
    gyrs_calibrated = imu_calib.correct_gyr(gyrs, theta_gyr)

    if args.verbose:
        plot_utils.plot_gyroscope_before_and_after(gyrs, gyrs_calibrated, times)

    if args.verbose:
        plot_utils.plot_theta(theta_acc, theta_gyr, cov_theta_acc, cov_theta_gyr)

    if args.verbose:
        print("reconstruct calibration states...")
        p, q, v = imu_calib.evaluate_states_separately(accs, gyrs, dt, g, standstill_indices)
        p_, q_, v_ = imu_calib.evaluate_states_separately(accs_calibrated, gyrs_calibrated, dt, g, standstill_indices)
        plot_utils.plot_state_before_and_after(p, q, v, p_, q_, v_, standstill_indices, times)

    calib_path = args.calib if args.calib is not None else Path(args.imu).with_suffix(".json")
    print("Write to file: ", calib_path)
    theta_json = {
        "theta_acc": {
            "scale":   [float(theta_acc[0]), float(theta_acc[1]), float(theta_acc[2])],
            "oblique": [float(theta_acc[3]), float(theta_acc[4]), float(theta_acc[5])],
            "bias":    [float(theta_acc[6]), float(theta_acc[7]), float(theta_acc[8])],
        },
        "theta_gyr": {
            "scale":    [float(theta_gyr[0]), float(theta_gyr[1]), float(theta_gyr[2])],
            "oblique":  [float(theta_gyr[3]), float(theta_gyr[4]), float(theta_gyr[5])],
            "bias":     [float(theta_gyr[6]), float(theta_gyr[7]), float(theta_gyr[8])],
            "misalign": [float(theta_gyr[9]), float(theta_gyr[10]), float(theta_gyr[11])],
        },
    }
    with open(calib_path, "w") as output_file:
        json.dump(theta_json, output_file, indent=2)

if __name__ == '__main__':
    main()
