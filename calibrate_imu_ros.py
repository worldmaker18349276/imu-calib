#!/usr/bin/env python3

import rosbag
from pathlib import Path
import sys
import argparse
import calibrate_imu

def img_ros_to_csv(bag_path, topic, output_path):
    with rosbag.Bag(bag_path, "r") as bag, open(output_path, "w") as f:
        for _, msg, t in bag.read_messages(topics=[topic]):
            # acceleration
            ax = msg.linear_acceleration.x
            ay = msg.linear_acceleration.y
            az = msg.linear_acceleration.z

            # angular velocity
            wx = msg.angular_velocity.x
            wy = msg.angular_velocity.y
            wz = msg.angular_velocity.z

            # timestamp (seconds)
            ts = msg.header.stamp.to_sec()

            f.write(f"{ax} {ay} {az} {wx} {wy} {wz} {ts}\n")

def main():
    parser = argparse.ArgumentParser(description = 'Run calibration on real data from IMU.')
    parser.add_argument('-i', '--imu', help = 'Path to bag file with IMU data.', required = True, type = str)
    parser.add_argument('-t', '--topic', help = 'imu topic in the bag file.', required = True, type = str)
    parser.add_argument('-o', '--calib', help = 'Path to output json file for calibration paramters.', required = False, default = "data/calib.json", type = str)
    parser.add_argument('-v', '--verbose', action = 'store_true', help = 'Plot report')
    args = parser.parse_args()

    imu_csv_path = Path(args.imu).with_suffix(".log")
    img_ros_to_csv(args.imu, args.topic, imu_csv_path)
    sys.argv[1:] = ["--imu", str(imu_csv_path), "--calib", str(args.calib), *(["--verbose"] if args.verbose else [])]
    calibrate_imu.main()

if __name__ == '__main__':
    main()
