import numpy as np
import argparse
import json

from imu_calib import monte_carlo


def main():
    parser = argparse.ArgumentParser(description = 'Run monte carlo simulations for accelerometer and gyroscope calibration.')
    parser.add_argument('-i', '--calib', help = 'Path to input json file of theta parameters, will generate one if not given.',
        required = False, type = str)
    parser.add_argument('-f', '--sampling-frequency', help = 'Sampling frequency for synthesized IMU measurements.', 
        required = False, default = 100, type = float)
    parser.add_argument('-r', '--randomize-rotations', help = 'Flag to whether or not randomize the rotations.',
        required = False, default = True, type = bool)
    parser.add_argument('-o', '--imu', help = 'Path to output csv file for generated measurement data (accs, gyrs, times).',
        required = False, default="data/imu_sim.log", type = str)
    parser.add_argument('-ot', '--imu-true', help = 'Path to output csv file for true state data (accs_true, gyrs_true, times).',
        required = False, default="data/imu_true.log", type = str)
    args = parser.parse_args()

    dt = 1 / args.sampling_frequency
    N_samples = 400
    randomize = args.randomize_rotations

    if args.calib:
        with open(args.calib, "r") as calib_output_file:
            theta_json = json.load(calib_output_file)
            theta_acc = np.array(theta_json["theta_acc"])
            theta_gyr = np.array(theta_json["theta_gyr"])

    else:
        theta_acc = (
            np.random.randint(100, size=9)
            /
            np.array([
                1000 * np.random.choice([-1,1]), 1000 * np.random.choice([-1,1]), 1000 * np.random.choice([-1,1]), 
                1000 * np.random.choice([-1,1]), 1000 * np.random.choice([-1,1]), 1000 * np.random.choice([-1,1]),
                100  * np.random.choice([-1,1]), 100  * np.random.choice([-1,1]), 100  * np.random.choice([-1,1])
            ])
        )
        theta_gyr = (
            np.random.randint(100, size=12)
            /
            np.array([
                10000 * np.random.choice([-1,1]), 10000 * np.random.choice([-1,1]), 10000 * np.random.choice([-1,1]),
                10000 * np.random.choice([-1,1]), 10000 * np.random.choice([-1,1]), 10000 * np.random.choice([-1,1]), 
                10000 * np.random.choice([-1,1]), 10000 * np.random.choice([-1,1]), 10000 * np.random.choice([-1,1]),
                10000 * np.random.choice([-1,1]), 10000 * np.random.choice([-1,1]), 10000 * np.random.choice([-1,1])
            ])
        )

    print("simulated theta_acc: ", theta_acc)
    print("simulated theta_gyr: ", theta_gyr)

    simulation_result = monte_carlo.monte_carlo_cycle(N_samples, dt, theta_acc, theta_gyr, randomize)
    
    print("Write to file: ", args.imu, args.imu_true)

    with open(args.imu, "w") as imu_output_file:
        for i in range(len(simulation_result["imu"]["accs"])):
            imu_output_file.write(" ".join(map("{:.06f}".format, [
                *simulation_result["imu"]["accs"][i],
                *simulation_result["imu"]["gyrs"][i],
                simulation_result["imu"]["times"][i],
            ])) + "\n")

    with open(args.imu_true, "w") as imu_true_output_file:
        for i in range(len(simulation_result["imu_true"]["accs"])):
            imu_true_output_file.write(" ".join(map("{:.06f}".format, [
                *simulation_result["imu_true"]["accs"][i],
                *simulation_result["imu_true"]["gyrs"][i],
                simulation_result["imu_true"]["times"][i],
            ])) + "\n")

if __name__ == '__main__':
    main()
