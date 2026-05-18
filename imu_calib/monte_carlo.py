import numpy as np
import imu_calib
from imu_calib.utils import *


def generate_imu_measurements_between_orientations(q_from, q_to, N, dt):
    orientations = [slerp(q_from, q_to, fraction) for fraction in np.linspace(0.0, 1.0, N)]
    prev_orientations = [orientations[0], *orientations[:-1]]
    angular_velocities = [
        2.0 * quat_mul(quat_conj(q_prev), q - q_prev)[1:] / dt
        for q_prev, q in zip(prev_orientations, orientations)
    ]
    accelerations = [
        Rmat(q).T @ np.array([0.0, 0.0, imu_calib.g])
        for q in orientations
    ]
    return np.array(angular_velocities), np.array(accelerations)

def generate_ideal_imu_measurements_for_rotation_sequence(
        start_roll, start_yaw, start_pitch,
        yaw_increment, N_samples = 100,
        dt = 0.01, randomized = False
    ):

    # generate first orientation chunk while the sensor was motionless
    q0 = euler_to_quat(start_roll, start_yaw, start_pitch)
    gyrs, accs = generate_imu_measurements_between_orientations(q0, q0, N_samples, dt)
    times = np.arange(N_samples) * dt
    standstill = np.ones(len(gyrs))
    
    roll, yaw, pitch = start_roll, start_pitch, start_yaw
    roll_prev, yaw_prev, pitch_prev = start_roll, start_yaw, start_pitch

    for pitch in np.linspace(start_pitch + np.pi/4, start_pitch + np.pi, 4):
        for roll in np.linspace(start_roll, start_roll + 2 * np.pi, 6):
            if randomized:
                roll += np.deg2rad(np.random.randint(60)) * np.random.choice([-1, 0, 1])
                pitch += np.deg2rad(np.random.randint(45)) * np.random.choice([-1, 0, 1])
            yaw += yaw_increment
            
            # generate data through rotation
            q_from = euler_to_quat(roll_prev, yaw_prev, pitch_prev)
            q_to = euler_to_quat(roll, yaw, pitch)
            ang, acc = generate_imu_measurements_between_orientations(q_from, q_to, N_samples, dt)
            times = np.concatenate((times, times[-1] + np.arange(1, N_samples+1) * dt), axis=0)
            
            gyrs = np.concatenate((gyrs, ang), axis=0)
            accs = np.concatenate((accs, acc), axis=0)
            standstill = np.concatenate((standstill, np.zeros(N_samples)), axis=0)

            # generate data for standstill
            ang, acc = generate_imu_measurements_between_orientations(q_to, q_to, N_samples, dt)
            times = np.concatenate((times, times[-1] + np.arange(1, N_samples+1) * dt), axis=0)
            
            gyrs = np.concatenate((gyrs, ang), axis=0)
            accs = np.concatenate((accs, acc), axis=0)
            standstill = np.concatenate((standstill, np.ones(N_samples)), axis=0)
            
            roll_prev, pitch_prev, yaw_prev = roll, pitch, yaw

    return times, gyrs, accs, standstill

def corrupt_imu_measurements(theta_acc, theta_gyr, gyrs, accs, gyr_noise_std = 0.001, acc_noise_std = 0.04):
    gyrs = np.copy(gyrs)
    accs = np.copy(accs)
    # according to equations (1), (2), (5) and (6) in the paper
    Ma, Ba = imu_calib.sensor_error_model(theta_acc[0:9])
    Mg, Bg = imu_calib.sensor_error_model(theta_gyr[0:9])

    Rga = imu_calib.misalignment(theta_gyr[-3:])
    for i, _ in enumerate(gyrs):
        gyrs[i] = Rga @ (Mg @ gyrs[i] + Bg + np.random.normal(0, gyr_noise_std, 3))
        accs[i] = Ma @ (accs[i]) + Ba + np.random.normal(0, acc_noise_std, 3)
        
    return gyrs, accs


np.set_printoptions(edgeitems=30, linewidth=1000, formatter={'float': '{: 0.4f}'.format})


def monte_carlo_cycle(N_samples, dt, theta_acc, theta_gyr, randomized_rotations = True, gyr_noise_std = 0.004, acc_noise_std = 0.04):
    # start orientation
    roll, yaw, pitch = np.deg2rad([-150, -150, -75])
    yaw_increment = np.deg2rad(18)

    times, gyrs_true, accs_true, standstill_flags = generate_ideal_imu_measurements_for_rotation_sequence(
            roll, yaw, pitch, yaw_increment, N_samples, dt, randomized_rotations)

    # p_true, q_true, v_true = imu_calib.evaluate_states(accs_true, gyrs_true, dt, g, p0=None, q0=euler_to_quat(roll, pitch, yaw), v0=None)

    gyrs, accs = corrupt_imu_measurements(theta_acc, theta_gyr, gyrs_true, accs_true, gyr_noise_std, acc_noise_std)

    simulation_results = {
        "imu_true": {
            'accs': accs_true,
            'gyrs': gyrs_true,
            'times': times,
        },

        "imu": {
            'accs': accs,
            'gyrs': gyrs,
            'times': times,
        },
    }
    return simulation_results
