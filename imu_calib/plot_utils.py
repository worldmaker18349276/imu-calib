import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from imu_calib import imu_calib


def plot_imu_data_and_standstill(imu_data, standstill_flags, times):
    fig, ax = plt.subplots(2, 1, figsize=(16, 9))
    ax[0].plot(times, imu_data[:, 0:3])
    ax[0].legend(["ax", "ay", "az"])
    ax[1].plot(times, imu_data[:, 3:6])
    ax[1].legend(["wx", "wy", "wz"])

    standstill_changes = np.hstack((0, np.diff(standstill_flags)))
    motion_starts = np.where(standstill_changes == -1)
    motion_ends = np.where(standstill_changes == 1)
    motion_start_end_idxs = np.array([[s, e] for s, e in zip(*motion_starts, *motion_ends)], dtype=int)

    for xmin, xmax in motion_start_end_idxs:
        ax[0].axvspan(times[xmin], times[xmax], color='green', alpha=0.2)
        ax[1].axvspan(times[xmin], times[xmax], color='green', alpha=0.2)

    plt.show()

def plot_accelerations_before_and_after(accs, accs_calibrated, times):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(times, np.linalg.norm(accs, axis=1), alpha = 0.5)
    ax.plot(times, np.linalg.norm(accs_calibrated, axis=1), alpha = 0.5)
    ax.legend(["Uncalibrated norm", "Calibrated norm"])
    ax.set(xlabel='$time, s$', ylabel='$m/s^2$', ylim = [8.81, 10.81])
    plt.show()

def plot_gyro_before_and_after(accs, accs_calibrated, gyrs, gyrs_calibrated, dt, g, times, compare=None):
    p, q, v = imu_calib.evaluate_states(accs, gyrs, dt, g)
    p_, q_, v_ = imu_calib.evaluate_states(accs_calibrated, gyrs_calibrated, dt, g)
    p0, q0, v0 = imu_calib.evaluate_states(compare[:, 0:3], compare[:, 3:6], dt, g) if compare is not None else (None, None, None)

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

def plot_theta(theta_acc, theta_gyr):
    def draw_axis(ax, v0, v1, color, alpha):
        ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], color=color, alpha=alpha)

    ax_colors = ['red', 'green', 'blue']

    fig = plt.figure(figsize=(14, 6))
    fig.suptitle('IMU Axis Calibration')

    for idx, (theta, title) in enumerate([(theta_acc, 'Accelerometer'), (theta_gyr, 'Gyroscope')]):
        ax = fig.add_subplot(1, 2, idx+1, projection='3d')
        ax.set_title(title)
        
        o = np.array([0., 0., 0.])
        x = np.array([1., 0., 0.])
        y = np.array([0., 1., 0.])
        z = np.array([0., 0., 1.])

        o_ = imu_calib.distort(theta, o)
        x_ = imu_calib.distort(theta, x)
        y_ = imu_calib.distort(theta, y)
        z_ = imu_calib.distort(theta, z)

        # calibrated
        for v, color in zip([x, y, z], ax_colors):
            draw_axis(ax, o, v, color, 0.5)

        for v, color in zip([x, y, z], ax_colors):
            draw_axis(ax, o_ + o, o_ + v, color, 0.2)

        # uncalibrated
        for v_, color in zip([x_, y_, z_], ax_colors):
            draw_axis(ax, o_, v_, color, 1.0)

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.view_init(elev=20, azim=30)

    plt.tight_layout()
    plt.show()

