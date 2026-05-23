import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from imu_calib import imu_calib


def plot_imu_data_and_standstill(imu_data, standstill_indices, times):
    fig, ax = plt.subplots(2, 1, figsize=(16, 9))
    ax[0].plot(times, imu_data[:, 0:3])
    ax[0].legend(["ax", "ay", "az"])
    ax[1].plot(times, imu_data[:, 3:6])
    ax[1].legend(["wx", "wy", "wz"])

    motion_indices = standstill_indices.flatten()[1:-1].reshape((-1, 2))
    for idxs in motion_indices:
        ax[0].axvspan(times[idxs[0]], times[idxs[1]], color='green', alpha=0.2)
        ax[1].axvspan(times[idxs[0]], times[idxs[1]], color='green', alpha=0.2)

    plt.show()

def plot_accelerations_before_and_after(accs, accs_calibrated, times, g):
    fig, ax = plt.subplots(1, 1, figsize=(8, 3))
    ax.plot(times, np.linalg.norm(accs, axis=1), alpha = 0.5)
    ax.plot(times, np.linalg.norm(accs_calibrated, axis=1), alpha = 0.5)
    ax.axhline(g, linewidth=1, color="black")
    ax.legend(["Uncalibrated norm", "Calibrated norm"])
    ax.set(xlabel='$time, s$', ylabel='$m/s^2$', ylim = [g-1, g+1])
    plt.show()

def quat_to_z(v):
    v = np.asarray(v, dtype=float)
    v /= np.linalg.norm(v)

    z = np.array([0.0, 0.0, 1.0])
    c = np.cross(v, z)
    d = np.dot(v, z)

    q = np.array([1.0 + d, c[0], c[1], c[2]])
    if q[0] < 1e-10:
        q = np.array([0.0, 1.0, 0.0, 0.0])
    q /= np.linalg.norm(q)
    return q

def plot_state_before_and_after(accs, accs_calibrated, gyrs, gyrs_calibrated, dt, standstill_indices, g, times, compare=None):
    up0 = np.mean(accs_calibrated[standstill_indices[0,0]:standstill_indices[0,1], :], axis=0)
    q0 = quat_to_z(up0)
    
    p, q, v = imu_calib.evaluate_states(accs, gyrs, dt, g, q0=q0)
    p_, q_, v_ = imu_calib.evaluate_states(accs_calibrated, gyrs_calibrated, dt, g, q0=q0)
    p0, q0, v0 = imu_calib.evaluate_states(compare[:, 0:3], compare[:, 3:6], dt, g, q0=q0) if compare is not None else (None, None, None)

    fig, ax = plt.subplots(3, 2, figsize=(16, 9))
    ax[0, 0].plot(times, q[:, 1], label = 'uncalibrated')
    ax[0, 0].plot(times, q_[:, 1], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated')
    if q0 is not None:
        ax[0, 0].plot(times, q0[:, 1], marker = 'o', markersize = 2, linestyle = 'none', label = 'true')
    ax[0, 0].legend()
    ax[0, 0].set(xlabel = "time", ylim = [-1.0, 1.0], title = "orientation q.x")

    ax[1, 0].plot(times, q[:, 2], label = 'uncalibrated')
    ax[1, 0].plot(times, q_[:, 2], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated')
    if q0 is not None:
        ax[1, 0].plot(times, q0[:, 2], marker = 'o', markersize = 2, linestyle = 'none', label = 'true')
    ax[1, 0].legend()
    ax[1, 0].set(xlabel = "time", ylim = [-1.0, 1.0], title = "orientation q.y")

    ax[2, 0].plot(times, q[:, 3], label = 'uncalibrated')
    ax[2, 0].plot(times, q_[:, 3], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated')
    if q0 is not None:
        ax[2, 0].plot(times, q0[:, 3], marker = 'o', markersize = 2, linestyle = 'none', label = 'true')
    ax[2, 0].legend()
    ax[2, 0].set(xlabel = "time", ylim = [-1.0, 1.0], title = "orientation q.z")

    ax[0, 1].plot(times, v[:, 0], label = 'uncalibrated')
    ax[0, 1].plot(times, v_[:, 0], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated')
    if v0 is not None:
        ax[0, 1].plot(times, v0[:, 0], marker = 'o', markersize = 2, linestyle = 'none', label = 'true')
    ax[0, 1].legend()
    ax[0, 1].set(xlabel = "time", ylim = [-10.0, 10.0], title = "velocity v.x")

    ax[1, 1].plot(times, v[:, 1], label = 'uncalibrated')
    ax[1, 1].plot(times, v_[:, 1], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated')
    if v0 is not None:
        ax[1, 1].plot(times, v0[:, 1], marker = 'o', markersize = 2, linestyle = 'none', label = 'true')
    ax[1, 1].legend()
    ax[1, 1].set(xlabel = "time", ylim = [-10.0, 10.0], title = "velocity v.y")

    ax[2, 1].plot(times, v[:, 2], label = 'uncalibrated')
    ax[2, 1].plot(times, v_[:, 2], marker = 'x', markersize = 2, linestyle = 'none', label = 'calibrated')
    if v0 is not None:
        ax[2, 1].plot(times, v0[:, 2], marker = 'o', markersize = 2, linestyle = 'none', label = 'true')
    ax[2, 1].legend()
    ax[2, 1].set(xlabel = "time", ylim = [-10.0, 10.0], title = "velocity v.z")

    motion_indices = standstill_indices.flatten()[1:-1].reshape((-1, 2))
    for i in range(3):
        for j in range(2):
            for idxs in motion_indices:
                ax[i,j].axvspan(times[idxs[0]], times[idxs[1]], color='green', alpha=0.2)
                ax[i,j].axvspan(times[idxs[0]], times[idxs[1]], color='green', alpha=0.2)

    plt.tight_layout()
    plt.show()

def plot_theta(theta_acc, theta_gyr, cov_theta_acc, cov_theta_gyr):
    def draw_axis(ax, v0, v1, color, alpha):
        ax.plot([v0[0], v1[0]], [v0[1], v1[1]], [v0[2], v1[2]], color=color, alpha=alpha)

    ax_colors = ['red', 'green', 'blue']

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle('IMU Axis Calibration')

    for idx, (theta, title) in enumerate([(theta_acc, 'Accelerometer'), (theta_gyr, 'Gyroscope')]):
        ax = fig.add_subplot(2, 2, idx+1, projection='3d')
        ax.set_title(title)
        
        o = np.array([0., 0., 0.])
        x = np.array([1., 0., 0.])
        y = np.array([0., 1., 0.])
        z = np.array([0., 0., 1.])

        o_ = imu_calib.distort(theta, o)
        x_ = imu_calib.distort(theta, x)
        y_ = imu_calib.distort(theta, y)
        z_ = imu_calib.distort(theta, z)

        # only consider bias and misalign, so that oblique and scale are easy to see
        theta2 = np.copy(theta)
        theta2[0:6] = 0
        o2 = imu_calib.distort(theta2, o)
        x2 = imu_calib.distort(theta2, x)
        y2 = imu_calib.distort(theta2, y)
        z2 = imu_calib.distort(theta2, z)

        # calibrated
        for v, color in zip([x, y, z], ax_colors):
            draw_axis(ax, o, v, color, 0.5)

        for v2, color in zip([x2, y2, z2], ax_colors):
            draw_axis(ax, o2, v2, color, 0.2)

        # uncalibrated
        for v_, color in zip([x_, y_, z_], ax_colors):
            draw_axis(ax, o_, v_, color, 1.0)

        ax.set_xlim(-1, 1); ax.set_ylim(-1, 1); ax.set_zlim(-1, 1)
        ax.view_init(elev=20, azim=30)

    covs = [
        (cov_theta_acc, ["sx", "sy", "sz", "ox", "oy", "oz", "bx", "by", "bz"]),
        (cov_theta_gyr, ["sx", "sy", "sz", "ox", "oy", "oz", "bx", "by", "bz", "ex", "ey", "ez"]),
    ]
    for idx, (cov, labels) in enumerate(covs):
        ax = fig.add_subplot(2, 2, idx+3)
        v = np.abs(cov).max()
        im = ax.imshow(cov, cmap="seismic", vmin=-v, vmax=v)
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.colorbar(im)

    plt.tight_layout()
    plt.show()

