import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from imu_calib import imu_calib
from imu_calib.utils import *


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

def plot_gyroscope_before_and_after(gyrs, gyrs_calibrated, times):
    fig, ax = plt.subplots(3, 1, figsize=(8, 6), sharex=True, sharey=True)
    ax[0].plot(times, gyrs[:, 0], alpha = 0.5)
    ax[1].plot(times, gyrs[:, 1], alpha = 0.5)
    ax[2].plot(times, gyrs[:, 2], alpha = 0.5)
    ax[0].plot(times, gyrs_calibrated[:, 0], alpha = 0.5)
    ax[1].plot(times, gyrs_calibrated[:, 1], alpha = 0.5)
    ax[2].plot(times, gyrs_calibrated[:, 2], alpha = 0.5)
    for i in range(3):
        ax[i].legend(["Uncalibrated", "Calibrated"])
        ax[i].axhline(0, linewidth=1, color="black")
        ax[i].set(ylabel='$rad/s$', ylim = [-0.2, 0.2])
    ax[2].set(xlabel='$time, s$')
    plt.show()

def plot_state_before_and_after(p, q, v, p_, q_, v_, standstill_indices, times):
    def compute_ups(q):
        u = np.zeros((len(q), 3))
        for i in range(len(q)):
            u[i, :] = Rmat(q[i, :]).T @ np.array([0.0, 0.0, 1.0])
        return u

    ups = compute_ups(q)
    ups_ = compute_ups(q_)

    fig, ax = plt.subplots(3, 2, figsize=(16, 9), sharex=True)
    ax[0,0].plot(times, ups[:,0], label = 'uncalibrated')
    ax[0,0].plot(times, ups_[:,0], label = 'calibrated')
    ax[0,0].legend()
    ax[0,0].set(xlabel = "time", ylim = [-1.0, 1.0], title = "up.x")

    ax[1,0].plot(times, ups[:,1], label = 'uncalibrated')
    ax[1,0].plot(times, ups_[:,1], label = 'calibrated')
    ax[1,0].legend()
    ax[1,0].set(xlabel = "time", ylim = [-1.0, 1.0], title = "up.y")

    ax[2,0].plot(times, ups[:,2], label = 'uncalibrated')
    ax[2,0].plot(times, ups_[:,2], label = 'calibrated')
    ax[2,0].legend()
    ax[2,0].set(xlabel = "time", ylim = [-1.0, 1.0], title = "up.z")

    ax[0,1].plot(times, v[:, 2], label = 'uncalibrated')
    ax[0,1].plot(times, v_[:, 2], label = 'calibrated')
    ax[0,1].legend()
    ax[0,1].set(xlabel = "time", ylim = [-2.0, 2.0], title = "velocity v.z")

    ax[1,1].plot(times, (v[:, 1]**2 + v[:, 1]**2)**0.5, label = 'uncalibrated')
    ax[1,1].plot(times, (v_[:, 1]**2 + v_[:, 1]**2)**0.5, label = 'calibrated')
    ax[1,1].legend()
    ax[1,1].set(xlabel = "time", ylim = [0.0, 4.0], title = "velocity v.s")

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

