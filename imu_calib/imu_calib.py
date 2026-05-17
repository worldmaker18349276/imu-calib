import numpy as np
import scipy.optimize as optimize
from imu_calib.utils import *


g = 9.81

def sensor_error_model(theta):
    KX, KY, KZ, NOX, NOY, NOZ, BX, BY, BZ = theta
    M = np.array([[1.0 + KX, NOY, NOZ], 
                  [0.0, 1.0 + KY, NOX], 
                  [0.0, 0.0, 1.0 + KZ]])
    b = np.array([BX, BY, BZ])
    
    return M, b

def misalignment(epsilon):
    return np.eye(3) + skew(epsilon)

def make_residual_acc(accs):
    '''
    Accelerometer cost function according to equation (10) in the paper
    '''
    def residual_func(theta):
        M, B = sensor_error_model(theta[0:9])
        C = np.linalg.inv(M)
        accs_ = (accs - B[None, :]) @ C.T
        z = accs_ * (g / np.sqrt(np.sum(accs_**2, axis=1, keepdims=True)) - 1)
        
        return z.flatten()

    return residual_func

def make_residual_gyr(gyrs, accs, dt, standstill_flags):
    '''
    Gyroscope cost function according to equation (16) in the paper
    '''
    # find the indexes, when the sensor was rotated
    standstill_changes = np.hstack((0, np.diff(standstill_flags)))
    motion_starts = np.where(standstill_changes == -1)
    motion_ends = np.where(standstill_changes == 1)
    motion_start_end_idxs = np.array([[max(0, s - 5), e + 5] for s, e in zip(*motion_starts, *motion_ends)], dtype=int)
    standstill_start_end_idxs = np.array([0, *motion_start_end_idxs.flatten(), len(accs)]).reshape((-1, 2))
    
    standstill_up = []
    for idx_standstill_start, idx_standstill_end in standstill_start_end_idxs:
        acc = np.mean(accs[idx_standstill_start:idx_standstill_end, :], axis=0)
        standstill_up.append(acc / np.linalg.norm(acc))

    def residual_func(theta):
        M, B = sensor_error_model(theta[0:9])
        C = np.linalg.inv(M)
        R = np.linalg.inv(misalignment(theta[-3:]))

        z = np.zeros(len(motion_start_end_idxs))
        for i, idx_motion_start_end in enumerate(motion_start_end_idxs):
            ws = gyrs[idx_motion_start_end[0]:idx_motion_start_end[1],:]
            ws_b = ws @ (R.T @ C.T) - B[None, :] @ C.T
            dt_ = dt if np.isscalar(dt) else dt[idx_motion_start_end[0]:idx_motion_start_end[1], None]
            Qws = Qright(ws_b * dt_)

            q = np.array([1.0, 0.0, 0.0, 0.0])
            for Qw in Qws:
                q = Qw @ q
                q /= np.linalg.norm(q)

            z[i] = np.linalg.norm(Rmat(q) @ standstill_up[i+1] - standstill_up[i])

        return z.flatten()

    return residual_func

def generate_standstill_flags(imu_data, standstill_gyr_threshold = 0.13):
    standstill = np.zeros(len(imu_data), dtype=np.int8)
    MARGIN = 60
    counter_after_motion = MARGIN
    # generate standstill flags
    for i, m in enumerate(imu_data):
        if np.linalg.norm(m[3:6]) < standstill_gyr_threshold:
            counter_after_motion += 1
            if counter_after_motion > MARGIN:
                standstill[i] = 1
        else:
            standstill[i] = 0
            standstill[:i][-MARGIN:] = 0
            counter_after_motion = 0

    return standstill

def calib_acc(accs, standstill_flags):
    '''
    Find calibration parameters according to cost function from eq. (10)
    For details see make_residual_acc function inside cost_functions.py
    '''
    initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    res = optimize.least_squares(make_residual_acc(accs[standstill_flags > 0]),
                                 initial,
                                 max_nfev = 25,
                                 x_scale = [10., 10., 10., 10., 10., 10., 1., 1., 1.],
                                 method='trf', loss='soft_l1',
                                 bounds = [
                                       (-0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -1.1, -1.1, -1.1),
                                       (0.11,   0.11,  0.11,  0.11,  0.11,  0.11,  1.1,  1.1,  1.1)],
        )
    return res.x

def calib_gyr(gyrs, accs, dt, standstill_flags):
    '''
    Find calibration parameters according to cost function from eq. (16)
    For details see make_residual_gyr function inside cost_functions.py
    '''
    intial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    intial[-6:-3] = np.mean(gyrs[0:100,:], axis=0)

    res = optimize.least_squares(
            make_residual_gyr(gyrs, accs, dt, standstill_flags),
            intial, 
            max_nfev = 50,
            x_scale = [10., 10., 10., 10., 10., 10., 10., 10., 10., 10, 10, 10],
            method='trf', loss='soft_l1',
            bounds = [
                    (-0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11),
                    (0.11,   0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11)],
        )
    return res.x

def correct_acc(accs, theta_acc):
    '''
    Calibrate sensor measurements according to the model.
    See eq. (7).
    '''
    M, B = sensor_error_model(theta_acc[0:9])
    C = np.linalg.inv(M)

    # a' = (I + diag(S) + tri(NO)) @ a + B
    return (accs - B[None, :]) @ C.T

def correct_gyr(gyrs, theta_gyr):
    '''
    Calibrate sensor measurements according to the model.
    See eq. (8).
    '''
    M, B = sensor_error_model(theta_gyr[0:9])
    C = np.linalg.inv(M)
    R = np.linalg.inv(misalignment(theta_gyr[9:]))

    # w' = (I + X(E)) @ ((I + diag(S) + tri(NO)) @ w + B)
    return gyrs @ (R.T @ C.T) - B[None, :] @ C.T

def evaluate_states(accs, gyrs, dt, p0=None, q0=None, v0=None):
    p0 = p0 if p0 is not None else np.array([0.0, 0.0, 0.0])
    q0 = q0 if q0 is not None else np.array([1.0, 0.0, 0.0, 0.0])
    v0 = v0 if v0 is not None else np.array([0.0, 0.0, 0.0])

    N = accs.shape[0]

    ps = np.zeros((N, 3))
    vs = np.zeros((N, 3))
    qs = np.zeros((N, 4))

    ps[0, :] = p0
    vs[0, :] = v0
    qs[0, :] = q0

    acc_g = np.mean(accs[0:100,:], axis=0)
    acc_g = acc_g / np.linalg.norm(acc_g) * g

    dt = np.repeat(dt, N) if np.isscalar(dt) else dt
    Qws = Qright(gyrs * dt[:, None])
    for i in range(1, N):
        qs[i, :] = Qws[i-1] @ qs[i-1, :]
        qs[i, :] /= np.linalg.norm(qs[i, :])
        R_i = Rmat(qs[i, :])
        ps[i, :] = ps[i-1, :] + vs[i-1, :] * dt[i] + (R_i @ accs[i, :] - acc_g) * 0.5 * dt[i]**2
        vs[i, :] = vs[i-1, :] + (R_i @ accs[i, :] - acc_g) * dt[i]

    return ps, qs, vs
