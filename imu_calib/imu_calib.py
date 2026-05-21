import numpy as np
import scipy.optimize as optimize
from imu_calib.utils import *


def sensor_error_model(theta):
    KX, KY, KZ, NOX, NOY, NOZ, BX, BY, BZ = theta
    M = np.array([[1.0 + KX, -NOZ, NOY], 
                  [0.0, 1.0 + KY, -NOX], 
                  [0.0, 0.0, 1.0 + KZ]])
    B = np.array([BX, BY, BZ])
    
    return M, B

def misalignment(epsilon):
    return np.eye(3) + skew(epsilon)

def distort(theta, v):
    # v' = (I + skew(E)) @ ((I + diag(S) + tri(NO)) @ v + B)
    M, B = sensor_error_model(theta[0:9])
    R = misalignment(theta[9:12]) if len(theta) == 12 else np.eye(3)
    return R @ (M @ v + B)

def explain(theta, name):
    KX, KY, KZ, NOX, NOY, NOZ, BX, BY, BZ = theta[0:9]
    RD = 180 / np.pi
    res = f"Your sensor {name} are:\n"
    if len(theta) == 12:
        EX, EY, EZ = theta[9:12]
        res += (
            f"  mis-aligned by    {EX*RD:+8.4f} {EY*RD:+8.4f} {EZ*RD:+8.4f} deg;\n"
        )
    res += (
            f"  non-orthogonal by {NOX*RD:+8.4f} {NOY*RD:+8.4f} {NOZ*RD:+8.4f} deg;\n"
            f"  enlarged by       {KX*100:+8.4f} {KY*100:+8.4f} {KZ*100:+8.4f} %;\n"
            f"  offset by         {BX*100:+8.4f} {BY*100:+8.4f} {BZ*100:+8.4f} cent unit"
    )
    return res

def make_residual_acc(accs, g, standstill_flags):
    '''
    Accelerometer cost function according to equation (10) in the paper
    '''
    standstill_changes = np.hstack((0, np.diff(standstill_flags)))
    motion_starts = np.where(standstill_changes == -1)
    motion_ends = np.where(standstill_changes == 1)
    motion_start_end_idxs = np.array([[max(0, s - 5), e + 5] for s, e in zip(*motion_starts, *motion_ends)], dtype=int)
    standstill_start_end_idxs = np.array([0, *motion_start_end_idxs.flatten(), len(accs)]).reshape((-1, 2))
    
    standstill_up = []
    for idxs in standstill_start_end_idxs:
        acc = np.mean(accs[idxs[0]:idxs[1], :], axis=0)
        standstill_up.append(acc / np.linalg.norm(acc))

    def residual_func(theta_up):
        M, B = sensor_error_model(theta_up[0:9])
        C = np.linalg.inv(M)
        z = np.zeros((sum(standstill_flags > 0), 3))
        j = 0
        for idxs, up in zip(standstill_start_end_idxs, theta_up[9:].reshape((-1, 3))):
            accs_ = (accs[idxs[0]:idxs[1], :] - B[None, :]) @ C.T
            g_up = up * (g / np.linalg.norm(up))
            j_next = j + idxs[1] - idxs[0]
            z[j:j_next, :] = g_up - accs_
            j = j_next
        return z.flatten()

    initial = np.concatenate([np.zeros((9,)), *standstill_up])
    return initial, residual_func

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

        z = np.zeros((len(motion_start_end_idxs), 3))
        for i, idxs in enumerate(motion_start_end_idxs):
            dt_ = np.repeat(dt, idxs[1]-idxs[0]) if np.isscalar(dt) else dt[idxs[0]:idxs[1]]
            ws = gyrs[idxs[0]:idxs[1],:]
            ws = ws @ (R.T @ C.T) - B[None, :] @ C.T
            Ws = Omega(ws)

            q = np.array([1.0, 0.0, 0.0, 0.0])
            for j in range(len(Ws))[:-1]:
                # RK4
                k1 = Ws[j] @ q
                k2 = (Ws[j] + Ws[j+1])/2 @ (q + k1 * dt_[j]/2)
                k3 = (Ws[j] + Ws[j+1])/2 @ (q + k2 * dt_[j]/2)
                k4 = Ws[j+1] @ (q + k3 * dt_[j])
                q = q + (k1 + k2*2 + k3*2 + k4)/6 * dt_[j]
                q /= np.linalg.norm(q)

            z[i, :] = Rmat(q) @ standstill_up[i+1] - standstill_up[i]

        return z.flatten()

    initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    initial[-6:-3] = np.mean(gyrs[standstill_flags > 0,:], axis=0)

    return initial, residual_func

def compute_jacobian(residual, x, eps=1.0e-8):
    r = residual(x)
    J = np.zeros((len(r), len(x)))
    for i, e in enumerate(np.eye(len(x))):
        r_ = residual(x + e * eps)
        J[:, i] = (r_ - r) / eps
    return J

def compute_covariance(residual, x, eps=1.0e-8):
    r = residual(x)
    J = compute_jacobian(residual, x, eps=eps)
    sigma = 1.4826 * np.median(np.abs(r))
    cov = sigma**2 * np.linalg.pinv(J.T @ J)
    return cov

def generate_standstill_flags(imu_data, motion_margn_frame = 60, standstill_gyr_threshold = 0.13):
    standstill = np.zeros(len(imu_data), dtype=np.int8)
    counter_after_motion = motion_margn_frame
    # generate standstill flags
    for i, m in enumerate(imu_data):
        if np.linalg.norm(m[3:6]) < standstill_gyr_threshold:
            counter_after_motion += 1
            if counter_after_motion > motion_margn_frame:
                standstill[i] = 1
        else:
            standstill[i] = 0
            standstill[:i][-motion_margn_frame:] = 0
            counter_after_motion = 0

    return standstill

def calib_acc(accs, standstill_flags, g):
    '''
    Find calibration parameters according to cost function from eq. (10)
    For details see make_residual_acc function inside cost_functions.py
    '''
    initial, residual = make_residual_acc(accs, g, standstill_flags)
    M = initial.shape[0] - 9
    res = optimize.least_squares(residual,
                                 initial,
                                 max_nfev = 25,
                                 x_scale = [10., 10., 10., 10., 10., 10., 1., 1., 1., *[1.]*M],
                                 method='trf', loss='soft_l1',
                                 bounds = [
                                       (-0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -1.1, -1.1, -1.1, *[-1.]*M),
                                       (0.11,   0.11,  0.11,  0.11,  0.11,  0.11,  1.1,  1.1,  1.1,  *[1.]*M)],
        )

    cov = compute_covariance(residual, res.x)
    return res.x, cov[:9, :9]

def calib_gyr(gyrs, accs, dt, standstill_flags):
    '''
    Find calibration parameters according to cost function from eq. (16)
    For details see make_residual_gyr function inside cost_functions.py
    '''
    initial, residual = make_residual_gyr(gyrs, accs, dt, standstill_flags)
    res = optimize.least_squares(
            residual,
            initial, 
            max_nfev = 50,
            x_scale = [10., 10., 10., 10., 10., 10., 10., 10., 10., 10, 10, 10],
            method='trf', loss='soft_l1',
            bounds = [
                    (-0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11, -0.11),
                    (0.11,   0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11,  0.11)],
        )
    cov = compute_covariance(residual, res.x)
    return res.x, cov

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

def evaluate_states(accs, gyrs, dt, g, p0=None, q0=None, v0=None):
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
    Ws = Omega(gyrs)

    for j in range(len(Ws))[:-1]:
        # RK4
        k1 = Ws[j] @ qs[j]
        k2 = (Ws[j] + Ws[j+1])/2 @ (qs[j] + k1 * dt[j]/2)
        k3 = (Ws[j] + Ws[j+1])/2 @ (qs[j] + k2 * dt[j]/2)
        k4 = Ws[j+1] @ (qs[j] + k3 * dt[j])
        qs[j+1] = qs[j] + (k1 + k2*2 + k3*2 + k4)/6 * dt[j]
        qs[j+1] /= np.linalg.norm(qs[j+1])

        R0 = Rmat(qs[j])
        R1 = Rmat(qs[j+1])
        vs[j+1] = vs[j] + ((R0 @ accs[j] + R1 @ accs[j+1])/2 - acc_g) * dt[j]
        ps[j+1] = ps[j] + (vs[j] + vs[j+1])/2 * dt[j] + (R0 @ accs[j] - acc_g) * 0.5 * dt[j]**2

    return ps, qs, vs
