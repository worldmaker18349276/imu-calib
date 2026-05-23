import numpy as np


def normalize(v):
    return v / np.linalg.norm(v)

def quat_conj(q):
    return np.array([q[0], *(-q[1:])])

def quat_mul(q1, q2):
    w = q1[0] * q2[0] - q1[1:].dot(q2[1:])
    v = q1[0] * q2[1:] + q2[0] * q1[1:] + np.cross(q1[1:], q2[1:])
    return np.array([w, v[0], v[1], v[2]])

def quat_to_euler(q):
    t0 = 2.0 * (q[0] * q[1] + q[2] * q[3])
    t1 = 1.0 - 2.0 * (q[1] * q[1] + q[2] * q[2])
    roll = np.arctan2(t0, t1)
    t2 = 2.0 * (q[0] * q[2] - q[3] * q[1])
    t2 = 1.0 if t2 > 1.0 else t2
    t2 = -1.0 if t2 < -1.0 else t2
    pitch = np.arcsin(t2)
    t3 = 2.0 * (q[0] * q[3] + q[1] * q[2])
    t4 = 1.0 - 2.0 * (q[2] * q[2] + q[3] * q[3])
    yaw = np.arctan2(t3, t4)
    return roll, pitch, yaw

def euler_to_quat(roll, yaw, pitch):
    halfRoll = roll / 2.0
    halfYaw = yaw / 2.0
    halfPitch = pitch / 2.0
    w = np.cos(halfRoll) * np.cos(halfPitch) * np.cos(halfYaw) + np.sin(halfRoll) * np.sin(halfPitch) * np.sin(halfYaw)
    x = np.sin(halfRoll) * np.cos(halfPitch) * np.cos(halfYaw) - np.cos(halfRoll) * np.sin(halfPitch) * np.sin(halfYaw)
    y = np.cos(halfRoll) * np.sin(halfPitch) * np.cos(halfYaw) + np.sin(halfRoll) * np.cos(halfPitch) * np.sin(halfYaw)
    z = np.cos(halfRoll) * np.cos(halfPitch) * np.sin(halfYaw) - np.sin(halfRoll) * np.sin(halfPitch) * np.cos(halfYaw)
    return np.array([w, x, y, z])

def quat_to_z(v):
    v = np.asarray(v, dtype=float)
    v /= np.linalg.norm(v)

    z = np.array([0.0, 0.0, 1.0])
    s = np.cross(v, z)
    c = np.dot(v, z)

    q = np.array([1.0 + c, s[0], s[1], s[2]])
    q /= np.linalg.norm(q)
    return q

def slerp(q_from, q_to, fraction):
    theta = q_from.T @ q_to
    q = np.sin((1.0 - fraction) * theta) / np.sin(theta) * q_from \
      + np.sin(       fraction  * theta) / np.sin(theta) * q_to
    return normalize(q)

def skew(v):
    return np.array([[0, -v[2], v[1]], 
                     [v[2], 0, -v[0]], 
                     [-v[1], v[0], 0]])

Omega_indices = np.array([
    [0, 0, 1, 2],
    [0, 0, 2, 1],
    [1, 2, 0, 0],
    [2, 1, 0, 0],
])
Omega_sign = np.array([
    [0, -1, -1, -1],
    [1,  0,  1, -1],
    [1, -1,  0,  1],
    [1,  1, -1,  0],
])

def Omega(phi):
    """
    for small phi,
    quat_mul(q, rvec_to_quat(phi)) = q + Omega(phi) @ q
    """
    return phi[..., Omega_indices] * (Omega_sign / 2)

def Rmat(q):
    """
    Rmat(q1) @ Rmat(q2) = Rmat(quat_mul(q1, q2))
    """
    w = q[0]
    v = q[1:]
    return (w**2 - v.dot(v)) * np.eye(3) + 2 * v[:,None] * v[None,:] + 2 * w * skew(v)
