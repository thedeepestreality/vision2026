"""IBVS geometry for the planar arm; twists use optical camera coordinates."""
import numpy as np
import pybullet as p


def set_joints(scene, q):
    for i, angle in enumerate(q):
        p.resetJointState(scene.robot, i, float(angle), physicsClientId=scene.client)


def normalize(scene, pixels):
    # OpenGL projection gives fx = P00*width/2, fy = P11*height/2.
    from scene import WIDTH, HEIGHT
    projection = np.asarray(scene.projection).reshape(4, 4, order="F")
    focal = np.array([projection[0, 0]*WIDTH/2, projection[1, 1]*HEIGHT/2])
    return (np.asarray(pixels) - [WIDTH/2, HEIGHT/2]) / focal


def interaction_matrix(points, depth):
    """ds/dt = L [vx, vy, wz]; x right, y down, z forward."""
    points = np.asarray(points).reshape(-1, 2)
    depths = np.broadcast_to(depth, (len(points),))
    if np.any(depths <= 0):
        raise ValueError("Depth must be positive")
    matrix = np.zeros((2*len(points), 3))
    matrix[0::2, 0] = -1/depths
    matrix[1::2, 1] = -1/depths
    matrix[0::2, 2] = points[:, 1]
    matrix[1::2, 2] = -points[:, 0]
    return matrix


def camera_jacobian(scene, q):
    """Map qdot to optical [vx, vy, wz] at the fixed camera link origin."""
    linear, angular = p.calculateJacobian(
        scene.robot, 3, [0, 0, 0], list(q), [0.0]*3, [0.0]*3,
        physicsClientId=scene.client)
    _, flange_rotation = scene.camera_pose()
    optical_rotation = flange_rotation @ np.diag([1, -1, -1])
    return np.vstack(((optical_rotation.T @ np.asarray(linear))[:2],
                      (optical_rotation.T @ np.asarray(angular))[2]))
