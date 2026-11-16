"""Independent central-difference checks of L, J and the composed image Jacobian."""
from pathlib import Path
import sys

import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scene import TriangleScene
from control import set_joints, normalize, interaction_matrix, camera_jacobian


def main():
    scene = TriangleScene()
    worst = np.zeros(3)
    try:
        for q in ([0.35, -0.59, 0.3], [0.43, -0.69, 0.45],
                  [-0.2, 0.8, -0.4], [0.7, -1.1, 0.6]):
            q = np.asarray(q)
            set_joints(scene, q)
            eye, rotation = scene.camera_pose()
            optical = rotation @ np.diag([1, -1, -1])
            xyz = (scene.marker_vertices-eye) @ optical
            points = xyz[:, :2] / xyz[:, 2:3]
            L = interaction_matrix(points, xyz[:, 2])
            J = camera_jacobian(scene, q)
            # Move the optical camera in translation and yaw; keep world points fixed.
            numeric_L = np.empty_like(L)
            eps = 1e-6
            for axis in range(3):
                samples = []
                for sign in (1, -1):
                    delta = sign*eps
                    translated = xyz.copy()
                    if axis < 2:
                        translated[:, axis] -= delta
                    else:
                        c, s = np.cos(delta), np.sin(delta)
                        translated = xyz @ np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
                    samples.append((translated[:, :2]/translated[:, 2:3]).ravel())
                numeric_L[:, axis] = (samples[0]-samples[1])/(2*eps)
            # Bullet link poses are float32, so use a larger central-difference step.
            eps = 1e-3
            numeric_J = np.empty_like(J)
            numeric_image = np.empty((6, 3))
            for axis in range(3):
                poses, features = [], []
                for sign in (1, -1):
                    shifted = q.copy()
                    shifted[axis] += sign*eps
                    set_joints(scene, shifted)
                    position, orientation = scene.camera_pose()
                    yaw = np.arctan2(orientation[1, 0], orientation[0, 0])
                    poses.append(np.r_[position, yaw])
                    features.append(normalize(scene, scene.projected_vertices()).ravel())
                velocity = (poses[0]-poses[1])/(2*eps)
                numeric_J[:, axis] = np.r_[(optical.T @ velocity[:3])[:2], -velocity[3]]
                numeric_image[:, axis] = (features[0]-features[1])/(2*eps)
            errors = [np.max(np.abs(L-numeric_L)), np.max(np.abs(J-numeric_J)),
                      np.max(np.abs(L@J-numeric_image))]
            worst = np.maximum(worst, errors)
            np.testing.assert_allclose(L, numeric_L, atol=1e-8, rtol=1e-7)
            np.testing.assert_allclose(J, numeric_J, atol=3e-4, rtol=1e-3)
            np.testing.assert_allclose(L@J, numeric_image, atol=5e-4, rtol=1e-3)
        print(f"PASS: 4 configurations; max absolute errors L={worst[0]:.2e}, "
              f"J={worst[1]:.2e}, L@J={worst[2]:.2e}")
    finally:
        scene.close()


if __name__ == "__main__":
    main()
