# Triangle image-based visual servoing

Run from the repository root using the same dependencies as `triangle_sim`:

```powershell
python triangle_sim/ibvs/run.py
python triangle_sim/ibvs/run.py --headless
python triangle_sim/ibvs/verify.py
```

`--gui` additionally opens the PyBullet view. Q or Esc exits the live image.
The reference image is detected at `scene.joint_angles(0)`. The arm starts with
joint offsets `[0.24, -0.30, 0.45]` radians and moves until RMS vertex error is
at most one pixel. Red circles show desired vertices. Initial/final images and
`history.csv` are saved in `triangle_sim/ibvs/output` (`--output-dir` overrides).
Headless execution fails if detection is lost or convergence is not achieved.

This reuses `TriangleScene` and the RGB triangle detector, following the reference
`visual_feedback2024/combined/run.py` workflow: capture desired features, offset
the arm, detect current features, calculate image error, and command joint
velocities. Like the existing triangle simulation, it uses ideal kinematic
tracking (`q_next = q + dt*qdot` and `resetJointState`), not simulated motor
dynamics. `--dt` is simulated control time; rendering/detection may take longer
than this interval in live operation.

## Interaction matrix and coordinate conventions

Optical coordinates are **x right, y down, z forward**. The scene's camera looks
along flange -Z with image up along flange +Y, so its optical-to-world rotation is
`R_wc = R_flange @ diag(1, -1, -1)`.

Normalize pixels with actual focal lengths from the renderer projection:
`x=(u-WIDTH/2)/fx`, `y=(v-HEIGHT/2)/fy`, where
`fx=P[0,0]*WIDTH/2` and `fy=P[1,1]*HEIGHT/2`.
Dividing both coordinates by half the image width, as in the reference, would
give incorrect calibration for this camera. Its 440x280 image is a central crop
of the original 640x480, 75-degree vertical FoV image: 100 pixels are removed
from each edge. The projection preserves fx and fy, with the principal point
shifted to (220, 140), so the image scale stays unchanged.

For each fixed triangle point, the reduced interaction matrix is

```text
[xdot]   [-1/Z     0     y] [vx]
[ydot] = [   0  -1/Z    -x] [vy]
                           [wz]
```

Stack the three vertex blocks to obtain `L` of shape 6x3. This restriction of the
usual point-feature interaction matrix is valid because the planar arm cannot
produce optical vz, wx or wy. The board and camera stay parallel; all vertex
depths are the constant `Z = camera_world_z - board_marker_z = 0.558 m`.
Known model depth is used, while current/desired feature coordinates come only
from RGB detections. Projected ground truth is used solely by `verify.py`.

## Robot Jacobian

Call PyBullet `calculateJacobian` on **fixed camera link 3**, local point `[0,0,0]`,
with the three movable joint coordinates in their native order. Rotate its world
linear and angular Jacobians into the optical frame before selecting rows:

```text
J = stack((R_wc.T @ J_linear)[0:2], (R_wc.T @ J_angular)[2])
```

This includes all three link lengths and the fixed camera mounting offset. In
this planar model the vertical offset contributes no linear velocity. For an
independent analytic interpretation, let `theta_i=sum(q[:i+1])`. The world
linear column j is
`[-sum_i>=j length_i*sin(theta_i), sum_i>=j length_i*cos(theta_i), 0]`,
and every world angular column is `[0,0,1]`. Optical angular columns therefore
equal **-1**, since optical +Z points downward. No joint reordering or manual
velocity sign correction from the old robot is appropriate here.

The composed feature Jacobian is `A = L @ J`, and the control law is
`qdot = -gain * pinv(A) @ (s - s_desired)`. A uniform velocity scale caps joint
speeds at 0.5 rad/s. The pseudoinverse avoids explicit normal-matrix inversion;
it does not guarantee convergence from arbitrary starts or at arm singularities.

## Verification

`verify.py` independently checks:

- L against central differences of pinhole projections after optical camera
  translations and rotations.
- J against central differences of actual PyBullet camera poses.
- L@J against central differences of the renderer's projected marker vertices.

Four configurations are checked, including both elbow signs. Pose differences
use a 0.001 rad step because PyBullet link poses have float32 precision.
Observed maximum absolute discrepancies: L `2.03e-11`, J `2.00e-5`, L@J `9.56e-5`.
The default RGB-feedback run at cropped 440x280 reduced RMS pixel error from 64.843 to
1.000 in 57 observations.
The one-pixel stopping tolerance accommodates rasterization
and detector quantization.
