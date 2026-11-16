"""Triangle IBVS with ideal joint velocity tracking, using rendered detections."""
import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from scene import TriangleScene, joint_angles
from find_triangle import find_triangles, annotate_triangles
from control import set_joints, normalize, interaction_matrix, camera_jacobian


def detect(scene):
    image = scene.render()
    triangles = find_triangles(image)
    if len(triangles) != 1:
        raise RuntimeError(f"Expected one triangle, found {len(triangles)}; stopped")
    return image, triangles[0]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--gui", action="store_true")
    parser.add_argument("--steps", type=int, default=240)
    parser.add_argument("--dt", type=float, default=0.05)
    parser.add_argument("--gain", type=float, default=1.5)
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "output")
    args = parser.parse_args()
    if args.steps < 1 or args.dt <= 0 or args.gain <= 0 or (args.gui and args.headless):
        parser.error("Use positive steps, dt and gain; --gui conflicts with --headless")
    scene = TriangleScene(gui=args.gui)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    try:
        desired_q = np.asarray(joint_angles(0))
        _, desired_pixels = detect(scene)
        desired = normalize(scene, desired_pixels)
        # Board and camera stay parallel, so every point has the same known depth.
        depth = scene.camera_pose()[0][2] - scene.marker_center[2]
        q = desired_q + [0.24, -0.30, 0.45]
        for step in range(args.steps):
            started = time.perf_counter()
            set_joints(scene, q)
            image, pixels = detect(scene)
            points = normalize(scene, pixels)
            error = (points - desired).ravel()
            rms = float(np.sqrt(np.mean((pixels-desired_pixels)**2)))
            history.append([step*args.dt, rms, *q])
            display = annotate_triangles(image, [pixels])
            for point in desired_pixels:
                cv2.circle(display, tuple(np.rint(point).astype(int)), 4, (0, 0, 255), 1)
            if step == 0:
                cv2.imwrite(str(args.output_dir / "initial.png"), display)
            if rms <= 1.0:
                break
            # Compose the image and robot Jacobians without reference-specific swaps.
            image_jacobian = interaction_matrix(points, depth) @ camera_jacobian(scene, q)
            dq = -args.gain * np.linalg.pinv(image_jacobian, rcond=1e-6) @ error
            # Uniform scaling preserves the descent direction.
            dq *= min(1.0, 0.5 / max(np.max(np.abs(dq)), 1e-12))
            q = q + args.dt*dq
            if not args.headless:
                cv2.imshow("Triangle IBVS (red: desired) | Q/Esc: quit", display)
                delay = max(1, round(1000*(args.dt-(time.perf_counter()-started))))
                if cv2.waitKey(delay) & 0xFF in (27, ord("q")):
                    break
        cv2.imwrite(str(args.output_dir / "final.png"), display)
        np.savetxt(args.output_dir / "history.csv", history, delimiter=",",
                   header="time,rms_pixels,q1,q2,q3", comments="")
        print(f"RMS pixel error: {history[0][1]:.3f} -> {rms:.3f}; {len(history)} observations")
        if args.headless and rms > 1.0:
            raise AssertionError("IBVS did not converge to <= 1 pixel RMS")
    finally:
        scene.close()
        if not args.headless:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
