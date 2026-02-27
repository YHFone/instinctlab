#!/usr/bin/env python3
"""
Visualize the camera quaternion [0.7071, 0, 0, -0.7071] (Rz(-90°)) and its
effect on the MuJoCo camera coordinate system.

This script shows:
  1. The parent body (torso_link) coordinate frame
  2. The camera coordinate frame after the quaternion rotation
  3. The camera's look direction and frustum
  4. Comparison with the original (incorrect) XML quaternion

Usage:
    python deploy/visualize_cam_quat.py
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def quat_to_rot(w, x, y, z):
    """Quaternion (w, x, y, z) → 3×3 rotation matrix."""
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def draw_frame(ax, origin, R, length=0.5, labels=None, alpha=1.0, lw=2.5):
    """Draw a coordinate frame (3 arrows: X=red, Y=green, Z=blue)."""
    colors = ['red', 'green', 'blue']
    default_labels = ['X', 'Y', 'Z']
    if labels is None:
        labels = default_labels
    for i in range(3):
        end = origin + R[:, i] * length
        ax.quiver(*origin, *(R[:, i] * length),
                  color=colors[i], alpha=alpha, linewidth=lw,
                  arrow_length_ratio=0.15)
        ax.text(*end, f"  {labels[i]}", color=colors[i], fontsize=10,
                fontweight='bold', alpha=alpha)


def draw_camera_frustum(ax, origin, R, depth=0.6, fov_h=60, fov_w=90,
                         color='cyan', alpha=0.15):
    """Draw a simple camera frustum pyramid.

    MuJoCo camera convention:
        look direction = -Z_cam
        up direction   = +Y_cam
        right          = +X_cam
    """
    look = -R[:, 2]  # -Z
    up = R[:, 1]      # +Y
    right = R[:, 0]   # +X

    half_h = depth * np.tan(np.radians(fov_h / 2))
    half_w = depth * np.tan(np.radians(fov_w / 2))

    center = origin + look * depth
    corners = [
        center + up * half_h + right * half_w,
        center + up * half_h - right * half_w,
        center - up * half_h - right * half_w,
        center - up * half_h + right * half_w,
    ]

    # Draw edges from origin to each corner
    for c in corners:
        ax.plot3D(*zip(origin, c), color=color, alpha=0.4, linewidth=1)

    # Draw far plane rectangle
    for i in range(4):
        ax.plot3D(*zip(corners[i], corners[(i + 1) % 4]),
                  color=color, alpha=0.6, linewidth=1.5)

    # Fill the far plane
    verts = [list(c) for c in corners]
    poly = Poly3DCollection([verts], alpha=alpha, facecolor=color, edgecolor=color)
    ax.add_collection3d(poly)

    # Draw look direction arrow (dashed)
    ax.quiver(*origin, *look * depth * 0.8,
              color='magenta', alpha=0.8, linewidth=2,
              arrow_length_ratio=0.1, linestyle='dashed')


def main():
    fig = plt.figure(figsize=(18, 8))

    # =====================================================================
    # Quaternion definitions
    # =====================================================================
    quats = {
        'Identity\n(default camera)': (1.0, 0.0, 0.0, 0.0),
        'Rz(-90) CORRECT\n[0.7071, 0, 0, -0.7071]': (0.7071, 0.0, 0.0, -0.7071),
        'Rx(-90) WRONG (XML original)\n[0.7071, -0.7071, 0, 0]': (0.7071, -0.7071, 0.0, 0.0),
    }

    for idx, (title, (w, x, y, z)) in enumerate(quats.items()):
        ax = fig.add_subplot(1, 3, idx + 1, projection='3d')
        R = quat_to_rot(w, x, y, z)

        origin = np.array([0.0, 0.0, 0.0])

        # Draw parent body frame (light, dashed-like)
        draw_frame(ax, origin, np.eye(3), length=0.7,
                   labels=['+X (Fwd)', '+Y (Left)', '+Z (Up)'],
                   alpha=0.25, lw=1.5)

        # Draw camera frame after rotation
        cam_labels = ['+X_cam\n(img right)', '+Y_cam\n(img up)', '+Z_cam']
        draw_frame(ax, origin, R, length=0.55, labels=cam_labels, alpha=0.9, lw=3)

        # Camera look direction = -Z_cam
        look_dir = -R[:, 2]
        ax.quiver(*origin, *look_dir * 0.7,
                  color='magenta', linewidth=3, arrow_length_ratio=0.12)
        end = origin + look_dir * 0.75
        ax.text(*end, '  LOOK\n  (-Z_cam)', color='magenta', fontsize=9,
                fontweight='bold')

        # Draw frustum
        draw_camera_frustum(ax, origin, R, depth=0.5, fov_h=50, fov_w=80)

        # Formatting
        ax.set_xlim([-0.8, 0.8])
        ax.set_ylim([-0.8, 0.8])
        ax.set_zlim([-0.8, 0.8])
        ax.set_xlabel('X (Forward)')
        ax.set_ylabel('Y (Left)')
        ax.set_zlabel('Z (Up)')
        ax.set_title(title, fontsize=11, fontweight='bold', pad=15)
        ax.view_init(elev=25, azim=-60)

    fig.suptitle(
        'MuJoCo Camera Quaternion Orientation Comparison\n'
        'Light = parent body frame (torso_link)  |  Bold = camera frame  |  Magenta = look direction (-Z_cam)',
        fontsize=12, fontweight='bold', y=0.98
    )
    plt.tight_layout(rect=[0, 0, 1, 0.92])
    save_path = 'deploy/cam_quat_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")
    plt.show()

    # =====================================================================
    # Print numerical summary
    # =====================================================================
    print("\n" + "=" * 70)
    print("Numerical Analysis")
    print("=" * 70)
    for title, (w, x, y, z) in quats.items():
        R = quat_to_rot(w, x, y, z)
        look = -R[:, 2]
        up = R[:, 1]
        right = R[:, 0]
        # Compute pitch angle from horizontal
        pitch_deg = np.degrees(np.arcsin(-look[2]))
        print(f"\n--- {title.replace(chr(10), ' ')} ---")
        print(f"  quat [w,x,y,z] = [{w}, {x}, {y}, {z}]")
        print(f"  Rotation matrix R =")
        for row in R:
            print(f"    [{row[0]:+6.3f}, {row[1]:+6.3f}, {row[2]:+6.3f}]")
        print(f"  Look direction  (world) = [{look[0]:+.3f}, {look[1]:+.3f}, {look[2]:+.3f}]")
        print(f"  Image up        (world) = [{up[0]:+.3f}, {up[1]:+.3f}, {up[2]:+.3f}]")
        print(f"  Image right     (world) = [{right[0]:+.3f}, {right[1]:+.3f}, {right[2]:+.3f}]")
        print(f"  Pitch angle             = {pitch_deg:+.1f} deg (>0 = looking down)")


if __name__ == "__main__":
    main()
