#!/usr/bin/env python3
"""
Comprehensive visualization of the depth camera quaternion chain in MuJoCo.

The camera is defined with TWO levels of rotation:
  1. depth_camera_body quat = [0.9135, 0, -0.4067, 0]  →  Ry(-48°)  (body tilt)
  2. camera            quat = [0.7071, 0, 0, -0.7071]   →  Rz(-90°)  (camera frame)

Total orientation: R_total = R_body @ R_cam

This script visualizes the step-by-step rotation and the resulting look direction.

Usage:
    python deploy/visualize_cam_quat_full.py
"""
import matplotlib
matplotlib.use('Agg')

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.art3d import Poly3DCollection


def quat_to_rot(w, x, y, z):
    """Quaternion (w,x,y,z) → 3x3 rotation matrix."""
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-w*z), 2*(x*z+w*y)],
        [2*(x*y+w*z), 1-2*(x*x+z*z), 2*(y*z-w*x)],
        [2*(x*z-w*y), 2*(y*z+w*x), 1-2*(x*x+y*y)]])


def draw_axes(ax, origin, R, length=0.4, labels=None, alpha=1.0, lw=2.5, fs=9):
    """Draw XYZ axis arrows (red/green/blue)."""
    colors = ['#E74C3C', '#27AE60', '#2980B9']  # red, green, blue
    if labels is None:
        labels = ['X', 'Y', 'Z']
    for i in range(3):
        ax.quiver(*origin, *(R[:, i]*length), color=colors[i],
                  alpha=alpha, linewidth=lw, arrow_length_ratio=0.13)
        end = origin + R[:, i] * (length * 1.1)
        ax.text(*end, f' {labels[i]}', color=colors[i], fontsize=fs,
                fontweight='bold', alpha=alpha, ha='left')


def draw_frustum(ax, origin, R, depth=0.35, fov_h=50, fov_w=80,
                 color='cyan', alpha=0.10):
    """Draw camera frustum. MuJoCo: look = -Z_cam, up = +Y_cam."""
    look = -R[:, 2]
    up = R[:, 1]
    right = R[:, 0]
    hh = depth * np.tan(np.radians(fov_h / 2))
    hw = depth * np.tan(np.radians(fov_w / 2))
    c = origin + look * depth
    corners = [c+up*hh+right*hw, c+up*hh-right*hw,
               c-up*hh-right*hw, c-up*hh+right*hw]
    for co in corners:
        ax.plot3D(*zip(origin, co), color=color, alpha=0.3, lw=0.8)
    for i in range(4):
        ax.plot3D(*zip(corners[i], corners[(i+1) % 4]),
                  color=color, alpha=0.5, lw=1.2)
    poly = Poly3DCollection([[list(co) for co in corners]],
                            alpha=alpha, facecolor=color, edgecolor=color)
    ax.add_collection3d(poly)


def draw_look_arrow(ax, origin, R, length=0.55, label='LOOK'):
    """Draw the look direction arrow (magenta, -Z_cam)."""
    look = -R[:, 2]
    ax.quiver(*origin, *look*length, color='#9B59B6', linewidth=3,
              arrow_length_ratio=0.1)
    end = origin + look * (length * 1.05)
    ax.text(*end, f'  {label}', color='#9B59B6', fontsize=10,
            fontweight='bold')


def setup_ax(ax, title, elev=25, azim=-55):
    ax.set_xlim([-0.7, 0.7])
    ax.set_ylim([-0.7, 0.7])
    ax.set_zlim([-0.7, 0.7])
    ax.set_xlabel('X (Forward)', fontsize=8)
    ax.set_ylabel('Y (Left)', fontsize=8)
    ax.set_zlabel('Z (Up)', fontsize=8)
    ax.set_title(title, fontsize=10, fontweight='bold', pad=10)
    ax.view_init(elev=elev, azim=azim)


def main():
    # =====================================================================
    # Quaternion chain from XML:
    #   torso_link
    #     └─ depth_camera_body  pos=(0.049, 0, 0.40)  quat=(0.9135, 0, -0.4067, 0)
    #          └─ camera        pos=(0, 0, 0)          quat=(was 0.7071,-0.7071,0,0)
    #                                                       (now 0.7071, 0, 0,-0.7071)
    # =====================================================================
    R_body = quat_to_rot(0.9135, 0.0, -0.4067, 0.0)   # Ry(-48°)
    R_cam_correct = quat_to_rot(0.7071, 0.0, 0.0, -0.7071)  # Rz(-90°)
    R_cam_wrong = quat_to_rot(0.7071, -0.7071, 0.0, 0.0)    # Rx(-90°)

    R_total_correct = R_body @ R_cam_correct
    R_total_wrong = R_body @ R_cam_wrong

    I = np.eye(3)
    o = np.zeros(3)

    fig = plt.figure(figsize=(20, 14))

    # ----- Row 1: Step-by-step decomposition (CORRECT) -----
    # 1a: Torso frame (identity)
    ax1 = fig.add_subplot(2, 4, 1, projection='3d')
    draw_axes(ax1, o, I, length=0.5,
              labels=['+X\nFwd', '+Y\nLeft', '+Z\nUp'])
    setup_ax(ax1, 'Step 0: torso_link frame\n(world reference)')

    # 1b: After body rotation Ry(-48°)
    ax2 = fig.add_subplot(2, 4, 2, projection='3d')
    draw_axes(ax2, o, I, length=0.5,
              labels=['+X', '+Y', '+Z'], alpha=0.2)
    draw_axes(ax2, o, R_body, length=0.5,
              labels=["+X'", "+Y'", "+Z'"])
    ax2.annotate('', xy=(0.5, 0.8), xytext=(0.2, 0.8),
                 xycoords='axes fraction',
                 arrowprops=dict(arrowstyle='->', color='orange', lw=2))
    setup_ax(ax2, 'Step 1: body quat Ry(-48)\n[0.9135, 0, -0.4067, 0]')

    # 1c: After cam rotation Rz(-90°)
    ax3 = fig.add_subplot(2, 4, 3, projection='3d')
    draw_axes(ax3, o, R_body, length=0.5,
              labels=["+X'", "+Y'", "+Z'"], alpha=0.2)
    draw_axes(ax3, o, R_total_correct, length=0.5,
              labels=['+Xc\nimgR', '+Yc\nimgUp', '+Zc'])
    draw_look_arrow(ax3, o, R_total_correct, length=0.5, label='LOOK')
    draw_frustum(ax3, o, R_total_correct)
    setup_ax(ax3, 'Step 2: cam quat Rz(-90)\n[0.7071, 0, 0, -0.7071]')

    # 1d: Final result (CORRECT)
    ax4 = fig.add_subplot(2, 4, 4, projection='3d')
    draw_axes(ax4, o, I, length=0.5,
              labels=['+X\nFwd', '+Y\nLeft', '+Z\nUp'], alpha=0.15)
    draw_axes(ax4, o, R_total_correct, length=0.5,
              labels=['+Xc\nimgR', '+Yc\nimgUp', '+Zc'])
    draw_look_arrow(ax4, o, R_total_correct, length=0.5, label='LOOK')
    draw_frustum(ax4, o, R_total_correct, depth=0.4)
    setup_ax(ax4, 'CORRECT: looks forward+down ~42\nMatches training camera')

    # ----- Row 2: Comparison -----
    # 2a: cam_quat alone: identity
    ax5 = fig.add_subplot(2, 4, 5, projection='3d')
    draw_axes(ax5, o, I, length=0.5,
              labels=['+X', '+Y', '+Z'])
    draw_look_arrow(ax5, o, I, length=0.45, label='LOOK\n(down)')
    draw_frustum(ax5, o, I)
    setup_ax(ax5, 'cam_quat = identity\nLook: straight down')

    # 2b: cam_quat = Rz(-90°) alone (no body tilt)
    ax6 = fig.add_subplot(2, 4, 6, projection='3d')
    draw_axes(ax6, o, I, length=0.5,
              labels=['+X', '+Y', '+Z'], alpha=0.2)
    draw_axes(ax6, o, R_cam_correct, length=0.5,
              labels=['+Xc\nimgR', '+Yc\nimgUp', '+Zc'])
    draw_look_arrow(ax6, o, R_cam_correct, length=0.45, label='LOOK\n(down)')
    draw_frustum(ax6, o, R_cam_correct)
    setup_ax(ax6, 'Rz(-90) alone: still down\nbut imgUp = +X (forward)')

    # 2c: WRONG - Rx(-90°) total
    ax7 = fig.add_subplot(2, 4, 7, projection='3d')
    draw_axes(ax7, o, I, length=0.5,
              labels=['+X\nFwd', '+Y\nLeft', '+Z\nUp'], alpha=0.15)
    draw_axes(ax7, o, R_total_wrong, length=0.5,
              labels=['+Xc', '+Yc', '+Zc'])
    draw_look_arrow(ax7, o, R_total_wrong, length=0.5, label='LOOK\n(RIGHT!)')
    draw_frustum(ax7, o, R_total_wrong, color='red')
    setup_ax(ax7, 'WRONG: Rx(-90) total\nLooks sideways (-Y)!')

    # 2d: Summary text panel
    ax8 = fig.add_subplot(2, 4, 8)
    ax8.axis('off')

    # Compute final vectors
    look_c = -R_total_correct[:, 2]
    up_c = R_total_correct[:, 1]
    right_c = R_total_correct[:, 0]
    pitch_c = np.degrees(np.arcsin(np.clip(-look_c[2], -1, 1)))

    look_w = -R_total_wrong[:, 2]
    pitch_w = np.degrees(np.arcsin(np.clip(-look_w[2], -1, 1)))

    summary = (
        "CAMERA TRANSFORM CHAIN\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "\n"
        "XML hierarchy:\n"
        "  torso_link\n"
        "    depth_camera_body\n"
        "      quat=[0.9135, 0, -0.4067, 0]\n"
        "      = Ry(-48°)   (tilt forward)\n"
        "    camera (depth_camera)\n"
        "      quat=[0.7071, 0, 0, -0.7071]\n"
        "      = Rz(-90°)   (rotate image)\n"
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        f"CORRECT  (Rz(-90°)):\n"
        f"  Look = [{look_c[0]:+.3f},{look_c[1]:+.3f},{look_c[2]:+.3f}]\n"
        f"        = forward + down {abs(pitch_c):.0f}°\n"
        f"  ImgUp = [{up_c[0]:+.3f},{up_c[1]:+.3f},{up_c[2]:+.3f}]\n"
        f"        = tilted forward-up\n"
        f"  ImgR  = [{right_c[0]:+.3f},{right_c[1]:+.3f},{right_c[2]:+.3f}]\n"
        f"        = robot's right (-Y)\n"
        "\n"
        f"WRONG  (Rx(-90°)):\n"
        f"  Look = [{look_w[0]:+.3f},{look_w[1]:+.3f},{look_w[2]:+.3f}]\n"
        f"        = sideways! pitch={pitch_w:.0f}°\n"
        "\n"
        "━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
        "Why Rz(-90°) is needed:\n"
        "  MuJoCo camera default:\n"
        "    look=-Z, imgUp=+Y\n"
        "  After Rz(-90°):\n"
        "    look=-Z (unchanged)\n"
        "    imgUp → +X (forward)\n"
        "  Combined with Ry(-48°) body:\n"
        "    camera tilts to look\n"
        "    forward+down at ~42°"
    )
    ax8.text(0.05, 0.95, summary, transform=ax8.transAxes,
             fontsize=9, fontfamily='monospace', verticalalignment='top',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                       alpha=0.8))

    fig.suptitle(
        'Depth Camera Quaternion: Complete Transform Chain Analysis\n'
        'cam_quat = [0.7071, 0, 0, -0.7071] = Rz(-90°)',
        fontsize=14, fontweight='bold', y=0.99
    )
    plt.tight_layout(rect=[0, 0, 1, 0.94])

    save_path = 'deploy/cam_quat_visualization.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Saved to {save_path}")

    # =========================================================================
    # Print detailed numerical analysis
    # =========================================================================
    print("\n" + "=" * 70)
    print("NUMERICAL ANALYSIS")
    print("=" * 70)

    print("\n[1] Body quaternion: [0.9135, 0, -0.4067, 0] = Ry(-48°)")
    print(f"    R_body =\n{np.array2string(R_body, precision=3, suppress_small=True)}")
    body_fwd = R_body[:, 0]
    print(f"    Body +X direction: [{body_fwd[0]:+.3f}, {body_fwd[1]:+.3f}, {body_fwd[2]:+.3f}]")
    print(f"    → The body frame is tilted forward by ~48°")

    print("\n[2] Camera quaternion: [0.7071, 0, 0, -0.7071] = Rz(-90°)")
    print(f"    R_cam =\n{np.array2string(R_cam_correct, precision=3, suppress_small=True)}")
    print(f"    In camera's body frame:")
    print(f"      look direction (-Z) = [0, 0, -1] → straight down")
    print(f"      image up      (+Y) = [1, 0,  0] → +X = forward")
    print(f"      image right   (+X) = [0,-1,  0] → -Y = right")
    print(f"    → This rotates the image so 'up in image' = robot forward")

    print(f"\n[3] Total: R_total = R_body @ R_cam")
    print(f"    R_total =\n{np.array2string(R_total_correct, precision=3, suppress_small=True)}")
    print(f"    Look direction  = [{look_c[0]:+.3f}, {look_c[1]:+.3f}, {look_c[2]:+.3f}]")
    print(f"    Pitch           = {pitch_c:+.1f}° (positive = looking down)")
    print(f"    Image up        = [{up_c[0]:+.3f}, {up_c[1]:+.3f}, {up_c[2]:+.3f}]")
    print(f"    Image right     = [{right_c[0]:+.3f}, {right_c[1]:+.3f}, {right_c[2]:+.3f}]")
    print(f"    → Camera looks forward and ~42° downward ✓")

    print(f"\n[4] WRONG Rx(-90°) comparison:")
    up_w = R_total_wrong[:, 1]
    right_w = R_total_wrong[:, 0]
    print(f"    Look = [{look_w[0]:+.3f}, {look_w[1]:+.3f}, {look_w[2]:+.3f}]")
    print(f"    → Looks sideways (-Y direction) = robot's right ✗")


if __name__ == "__main__":
    main()
