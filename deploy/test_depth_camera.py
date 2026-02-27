#!/usr/bin/env python3
"""
Depth Camera Test Script for G1 Robot in MuJoCo
Visualizes the depth image from the robot's onboard depth camera.

Usage:
    python deploy/test_depth_camera.py
    python deploy/test_depth_camera.py --duration 20
    python deploy/test_depth_camera.py --save_images
"""

import argparse
import os
import time

import cv2
import mujoco
import numpy as np


def main():
    parser = argparse.ArgumentParser(description="Test depth camera visualization")
    parser.add_argument("--xml_path", type=str,
                        default=os.path.expanduser(
                            "~/RL_robot/project-instinct/instinctlab/source/instinctlab/"
                            "instinctlab/assets/resources/unitree_g1/scene_29dof.xml"),
                        help="Path to MuJoCo scene XML")
    parser.add_argument("--camera_name", type=str, default="depth_camera",
                        help="Name of the depth camera in the XML")
    parser.add_argument("--duration", type=float, default=15.0,
                        help="Duration in seconds")
    parser.add_argument("--depth_min", type=float, default=0.0,
                        help="Minimum depth range (m)")
    parser.add_argument("--depth_max", type=float, default=2.5,
                        help="Maximum depth range (m)")
    parser.add_argument("--save_images", action="store_true",
                        help="Save depth images to deploy/depth_images/")
    parser.add_argument("--display_scale", type=int, default=8,
                        help="Scale factor for depth image display window")
    args = parser.parse_args()

    # -------------------------------------------------------------------------
    # Load model
    # -------------------------------------------------------------------------
    print(f"Loading MuJoCo model from {args.xml_path}")
    model = mujoco.MjModel.from_xml_path(args.xml_path)
    data = mujoco.MjData(model)

    # Verify camera exists
    cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, args.camera_name)
    if cam_id < 0:
        print(f"ERROR: Camera '{args.camera_name}' not found in the model!")
        print("Available cameras:")
        for i in range(model.ncam):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_CAMERA, i)
            print(f"  [{i}] {name}")
        return
    print(f"Found camera '{args.camera_name}' (id={cam_id})")

    # -------------------------------------------------------------------------
    # Set initial robot pose
    # -------------------------------------------------------------------------
    initial_qpos = [
        0, 0, 0.855,           # base position
        1, 0, 0, 0,            # base orientation (wxyz)
        0, 0, 0,               # waist (pitch, roll, yaw)
        -0.312, 0, 0, 0.669, -0.363, 0,   # left leg
        -0.312, 0, 0, 0.669, -0.363, 0,   # right leg
        0.2, 0.2, 0, 0.6, 0, 0, 0,        # left arm
        0.2, -0.2, 0, 0.6, 0, 0, 0,       # right arm
    ]
    data.qpos[:len(initial_qpos)] = initial_qpos
    mujoco.mj_forward(model, data)

    # -------------------------------------------------------------------------
    # PD control to hold initial pose
    # -------------------------------------------------------------------------
    num_actuators = model.nu
    kp = 300.0
    kd = 8.0
    target_q = data.qpos[7:7 + num_actuators].copy()  # target = initial joint angles
    print(f"PD control: KP={kp}, KD={kd}, {num_actuators} actuators")

    # -------------------------------------------------------------------------
    # Create renderers
    # -------------------------------------------------------------------------
    # Depth renderer (native camera resolution from XML: 64x36)
    depth_renderer = mujoco.Renderer(model, height=36, width=64)
    depth_renderer.enable_depth_rendering()

    # RGB renderer from depth camera (first-person view)
    rgb_renderer = mujoco.Renderer(model, height=36, width=64)

    # Third-person renderer (shows robot + terrain from outside)
    tp_width, tp_height = 640, 480
    tp_renderer = mujoco.Renderer(model, height=tp_height, width=tp_width)

    # Third-person camera parameters (will track the robot)
    tp_cam = mujoco.MjvCamera()
    tp_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
    tp_cam.trackbodyid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "torso_link")
    tp_cam.distance = 3.0
    tp_cam.azimuth = 135
    tp_cam.elevation = -20
    tp_cam.lookat[:] = [0, 0, 0.8]

    print("Renderers created (third-person + depth camera)")

    # Save directory
    if args.save_images:
        save_dir = os.path.join(os.path.dirname(__file__), "depth_images")
        os.makedirs(save_dir, exist_ok=True)
        print(f"Saving images to {save_dir}")

    # -------------------------------------------------------------------------
    # Main loop
    # -------------------------------------------------------------------------
    display_w = 64 * args.display_scale
    display_h = 36 * args.display_scale

    print(f"\nDepth camera test running for {args.duration}s")
    print(f"  Depth camera resolution: 64 x 36")
    print(f"  Third-person view: {tp_width} x {tp_height}")
    print(f"  Depth range: [{args.depth_min}, {args.depth_max}] m")
    print("Press 'q' in the window or Ctrl+C to exit.\n")

    frame_count = 0
    start_time = time.time()
    sim_dt = model.opt.timestep

    try:
        while time.time() - start_time < args.duration:
            # PD control: hold initial joint positions
            current_q = data.qpos[7:7 + num_actuators]
            current_dq = data.qvel[6:6 + num_actuators]
            data.ctrl[:] = kp * (target_q - current_q) - kd * current_dq

            # Step physics
            mujoco.mj_step(model, data)

            # Render at ~30 fps (skip physics-only frames)
            frame_count += 1
            if frame_count % max(1, int(1.0 / (30.0 * sim_dt))) != 0:
                continue

            # --- Third-person view (robot + terrain) ---
            tp_renderer.update_scene(data, tp_cam)
            tp_image = tp_renderer.render()
            tp_bgr = cv2.cvtColor(tp_image, cv2.COLOR_RGB2BGR)

            # --- Depth image (from onboard camera) ---
            depth_renderer.update_scene(data, camera=args.camera_name)
            raw_depth = depth_renderer.render()  # float32, distance in meters

            # Clip and normalize
            depth_clipped = np.clip(raw_depth, args.depth_min, args.depth_max)
            depth_norm = (depth_clipped - args.depth_min) / (args.depth_max - args.depth_min)

            # Colormap for visualization (TURBO gives nice gradient)
            depth_u8 = (depth_norm * 255).astype(np.uint8)
            depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_TURBO)

            # --- RGB image from depth camera (first-person) ---
            rgb_renderer.update_scene(data, camera=args.camera_name)
            rgb_image = rgb_renderer.render()
            rgb_bgr = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2BGR)

            # --- Layout: third-person on top, depth camera views on bottom ---
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Top row: third-person view (already at tp_width x tp_height)
            tp_display = tp_bgr.copy()
            cv2.putText(tp_display, "Third-Person (Robot + Terrain)", (10, 30),
                        font, 0.8, (0, 255, 0), 2)
            sim_time = frame_count * sim_dt
            cv2.putText(tp_display, f"t={sim_time:.1f}s", (tp_width - 120, 30),
                        font, 0.7, (0, 255, 0), 2)

            # Bottom row: depth camera RGB + Depth side by side
            # Scale to half of tp_width each
            half_w = tp_width // 2
            bottom_h = tp_height // 2

            rgb_display = cv2.resize(rgb_bgr, (half_w, bottom_h),
                                     interpolation=cv2.INTER_LINEAR)
            depth_display = cv2.resize(depth_color, (half_w, bottom_h),
                                       interpolation=cv2.INTER_NEAREST)

            # Labels
            cv2.putText(rgb_display, "Depth Cam RGB", (10, 25),
                        font, 0.6, (255, 255, 255), 2)
            cv2.putText(depth_display, "Depth Map", (10, 25),
                        font, 0.6, (255, 255, 255), 2)

            # Depth stats
            valid = raw_depth[raw_depth < args.depth_max]
            if len(valid) > 0:
                stats = f"min={valid.min():.2f}m max={valid.max():.2f}m mean={valid.mean():.2f}m"
            else:
                stats = "No valid depth"
            cv2.putText(depth_display, stats, (10, bottom_h - 10),
                        font, 0.4, (255, 255, 255), 1)

            bottom_row = np.hstack([rgb_display, depth_display])

            # Colorbar
            bar_h = 20
            gradient = np.linspace(0, 255, tp_width).astype(np.uint8)
            gradient = np.tile(gradient, (bar_h, 1))
            colorbar = cv2.applyColorMap(gradient, cv2.COLORMAP_TURBO)
            cv2.putText(colorbar, f"{args.depth_min:.1f}m", (5, 15),
                        font, 0.4, (255, 255, 255), 1)
            cv2.putText(colorbar, f"{args.depth_max:.1f}m",
                        (tp_width - 50, 15), font, 0.4, (255, 255, 255), 1)

            # Stack: top (third-person) + bottom (depth cam views) + colorbar
            final = np.vstack([tp_display, bottom_row, colorbar])
            cv2.imshow("G1 Depth Camera Test", final)

            # Save if requested
            if args.save_images:
                cv2.imwrite(os.path.join(save_dir, f"combined_{frame_count:06d}.png"), final)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                print("Quit key pressed.")
                break

    except KeyboardInterrupt:
        print("Interrupted by user.")
    finally:
        cv2.destroyAllWindows()
        elapsed = time.time() - start_time
        print(f"Done. Ran for {elapsed:.1f}s, rendered {frame_count} physics frames.")


if __name__ == "__main__":
    main()
