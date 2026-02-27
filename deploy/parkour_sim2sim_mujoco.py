#!/usr/bin/env python3
"""
Parkour Sim2Sim Deployment Script for G1 Robot in MuJoCo

Runs the trained parkour policy (depth encoder + actor) in MuJoCo simulation.
Mirrors the observation construction and action application from the real-robot
ParkourAgent (instinct_onboard) for accurate sim2sim transfer.

Usage:
    python deploy/parkour_sim2sim_mujoco.py
    python deploy/parkour_sim2sim_mujoco.py --vx 0.8 --duration 30
    python deploy/parkour_sim2sim_mujoco.py --no_render
    python deploy/parkour_sim2sim_mujoco.py --record
"""

from __future__ import annotations

import argparse
import os
import re
import time
from collections import OrderedDict

import cv2
import mujoco
import numpy as np
import onnxruntime as ort
import yaml

# =============================================================================
# Isaac Lab simulation joint order (from robot_cfgs.py G1_29Dof_TorsoBase)
# This is the order the policy was trained with.
# =============================================================================
ISAAC_JOINT_NAMES = [
    "left_shoulder_pitch_joint",   # 0
    "right_shoulder_pitch_joint",  # 1
    "waist_pitch_joint",           # 2
    "left_shoulder_roll_joint",    # 3
    "right_shoulder_roll_joint",   # 4
    "waist_roll_joint",            # 5
    "left_shoulder_yaw_joint",     # 6
    "right_shoulder_yaw_joint",    # 7
    "waist_yaw_joint",             # 8
    "left_elbow_joint",            # 9
    "right_elbow_joint",           # 10
    "left_hip_pitch_joint",        # 11
    "right_hip_pitch_joint",       # 12
    "left_wrist_roll_joint",       # 13
    "right_wrist_roll_joint",      # 14
    "left_hip_roll_joint",         # 15
    "right_hip_roll_joint",        # 16
    "left_wrist_pitch_joint",      # 17
    "right_wrist_pitch_joint",     # 18
    "left_hip_yaw_joint",          # 19
    "right_hip_yaw_joint",         # 20
    "left_wrist_yaw_joint",        # 21
    "right_wrist_yaw_joint",       # 22
    "left_knee_joint",             # 23
    "right_knee_joint",            # 24
    "left_ankle_pitch_joint",      # 25
    "right_ankle_pitch_joint",     # 26
    "left_ankle_roll_joint",       # 27
    "right_ankle_roll_joint",      # 28
]

NUM_JOINTS = 29


# =============================================================================
# Circular Buffer (matches instinct_onboard.utils.CircularBuffer)
# =============================================================================
class CircularBuffer:
    """Fixed-length circular buffer. On first append, fills entire buffer with that value."""

    def __init__(self, length: int):
        self._buffer: np.ndarray | None = None
        self._length = length
        self._num_pushes = 0

    def append(self, value: np.ndarray):
        if self._buffer is None:
            self._buffer = np.zeros((self._length,) + tuple(value.shape), dtype=np.float32)
        if self._num_pushes == 0:
            self._buffer[:] = value
        else:
            self._buffer = np.roll(self._buffer, -1, axis=0)
            self._buffer[-1] = value
        self._num_pushes += 1

    @property
    def buffer(self):
        return self._buffer

    def reset(self):
        if self._buffer is None:
            return
        self._buffer[:] = 0.0
        self._num_pushes = 0


# =============================================================================
# Quaternion utilities
# =============================================================================
def quat_to_rot_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    """Convert quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = quat_wxyz
    return np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)],
    ])


def quat_rotate_inverse(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector by the inverse of the quaternion (world → body frame)."""
    R = quat_to_rot_matrix(quat_wxyz)
    return R.T @ vec


# =============================================================================
# Main Sim2Sim class
# =============================================================================
class ParkourSim2Sim:
    """Runs the trained G1 parkour policy in MuJoCo."""

    def __init__(self, cfg: dict, args):
        self.cfg = cfg
        self.args = args

        # -- Paths --
        self.model_dir = os.path.expanduser(cfg["paths"]["model_dir"])
        self.xml_path = os.path.expanduser(cfg["paths"]["xml_path"])

        # -- Simulation params --
        self.sim_dt = cfg["simulation"]["sim_dt"]       # 0.001
        self.decimation = cfg["simulation"]["decimation"]  # 20
        self.control_dt = self.sim_dt * self.decimation  # 0.02s = 50 Hz

        # -- Joint index mappings --
        self.isaac_to_mujoco = np.array(cfg["mappings"]["isaac_to_mujoco"], dtype=np.int32)
        self.mujoco_to_isaac = np.array(cfg["mappings"]["mujoco_to_isaac"], dtype=np.int32)

        # -- Default joint angles (MuJoCo order) → also build Isaac order --
        self.default_pos_muj = np.array(cfg["initial_joint_angles"], dtype=np.float32)
        self.default_pos_isaac = np.zeros(NUM_JOINTS, dtype=np.float32)
        for mj_idx in range(NUM_JOINTS):
            isaac_idx = self.mujoco_to_isaac[mj_idx]
            self.default_pos_isaac[isaac_idx] = self.default_pos_muj[mj_idx]

        # -- PD gains (MuJoCo order) — from training ImplicitActuator config --
        # IsaacLab's ImplicitActuator solves PD within the constraint solver (very stiff).
        # MuJoCo's position actuators apply PD forces explicitly, requiring higher gains
        # to match the same joint stiffness. kp_scale compensates for this difference.
        self.kp_scale = args.kp_scale
        self.kd_scale = args.kd_scale
        self.kp = np.array(cfg["joints"]["kp"], dtype=np.float32) * self.kp_scale
        self.kd = np.array(cfg["joints"]["kd"], dtype=np.float32) * self.kd_scale

        # -- Training effort limits per joint (MuJoCo order) --
        self._build_effort_limits()

        # -- Training armature per joint (MuJoCo order) --
        self._build_armature()

        # -- Action scales (Isaac order) — build from env.yaml regex rules --
        self._build_action_scales()

        # -- Depth camera config --
        dc = cfg["depth_camera"]
        self.depth_width = dc["width"]       # 32 (after crop)
        self.depth_height = dc["height"]     # 18 (after crop)
        self.depth_history = dc["history"]   # 8
        self.depth_skip = dc["history_skip_frames"]  # 5
        self.depth_orig_w = dc["original_width"]   # 64
        self.depth_orig_h = dc["original_height"]  # 36

        # Crop region: [top, bottom, left, right] = [18, 0, 16, 16]
        self.crop_region = (18, 0, 16, 16)
        self.depth_range = (0.0, 2.5)
        self.gaussian_kernel = (3, 3)
        self.gaussian_sigma = 1.0

        # Depth buffer: store enough frames for skip-based downsampling
        # Training: 37 frames stored, skip 5, output 8 frames
        # In sim at 50 Hz control, we run 1 depth frame per control step
        # depth_obs_indices picks 8 frames from buffer with skip spacing
        self.depth_buffer_len = 50  # >= 37
        self.depth_buffer = CircularBuffer(length=self.depth_buffer_len)
        # Compute downsample indices: 8 frames, spaced by skip_frames
        # Matching parkour_agent.py logic:
        #   frames = (data_histories - 1) / downsample_factor + 1 = (37-1)/5 + 1 = 8.2 -> int(8)
        #   In sim, sim_frequency = 50 Hz (control rate), real_downsample = 1 * skip = 5
        #   indices = linspace(-1 - 5*(8-1), -1, 8) = linspace(-36, -1, 8)
        self.depth_obs_indices = np.linspace(
            -1 - self.depth_skip * (self.depth_history - 1), -1, self.depth_history
        ).astype(int)

        # -- Observation history buffers (all with history_length=8) --
        self.obs_history_len = 8
        self.obs_history_buffers: OrderedDict[str, CircularBuffer] = OrderedDict()
        for name in ["base_ang_vel", "projected_gravity", "velocity_commands",
                      "joint_pos", "joint_vel", "actions"]:
            self.obs_history_buffers[name] = CircularBuffer(length=self.obs_history_len)

        # Observation scales
        self.obs_scales = {
            "base_ang_vel": 0.25,
            "joint_vel": 0.05,
        }

        # -- Last action (Isaac order, 29-dim) --
        self.last_action = np.zeros(NUM_JOINTS, dtype=np.float32)

        # -- Velocity command --
        self.velocity_command = np.array(
            [args.vx, args.vy, args.yaw_rate], dtype=np.float32
        )

        # -- Load MuJoCo model --
        self._init_mujoco()

        # -- Load ONNX models --
        self._load_onnx()

        # -- Depth renderer (always needed for policy) --
        self.depth_renderer = mujoco.Renderer(
            self.model, height=self.depth_orig_h, width=self.depth_orig_w
        )
        self.depth_renderer.enable_depth_rendering()

        # -- Create visualization renderers --
        if not args.no_render or args.record:
            self._init_renderers()

        print(f"ParkourSim2Sim initialized:")
        print(f"  Sim dt: {self.sim_dt}s, Decimation: {self.decimation}, Control: {self.control_dt}s ({1/self.control_dt:.0f} Hz)")
        print(f"  Velocity command: vx={args.vx}, vy={args.vy}, yaw={args.yaw_rate}")
        print(f"  Depth encoder input: ({self.depth_history}, {self.depth_height}, {self.depth_width})")
        print(f"  Depth obs indices: {self.depth_obs_indices}")

    def _build_effort_limits(self):
        """Build per-joint effort limits in MuJoCo order from training config."""
        # From env.yaml actuators.*.effort_limit_sim
        effort_rules = {
            r".*_hip_yaw_joint": 88.0,
            r".*_hip_roll_joint": 139.0,
            r".*_hip_pitch_joint": 88.0,
            r".*_knee_joint": 139.0,
            r".*_ankle_pitch_joint": 50.0,
            r".*_ankle_roll_joint": 50.0,
            r"waist_roll_joint": 50.0,
            r"waist_pitch_joint": 50.0,
            r"waist_yaw_joint": 88.0,
            r".*_shoulder_pitch_joint": 25.0,
            r".*_shoulder_roll_joint": 25.0,
            r".*_shoulder_yaw_joint": 25.0,
            r".*_elbow_joint": 25.0,
            r".*_wrist_roll_joint": 25.0,
            r".*_wrist_pitch_joint": 5.0,
            r".*_wrist_yaw_joint": 5.0,
        }
        MUJOCO_NAMES = [
            'waist_pitch_joint', 'waist_roll_joint', 'waist_yaw_joint',
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
        ]
        self.effort_limit = np.zeros(NUM_JOINTS, dtype=np.float32)
        for i, name in enumerate(MUJOCO_NAMES):
            for pattern, limit in effort_rules.items():
                if re.search(pattern, name):
                    self.effort_limit[i] = limit
                    break

    def _build_armature(self):
        """Build per-joint armature values in MuJoCo order from training config."""
        armature_rules = {
            r".*_hip_pitch_joint": 0.01017752,
            r".*_hip_roll_joint": 0.025101925,
            r".*_hip_yaw_joint": 0.01017752,
            r".*_knee_joint": 0.025101925,
            r".*_ankle_pitch_joint": 0.00721945,
            r".*_ankle_roll_joint": 0.00721945,
            r"waist_roll_joint": 0.00721945,
            r"waist_pitch_joint": 0.00721945,
            r"waist_yaw_joint": 0.01017752,
            r".*_shoulder_pitch_joint": 0.003609725,
            r".*_shoulder_roll_joint": 0.003609725,
            r".*_shoulder_yaw_joint": 0.003609725,
            r".*_elbow_joint": 0.003609725,
            r".*_wrist_roll_joint": 0.003609725,
            r".*_wrist_pitch_joint": 0.003609725,
            r".*_wrist_yaw_joint": 0.003609725,
        }
        MUJOCO_NAMES = [
            'waist_pitch_joint', 'waist_roll_joint', 'waist_yaw_joint',
            'left_hip_pitch_joint', 'left_hip_roll_joint', 'left_hip_yaw_joint',
            'left_knee_joint', 'left_ankle_pitch_joint', 'left_ankle_roll_joint',
            'right_hip_pitch_joint', 'right_hip_roll_joint', 'right_hip_yaw_joint',
            'right_knee_joint', 'right_ankle_pitch_joint', 'right_ankle_roll_joint',
            'left_shoulder_pitch_joint', 'left_shoulder_roll_joint', 'left_shoulder_yaw_joint',
            'left_elbow_joint', 'left_wrist_roll_joint', 'left_wrist_pitch_joint', 'left_wrist_yaw_joint',
            'right_shoulder_pitch_joint', 'right_shoulder_roll_joint', 'right_shoulder_yaw_joint',
            'right_elbow_joint', 'right_wrist_roll_joint', 'right_wrist_pitch_joint', 'right_wrist_yaw_joint',
        ]
        self.armature = np.zeros(NUM_JOINTS, dtype=np.float32)
        for i, name in enumerate(MUJOCO_NAMES):
            for pattern, val in armature_rules.items():
                if re.search(pattern, name):
                    self.armature[i] = val
                    break

    def _build_action_scales(self):
        """Build per-joint action scales in Isaac order from env.yaml regex rules."""
        # Action scale rules from training env.yaml
        scale_rules = {
            r".*_hip_yaw_joint": 0.5475464652142303,
            r".*_hip_roll_joint": 0.3506614663788243,
            r".*_hip_pitch_joint": 0.5475464652142303,
            r".*_knee_joint": 0.3506614663788243,
            r".*_ankle_pitch_joint": 0.43857731392336724,
            r".*_ankle_roll_joint": 0.43857731392336724,
            r"waist_roll_joint": 0.43857731392336724,
            r"waist_pitch_joint": 0.43857731392336724,
            r"waist_yaw_joint": 0.5475464652142303,
            r".*_shoulder_pitch_joint": 0.43857731392336724,
            r".*_shoulder_roll_joint": 0.43857731392336724,
            r".*_shoulder_yaw_joint": 0.43857731392336724,
            r".*_elbow_joint": 0.43857731392336724,
            r".*_wrist_roll_joint": 0.43857731392336724,
            r".*_wrist_pitch_joint": 0.07450087032950714,
            r".*_wrist_yaw_joint": 0.07450087032950714,
        }
        self.action_scale_isaac = np.zeros(NUM_JOINTS, dtype=np.float32)
        for i, name in enumerate(ISAAC_JOINT_NAMES):
            for pattern, scale in scale_rules.items():
                if re.search(pattern, name):
                    self.action_scale_isaac[i] = scale
                    break
        print(f"Action scales (Isaac order): {self.action_scale_isaac}")

    def _init_mujoco(self):
        """Load MuJoCo model and set initial state."""
        print(f"Loading MuJoCo model: {self.xml_path}")
        self.model = mujoco.MjModel.from_xml_path(self.xml_path)
        self.model.opt.timestep = self.sim_dt
        # Ensure offscreen framebuffer is large enough for visualization
        self.model.vis.global_.offwidth = max(self.model.vis.global_.offwidth, 1280)
        self.model.vis.global_.offheight = max(self.model.vis.global_.offheight, 720)
        self.data = mujoco.MjData(self.model)

        # Verify camera
        self.cam_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_CAMERA, "depth_camera"
        )
        if self.cam_id < 0:
            print("WARNING: 'depth_camera' not found, listing available cameras:")
            for i in range(self.model.ncam):
                name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_CAMERA, i)
                print(f"  [{i}] {name}")

        # Get torso body id for tracking
        self.torso_body_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )

        # Set initial pose: base position + orientation + joint angles (MuJoCo order)
        # qpos layout: [x, y, z, qw, qx, qy, qz, joint0, joint1, ..., joint28]
        self.data.qpos[0:3] = [0, 0, 0.855]        # base position
        self.data.qpos[3:7] = [1, 0, 0, 0]          # base orientation (wxyz)
        self.data.qpos[7:7 + NUM_JOINTS] = self.default_pos_muj

        # Remove built-in joint damping to avoid double-counting with PD kd
        self.model.dof_damping[:] = 0.0

        # Convert motor actuators to position-mode actuators matching training
        # ImplicitActuator: force = kp * (ctrl - pos) - kd * vel, clamped to effort_limit
        for i in range(self.model.nu):
            # gaintype = FIXED (0): gain = gainprm[0]
            self.model.actuator_gaintype[i] = 0
            self.model.actuator_gainprm[i, 0] = self.kp[i]
            # biastype = AFFINE (1): bias = biasprm[0] + biasprm[1]*pos + biasprm[2]*vel
            self.model.actuator_biastype[i] = 1
            self.model.actuator_biasprm[i, 0] = 0.0
            self.model.actuator_biasprm[i, 1] = -self.kp[i]
            self.model.actuator_biasprm[i, 2] = -self.kd[i]
            # ctrlrange: wide enough for target positions (joint limits)
            self.model.actuator_ctrlrange[i] = [-2 * np.pi, 2 * np.pi]
            self.model.actuator_ctrllimited[i] = 1  # enable ctrlrange clamp
            # forcerange: scale proportionally with kp_scale to preserve the
            # same torque saturation profile as training. Without scaling,
            # the high kp makes the actuator saturate at much smaller errors.
            scaled_effort = self.effort_limit[i] * self.kp_scale
            self.model.actuator_forcerange[i] = [-scaled_effort, scaled_effort]
            self.model.actuator_forcelimited[i] = 1

        # Set armature (rotational inertia) per DOF to match training
        # dof_armature: first 6 DOFs are free joint (translation + rotation), then 29 joint DOFs
        for i in range(NUM_JOINTS):
            self.model.dof_armature[6 + i] = self.armature[i]

        # Move torso_link geoms to group 4 so they can be excluded from depth
        # rendering.  Training uses RayCaster with mesh_prim_paths that
        # excludes torso_link -- the camera is mounted on it and shouldn't
        # see its own housing.
        torso_body_id_local = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_BODY, "torso_link"
        )
        for gi in range(self.model.ngeom):
            if self.model.geom_bodyid[gi] == torso_body_id_local:
                self.model.geom_group[gi] = 4

        # Fix depth camera orientation: the XML inner quaternion [0.7071,-0.7071,0,0]
        # (Rx(-90°)) makes the camera look sideways (-Y).  The correct orientation
        # for a forward-looking camera is Rz(-90°) = [0.7071, 0, 0, -0.7071].
        # This makes the camera look along +X (forward) and down at ~42° below
        # horizontal, matching the training RayCasterCamera convention.
        # self.model.cam_quat[self.cam_id] = [0.7071, 0.0, 0.0, -0.7071]

        # Build mjvOption for depth rendering: show groups 0,1 (floor + robot
        # body parts) but hide group 4 (torso_link).
        self._depth_scene_option = mujoco.MjvOption()
        # geomgroup is a bool array[6]; by default all True.  Disable group 4.
        self._depth_scene_option.geomgroup[4] = 0

        mujoco.mj_forward(self.model, self.data)
        print(f"MuJoCo model loaded: {self.model.nu} actuators, {self.model.nq} qpos, {self.model.nv} qvel")

    def _load_onnx(self):
        """Load ONNX models for depth encoder and actor."""
        providers = ort.get_available_providers()
        print(f"ONNX Runtime providers: {providers}")

        depth_path = os.path.join(self.model_dir, "0-depth_encoder.onnx")
        actor_path = os.path.join(self.model_dir, "actor.onnx")

        self.depth_encoder = ort.InferenceSession(depth_path, providers=providers)
        self.actor = ort.InferenceSession(actor_path, providers=providers)

        # Print input/output info
        for name, sess in [("depth_encoder", self.depth_encoder), ("actor", self.actor)]:
            inputs = [(inp.name, inp.shape) for inp in sess.get_inputs()]
            outputs = [(out.name, out.shape) for out in sess.get_outputs()]
            print(f"  {name}: inputs={inputs}, outputs={outputs}")

    def _init_renderers(self):
        """Create MuJoCo renderers for visualization."""
        # Third-person renderer
        self.tp_width, self.tp_height = 960, 540
        self.tp_renderer = mujoco.Renderer(
            self.model, height=self.tp_height, width=self.tp_width
        )

        # Third-person camera
        self.tp_cam = mujoco.MjvCamera()
        self.tp_cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        self.tp_cam.trackbodyid = self.torso_body_id
        self.tp_cam.distance = 3.5
        self.tp_cam.azimuth = 135
        self.tp_cam.elevation = -20
        self.tp_cam.lookat[:] = [0, 0, 0.8]

        # Visualization scene option: default MjvOption only shows groups 0-2.
        # Torso geoms were moved to group 4 (for depth exclusion), so enable
        # group 4 here to keep them visible in the third-person view.
        self._viz_scene_option = mujoco.MjvOption()
        self._viz_scene_option.geomgroup[4] = 1

        # Video recording
        self.video_writer = None
        if self.args.record:
            video_dir = os.path.join(os.path.dirname(__file__), "videos")
            os.makedirs(video_dir, exist_ok=True)
            video_path = os.path.join(
                video_dir,
                f"parkour_{time.strftime('%Y%m%d_%H%M%S')}.mp4"
            )
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self.video_writer = cv2.VideoWriter(
                video_path, fourcc, 30.0, (self.tp_width, self.tp_height + 240)
            )
            print(f"Recording to {video_path}")

    # =========================================================================
    # Observation functions
    # =========================================================================
    def _get_base_ang_vel(self) -> np.ndarray:
        """Get base angular velocity in body frame. Shape: (3,)
        MuJoCo qvel[3:6] for free joints is already in the body (local) frame.
        """
        ang_vel_body = self.data.qvel[3:6].copy()
        return ang_vel_body.astype(np.float32)

    def _get_projected_gravity(self) -> np.ndarray:
        """Get gravity vector projected into body frame. Shape: (3,)"""
        quat_wxyz = self.data.qpos[3:7].copy()
        gravity_world = np.array([0.0, 0.0, -1.0])
        proj_grav = quat_rotate_inverse(quat_wxyz, gravity_world)
        return proj_grav.astype(np.float32)

    def _get_velocity_commands(self) -> np.ndarray:
        """Get velocity commands (vx, vy, yaw_rate). Shape: (3,)"""
        return self.velocity_command.copy()

    def _get_joint_pos_rel(self) -> np.ndarray:
        """Get joint positions relative to default, in Isaac order. Shape: (29,)"""
        # Read MuJoCo joint positions and map to Isaac order
        mj_pos = self.data.qpos[7:7 + NUM_JOINTS].copy()
        isaac_pos = np.zeros(NUM_JOINTS, dtype=np.float32)
        for i in range(NUM_JOINTS):
            isaac_pos[i] = mj_pos[self.isaac_to_mujoco[i]]
        return (isaac_pos - self.default_pos_isaac).astype(np.float32)

    def _get_joint_vel_rel(self) -> np.ndarray:
        """Get joint velocities in Isaac order. Shape: (29,)"""
        mj_vel = self.data.qvel[6:6 + NUM_JOINTS].copy()
        isaac_vel = np.zeros(NUM_JOINTS, dtype=np.float32)
        for i in range(NUM_JOINTS):
            isaac_vel[i] = mj_vel[self.isaac_to_mujoco[i]]
        return isaac_vel.astype(np.float32)

    def _get_last_action(self) -> np.ndarray:
        """Get last action in Isaac order. Shape: (29,)"""
        return self.last_action.copy()

    # =========================================================================
    # Depth image processing
    # =========================================================================
    def _render_depth(self) -> np.ndarray:
        """Render depth from MuJoCo and process it matching the training pipeline.
        Returns processed depth image, shape: (depth_height, depth_width) = (18, 32).
        Values normalized to [0, 1].
        """
        self.depth_renderer.update_scene(
            self.data, camera="depth_camera",
            scene_option=self._depth_scene_option,
        )
        raw_depth = self.depth_renderer.render().copy()  # float32, shape (36, 64)

        # MuJoCo depth renderer returns z-distance from the camera plane
        # (i.e. distance_to_image_plane), matching the training data type.

        # 0. Apply min_distance clipping (training: min_distance=0.1).
        # Anything closer than 0.1 m is set to max range so the policy
        # sees it as "far" (same as training RayCaster behaviour).
        d_min_clip = 0.1
        d_max = self.depth_range[1]  # 2.5
        raw_depth[raw_depth < d_min_clip] = d_max

        # 1. Crop: [top=18, bottom=0, left=16, right=16]
        top, bottom, left, right = self.crop_region
        h, w = raw_depth.shape
        if bottom == 0:
            depth_cropped = raw_depth[top:, left:w - right]
        else:
            depth_cropped = raw_depth[top:h - bottom, left:w - right]
        # Result: (18, 32)

        # 2. Gaussian blur
        depth_blurred = cv2.GaussianBlur(
            depth_cropped, self.gaussian_kernel,
            self.gaussian_sigma, self.gaussian_sigma
        )

        # 3. Clip and normalize to [0, 1]
        d_min, d_max2 = self.depth_range
        depth_clipped = np.clip(depth_blurred, d_min, d_max2)
        depth_norm = (depth_clipped - d_min) / (d_max2 - d_min)

        return depth_norm.astype(np.float32)

    def _get_depth_obs(self) -> np.ndarray:
        """Get depth observation: render current frame, store in buffer, return
        downsampled history. Shape: (depth_history, depth_height, depth_width) = (8, 18, 32).
        """
        depth_frame = self._render_depth()
        self.depth_buffer.append(depth_frame)
        # Downsample from buffer
        return self.depth_buffer.buffer[self.depth_obs_indices, ...]

    # =========================================================================
    # Build observation vector
    # =========================================================================
    def _build_proprio_obs(self) -> np.ndarray:
        """Build proprioceptive observation vector. Shape: (1, 768).
        Order: base_ang_vel, projected_gravity, velocity_commands,
               joint_pos_rel, joint_vel_rel, last_action
        Each term is scaled, put into history buffer, and flattened.
        """
        obs_terms = OrderedDict()
        obs_terms["base_ang_vel"] = self._get_base_ang_vel()       # (3,)
        obs_terms["projected_gravity"] = self._get_projected_gravity()  # (3,)
        obs_terms["velocity_commands"] = self._get_velocity_commands()  # (3,)
        obs_terms["joint_pos"] = self._get_joint_pos_rel()         # (29,)
        obs_terms["joint_vel"] = self._get_joint_vel_rel()         # (29,)
        obs_terms["actions"] = self._get_last_action()             # (29,)

        proprio_parts = []
        for name, value in obs_terms.items():
            # Apply scale
            if name in self.obs_scales:
                value = value * self.obs_scales[name]
            # Push to history buffer and read flattened history
            self.obs_history_buffers[name].append(value)
            hist = self.obs_history_buffers[name].buffer  # (8, dim)
            proprio_parts.append(hist.flatten())

        proprio = np.concatenate(proprio_parts).astype(np.float32)
        return proprio.reshape(1, -1)

    # =========================================================================
    # Policy inference step
    # =========================================================================
    def policy_step(self) -> np.ndarray:
        """Run one policy inference step.
        Returns target joint positions in MuJoCo order, shape: (29,).
        """
        # 1. Build proprioceptive observation
        proprio_obs = self._build_proprio_obs()  # (1, 768)

        # 2. Build depth observation
        depth_obs = self._get_depth_obs()  # (8, 18, 32)
        depth_input = depth_obs.reshape(
            1, self.depth_history, self.depth_height, self.depth_width
        ).astype(np.float32)

        # 3. Run depth encoder
        depth_enc_input_name = self.depth_encoder.get_inputs()[0].name
        depth_latent = self.depth_encoder.run(
            None, {depth_enc_input_name: depth_input}
        )[0]  # (1, 128)

        # 4. Concatenate proprio + depth latent → actor input
        actor_input = np.concatenate([proprio_obs, depth_latent], axis=1)  # (1, 896)

        # 5. Run actor
        actor_input_name = self.actor.get_inputs()[0].name
        action = self.actor.run(None, {actor_input_name: actor_input})[0]  # (1, 29)
        action = action.reshape(-1)  # (29,) Isaac order

        # Store as last action
        self.last_action = action.copy()

        # 6. Compute target joint positions: target = action * scale + default
        target_isaac = action * self.action_scale_isaac + self.default_pos_isaac

        # 7. Map to MuJoCo order
        target_muj = np.zeros(NUM_JOINTS, dtype=np.float32)
        for i in range(NUM_JOINTS):
            target_muj[self.isaac_to_mujoco[i]] = target_isaac[i]

        return target_muj

    # =========================================================================
    # Actuator control step
    # =========================================================================
    def set_targets(self, target_muj: np.ndarray):
        """Set target joint positions for position-mode actuators.
        MuJoCo handles PD control internally:
            force = kp * (ctrl - pos) - kd * vel, clamped to forcerange.
        """
        self.data.ctrl[:NUM_JOINTS] = target_muj

    # =========================================================================
    # Main simulation loop
    # =========================================================================
    def run(self):
        """Main simulation loop."""
        duration = self.args.duration
        render = not self.args.no_render

        print(f"\nStarting parkour sim2sim for {duration}s...")
        print("Press 'q' in the window or Ctrl+C to exit.\n")

        sim_step = 0
        control_step = 0
        target_muj = self.default_pos_muj.copy()
        start_time = time.time()

        # Warm up: run a few steps with default pose to let robot settle
        print("Warming up (1s standing)...")
        warmup_steps = int(1.0 / self.sim_dt)
        for _ in range(warmup_steps):
            self.set_targets(self.default_pos_muj)
            mujoco.mj_step(self.model, self.data)
        print("Warm-up done. Running policy...\n")

        # Initialize depth buffer with the current depth frame
        for _ in range(self.depth_buffer_len):
            depth_frame = self._render_depth()
            self.depth_buffer.append(depth_frame)

        try:
            total_sim_steps = int(duration / self.sim_dt)
            next_log_step = int(0.5 / self.sim_dt)  # first log at 0.5s
            wall_start = time.time()
            for sim_step in range(total_sim_steps):
                # Policy runs every `decimation` physics steps
                if sim_step % self.decimation == 0:
                    target_muj = self.policy_step()
                    control_step += 1

                    # Visualization at ~30 fps
                    if (render or self.args.record) and control_step % max(1, int(1.0 / (30.0 * self.control_dt))) == 0:
                        self._visualize(sim_step)

                # Set target positions + physics step
                self.set_targets(target_muj)
                mujoco.mj_step(self.model, self.data)

                # Real-time synchronization: sleep until wall-clock catches up
                # to simulation time. This keeps the display at 1x speed.
                sim_time_now = (sim_step + 1) * self.sim_dt
                wall_elapsed = time.time() - wall_start
                sleep_time = sim_time_now - wall_elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)

                # Periodic state logging
                if sim_step == next_log_step:
                    t = sim_step * self.sim_dt
                    bp = self.data.qpos[0:3]
                    q = self.data.qpos[3:7]
                    av = self.data.qvel[3:6]
                    print(f"  t={t:5.1f}s  pos=[{bp[0]:+6.3f},{bp[1]:+6.3f},{bp[2]:+6.3f}]  "
                          f"quat=[{q[0]:.3f},{q[1]:+.3f},{q[2]:+.3f},{q[3]:+.3f}]  "
                          f"angvel=[{av[0]:+.2f},{av[1]:+.2f},{av[2]:+.2f}]")
                    next_log_step += int(0.5 / self.sim_dt)

        except KeyboardInterrupt:
            print("\nInterrupted by user.")
        finally:
            if hasattr(self, "video_writer") and self.video_writer is not None:
                self.video_writer.release()
                print("Video saved.")
            if render:
                cv2.destroyAllWindows()

            elapsed = time.time() - start_time
            sim_time = sim_step * self.sim_dt
            rtf = sim_time / elapsed if elapsed > 0 else 0
            print(f"\nDone. Sim time: {sim_time:.1f}s, Wall time: {elapsed:.1f}s, RTF: {rtf:.2f}x")
            print(f"  Control steps: {control_step}, Physics steps: {sim_step}")
            # Print final base position
            base_pos = self.data.qpos[0:3]
            print(f"  Final base position: x={base_pos[0]:.3f}, y={base_pos[1]:.3f}, z={base_pos[2]:.3f}")

    # =========================================================================
    # Visualization
    # =========================================================================
    def _visualize(self, sim_step: int):
        """Render third-person view + depth visualization."""
        font = cv2.FONT_HERSHEY_SIMPLEX

        # -- Third person view --
        self.tp_renderer.update_scene(self.data, self.tp_cam, scene_option=self._viz_scene_option)
        tp_image = self.tp_renderer.render()
        tp_bgr = cv2.cvtColor(tp_image, cv2.COLOR_RGB2BGR)

        sim_time = sim_step * self.sim_dt
        base_pos = self.data.qpos[0:3]
        cv2.putText(tp_bgr, f"t={sim_time:.1f}s  pos=({base_pos[0]:.2f}, {base_pos[1]:.2f}, {base_pos[2]:.2f})",
                     (10, 30), font, 0.7, (0, 255, 0), 2)
        cv2.putText(tp_bgr, f"cmd: vx={self.velocity_command[0]:.2f} vy={self.velocity_command[1]:.2f} yaw={self.velocity_command[2]:.2f}",
                     (10, 60), font, 0.6, (0, 200, 255), 2)

        # -- Depth visualization --
        # Show the latest processed depth frame and 8-frame history grid
        if self.depth_buffer.buffer is not None:
            depth_obs = self.depth_buffer.buffer[self.depth_obs_indices, ...]  # (8, 18, 32)
            rows, cols = 2, 4
            tile_h, tile_w = self.depth_height, self.depth_width
            grid = np.zeros((rows * tile_h, cols * tile_w), dtype=np.uint8)
            for idx in range(min(depth_obs.shape[0], rows * cols)):
                r, c = divmod(idx, cols)
                tile = (np.clip(depth_obs[idx], 0, 1) * 255).astype(np.uint8)
                grid[r * tile_h:(r + 1) * tile_h, c * tile_w:(c + 1) * tile_w] = tile
            # Apply colormap and resize
            grid_color = cv2.applyColorMap(grid, cv2.COLORMAP_TURBO)
            depth_display_w = self.tp_width
            depth_display_h = 240
            grid_resized = cv2.resize(grid_color, (depth_display_w, depth_display_h),
                                       interpolation=cv2.INTER_NEAREST)
            cv2.putText(grid_resized, "Depth History (8 frames)", (10, 20),
                        font, 0.5, (255, 255, 255), 1)

            # Stack vertically
            final = np.vstack([tp_bgr, grid_resized])
        else:
            final = tp_bgr

        if not self.args.no_render:
            cv2.imshow("G1 Parkour Sim2Sim", final)

        if self.video_writer is not None:
            self.video_writer.write(final)

        if not self.args.no_render:
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                raise KeyboardInterrupt("User pressed 'q'")


# =============================================================================
# Entry point
# =============================================================================
def main():
    parser = argparse.ArgumentParser(
        description="G1 Parkour Sim2Sim Deployment in MuJoCo"
    )
    parser.add_argument(
        "--config", type=str,
        default=os.path.join(os.path.dirname(__file__), "config.yaml"),
        help="Path to deploy config YAML"
    )
    parser.add_argument("--duration", type=float, default=30.0,
                        help="Simulation duration (seconds)")
    parser.add_argument("--vx", type=float, default=0.5,
                        help="Forward velocity command (m/s)")
    parser.add_argument("--vy", type=float, default=0.0,
                        help="Lateral velocity command (m/s)")
    parser.add_argument("--yaw_rate", type=float, default=0.0,
                        help="Yaw rate command (rad/s)")
    parser.add_argument("--no_render", action="store_true",
                        help="Disable visualization")
    parser.add_argument("--record", action="store_true",
                        help="Record video to deploy/videos/")
    parser.add_argument("--kp_scale", type=float, default=1.0,
                        help="Scale factor for position PD stiffness (1.0 = training gains)")
    parser.add_argument("--kd_scale", type=float, default=1.0,
                        help="Scale factor for position PD damping (1.0 = training gains)")
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    sim = ParkourSim2Sim(cfg, args)
    sim.run()


if __name__ == "__main__":
    main()
