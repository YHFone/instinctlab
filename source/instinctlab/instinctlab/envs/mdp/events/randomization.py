from __future__ import annotations

import torch
from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.mdp.events import _randomize_prop_by_op
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg
from isaaclab.sensors import RayCaster

from instinctlab.sensors.grouped_ray_caster import GroupedRayCaster

if TYPE_CHECKING:
    from isaaclab.sensors import Camera, RayCasterCamera

    from instinctlab.sensors.grouped_ray_caster import GroupedRayCasterCamera


def randomize_body_com_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_pose_range: dict[str, tuple[float, float]] = dict(),
    operation: Literal["add", "abs"] = "add",
):
    """Randomizes the center of mass (COM) of the bodies in the asset.

    Args:
        - com_pose_range: (dict[str, tuple[float, float]])
            where keys are ["x", "y", "z", "roll", "pitch", "yaw"]
            and values are tuples representing the range for each component.
        - operation: (str) "add" or "abs", determines how the randomization is applied.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    coms = asset.root_physx_view.get_coms().clone()  # (num_envs, num_bodies, 7) (x, y, z, qx, qy, qz, qw)
    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(coms.shape[1], dtype=torch.long, device=coms.device)
        num_bodies = coms.shape[1]
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.long, device=coms.device)
        num_bodies = len(body_ids)
    # flatten env_ids and body_ids for indexing
    env_ids = env_ids.to(coms.device)
    env_ids_, body_ids_ = torch.meshgrid(env_ids, body_ids, indexing="ij")
    env_ids_ = env_ids_.flatten()  # (num_envs * num_bodies,)
    body_ids_ = body_ids_.flatten()  # (num_envs * num_bodies,)

    range_list = [com_pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=coms.device)  # (6, 2)
    rand_samples = math_utils.sample_uniform(
        ranges[:, 0],
        ranges[:, 1],
        (len(env_ids) * num_bodies, 6),
        device=coms.device,
    )
    position_samples = rand_samples[..., :3]  # (num_envs * num_bodies, 3)
    rotation_samples = math_utils.quat_from_euler_xyz(
        rand_samples[..., 3],
        rand_samples[..., 4],
        rand_samples[..., 5],
    )  # (num_envs * num_bodies, 4) (w, x, y, z)
    if operation == "add":
        coms[env_ids_, body_ids_, :3] += position_samples
        coms[env_ids_, body_ids_, 3:] = math_utils.convert_quat(
            math_utils.quat_mul(
                math_utils.convert_quat(coms[env_ids_, body_ids_, 3:], to="wxyz"),
                rotation_samples,
            ),
            to="xyzw",
        )
    elif operation == "abs":
        coms[env_ids_, body_ids_, :3] = position_samples
        coms[env_ids_, body_ids_, 3:] = math_utils.convert_quat(
            rotation_samples,
            to="xyzw",
        )
    else:
        raise ValueError(f"Invalid operation: {operation}. Use 'add' or 'abs'.")

    # set the new COMs to the asset
    asset.root_physx_view.set_coms(coms.cpu(), env_ids.cpu())


def randomize_rigid_body_coms(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    coms_x_distribution_params: tuple[float, float],
    coms_y_distribution_params: tuple[float, float],
    coms_z_distribution_params: tuple[float, float],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    .. tip::
        This function uses CPU tensors to assign the body coms. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # get the Center of Mass
    coms = asset.root_physx_view.get_coms()

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=coms.device)
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device=coms.device)
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device=coms.device)

    # sample from given range
    # note: we modify the coms in-place for all environments
    #   however, the setter takes care that only the coms of the specified environments are modified
    coms[..., 0] = _randomize_prop_by_op(
        coms[..., 0],
        coms_x_distribution_params,
        env_ids,
        body_ids,
        operation="add",
        distribution=distribution,
    )
    coms[..., 1] = _randomize_prop_by_op(
        coms[..., 1],
        coms_y_distribution_params,
        env_ids,
        body_ids,
        operation="add",
        distribution=distribution,
    )
    coms[..., 2] = _randomize_prop_by_op(
        coms[..., 2],
        coms_z_distribution_params,
        env_ids,
        body_ids,
        operation="add",
        distribution=distribution,
    )

    # set the new coms
    asset.root_physx_view.set_coms(coms, env_ids)


def randomize_default_joint_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if offset_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, offset_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]  # type: ignore
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_ray_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_pose_ranges: dict[str, tuple[float, float]],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the ray_starts and ray_directions of the sensor to mimic the sensor installation errors.

    Args:
        - offset_pose_ranges: (dict[str, tuple[float, float]])
            where keys are ["x", "y", "z", "roll", "pitch", "yaw"]
            and values are tuples representing the range for each component.
        - distribution: (str) "uniform" or "log_uniform" or "gaussian", determines the distribution of the randomization.
    """
    num_env_ids = env.scene.num_envs if env_ids is None else len(env_ids)
    # extract the used quantities (to enable type-hinting)
    sensor: RayCaster | GroupedRayCaster = env.scene[asset_cfg.name]
    ray_starts = sensor.ray_starts[env_ids]  # (num_envs, num_rays, 3)
    ray_directions = sensor.ray_directions[env_ids]  # (num_envs, num_rays, 3)
    # sample from given range
    range_list = [offset_pose_ranges.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=ray_starts.device)  # (6, 2)
    rand_samples = (
        math_utils.sample_uniform(
            ranges[:, 0],
            ranges[:, 1],
            (num_env_ids, 6),
            device=ray_starts.device,
        )[..., None, :]
        .repeat(1, sensor.num_rays, 1)
        .flatten(0, 1)
    )
    position_samples = rand_samples[..., :3]  # (num_envs * num_rays, 3)
    rotation_samples = math_utils.quat_from_euler_xyz(
        rand_samples[..., 3],
        rand_samples[..., 4],
        rand_samples[..., 5],
    )  # (num_envs * num_rays, 4) (w, x, y, z)
    # apply the randomization
    ray_starts += position_samples.reshape(ray_starts.shape)
    ray_directions = math_utils.quat_apply(rotation_samples.reshape(*ray_directions.shape[:-1], 4), ray_directions)

    sensor.ray_starts[env_ids] = ray_starts
    sensor.ray_directions[env_ids] = ray_directions


def randomize_camera_offsets(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    offset_pose_ranges: dict[str, tuple[float, float]],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the camera offset pose to mimic the sensor installation errors, which lead to imperfect camera calibration.

    Args:
        - offset_pose_ranges: (dict[str, tuple[float, float]])
            where keys are ["x", "y", "z", "roll", "pitch", "yaw"]
            and values are tuples representing the range for each component.
        - distribution: (str) "uniform" or "log_uniform" or "gaussian", determines the distribution of the randomization.
    """
    # extract the used quantities (to enable type-hinting), as well as all inherited classes
    sensor: Camera | RayCasterCamera | GroupedRayCasterCamera = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=sensor._device)

    # get the camera pose
    camera_offset_pos = torch.tensor(list(sensor.cfg.offset.pos), device=sensor._device).repeat(sensor._view.count, 1)
    camera_quat_w = math_utils.convert_camera_frame_orientation_convention(
        torch.tensor([sensor.cfg.offset.rot], device=sensor._device),
        origin=sensor.cfg.offset.convention,
        target="world",
    )
    camera_offset_quat = camera_quat_w.repeat(sensor._view.count, 1)
    camera_offset_pos = camera_offset_pos[env_ids]
    camera_offset_quat = camera_offset_quat[env_ids]

    # sample from given range
    camera_offset_pos[..., 0] = _randomize_prop_by_op(
        camera_offset_pos[..., 0].unsqueeze(-1),
        offset_pose_ranges.get("x", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_offset_pos[..., 1] = _randomize_prop_by_op(
        camera_offset_pos[..., 1].unsqueeze(-1),
        offset_pose_ranges.get("y", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_offset_pos[..., 2] = _randomize_prop_by_op(
        camera_offset_pos[..., 2].unsqueeze(-1),
        offset_pose_ranges.get("z", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_euler_w = math_utils.euler_xyz_from_quat(camera_offset_quat)

    camera_euler_roll = _randomize_prop_by_op(
        camera_euler_w[0].unsqueeze(-1),
        offset_pose_ranges.get("roll", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_euler_pitch = _randomize_prop_by_op(
        camera_euler_w[1].unsqueeze(-1),
        offset_pose_ranges.get("pitch", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_euler_yaw = _randomize_prop_by_op(
        camera_euler_w[2].unsqueeze(-1),
        offset_pose_ranges.get("yaw", (0.0, 0.0)),
        None,
        slice(None),
        operation="add",
        distribution=distribution,
    ).squeeze(-1)

    camera_offset_quat = math_utils.quat_from_euler_xyz(
        camera_euler_roll,
        camera_euler_pitch,
        camera_euler_yaw,
    )
    camera_pos_w, camera_quat_w = sensor._compute_view_world_poses(env_ids)
    camera_pos_w += math_utils.quat_apply(camera_quat_w, camera_offset_pos)
    camera_quat_w = math_utils.quat_mul(camera_quat_w, camera_offset_quat)

    # set the new camera pose
    # Note: the offset will be updated automatically,
    # and the attachment relation is kept.
    sensor.set_world_poses(
        camera_pos_w,
        camera_quat_w,
        env_ids=env_ids,
        convention=sensor.cfg.offset.convention,
    )
