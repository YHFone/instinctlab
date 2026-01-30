from __future__ import annotations

import numpy as np
import re
import torch
from collections.abc import Sequence
from typing import TYPE_CHECKING

import isaacsim.core.utils.prims as prim_utils
import omni.log
import omni.physics.tensors.impl.api as physx
import warp as wp
from isaacsim.core.prims import XFormPrim
from isaacsim.core.simulation_manager import SimulationManager
from pxr import Gf, Usd, UsdGeom, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.sensors.ray_caster import RayCaster
from isaaclab.terrains.trimesh.utils import make_plane
from isaaclab.utils.warp import convert_to_warp_mesh

from instinctlab.utils.warp.raycast import raycast_mesh_grouped

if TYPE_CHECKING:
    from .grouped_ray_caster_cfg import GroupedRayCasterCfg


class GroupedRayCaster(RayCaster):
    """Grouped Ray Caster sensor reads multiple isaacsim prim path and keep updating the mesh
    positions before casting rays.
    """

    cfg: GroupedRayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: GroupedRayCasterCfg):
        super().__init__(cfg)
        self.meshes: dict[str, list[wp.Mesh]] = dict()  # {prim_path: [warp_meshes]}
        self.mesh_transforms: torch.Tensor | None = None  # shape (N, 4, 4)
        self.mesh_inv_transforms: torch.Tensor | None = None  # shape (N, 4, 4)
        self.mesh_prototype_ids: torch.Tensor | list = []  # int64 shape (N,)
        self.mesh_collision_groups: torch.Tensor | list = []  # int32 shape (N,)
        self.rigid_body_mesh_transform_segments: dict[str, slice] = dict()
        self.rigid_body_views: dict[str, physx.RigidBodyView] = dict()  # {prim_path: rigid_body_view}

    def _get_rigid_body_view(
        self, env_prim_path_expr: str, matched_leaf_names: list[str]
    ) -> physx.RigidBodyView | None:
        """Get the rigid body view for a given prim path, which should be the root prim of an articulation or a rigid body.
        NOTE: This logic is copied from contact_sensor, which may not be the most efficient way to get the rigid body view.
        TODO: The rigid_body_view acquiring logic may be improved... later.
        """
        parent_prim_paths = sim_utils.find_matching_prim_paths(env_prim_path_expr)
        body_names = list()
        for potential_body_name in matched_leaf_names:
            prim = prim_utils.get_prim_at_path(parent_prim_paths[0] + "/" + potential_body_name)
            if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                prim_path = prim.GetPath().pathString
                body_names.append(prim_path.rsplit("/", 1)[-1])

        # check that there is at least one body with contact reporter API
        if body_names:
            # construct regex expression for the body names
            body_names_regex = r"(" + "|".join(body_names) + r")"
            body_names_regex = f"{env_prim_path_expr}/{body_names_regex}"
            # convert regex expressions to glob expressions for PhysX
            body_names_glob = body_names_regex.replace(".*", "*")

            # create a rigid prim view for the given body names
            rigid_body_view = self._physics_sim_view.create_rigid_body_view(body_names_glob)
            return rigid_body_view

        else:
            # try to get the rigid body view for the parent prim if it is a single rigid body prim
            prim = prim_utils.get_prim_at_path(parent_prim_paths[0])
            if prim.IsValid() and prim.HasAPI(UsdPhysics.RigidBodyAPI):
                body_name_glob = env_prim_path_expr.replace(".*", "*")
                return self._physics_sim_view.create_rigid_body_view(body_name_glob)
            else:
                return None

    def _get_collision_groups_by_env_str(
        self,
        rigid_body_view: physx.RigidBodyView,
        env_name_pattern: str = "/World/envs/env_\\d+",
    ) -> list[int]:
        """Get the collision groups for each rigid body in the given rigid body view.
        This is used to get the collision groups for the meshes in the ray caster.
        """
        collision_groups = []
        for prim_path in rigid_body_view.prim_paths:
            # match the environment name pattern
            match = re.search(env_name_pattern, prim_path)
            if match:
                env_id = int(match.group(0).split("_")[-1])
                collision_groups.append(env_id)
            else:
                omni.log.warn(f"Could not match environment name pattern in {prim_path}.")
                collision_groups.append(-1)  # default to -1 if no match
        return collision_groups

    def _get_merged_mesh_from_xform_prim(self, xform_prim_path: str) -> tuple[np.ndarray, np.ndarray]:
        """Search through the xform prim and its children to find all meshes and merge them into a single warp mesh in
        its local frame.
        """
        # prepare buffer to merge all the meshes
        points, indices = [], []
        indices_offset = 0
        # prepare xform to search
        xform_leaf = xform_prim_path.split("/")[-1]
        aux_mesh_name_search_list = [xform_leaf] + list(self.cfg.aux_mesh_and_link_names.keys())

        for prim_leaf_name in aux_mesh_name_search_list:
            mesh_prim_path = "/".join([xform_prim_path, "visuals", prim_leaf_name, "mesh"])
            mesh_prim = prim_utils.get_prim_at_path(mesh_prim_path)
            if not mesh_prim.IsValid():
                continue
            mesh = UsdGeom.Mesh(mesh_prim)
            local_points = np.asarray(mesh.GetPointsAttr().Get())
            if len(local_points) == 0:
                continue
            local_indices = np.asarray(mesh.GetFaceVertexIndicesAttr().Get())

            local_transform = Gf.Matrix4d(1.0)
            # compute local-to-parent transform if possible
            if self.cfg.aux_mesh_and_link_names.get(prim_leaf_name, None) is not None:
                fixed_link_prim_path = "/".join([xform_prim_path, self.cfg.aux_mesh_and_link_names[prim_leaf_name]])
                fixed_link_prim = prim_utils.get_prim_at_path(fixed_link_prim_path)
                if fixed_link_prim.IsValid():
                    xformable = UsdGeom.Xformable(fixed_link_prim)
                    xform_ops = xformable.GetOrderedXformOps()
                    for op in xform_ops:
                        local_transform = local_transform * op.GetOpTransform(Usd.TimeCode.Default())

            transformed_points = np.array(
                [local_transform.Transform(Gf.Vec3f(float(p[0]), float(p[1]), float(p[2])))[:3] for p in local_points]
            )

            points.append(transformed_points)
            indices.append(local_indices + indices_offset)
            indices_offset += len(transformed_points)
        if len(points) == 0:
            return None, None
        else:
            return np.concatenate(points, axis=0), np.concatenate(indices, axis=0)

    def _initialize_warp_meshes(self):
        """Initialize warp meshes for ray casting.
        ### NOTE
            This is a re-implementation of the `RayCaster._initialize_warp_meshes` method.
            This should support multiple meshes and update their positions.
            Use env_ids to specify the collision group ids for the ray caster.
        """

        """ Basic insights of getting rigid bodies mesh.
        prim_utils.get_prim_at_path('/World/envs/env_0/Robot/torso_link/visuals/{torso_link_file_name}/mesh').IsA(UsdGeom.Mesh)
        True

        mesh_prim = prim_utils.get_prim_at_path('/World/envs/env_0/Robot/torso_link/visuals/{torso_link_file_name}/mesh')

        However, {torso_link_file_name} could be different from the link name. This needs self.cfg.aux_mesh_and_link_names to be set.
        """

        # read prims to ray-cast
        for mesh_prim_path_regex in self.cfg.mesh_prim_paths:
            env_prim_path_expr, leaf_pattern = mesh_prim_path_regex.rsplit("/", 1)
            matched_mesh_prim_paths = sim_utils.find_matching_prim_paths(env_prim_path_expr)
            assert len(matched_mesh_prim_paths) >= 1, f"No matching mesh prim paths found for {env_prim_path_expr}."
            mesh_prims = sim_utils.get_all_matching_child_prims(
                matched_mesh_prim_paths[0],
                lambda prim: (
                    (prim.GetTypeName() == "Plane" or prim.GetTypeName() == "Mesh" or prim.GetTypeName() == "Xform")
                    and re.search("/".join([matched_mesh_prim_paths[0], leaf_pattern]), prim.GetPath().pathString)
                    is not None
                ),
            )
            # collect and acquire all the meshes in warp format
            wp_meshes = []
            wp_mesh_names = []
            wp_mesh_ids = []
            for mesh_prim in mesh_prims:
                mesh_name = mesh_prim.GetPath().pathString.rsplit("/", 1)[-1]
                if mesh_prim.GetTypeName() == "Xform":
                    # Get the mesh prim if it is proxy referenced from a Xform prim.
                    xform_prim_path = mesh_prim.GetPath().pathString
                    points, indices = self._get_merged_mesh_from_xform_prim(xform_prim_path)
                    if points is None:
                        continue
                    wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
                elif mesh_prim.GetTypeName() == "Mesh":
                    mesh_prim = UsdGeom.Mesh(mesh_prim)
                    # read the vertices and faces
                    points = np.asarray(mesh_prim.GetPointsAttr().Get())
                    #### >>> The code below are copied from original ray_caster, which is not used here.
                    # transform_matrix = np.array(omni.usd.get_world_transform_matrix(mesh_prim)).T
                    # points = np.matmul(points, transform_matrix[:3, :3].T)
                    # points += transform_matrix[:3, 3]
                    #### <<<
                    indices = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
                    wp_mesh = convert_to_warp_mesh(points, indices, device=self.device)
                elif mesh_prim.GetTypeName() == "Plane":
                    mesh = make_plane(size=(2e6, 2e6), height=0.0, center_zero=True)
                    wp_mesh = convert_to_warp_mesh(mesh.vertices, mesh.faces, device=self.device)
                else:
                    omni.log.warn(
                        f"Unsupported mesh type: {mesh_prim.GetTypeName()} in {mesh_prim.GetPath().pathString}."
                        " Skipping."
                    )
                    continue
                wp_mesh_names.append(mesh_name)
                wp_meshes.append(wp_mesh)
                wp_mesh_ids.append(wp_mesh.id)
            self.meshes[mesh_prim_path_regex] = wp_meshes
            # collect match rigid body names to mesh ids for transform updates
            rigid_body_view = self._get_rigid_body_view(env_prim_path_expr, wp_mesh_names)
            if rigid_body_view is None:
                # no rigid body view found, the mesh transform will not be updated at rollouts.
                # the meshes transforms will be set to identity.
                self.mesh_prototype_ids.extend(wp_mesh_ids)
                self.mesh_collision_groups.extend([-1] * len(wp_mesh_ids))
            else:
                # store the wp meshes and the rigid body views for updating the mesh positions
                assert rigid_body_view.count % len(wp_mesh_ids) == 0, (
                    f"Rigid body view count {rigid_body_view.count} is not divisible by the number of meshes "
                    f" {len(wp_mesh_ids)} in {mesh_prim_path_regex}."
                )
                # NOTE: This mesh_prototype_ids.extend is assuming the rigid_body_view has the order of
                # iterating over the mesh names and again and again across envs.
                self.mesh_prototype_ids.extend(wp_mesh_ids * (rigid_body_view.count // len(wp_mesh_ids)))
                self.rigid_body_mesh_transform_segments[mesh_prim_path_regex] = slice(
                    len(self.mesh_prototype_ids) - rigid_body_view.count, len(self.mesh_prototype_ids)
                )
                self.rigid_body_views[mesh_prim_path_regex] = rigid_body_view
                self.mesh_collision_groups.extend(self._get_collision_groups_by_env_str(rigid_body_view))
        # build buffers for all the meshes
        # which is batch-wize and has the same length as mesh_prototype_ids
        self.mesh_transforms_pyt = torch.zeros(
            len(self.mesh_prototype_ids), 7, dtype=torch.float32, device=self.device
        )  # shape (N, 7) in px, py, pz, qx, qy, qz, qw format
        self.mesh_transforms_pyt[:, -1] = 1.0  # set the w component of the quaternion to 1.0
        self.mesh_inv_transforms_pyt = torch.zeros_like(self.mesh_transforms_pyt)  # shape (N, 7)
        self.mesh_inv_transforms_pyt[:, -1] = 1.0  # set the w component of the quaternion to 1.0
        self.mesh_collision_groups_pyt = torch.tensor(self.mesh_collision_groups, dtype=torch.int32, device=self.device)
        self.mesh_prototype_ids_pyt = torch.tensor(self.mesh_prototype_ids, dtype=torch.int64, device=self.device)
        self.all_mesh_indices = torch.arange(len(self.mesh_prototype_ids), dtype=torch.int32, device=self.device)

    def _initialize_rays_impl(self):
        super()._initialize_rays_impl()
        # create buffer to store ray collision groups
        self._create_ray_collision_groups()

    def _create_ray_collision_groups(self):
        """Create buffer to store ray collision groups and mesh ids for group ids."""
        self._ray_collision_groups = (
            torch.arange(self._view.count, dtype=torch.int32, device=self.device).unsqueeze(1).repeat(1, self.num_rays)
        )
        unique_groups = torch.unique(self._ray_collision_groups)
        # NOTE: For code consistency, we do not put for following code in a separate function, since unique_groups is
        # only acquired from self._ray_collision_groups currently.
        self._mesh_ids_for_group = []
        self._mesh_ids_slice_for_group = []
        for group_id in unique_groups:
            negative_one_indices = torch.where(self.mesh_collision_groups_pyt == -1)[0]
            group_indices = torch.where(self.mesh_collision_groups_pyt == group_id)[0]
            ray_group = torch.cat([negative_one_indices, group_indices]).tolist()
            self._mesh_ids_for_group.append(ray_group)
            self._mesh_ids_slice_for_group.append(len(ray_group))
        self._mesh_ids_for_group = torch.tensor(self._mesh_ids_for_group, dtype=torch.int32, device=self._device)
        self._mesh_ids_for_group = self._mesh_ids_for_group.view(-1)
        self._mesh_ids_slice_for_group = [0] + [
            sum(self._mesh_ids_slice_for_group[: i + 1]) for i in range(len(self._mesh_ids_slice_for_group))
        ]
        self._mesh_ids_slice_for_group = torch.tensor(
            self._mesh_ids_slice_for_group, dtype=torch.int32, device=self._device
        )

    def _update_mesh_transforms(self, env_ids: torch.Tensor | None = None):
        """Update the mesh transforms for the given environment IDs.
        This will update the mesh transforms based on the rigid body views.
        """
        if not self.meshes:
            omni.log.warn("No meshes found for ray casting.")
            return

        # # update the mesh transforms based on the rigid body views
        for mesh_prim_path_regex, rigid_body_view in self.rigid_body_views.items():
            segment: slice = self.rigid_body_mesh_transform_segments[mesh_prim_path_regex]
            rigid_body_transforms = rigid_body_view.get_transforms().view(-1, 7)  # shape (N, 7)

            # get which mesh transforms to update
            rigid_body_env_ids = self.mesh_collision_groups_pyt[segment]
            rigid_body_view_mask = None if env_ids is None else torch.isin(rigid_body_env_ids, env_ids)
            # get the selected transforms in px, py, pz, qx, qy, qz, qw format
            selected_transforms = rigid_body_transforms[rigid_body_view_mask]
            mesh_tf_indices = self.all_mesh_indices[segment][rigid_body_view_mask]

            # update the mesh transforms in pytorch format
            self.mesh_transforms_pyt[mesh_tf_indices] = selected_transforms
            pos_inv, quat_wxyz_inv = math_utils.subtract_frame_transforms(
                selected_transforms[:, :3],
                math_utils.convert_quat(selected_transforms[:, 3:], to="wxyz"),
            )
            quat_xyzw_inv = math_utils.convert_quat(quat_wxyz_inv, to="xyzw")
            self.mesh_inv_transforms_pyt[mesh_tf_indices, 3:] = quat_xyzw_inv
            self.mesh_inv_transforms_pyt[mesh_tf_indices, :3] = pos_inv

    def _update_buffers_impl(self, env_ids: Sequence[int]):
        """Update the ray caster buffers with the current mesh positions and orientations.
        And also update the mesh points on given environment IDs (aka. collision group ids).

        Args:
            env_ids: The environment IDs for which to update the buffers.
        """
        self._update_mesh_transforms(env_ids)

        # obtain the poses of the sensors
        if isinstance(self._view, XFormPrim):
            pos_w, quat_w = self._view.get_world_poses(env_ids)
        elif isinstance(self._view, physx.ArticulationView):
            pos_w, quat_w = self._view.get_root_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        elif isinstance(self._view, physx.RigidBodyView):
            pos_w, quat_w = self._view.get_transforms()[env_ids].split([3, 4], dim=-1)
            quat_w = math_utils.convert_quat(quat_w, to="wxyz")
        else:
            raise RuntimeError(f"Unsupported view type: {type(self._view)}")
        # note: we clone here because we are read-only operations
        pos_w = pos_w.clone()
        quat_w = quat_w.clone()
        # apply drift
        pos_w += self.drift[env_ids]
        # store the poses
        self._data.pos_w[env_ids] = pos_w
        self._data.quat_w[env_ids] = quat_w

        # ray cast based on the sensor poses
        if self.cfg.attach_yaw_only:
            # only yaw orientation is considered and directions are not rotated
            ray_starts_w = math_utils.quat_apply_yaw(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = self.ray_directions[env_ids]
        else:
            # full orientation is considered
            ray_starts_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_starts[env_ids])
            ray_starts_w += pos_w.unsqueeze(1)
            ray_directions_w = math_utils.quat_apply(quat_w.repeat(1, self.num_rays), self.ray_directions[env_ids])
        ray_group_ids = self._ray_collision_groups[env_ids]

        self._data.ray_hits_w[env_ids] = raycast_mesh_grouped(
            mesh_prototypes=self.meshes,
            mesh_prototype_ids=self.mesh_prototype_ids_pyt,
            mesh_transforms=self.mesh_transforms_pyt,
            mesh_inv_transforms=self.mesh_inv_transforms_pyt,
            ray_group_ids=ray_group_ids,
            mesh_ids_for_group=self._mesh_ids_for_group,
            mesh_ids_slice_for_group=self._mesh_ids_slice_for_group,
            ray_starts=ray_starts_w,
            ray_directions=ray_directions_w,
            max_dist=self.cfg.max_distance,
            min_dist=self.cfg.min_distance,
        )[0]
