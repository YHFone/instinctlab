from dataclasses import MISSING

from isaaclab.markers import VisualizationMarkersCfg
from isaaclab.markers.config import RAY_CASTER_MARKER_CFG
from isaaclab.sensors.ray_caster import RayCasterCfg
from isaaclab.utils import configclass

from .grouped_ray_caster import GroupedRayCaster


@configclass
class GroupedRayCasterCfg(RayCasterCfg):
    """Configuration for the GroupedRayCaster sensor."""

    class_type: type = GroupedRayCaster

    min_distance: float = 0.0
    """The minimum distance from the sensor to ray cast to. aka ignore the hits closer than this distance."""

    aux_mesh_and_link_names: dict[str, str] = {}
    """The dictionary of merged mesh file (key) and link names (value). For the auxiliary mesh search when trying to dig
    out the mesh prim under a Xform prim. Please check all names where mesh file name is different from the link name.
    If the mesh file name is for the link name, set the `value` to None.
    For example, a torso link has a mesh file name `torso_link_rev_1_0`, but the link name is `torso_link`.
        It also has a `head_link` with a mesh file name `head_link` fixed in torso_link.
        Then, the `aux_mesh_and_link_names` should include:
        {
            "torso_link_rev_1_0": None,
            "head_link": "head_link",
        }
    """
