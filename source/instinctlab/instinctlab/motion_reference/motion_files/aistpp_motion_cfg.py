from __future__ import annotations

from isaaclab.utils import configclass

from .aistpp_motion import AistppMotion
from .amass_motion_cfg import AmassMotionCfg


@configclass
class AistppMotionCfg(AmassMotionCfg):
    """Configuration for AIST++ motion files."""

    class_type: type = AistppMotion
