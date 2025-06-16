import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

import os

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
TRACK_USD_PATH = os.path.join(WORKSPACE_ROOT, "custom_assets", "track_bend.usd")

TRACK_CFG = RigidObjectCfg(
    #prim_path="/World/envs/env_.*/Track",
    spawn=sim_utils.UsdFileCfg(
        usd_path=TRACK_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-10.0, 0.0, 0.0), # Initial position (x, y, z)
        rot=(0.0, 0.0, 0.0, 1.0),  # Quaternion (w, x, y, z)
    ),
)