import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg

import os

WORKSPACE_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__)))
OUTER_TRACK_V1_USD_PATH = os.path.join(WORKSPACE_ROOT, "custom_assets", "Outer_Track_V1.usd")
INNER_TRACK_V1_USD_PATH = os.path.join(WORKSPACE_ROOT, "custom_assets", "Inner_Track_V1.usd")

OUTER_TRACK_V2_USD_PATH = os.path.join(WORKSPACE_ROOT, "custom_assets", "Outer_Track_V2.usd")
INNER_TRACK_V2_USD_PATH = os.path.join(WORKSPACE_ROOT, "custom_assets", "Inner_Track_V2.usd")


# Individual track configs (kept for backward compatibility)
OUTER_TRACK_V1_CFG = RigidObjectCfg(
    #prim_path="/World/envs/env_.*/Outer_Track",
    spawn=sim_utils.UsdFileCfg(
        usd_path=OUTER_TRACK_V1_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
        ),
    ),
        init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-1.0, 1.0, 0.0), # Initial position (x, y, z)
        rot=(0.707, 0.0, 0.0, -0.707),  # Quaternion (w, x, y, z)
    ),
)

INNER_TRACK_V1_CFG = RigidObjectCfg(
    #prim_path="/World/envs/env_.*/Inner_Track",
    spawn=sim_utils.UsdFileCfg(
        usd_path=INNER_TRACK_V1_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
        ),
    ),

    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-1.0, 1.0, 0.0), # Initial position (x, y, z)
        rot=(0.707, 0.0, 0.0, -0.707),  # Quaternion (w, x, y, z)
    ),
)

OUTER_TRACK_V2_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=OUTER_TRACK_V2_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-1.0, 1.0, 0.0),
        rot=(0.707, 0.0, 0.0, -0.707),
    ),
)

INNER_TRACK_V2_CFG = RigidObjectCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=INNER_TRACK_V2_USD_PATH,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-1.0, 1.0, 0.0),
        rot=(0.707, 0.0, 0.0, -0.707),
    ),
)

# Multi-asset spawner configs for random track selection across environments
OUTER_TRACK_MULTI_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Outer_Track",
    spawn=sim_utils.MultiUsdFileCfg(
        usd_path=[
            OUTER_TRACK_V1_USD_PATH,
            OUTER_TRACK_V2_USD_PATH,
        ],
        random_choice=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-1.0, 1.0, 0.0),
        rot=(0.707, 0.0, 0.0, -0.707),
    ),
)

INNER_TRACK_MULTI_CFG = RigidObjectCfg(
    prim_path="/World/envs/env_.*/Inner_Track",
    spawn=sim_utils.MultiUsdFileCfg(
        usd_path=[
            INNER_TRACK_V1_USD_PATH,
            INNER_TRACK_V2_USD_PATH,
        ],
        random_choice=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            rigid_body_enabled=True,
            kinematic_enabled=True,
            disable_gravity=True,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            collision_enabled=True,
            contact_offset=0.01,
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(-1.0, 1.0, 0.0),
        rot=(0.707, 0.0, 0.0, -0.707),
    ),
)