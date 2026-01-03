from __future__ import annotations

import torch
from collections.abc import Sequence
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.sensors import RayCasterCfg, patterns, RayCaster
from .leatherback import LEATHERBACK_CFG
from .track import OUTER_TRACK_V1_CFG, OUTER_TRACK_V2_CFG, INNER_TRACK_V1_CFG, INNER_TRACK_V2_CFG
from isaacsim.sensors.physx import _range_sensor


from isaacsim.core.utils.extensions import enable_extension

@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 60.0
    action_space = 2
    observation_space = 540
    state_space = 540
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")

    # Individual track configs - spawn all variants and toggle visibility during resets
    outer_track_v1_cfg: RigidObjectCfg = OUTER_TRACK_V1_CFG.replace(prim_path="/World/envs/env_.*/Outer_Track_V1")
    outer_track_v2_cfg: RigidObjectCfg = OUTER_TRACK_V2_CFG.replace(prim_path="/World/envs/env_.*/Outer_Track_V2")
    inner_track_v1_cfg: RigidObjectCfg = INNER_TRACK_V1_CFG.replace(prim_path="/World/envs/env_.*/Inner_Track_V1")
    inner_track_v2_cfg: RigidObjectCfg = INNER_TRACK_V2_CFG.replace(prim_path="/World/envs/env_.*/Inner_Track_V2")
    
    # Number of track variants available
    num_track_variants = 2

    throttle_dof_name = [
        "Wheel__Knuckle__Front_Left",
        "Wheel__Knuckle__Front_Right",
        "Wheel__Upright__Rear_Right",
        "Wheel__Upright__Rear_Left"
    ]
    steering_dof_name = [
        "Knuckle__Upright__Front_Right",
        "Knuckle__Upright__Front_Left",
    ]

    env_spacing = 60.0
    # IMPORTANT: replicate_physics must be False when using MultiUsdFileCfg for random asset spawning
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=env_spacing, replicate_physics=False)

class LeatherbackEnv(DirectRLEnv):
    cfg: LeatherbackEnvCfg

    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._throttle_dof_idx, _ = self.leatherback.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.leatherback.find_joints(self.cfg.steering_dof_name)
        self._throttle_state = torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        self._steering_state = torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32)
        self.task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self.env_spacing = self.cfg.env_spacing
        
        # Lidar-based reward parameters
        self.forward_velocity_weight = 10.5  # Reward for driving fast
        self.centering_weight = 1.5  # Reward for staying centered on track
        self.wall_clearance_weight = 0.5  # Reward for maintaining safe distance from walls
        self.min_safe_distance = 0.5  # Minimum safe distance from walls (meters)
        self.optimal_safe_distance = 1.0  # Optimal distance from walls for max reward
        self.steering_smoothness_weight = 0.3  # Reward for smooth steering
        self.max_forward_velocity = 10.0  # Max velocity for normalization
        self.progress_weight = 3.5  # Reward for making forward progress
        self.max_lidar_range = 10.0  # Sensor fallback range for invalid readings
        self.collision_distance_threshold = 0.2  # Reset if any beam is this close to wall
        self.collision_penalty = -5.0  # Strong penalty when colliding with wall
        self.min_moving_speed = 0.5  # Threshold below which the agent is considered idle
        self.idle_steps_threshold = 30  # Number of consecutive idle steps before applying penalties
        self.idle_penalty_scale = 0.05  # Penalty applied per idle step beyond the threshold
        self.max_idle_penalty = 5.0  # Cap the idle penalty to avoid runaway negatives
        self.stuck_front_clearance_threshold = 1.0  # Considered stuck if facing a wall closer than this while idle
        self.stuck_penalty = -1.5  # Extra penalty for being idle while facing a nearby wall
        
        # Store previous steering for smoothness calculation
        self._previous_steering = torch.zeros((self.num_envs,), device=self.device, dtype=torch.float32)
        self._collision = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
        self._idle_steps = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
        
        # Store lidar depths for reward calculation
        self._lidar_depths = None
        
        # Track which track variant is active for each environment (0 or 1)
        self._active_track_variant = torch.zeros((self.num_envs,), device=self.device, dtype=torch.int32)
        


    def _setup_scene(self):
        enable_extension("isaacsim.sensors.physx")
        # Create a large ground plane without grid
        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                size=(500.0, 500.0),  # Much larger ground plane (500m x 500m)
                color=(0.2, 0.2, 0.2),  # Dark gray color
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )

        # Setup rest of the scene
        self.leatherback = Articulation(self.cfg.robot_cfg)
        
        # Spawn ALL track variants - we'll toggle visibility/position during resets
        # This allows random track selection at reset time rather than just at spawn time
        self.outer_track_v1 = RigidObject(self.cfg.outer_track_v1_cfg)
        self.outer_track_v2 = RigidObject(self.cfg.outer_track_v2_cfg)
        self.inner_track_v1 = RigidObject(self.cfg.inner_track_v1_cfg)
        self.inner_track_v2 = RigidObject(self.cfg.inner_track_v2_cfg)
        
        # Store track pairs for easy access during resets
        self._track_variants = [
            (self.outer_track_v1, self.inner_track_v1),  # Variant 0
            (self.outer_track_v2, self.inner_track_v2),  # Variant 1
        ]

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["leatherback"] = self.leatherback
        self.scene.rigid_objects["outer_track_v1"] = self.outer_track_v1
        self.scene.rigid_objects["outer_track_v2"] = self.outer_track_v2
        self.scene.rigid_objects["inner_track_v1"] = self.inner_track_v1
        self.scene.rigid_objects["inner_track_v2"] = self.inner_track_v2

        self.object_state = []
        

        # Add lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

        # import omni.usd
        self.lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
        # stage = omni.usd.get_context().get_stage()
        # lidar_prim_path = lambda env: f"/World/envs/env_{env}/Robot/Rigid_Bodies/Chassis/Lidar_01/Lidar"
        # self.lidar_prim = stage.GetPrimAtPath(lidar_prim_path)
        # self.lidar = self.lidar_prim
    



    def _pre_physics_step(self, actions: torch.Tensor) -> None:
        



        throttle_scale = 10
        throttle_max = 50
        steering_scale = 0.1
        steering_max = 0.75

        self._throttle_action = actions[:, 0].repeat_interleave(4).reshape((-1, 4)) * throttle_scale
        self.throttle_action = torch.clamp(self._throttle_action, -throttle_max, throttle_max)
        self._throttle_state = self._throttle_action
        
        self._steering_action = actions[:, 1].repeat_interleave(2).reshape((-1, 2)) * steering_scale
        self._steering_action = torch.clamp(self._steering_action, -steering_max, steering_max)
        self._steering_state = self._steering_action

        # print("-------------------------------")
        # print(self.scanner)
        # print(self.scanner.data)
        # print("-------------------------------")

    def _apply_action(self) -> None:
        self.leatherback.set_joint_velocity_target(self._throttle_action, joint_ids=self._throttle_dof_idx)
        self.leatherback.set_joint_position_target(self._steering_state, joint_ids=self._steering_dof_idx)

    def _get_observations(self) -> dict:
        self.depths = []
        for env in range(self.num_envs):
            import omni.usd
            stage = omni.usd.get_context().get_stage()
            #print(f"stage:{stage}")
            lidar_prim_path = f"/World/envs/env_{env}/Robot/Rigid_Bodies/Chassis/Lidar_01/Lidar"
            self.lidar_prim = stage.GetPrimAtPath(lidar_prim_path)
            self.lidar = self.lidar_prim
            #print(f"lidar:{self.lidar}")
            lidarInterface = _range_sensor.acquire_lidar_sensor_interface()
            depth = lidarInterface.get_linear_depth_data(lidar_prim_path)
            depth_tensor = torch.tensor(depth)  # Convert depth to tensor and add batch dimension
            
            self.depths.append(depth_tensor)
            #print(f"env_{env}, depth:{depth}")
        #print(f"depths:{self.depths}")  #TODO: Remove this print statement in production code
        depths_tensor = torch.stack(self.depths)
        # Replace invalid lidar readings before using them elsewhere
        depths_tensor = torch.nan_to_num(
            depths_tensor,
            nan=self.max_lidar_range,
            posinf=self.max_lidar_range,
            neginf=0.0,
        )
        depths_tensor = torch.clamp(depths_tensor, 0.0, self.max_lidar_range)
        
        # Store lidar depths for reward calculation
        self._lidar_depths = depths_tensor.squeeze(-1).to(self.device)
        
        obs = depths_tensor.squeeze(-1).to(self.device) 

        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        # ==================== POSITIVE REWARD-BASED SYSTEM FOR TRACK RACING ====================
        # Philosophy: Reward good behavior rather than punish bad behavior
        
        # 1. Forward Velocity Reward - encourage driving fast
        forward_velocity = torch.nan_to_num(
            self.leatherback.data.root_lin_vel_b[:, 0],
            nan=0.0,
            posinf=self.max_forward_velocity,
            neginf=-self.max_forward_velocity,
        )  # Forward velocity in body frame
        # Reward positive velocity, with bonus for going faster (up to max)
        velocity_reward = torch.clamp(forward_velocity / self.max_forward_velocity, 0.0, 1.0)
        # Extra bonus for maintaining good speed
        speed_bonus = torch.where(forward_velocity > 2.0, torch.ones_like(forward_velocity) * 0.2, torch.zeros_like(forward_velocity))
        not_moving = torch.abs(forward_velocity) < self.min_moving_speed
        self._idle_steps = torch.where(
            not_moving,
            self._idle_steps + 1,
            torch.zeros_like(self._idle_steps),
        )
        
        # 2. Track Centering Reward - use lidar to stay centered
        if self._lidar_depths is not None and self._lidar_depths.numel() > 0:
            # Replace NaN and inf values with a large distance (no obstacle detected)
            lidar_clean = self._lidar_depths.clone()
            lidar_clean = torch.where(torch.isnan(lidar_clean), torch.tensor(self.max_lidar_range, device=self.device), lidar_clean)
            lidar_clean = torch.where(torch.isinf(lidar_clean), torch.tensor(self.max_lidar_range, device=self.device), lidar_clean)
            lidar_clean = torch.clamp(lidar_clean, 0.0, self.max_lidar_range)
            
            num_rays = lidar_clean.shape[-1]
            
            # Split lidar into left and right halves
            left_half = lidar_clean[:, :num_rays // 2]  # Left side distances
            right_half = lidar_clean[:, num_rays // 2:]  # Right side distances
            
            # Get minimum distances on each side (closest obstacles)
            left_min_dist = torch.min(left_half, dim=-1).values
            right_min_dist = torch.min(right_half, dim=-1).values
            
            # Get mean distances for centering calculation
            left_mean_dist = torch.mean(left_half, dim=-1)
            right_mean_dist = torch.mean(right_half, dim=-1)
            
            # Centering reward: reward for balanced distances (being centered)
            centering_diff = torch.abs(left_mean_dist - right_mean_dist)
            centering_reward = torch.exp(-centering_diff * 0.5)  # Higher when centered
            
            # 3. Wall Clearance Reward - reward for maintaining safe distance from walls
            overall_min_dist = torch.min(left_min_dist, right_min_dist)

            # Flag collisions when any beam is within the threshold
            collision_mask = overall_min_dist < self.collision_distance_threshold
            self._collision = collision_mask
            collision_penalty = torch.where(
                collision_mask,
                torch.ones_like(overall_min_dist) * self.collision_penalty,
                torch.zeros_like(overall_min_dist),
            )
            
            # Reward based on how much clearance we have (more clearance = more reward)
            # Smoothly increases from 0 at min_safe_distance to 1 at optimal_safe_distance
            wall_clearance_reward = torch.clamp(
                (overall_min_dist - self.min_safe_distance) / (self.optimal_safe_distance - self.min_safe_distance),
                0.0, 1.0
            )
            # Bonus for having excellent clearance
            excellent_clearance_bonus = torch.where(
                overall_min_dist > self.optimal_safe_distance,
                torch.ones_like(overall_min_dist) * 0.3,
                torch.zeros_like(overall_min_dist)
            )
            
            # Bonus for having clear space ahead (front sector)
            front_sector_start = num_rays // 4
            front_sector_end = 3 * num_rays // 4
            front_distances = lidar_clean[:, front_sector_start:front_sector_end]
            front_clearance = torch.mean(front_distances, dim=-1)
            front_clearance_reward = torch.clamp(front_clearance / 10.0, 0.0, 1.0)  # Normalize
            
            # Additional reward for having open path ahead
            open_path_bonus = torch.where(
                front_clearance > 5.0,
                torch.ones_like(front_clearance) * 0.2,
                torch.zeros_like(front_clearance)
            )
        else:
            # Fallback if lidar data not available
            centering_reward = torch.zeros((self.num_envs,), device=self.device)
            wall_clearance_reward = torch.zeros((self.num_envs,), device=self.device)
            excellent_clearance_bonus = torch.zeros((self.num_envs,), device=self.device)
            front_clearance_reward = torch.zeros((self.num_envs,), device=self.device)
            open_path_bonus = torch.zeros((self.num_envs,), device=self.device)
            collision_penalty = torch.zeros((self.num_envs,), device=self.device)
            self._collision = torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool)
            front_clearance = torch.zeros((self.num_envs,), device=self.device)

        idle_excess = torch.clamp(self._idle_steps - self.idle_steps_threshold, min=0)
        idle_penalty = -torch.clamp(idle_excess.float() * self.idle_penalty_scale, max=self.max_idle_penalty)
        stuck_mask = torch.logical_and(not_moving, front_clearance < self.stuck_front_clearance_threshold)
        stuck_penalty = torch.where(stuck_mask, torch.ones_like(forward_velocity) * self.stuck_penalty, torch.zeros_like(forward_velocity))
        
        # 4. Steering Smoothness Reward - reward smooth steering
        current_steering = torch.nan_to_num(self._steering_state[:, 0], nan=0.0)
        steering_change = torch.abs(current_steering - torch.nan_to_num(self._previous_steering, nan=0.0))
        # Reward for smooth steering (low change = high reward)
        max_steering_change = 0.5  # Maximum expected steering change per step
        steering_smoothness_reward = 1.0 - torch.clamp(steering_change / max_steering_change, 0.0, 1.0)
        self._previous_steering = current_steering.clone()
        
        # 5. Alive/Survival Bonus - reward for staying on track
        alive_bonus = torch.ones((self.num_envs,), device=self.device) * 0.2
        
        # 6. Forward Progress Reward - reward for moving forward (not just velocity, but actual progress)
        # Encourages consistent forward movement
        forward_progress_reward = torch.where(
            forward_velocity > 0.5,
            torch.ones_like(forward_velocity) * 0.3,
            torch.zeros_like(forward_velocity)
        )
        
        # ==================== COMPOSITE REWARD (ALL POSITIVE) ====================
        composite_reward = (
            velocity_reward * self.forward_velocity_weight +
            speed_bonus +
            centering_reward * self.centering_weight +
            wall_clearance_reward * self.wall_clearance_weight +
            excellent_clearance_bonus +
            front_clearance_reward * 0.4 +
            open_path_bonus +
            steering_smoothness_reward * self.steering_smoothness_weight +
            alive_bonus +
            forward_progress_reward * self.progress_weight +
            idle_penalty +
            stuck_penalty +
            collision_penalty
        )
        composite_reward = torch.nan_to_num(composite_reward, nan=0.0, posinf=0.0, neginf=0.0)

        if torch.any(composite_reward.isnan()):
            print(composite_reward)
            raise ValueError("Rewards cannot be NAN")

        return composite_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        # Check for timeout
        task_failed = self.episode_length_buf > self.max_episode_length
        collided = getattr(self, "_collision", torch.zeros((self.num_envs,), device=self.device, dtype=torch.bool))
        return task_failed | collided, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.leatherback._ALL_INDICES
        super()._reset_idx(env_ids)

        # Randomly select track variant for each resetting environment
        # Convert env_ids to tensor if needed
        env_ids_tensor = torch.tensor(env_ids, device=self.device) if not isinstance(env_ids, torch.Tensor) else env_ids
        num_reset_envs = len(env_ids_tensor)
        
        # Randomly select which track variant to use (0 or 1)
        new_variants = torch.randint(0, self.cfg.num_track_variants, (num_reset_envs,), device=self.device, dtype=torch.int32)
        self._active_track_variant[env_ids_tensor] = new_variants
        
        # Position offset to "hide" inactive tracks (move them far away)
        hide_offset = torch.tensor([0.0, 0.0, -1000.0], device=self.device)  # Move underground
        show_offset = torch.tensor([0.0, 0.0, 0.0], device=self.device)  # Normal position
        
        # Update track positions based on which variant is active
        for env_idx, env_id in enumerate(env_ids_tensor):
            variant = new_variants[env_idx].item()
            env_origin = self.scene.env_origins[env_id]
            
            # For each track variant, show the active one and hide the inactive one
            for var_idx, (outer_track, inner_track) in enumerate(self._track_variants):
                # Get default positions from the track's initial state
                default_pos = torch.tensor([[-1.0, 1.0, 0.0]], device=self.device)
                default_rot = torch.tensor([[0.707, 0.0, 0.0, -0.707]], device=self.device)
                
                if var_idx == variant:
                    # Show this track variant (at normal position)
                    track_pos = default_pos + env_origin.unsqueeze(0) + show_offset
                else:
                    # Hide this track variant (move underground)
                    track_pos = default_pos + env_origin.unsqueeze(0) + hide_offset
                
                # Create full pose tensor [x, y, z, qw, qx, qy, qz]
                track_pose = torch.cat([track_pos, default_rot], dim=-1)
                
                # Write pose for this specific environment
                outer_track.write_root_pose_to_sim(track_pose, torch.tensor([env_id], device=self.device))
                inner_track.write_root_pose_to_sim(track_pose, torch.tensor([env_id], device=self.device))

        default_state = self.leatherback.data.default_root_state[env_ids]
        leatherback_pose = default_state[:, :7] # x, y, z, qw, qx, qy, qz
        leatherback_velocities = default_state[:, 7:] # vx, vy, vz, wx, wy, wz
        joint_positions = self.leatherback.data.default_joint_pos[env_ids]
        joint_velocities = self.leatherback.data.default_joint_vel[env_ids]

        leatherback_pose[:, :3] += self.scene.env_origins[env_ids]
        # Fixed spawn position - no randomization to ensure car spawns on track
        # leatherback_pose[:, 0] -= self.env_spacing / 2  # Removed X offset
        # leatherback_pose[:, 1] += 0.0  # No Y offset - spawn at track center
        
        # Fixed heading angle (facing forward, no rotation)
        # Quaternion for 0 rotation: [w=1, x=0, y=0, z=0]
        leatherback_pose[:, 3] = 1.0  # qw
        leatherback_pose[:, 4] = 0.0  # qx
        leatherback_pose[:, 5] = 0.0  # qy
        leatherback_pose[:, 6] = 0.0  # qz

        self.leatherback.write_root_pose_to_sim(leatherback_pose, env_ids)
        self.leatherback.write_root_velocity_to_sim(leatherback_velocities, env_ids)
        self.leatherback.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)

        self._idle_steps[env_ids] = 0
        
        # Reset steering tracking for smoothness penalty
        self._previous_steering[env_ids] = 0.0
