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
from .waypoint import WAYPOINT_CFG
from .leatherback import LEATHERBACK_CFG
from .track import TRACK_CFG
from isaaclab.markers import VisualizationMarkers
from isaacsim.sensors.physx import _range_sensor


from isaacsim.core.utils.extensions import enable_extension

@configclass
class LeatherbackEnvCfg(DirectRLEnvCfg):
    decimation = 4
    episode_length_s = 20.0
    action_space = 2
    observation_space = 540
    state_space = 540
    sim: SimulationCfg = SimulationCfg(dt=1 / 60, render_interval=decimation)
    robot_cfg: ArticulationCfg = LEATHERBACK_CFG.replace(prim_path="/World/envs/env_.*/Robot")
    waypoint_cfg = WAYPOINT_CFG
    track_cfg: RigidObjectCfg = TRACK_CFG.replace(prim_path="/World/envs/env_.*/Track")

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

    env_spacing = 35.0
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=env_spacing, replicate_physics=True)

class LeatherbackEnv(DirectRLEnv):
    cfg: LeatherbackEnvCfg

    def __init__(self, cfg: LeatherbackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._throttle_dof_idx, _ = self.leatherback.find_joints(self.cfg.throttle_dof_name)
        self._steering_dof_idx, _ = self.leatherback.find_joints(self.cfg.steering_dof_name)
        self._throttle_state = torch.zeros((self.num_envs,4), device=self.device, dtype=torch.float32)
        self._steering_state = torch.zeros((self.num_envs,2), device=self.device, dtype=torch.float32)
        self._goal_reached = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        self.task_completed = torch.zeros((self.num_envs), device=self.device, dtype=torch.bool)
        self._num_goals = 10
        self._target_positions = torch.zeros((self.num_envs, self._num_goals, 2), device=self.device, dtype=torch.float32)
        self._markers_pos = torch.zeros((self.num_envs, self._num_goals, 3), device=self.device, dtype=torch.float32)
        self.env_spacing = self.cfg.env_spacing
        self.course_length_coefficient = 2.5
        self.course_width_coefficient = 2.0
        self.position_tolerance = 0.15
        self.goal_reached_bonus = 10.0
        self.position_progress_weight = 1.0
        self.heading_coefficient = 0.25
        self.heading_progress_weight = 0.05
        self._target_index = torch.zeros((self.num_envs), device=self.device, dtype=torch.int32)
        


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
        
        self.track = RigidObject(self.cfg.track_cfg)

        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[])
        self.scene.articulations["leatherback"] = self.leatherback

        self.waypoints = VisualizationMarkers(self.cfg.waypoint_cfg)

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

        depths_tensor = torch.stack(self.depths)
        #print(f"depths:{depths_tensor.squeeze(-1)}") #TODO: Remove this print statement in production code
        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions - self.leatherback.data.root_pos_w[:, :2]
        self._previous_position_error = self._position_error.clone()
        self._position_error = torch.norm(self._position_error_vector, dim=-1)

        heading = self.leatherback.data.heading_w
        target_heading_w = torch.atan2(
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 1] - self.leatherback.data.root_link_pos_w[:, 1],
            self._target_positions[self.leatherback._ALL_INDICES, self._target_index, 0] - self.leatherback.data.root_link_pos_w[:, 0],
        )
        self.target_heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        
        #print(self._position_error.unsqueeze(dim=1))
        # obs = torch.cat(
        #     (
        #         self._position_error.unsqueeze(dim=1),
        #         torch.cos(self.target_heading_error).unsqueeze(dim=1),
        #         torch.sin(self.target_heading_error).unsqueeze(dim=1),
        #         self.leatherback.data.root_lin_vel_b[:, 0].unsqueeze(dim=1),
        #         self.leatherback.data.root_lin_vel_b[:, 1].unsqueeze(dim=1),
        #         self.leatherback.data.root_ang_vel_w[:, 2].unsqueeze(dim=1),
        #         #self.depths_tensor,  # Assuming depth is a single value for simplicity
        #         #self._throttle_state[:, 0].unsqueeze(dim=1),
        #         #self._steering_state[:, 0].unsqueeze(dim=1),
                
        #     ),
        #     dim=-1,
        # )

        obs = depths_tensor.squeeze(-1).to(self.device) 

        if torch.any(obs.isnan()):
            raise ValueError("Observations cannot be NAN")

        return {"policy": obs}
    
    def _get_rewards(self) -> torch.Tensor:
        position_progress_rew = self._previous_position_error - self._position_error
        target_heading_rew = torch.exp(-torch.abs(self.target_heading_error) / self.heading_coefficient)
        goal_reached = self._position_error < self.position_tolerance
        self._target_index = self._target_index + goal_reached
        self.task_completed = self._target_index > (self._num_goals -1)
        self._target_index = self._target_index % self._num_goals

        composite_reward = (
            position_progress_rew * self.position_progress_weight +
            target_heading_rew * self.heading_progress_weight +
            goal_reached * self.goal_reached_bonus
        )

        one_hot_encoded = torch.nn.functional.one_hot(self._target_index.long(), num_classes=self._num_goals)
        marker_indices = one_hot_encoded.view(-1).tolist()
        self.waypoints.visualize(marker_indices=marker_indices)

        if torch.any(composite_reward.isnan()):
            print(composite_reward)
            raise ValueError("Rewards cannot be NAN")

        return composite_reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        task_failed = self.episode_length_buf > self.max_episode_length
        return task_failed, self.task_completed

    def _reset_idx(self, env_ids: Sequence[int] | None):
        if env_ids is None:
            env_ids = self.leatherback._ALL_INDICES
        super()._reset_idx(env_ids)

        num_reset = len(env_ids)
        default_state = self.leatherback.data.default_root_state[env_ids]
        leatherback_pose = default_state[:, :7]
        leatherback_velocities = default_state[:, 7:]
        joint_positions = self.leatherback.data.default_joint_pos[env_ids]
        joint_velocities = self.leatherback.data.default_joint_vel[env_ids]

        leatherback_pose[:, :3] += self.scene.env_origins[env_ids]
        leatherback_pose[:, 0] -= self.env_spacing / 2
        leatherback_pose[:, 1] += 2.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device) * self.course_width_coefficient

        angles = torch.pi / 6.0 * torch.rand((num_reset), dtype=torch.float32, device=self.device)
        leatherback_pose[:, 3] = torch.cos(angles * 0.5)
        leatherback_pose[:, 6] = torch.sin(angles * 0.5)

        self.leatherback.write_root_pose_to_sim(leatherback_pose, env_ids)
        self.leatherback.write_root_velocity_to_sim(leatherback_velocities, env_ids)
        self.leatherback.write_joint_state_to_sim(joint_positions, joint_velocities, None, env_ids)

        self._target_positions[env_ids, :, :] = 0.0
        self._markers_pos[env_ids, :, :] = 0.0

        spacing = 2 / self._num_goals
        target_positions = torch.arange(-0.8, 1.1, spacing, device=self.device) * self.env_spacing / self.course_length_coefficient
        self._target_positions[env_ids, :len(target_positions), 0] = target_positions
        self._target_positions[env_ids, :, 1] = torch.rand((num_reset, self._num_goals), dtype=torch.float32, device=self.device) + self.course_length_coefficient
        self._target_positions[env_ids, :] += self.scene.env_origins[env_ids, :2].unsqueeze(1)

        self._target_index[env_ids] = 0
        self._markers_pos[env_ids, :, :2] = self._target_positions[env_ids]
        visualize_pos = self._markers_pos.view(-1, 3)
        self.waypoints.visualize(translations=visualize_pos)

        current_target_positions = self._target_positions[self.leatherback._ALL_INDICES, self._target_index]
        self._position_error_vector = current_target_positions[:, :2] - self.leatherback.data.root_pos_w[:, :2]
        self._position_error = torch.norm(self._position_error_vector, dim=-1)
        self._previous_position_error = self._position_error.clone()

        heading = self.leatherback.data.heading_w[:]
        target_heading_w = torch.atan2( 
            self._target_positions[:, 0, 1] - self.leatherback.data.root_pos_w[:, 1],
            self._target_positions[:, 0, 0] - self.leatherback.data.root_pos_w[:, 0],
        )
        self._heading_error = torch.atan2(torch.sin(target_heading_w - heading), torch.cos(target_heading_w - heading))
        self._previous_heading_error = self._heading_error.clone()


    def get_lidar_point_cloud(self, env_id: int = None):
        """
        Get point cloud data from lidar sensor for a specific environment or all environments.
        
        Args:
            env_id: Environment ID to get lidar data from. If None, returns data for all environments.
            
        Returns:
            Point cloud data as numpy array or list of arrays for all environments.
        """
        if env_id is not None:
            # Get data for specific environment
            path = self.lidar_path(env_id)
            return self.lidarInterface.get_point_cloud_data(path)
        else:
            # Get data for all environments
            point_clouds = []
            for env in range(self.num_envs):
                path = self.lidar_path(env)
                pointcloud = self.lidarInterface.get_point_cloud_data(path)
                point_clouds.append(pointcloud)
            return point_clouds

    def get_lidar_depth_data(self, env_id: int = None):
        """
        Get depth data from lidar sensor for a specific environment or all environments.
        
        Args:
            env_id: Environment ID to get lidar data from. If None, returns data for all environments.
            
        Returns:
            Depth data as numpy array or list of arrays for all environments.
        """
        if env_id is not None:
            # Get data for specific environment
            path = self.lidar_path(env_id)
            return self.lidarInterface.get_depth_data(path)
        else:
            # Get data for all environments
            depth_data = []
            for env in range(self.num_envs):
                path = self.lidar_path(env)
                depth = self.lidarInterface.get_depth_data(path)
                depth_data.append(depth)
            return depth_data

    def get_lidar_linear_depth_data(self, env_id: int = None):
        """
        Get linear depth data from lidar sensor for a specific environment or all environments.
        
        Args:
            env_id: Environment ID to get lidar data from. If None, returns data for all environments.
            
        Returns:
            Linear depth data as numpy array or list of arrays for all environments.
        """
        if env_id is not None:
            # Get data for specific environment
            path = self.lidar_path(env_id)
            return self.lidarInterface.get_linear_depth_data(path)
        else:
            # Get data for all environments
            linear_depth_data = []
            for env in range(self.num_envs):
                path = self.lidar_path(env)
                linear_depth = self.lidarInterface.get_linear_depth_data(path)
                linear_depth_data.append(linear_depth)
            return linear_depth_data

    def get_lidar_intensity_data(self, env_id: int = None):
        """
        Get intensity data from lidar sensor for a specific environment or all environments.
        
        Args:
            env_id: Environment ID to get lidar data from. If None, returns data for all environments.
            
        Returns:
            Intensity data as numpy array or list of arrays for all environments.
        """
        if env_id is not None:
            # Get data for specific environment
            path = self.lidar_path(env_id)
            return self.lidarInterface.get_intensity_data(path)
        else:
            # Get data for all environments
            intensity_data = []
            for env in range(self.num_envs):
                path = self.lidar_path(env)
                intensity = self.lidarInterface.get_intensity_data(path)
                intensity_data.append(intensity)
            return intensity_data

    def get_all_lidar_data(self, env_id: int = None):
        """
        Get all available lidar data (point cloud, depth, linear depth, intensity) for environments.
        
        Args:
            env_id: Environment ID to get lidar data from. If None, returns data for all environments.
            
        Returns:
            Dictionary containing all lidar data types.
        """
        return {
            'point_cloud': self.get_lidar_point_cloud(env_id),
            'depth': self.get_lidar_depth_data(env_id),
            'linear_depth': self.get_lidar_linear_depth_data(env_id),
            'intensity': self.get_lidar_intensity_data(env_id)
        }
