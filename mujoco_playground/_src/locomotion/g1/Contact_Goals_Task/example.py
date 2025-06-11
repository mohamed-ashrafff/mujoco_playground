# Copyright 2025 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Joystick task for Unitree G1."""

from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as jp
from ml_collections import config_dict
from mujoco import mjx
from mujoco.mjx._src import math
import numpy as np

from mujoco_playground._src import collision
from mujoco_playground._src import gait
from mujoco_playground._src import mjx_env
from mujoco_playground._src.collision import geoms_colliding
from mujoco_playground._src.locomotion.g1 import base as g1_base
from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from matplotlib.path import Path



def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=10,
      action_scale=0.5,
      history_len=1,
      restricted_joint_range=False,
      soft_joint_pos_limit_factor=0.95,
      noise_config=config_dict.create(
          level=1.0,  # Set to 0.0 to disable noise.
          scales=config_dict.create(
              joint_pos=0.03,
              joint_vel=1.5,
              gravity=0.05,
              linvel=0.1,
              gyro=0.2,
          ),
      ),
      reward_config=config_dict.create(
          scales=config_dict.create(
              stance_reward=1.0,
              swing_reward=1.0,
              touchdown_reward=1.0,
              touchdown_finegrained_reward=0.5,
              new_goal_reward=1.0,
              # Tracking related rewards.
              tracking_lin_vel=1.0,
              tracking_ang_vel=0.75,
              # Base related rewards.
              lin_vel_z=0.0,
              ang_vel_xy=-0.15,
              orientation=-2.0,
              base_height=0.0,
              # Energy related rewards.
              torques=0.0,
              action_rate=0.0,
              energy=0.0,
              dof_acc=0.0,
              # Feet related rewards.
              feet_clearance=0.0,
              feet_air_time=2.0,
              feet_slip=-0.25,
              feet_height=0.0,
              feet_phase=1.0,
              # Other rewards.
              alive=0.0,
              stand_still=-1.0,
              termination=-100.0,
              collision=-0.1,
              contact_force=-0.01,
              # Pose related rewards.
              joint_deviation_knee=-0.1,
              joint_deviation_hip=-0.25,
              dof_pos_limits=-1.0,
              pose=-0.1,
          ),
          tracking_sigma=0.25,
          max_foot_height=0.15,
          base_height_target=0.5,
          max_contact_force=500.0,
      ),
      push_config=config_dict.create(
          enable=True,
          interval_range=[5.0, 10.0],
          magnitude_range=[0.1, 2.0],
      ),
      command_config=config_dict.create(
          contact_horizon=25,
          step_length=1,
          resample_time=0.4,
          observation_horizon=2,
          angle_range=[0, 2 * jp.pi],
          stride_length_range=[0.5, 1.5],
          distance_threshold=0.8,
          goal_reached_threshold=2,
          touchdown_threshold=0.2,
          quad_delta=0.2,
          max_resample_count=25,   
      ),
      lin_vel_x=[-1.0, 1.0],
      lin_vel_y=[-0.5, 0.5],
      ang_vel_yaw=[-1.0, 1.0],
  )


class ContactEnv(g1_base.G1Env):
  """Track a joystick command."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ):
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init()

  def _post_init(self) -> None:
    self._init_q = jp.array(self._mj_model.keyframe("knees_bent").qpos)
    self._default_pose = jp.array(
        self._mj_model.keyframe("knees_bent").qpos[7:]
    )

    # Note: First joint is freejoint.
    self._lowers, self._uppers = self.mj_model.jnt_range[1:].T
    c = (self._lowers + self._uppers) / 2
    r = self._uppers - self._lowers
    self._soft_lowers = c - 0.5 * r * self._config.soft_joint_pos_limit_factor
    self._soft_uppers = c + 0.5 * r * self._config.soft_joint_pos_limit_factor

    waist_indices = []
    waist_joint_names = [
        "waist_yaw",
        "waist_roll",
        "waist_pitch",
    ]
    for joint_name in waist_joint_names:
      waist_indices.append(
          self._mj_model.joint(f"{joint_name}_joint").qposadr - 7
      )
    self._waist_indices = jp.array(waist_indices)

    arm_indices = []
    arm_joint_names = [
        "shoulder_roll",
        "shoulder_yaw",
        "wrist_roll",
        "wrist_pitch",
        "wrist_yaw",
    ]
    for side in ["left", "right"]:
      for joint_name in arm_joint_names:
        arm_indices.append(
            self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7
        )
    self._arm_indices = jp.array(arm_indices)

    hip_indices = []
    hip_joint_names = [
        "hip_roll",
        "hip_yaw",
    ]
    for side in ["left", "right"]:
      for joint_name in hip_joint_names:
        hip_indices.append(
            self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7
        )
    self._hip_indices = jp.array(hip_indices)

    knee_indices = []
    knee_joint_names = ["knee"]
    for side in ["left", "right"]:
      for joint_name in knee_joint_names:
        knee_indices.append(
            self._mj_model.joint(f"{side}_{joint_name}_joint").qposadr - 7
        )
    self._knee_indices = jp.array(knee_indices)

    # fmt: off
    self._weights = jp.array([
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # left leg.
        0.01, 1.0, 1.0, 0.01, 1.0, 1.0,  # right leg.
        1.0, 1.0, 1.0,  # waist.
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # left arm.
        1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0,  # right arm.
    ])
    # fmt: on

    self._torso_body_id = self._mj_model.body(consts.ROOT_BODY).id
    self._torso_mass = self._mj_model.body_subtreemass[self._torso_body_id]
    self._torso_imu_site_id = self._mj_model.site("imu_in_torso").id
    self._pelvis_imu_site_id = self._mj_model.site("imu_in_pelvis").id

    self._feet_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.FEET_SITES]
    )
    self._hands_site_id = np.array(
        [self._mj_model.site(name).id for name in consts.HAND_SITES]
    )
    self._floor_geom_id = self._mj_model.geom("floor").id
    self._feet_geom_id = np.array(
        [self._mj_model.geom(name).id for name in consts.FEET_GEOMS]
    )

    foot_linvel_sensor_adr = []
    for site in consts.FEET_SITES:
      sensor_id = self._mj_model.sensor(f"{site}_global_linvel").id
      sensor_adr = self._mj_model.sensor_adr[sensor_id]
      sensor_dim = self._mj_model.sensor_dim[sensor_id]
      foot_linvel_sensor_adr.append(
          list(range(sensor_adr, sensor_adr + sensor_dim))
      )
    self._foot_linvel_sensor_adr = jp.array(foot_linvel_sensor_adr)

    # self._cmd_a = jp.array(self._config.command_config.a)
    # self._cmd_b = jp.array(self._config.command_config.b)

    self._left_hand_geom_id = self._mj_model.geom("left_hand_collision").id
    self._right_hand_geom_id = self._mj_model.geom("right_hand_collision").id
    self._left_foot_geom_id = self._mj_model.geom("left_foot").id
    self._right_foot_geom_id = self._mj_model.geom("right_foot").id
    self._left_shin_geom_id = self._mj_model.geom("left_shin").id
    self._right_shin_geom_id = self._mj_model.geom("right_shin").id
    self._left_thigh_geom_id = self._mj_model.geom("left_thigh").id
    self._right_thigh_geom_id = self._mj_model.geom("right_thigh").id

    

    # self.left_foot_target_site = self._mj_model.site("left_foot_target").id
    # self.right_foot_target_site = self._mj_model.site("right_foot_target").id

    # self.left_target_joint_id = self._mj_model.joint('left_foot_target_joint').id
    # self.right_target_joint_id = self._mj_model.joint('right_foot_target_joint').id

    print(self._mj_model.nmocap)

    self.left_target_mocap_id = 0
    self.right_target_mocap_id = 1
    
    # breakpoint()
    

    self.contact_horizon = self._config.command_config.contact_horizon
    self.observation_horizon = self._config.command_config.observation_horizon
    self.angle_range = self._config.command_config.angle_range
    self.stride_length_range = self._config.command_config.stride_length_range
    self.goal_reached_threshold = self._config.command_config.goal_reached_threshold
    self.distance_threshold = self._config.command_config.distance_threshold
    self.quad_delta = self._config.command_config.quad_delta

    print("nq", self.mjx_model.nq)
    print("nv", self.mjx_model.nv)
    print("nu", self.mjx_model.nu)
    print(self._default_pose.shape)

  def _generate_contact_plan(self):
    contact_plan = jp.zeros((self.contact_horizon, 2), dtype=bool)
    pattern = jp.array([[True, True], [False, True], [True, True], [True, False]])


    # Repeat the pattern enough times to cover the entire horizon
    repeats = (self.contact_horizon + pattern.shape[0] - 1) // pattern.shape[0]  # ceil division
    tiled_pattern = jp.tile(pattern, (repeats, 1))

    # Slice to match exactly the contact_horizon
    contact_plan = tiled_pattern[:self.contact_horizon]

    return contact_plan
  
  def _generate_future_ee_positions(self, rng: jax.Array, data: mjx.Data, contact_plan: jax.Array) -> jax.Array:
    foot_positions = data.site_xpos[self._feet_site_id]  # shape: (2, 3), [left_foot, right_foot]
    horizon = self.contact_horizon

    # Step 2: Sample heading and stride
    rng, key = jax.random.split(rng)
    heading = jp.pi / 2
    direction = jp.array([jp.cos(heading), jp.sin(heading), 0.0])

    rng, key = jax.random.split(rng)
    stride_length = 0.4

    # Step 3: Initialize positions
    left_init, right_init = foot_positions[0], foot_positions[1]
    foot_pos_init = jp.stack([left_init, right_init])  # (2, 3)

    def step_fn(carry, t):
        foot_positions, _ = carry  # (2, 3)
        contact = contact_plan[t]

        new_positions = foot_positions

        # If a foot is in swing (contact=False), move it forward
        def move_foot(i, pos):
            new_pos = foot_positions[i] + direction * stride_length
            return jax.lax.cond(contact[i], lambda _: pos, lambda _: new_pos, operand=None)

        new_left = move_foot(0, foot_positions[0])
        new_right = move_foot(1, foot_positions[1])
        new_positions = jp.stack([new_left, new_right])  # (2, 3)

        return (new_positions, None), new_positions

    (_, _), positions_over_time = jax.lax.scan(step_fn, (foot_pos_init, None), jp.arange(horizon))

    return positions_over_time
  
  def convert_ee_to_base_frame(self, info):
        """
        Converts future end-effector positions from world frame to robot base frame using JAX.
        Works for a single environment.

        Args:
            future_ee_positions_w (jnp.ndarray): Future end-effector positions in world frame.
                Shape: (contact_horizon, 2, 3)
            robot_base_pos_w (jnp.ndarray): Robot base position in world frame.
                Shape: (3,)
            robot_base_quat_w (jnp.ndarray): Robot base orientation (quaternion) in world frame.
                Shape: (4,)

        Returns:
            jnp.ndarray: Future end-effector positions in robot base frame.
                Shape: (contact_horizon, 2, 3)
        """
        contact_horizon = self._config.command_config.contact_horizon
        robot_base_pos_w = info["robot_base_pos_w"]  # Get the robot base position in world frame
        robot_base_quat_w = info["robot_base_quat_w"]  # Get the robot base orientation (quaternion) in world frame

        # 1. Translate to Origin
        # Expand robot_base_pos_w to match the shape of future_ee_positions_w for broadcasting
        robot_base_pos_w_expanded = jp.expand_dims(jp.expand_dims(robot_base_pos_w, axis=0), axis=0)  # Shape: (1, 1, 3)
        robot_base_pos_w_expanded = jp.broadcast_to(robot_base_pos_w_expanded, (contact_horizon, 2, 3))
        future_ee_positions_b_translated = info["future_ee_positions_w"] - robot_base_pos_w_expanded

        # Expand robot_base_pos_w to match the shape of desired_ee_positions_w for broadcasting
        robot_base_pos_w_expanded_desired = jp.expand_dims(robot_base_pos_w, axis=0) # Shape: (1, 3)
        desired_ee_positions_b_translated = info["desired_ee_positions_w"] - robot_base_pos_w_expanded_desired

        # 2. Rotate to Base Frame
        # Inverse of a quaternion is its conjugate
        robot_base_quat_w_conj = jp.array([robot_base_quat_w[0], -robot_base_quat_w[1], -robot_base_quat_w[2], -robot_base_quat_w[3]])

        def quat_rotate_batched(q: jp.ndarray, v: jp.ndarray) -> jp.ndarray:
            """
            Rotates each vector `v[i]` by the quaternion `q[i]`.
            Args:
                q: (..., 4)
                v: (..., 3)
            Returns:
                Rotated vectors (..., 3)
            """
            w, x, y, z = jp.split(q, 4, axis=-1)
            q_vec = jp.concatenate([x, y, z], axis=-1)

            t = 2.0 * jp.cross(q_vec, v)
            return v + w * t + jp.cross(q_vec, t)


        # Expand the quaternion to match the shape of the vectors to be rotated
        quat_expanded = jp.expand_dims(jp.expand_dims(robot_base_quat_w_conj, axis=0), axis=0)
        quat_expanded = jp.broadcast_to(quat_expanded, (contact_horizon, 2, 4))
        info["future_ee_positions_b"] = quat_rotate_batched(quat_expanded, future_ee_positions_b_translated)

        quat_expanded_desired = jp.expand_dims(robot_base_quat_w_conj, axis=0)
        quat_expanded_desired = jp.broadcast_to(quat_expanded_desired, (2, 4))
        info["desired_ee_positions_b"] = quat_rotate_batched(quat_expanded_desired, desired_ee_positions_b_translated)

        return info
  
#   def _is_goal_reached(self, info: Dict) -> jp.ndarray:
#     """
#     JAX-compatible function to check whether the goal is reached based on end-effector positions
#     and contact plan. Returns a boolean JAX array (not Python bool).
#     """

#     actual_ee_xy = info["current_ee_positions_w"][:, :2]  # Shape: (2, 2)
#     desired_ee_xy = info["desired_ee_positions_w"][:, :2]  # Shape: (2, 2)

#     def case_no_contact(_):
#         return jp.array(True)

#     def case_both_contact(_):
#         distances = jp.linalg.norm(actual_ee_xy - desired_ee_xy, axis=1)  # Shape: (2,)
#         return jp.all(distances < self.distance_threshold) == True

#     def case_one_contact(_):
#         # Get index of the foot in contact (0 or 1)
#         left_in_contact = info["contact_command"][0]
#         dist = jp.where(
#             left_in_contact,
#             jp.linalg.norm(actual_ee_xy[0] - desired_ee_xy[0]),
#             jp.linalg.norm(actual_ee_xy[1] - desired_ee_xy[1]),
#         )
#         return  jp.all(dist < self.distance_threshold)

#     both_contact = jp.all(info["contact_command"])
#     no_contact = jp.all(~info["contact_command"])

#     return jax.lax.cond(
#         no_contact,
#         case_no_contact,
#         lambda _: jax.lax.cond(both_contact, case_both_contact, case_one_contact, operand=None),
#         operand=None
#     )
  
  def _is_goal_reached(self, info: Dict) -> jax.Array:
    time_left = info["time_left"]
    robot_base_pos_w = info["robot_base_pos_w"]
    desired_ee_positions_w = info["desired_ee_positions_w"]

    rect_corners = self._create_rectangle_from_midpoints(desired_ee_positions_w[0], desired_ee_positions_w[1])
    # Check if the robot base position is within the rectangle defined by the midpoints
    is_in_rectangle = self._is_point_in_rectangle(rect_corners, robot_base_pos_w)

    touchdown_phase = time_left <= self._config.command_config.touchdown_threshold

    condition = jp.logical_and(is_in_rectangle, jp.logical_or(touchdown_phase, jp.all(info["contact_plan"][info["current_goal_index"]])))

    return condition

  
  def _create_rectangle_from_midpoints(self, p1, p2):    
    # Vector between midpoints
    v = p2 - p1
    v_norm = jp.linalg.norm(v)
    

    # Perpendicular unit vector
    perp = jp.array([-v[1], v[0], 0]) / v_norm

    # Half side length
    offset = perp * (self.quad_delta / 2)

    # Corners
    p1_a = p1 + offset
    p1_b = p1 - offset
    p2_a = p2 + offset
    p2_b = p2 - offset

    return jp.array([p1_a, p2_a, p2_b, p1_b])
  
  def _is_point_in_rectangle(self, rect_corners, point):
    # Ensure corners are in a closed loop
    rect_corners = rect_corners[:, :2]
    point = point[:2]
    polygon = jp.vstack([rect_corners, rect_corners[0]])
    x, y = point
    x0 = polygon[:-1, 0]
    y0 = polygon[:-1, 1]
    x1 = polygon[1:, 0]
    y1 = polygon[1:, 1]

    # Ray casting conditions
    cond1 = (y0 > y) != (y1 > y)
    slope = (x1 - x0) * (y - y0) / (y1 - y0 + 1e-8) + x0
    cond2 = x < slope

    inside = jp.sum(jp.logical_and(cond1, cond2)) % 2 == 1
    return inside

  
  def _compute_command(self, info: Dict) -> jax.Array:
    info["time_left"] = info["time_left"] - self._config.ctrl_dt
    info = jax.lax.cond(info["time_left"] <= 0.0, lambda _: self._resample_command(info), lambda _: info, operand=None)
    info = self._update_command(info)
    return info
  
  
  def _resample_command(self, info: Dict) -> Dict:
    info["time_left"] = self._config.command_config.resample_time
    def new_command(info): 
      time_left = info["time_left"]
      time_left = self._config.command_config.resample_time
      info["time_left"] = time_left

      info["previous_goal_index"] = info["current_goal_index"]
      info["current_goal_index"] = info["current_goal_index"] + 1
      info["desired_ee_positions_w"] = info["future_ee_positions_w"][info["current_goal_index"]]
      return info    

    def old_command(info):
      return info
      
    goal_reached = self._is_goal_reached(info)
    first_goal = info["current_goal_index"] < 0
    goal_reached = jp.logical_and(goal_reached, info["reach_goal_timer"] >= self.goal_reached_threshold)
    condition = jp.logical_or(first_goal, goal_reached)

    return jax.lax.cond(condition, new_command, old_command, info)

  

  
  def _update_command(self, info: Dict) -> Dict:
    info = self.convert_ee_to_base_frame(info)

    def increment_timer(_):
        info["reach_goal_timer"] += 1
        return info

    def reset_timer(_):
        info["reach_goal_timer"] = 0
        return info

    info = jax.lax.cond(
        self._is_goal_reached(info),
        increment_timer,
        reset_timer,
        operand=None
    )

    start_idx = info["current_goal_index"]
    slice_len = self.observation_horizon

    # Ensure we don't go out of bounds
    max_start_idx = self.contact_horizon - slice_len
    start_idx = jp.minimum(start_idx, max_start_idx)

    info["position_command"] = jax.lax.dynamic_slice(
        info["future_ee_positions_b"],
        start_indices=(start_idx, 0, 0),  # assumes shape is (T, feet, xyz)
        slice_sizes=(slice_len, info["future_ee_positions_b"].shape[1], info["future_ee_positions_b"].shape[2])
    )
    info["contact_command"] = jax.lax.dynamic_slice(
        info["contact_plan"],
        start_indices=(start_idx, 0),  # slicing starts at (start_idx, column 0)
        slice_sizes=(slice_len, info["contact_plan"].shape[1])
    )
    return info


  def reset(self, rng: jax.Array) -> mjx_env.State:
    qpos = self._init_q
    qvel = jp.zeros(self.mjx_model.nv)

    # x=+U(-0.5, 0.5), y=+U(-0.5, 0.5), yaw=U(-3.14, 3.14).
    rng, key = jax.random.split(rng)
    dxy = jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
    qpos = qpos.at[0:2].set(qpos[0:2] + dxy)
    rng, key = jax.random.split(rng)
    yaw = jax.random.uniform(key, (1,), minval=-3.14, maxval=3.14)
    quat = math.axis_angle_to_quat(jp.array([0, 0, 1]), yaw)
    new_quat = math.quat_mul(qpos[3:7], quat)
    qpos = qpos.at[3:7].set(new_quat)

    # qpos[7:]=*U(0.5, 1.5)
    rng, key = jax.random.split(rng)
    qpos = qpos.at[7:].set(qpos[7:] * jax.random.uniform(key, qpos[7:].shape, minval=0.5, maxval=1.5))



    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )
    ctrl = jp.zeros(self.mjx_model.nu)


    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=ctrl)
     

    contact_plan = self._generate_contact_plan()
    future_ee_positions_w = self._generate_future_ee_positions(rng, data, contact_plan)    

    # # Phase, freq=U(1.0, 1.5)
    # rng, key = jax.random.split(rng)
    # gait_freq = jax.random.uniform(key, (1,), minval=1.25, maxval=1.5)
    # phase_dt = 2 * jp.pi * self.dt * gait_freq
    # phase = jp.array([0, jp.pi])

    # Sample push interval.
    rng, push_rng = jax.random.split(rng)
    push_interval = jax.random.uniform(
        push_rng,
        minval=self._config.push_config.interval_range[0],
        maxval=self._config.push_config.interval_range[1],
    )
    push_interval_steps = jp.round(push_interval / self.dt).astype(jp.int32)

    info = {
        "rng": rng,
        "step": 0,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "contact_status": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
        # # Phase related.
        # "phase_dt": phase_dt,
        # "phase": phase,
        # Push related.
        "push": jp.array([0.0, 0.0]),
        "push_step": 0,
        "push_interval_steps": push_interval_steps,
    }

    info["robot_base_pos_w"] = data.xpos[self._torso_body_id]
    info["robot_base_quat_w"] = data.xquat[self._torso_body_id]
    info["current_ee_positions_w"] = data.site_xpos[self._feet_site_id]
    info["contact_plan"] = contact_plan
    info["future_ee_positions_w"] = future_ee_positions_w
    info["time_left"] = 0.0
    info["previous_goal_index"] = -2
    info["current_goal_index"] = -1
    info["position_command"] = jp.zeros((self.observation_horizon, 2, 3))
    info["contact_command"] = jp.zeros((self.observation_horizon, 2), dtype=bool)
    info["reach_goal_timer"] = 0
    info["desired_ee_positions_w"] = jp.zeros((2, 3))
    info["resample_count"] = 0
    info["truncation"] = jp.zeros(1)

    info = self._compute_command(info)

    data = data.replace(mocap_pos=data.mocap_pos.at[0].set(info["future_ee_positions_w"][0][0]))
    data = data.replace(mocap_pos=data.mocap_pos.at[1].set(info["future_ee_positions_w"][0][1]))

    data = data.replace(mocap_pos=data.mocap_pos.at[2].set(info["future_ee_positions_w"][1][0]))
    data = data.replace(mocap_pos=data.mocap_pos.at[3].set(info["future_ee_positions_w"][1][1]))

    data = data.replace(mocap_pos=data.mocap_pos.at[4].set(info["future_ee_positions_w"][2][0]))
    data = data.replace(mocap_pos=data.mocap_pos.at[5].set(info["future_ee_positions_w"][2][1]))
       
    data = data.replace(mocap_pos=data.mocap_pos.at[6].set(info["future_ee_positions_w"][3][0]))
    data = data.replace(mocap_pos=data.mocap_pos.at[7].set(info["future_ee_positions_w"][3][1]))

    data = data.replace(mocap_pos=data.mocap_pos.at[8].set(info["future_ee_positions_w"][4][0]))
    data = data.replace(mocap_pos=data.mocap_pos.at[9].set(info["future_ee_positions_w"][4][1]))

    data = data.replace(mocap_pos=data.mocap_pos.at[10].set(info["future_ee_positions_w"][6][0]))
    data = data.replace(mocap_pos=data.mocap_pos.at[11].set(info["future_ee_positions_w"][6][1]))

    # data = data.replace(mocap_pos=data.mocap_pos.at[:].set(info["future_ee_positions_w"][:self._mj_model.nmocap][0]))

    metrics = {}
    for k in self._config.reward_config.scales.keys():
      metrics[f"reward/{k}"] = jp.zeros(())
    metrics["swing_peak"] = jp.zeros(())

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    obs = self._get_obs(data, info, contact)
    reward, done = jp.zeros(2)

    return mjx_env.State(data, obs, reward, done, metrics, info)

  def step(self, state: mjx_env.State, action: jax.Array) -> mjx_env.State:
    state.info["rng"], push1_rng, push2_rng = jax.random.split(
        state.info["rng"], 3
    )
    push_theta = jax.random.uniform(push1_rng, maxval=2 * jp.pi)
    push_magnitude = jax.random.uniform(
        push2_rng,
        minval=self._config.push_config.magnitude_range[0],
        maxval=self._config.push_config.magnitude_range[1],
    )
    push = jp.array([jp.cos(push_theta), jp.sin(push_theta)])
    push *= (
        jp.mod(state.info["push_step"] + 1, state.info["push_interval_steps"])
        == 0
    )
    push *= self._config.push_config.enable
    qvel = state.data.qvel
    qvel = qvel.at[:2].set(push * push_magnitude + qvel[:2])
    data = state.data.replace(qvel=qvel)
    # jax.debug.print("Before: {}", data.site_xpos[self.left_foot_target_site])

    # data = data.replace(qpos=data.qpos.at[self._mj_model.jnt_qposadr[self.left_target_joint_id]:
    #                self._mj_model.jnt_qposadr[self.left_target_joint_id] + 3].set([0.0, 0.0, 0.8]))
    # data = data.replace(qpos=data.qpos.at[self._mj_model.jnt_qposadr[self.right_target_joint_id]:
    #                self._mj_model.jnt_qposadr[self.right_target_joint_id] + 3].set([0.0, 0.3, 0.8]))

    

    

    # jax.debug.print("After: {}", data.site_xpos[self.left_foot_target_site])

    # jax.debug.print("###"* 30)
    state = state.replace(data=data)

    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )
    state.info["motor_targets"] = motor_targets

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    state.info["contact_status"] = contact
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt
    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data, state.info)
    

    rewards = self._get_reward(
        data, action, state.info, state.metrics, done, first_contact, contact
    )
    rewards = {
        k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    }
    reward = sum(rewards.values()) * self.dt
    state.info["truncation"] = self._get_truncation(state.info).astype(reward.dtype)

    state.info["push"] = push
    state.info["step"] += 1
    state.info["push_step"] += 1
    # phase_tp1 = state.info["phase"] + state.info["phase_dt"]
    # state.info["phase"] = jp.fmod(phase_tp1 + jp.pi, 2 * jp.pi) - jp.pi
    # NOTE(kevin): Enable this to make the policy stand still at 0 command.
    # state.info["phase"] = jp.where(
    #     jp.linalg.norm(state.info["command"]) > 0.01,
    #     state.info["phase"],
    #     jp.ones(2) * jp.pi,
    # )
    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["rng"], cmd_rng = jax.random.split(state.info["rng"])
    # state.info["command"] = jp.where(
    #     state.info["step"] > 500,
    #     self.sample_command(cmd_rng),
    #     state.info["command"],
    # )
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact

    state = state.replace(info=self._compute_command(state.info))


    for k, v in rewards.items():
      state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    # mjx.forward(self.mjx_model, data)
    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state
  
  def _quat_to_euler(self, q: jp.ndarray) -> jp.ndarray:
    # Assumes q = [w, x, y, z]
    w, x, y, z = q

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = jp.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    pitch = jp.where(jp.abs(sinp) >= 1.0, jp.sign(sinp) * (jp.pi / 2), jp.arcsin(sinp))

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = jp.arctan2(siny_cosp, cosy_cosp)

    return jp.array([roll, pitch, yaw])
  
  def _get_truncation(self, info: dict[str, Any]) -> jax.Array:
    # Truncation condition: number of resamples exceeds the limit
    truncation = info["resample_count"] >= self._config.command_config.max_resample_count
    return truncation


  def _get_termination(self, data: mjx.Data, info) -> jax.Array:
    fall_termination = self.get_gravity(data, "torso")[-1] < 0.0
    contact_termination = collision.geoms_colliding(
        data,
        self._right_foot_geom_id,
        self._left_foot_geom_id,
    )
    contact_termination |= collision.geoms_colliding(
        data,
        self._left_foot_geom_id,
        self._right_shin_geom_id,
    )
    contact_termination |= collision.geoms_colliding(
        data,
        self._right_foot_geom_id,
        self._left_shin_geom_id,
    )

    torso_q = data.xquat[self._torso_body_id]  # shape (4,)
    torso_euler = self._quat_to_euler(torso_q)       # shape (3,) -> [roll, pitch, yaw]
    excessive_roll = jp.abs(torso_euler[0]) > 0.872  # roll
    excessive_pitch = jp.abs(torso_euler[1]) > 0.872  # pitch

    return (
        fall_termination
        | contact_termination
        | excessive_roll
        | excessive_pitch
        | jp.isnan(data.qpos).any()
        | jp.isnan(data.qvel).any()
    )

  def _get_obs(
      self, data: mjx.Data, info: dict[str, Any], contact: jax.Array
  ) -> mjx_env.Observation:
    gyro = self.get_gyro(data, "pelvis")
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gyro = (
        gyro
        + (2 * jax.random.uniform(noise_rng, shape=gyro.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gyro
    )

    gravity = data.site_xmat[self._pelvis_imu_site_id].T @ jp.array([0, 0, -1])
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_gravity = (
        gravity
        + (2 * jax.random.uniform(noise_rng, shape=gravity.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.gravity
    )

    joint_angles = data.qpos[7:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_angles = (
        joint_angles
        + (2 * jax.random.uniform(noise_rng, shape=joint_angles.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_pos
    )

    joint_vel = data.qvel[6:]
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_joint_vel = (
        joint_vel
        + (2 * jax.random.uniform(noise_rng, shape=joint_vel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.joint_vel
    )

    # cos = jp.cos(info["phase"])
    # sin = jp.sin(info["phase"])
    # phase = jp.concatenate([cos, sin])

    linvel = self.get_local_linvel(data, "pelvis")
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )

    current_ee_positions_w = data.site_xpos[self._feet_site_id]
    desired_ee_positions_w = info["desired_ee_positions_w"]
    position_error = current_ee_positions_w - desired_ee_positions_w  # shape (num_feet, 3)
    norm_error = jp.linalg.norm(position_error, axis=-1)  # shape (num_feet,)


    state = jp.hstack([
        jp.ravel(noisy_linvel),              # → (3,)
        jp.ravel(noisy_gyro),               # → (3,)
        jp.ravel(noisy_gravity),            # → (3,)
        jp.ravel(info["contact_command"]),  # → (2,)
        jp.ravel(info["position_command"]), # → (3,) ← if it's (x, 3), this needs clarification
        jp.ravel(noisy_joint_angles - self._default_pose),  # → (29,)
        jp.ravel(noisy_joint_vel),          # → (29,)
        jp.ravel(info["last_act"]),         # → (29,)
        jp.ravel(info["time_left"]),    # → (29,)
        jp.ravel(norm_error),              
    ])

    accelerometer = self.get_accelerometer(data, "pelvis")
    global_angvel = self.get_global_angvel(data, "pelvis")
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr].ravel()
    root_height = data.qpos[2]

    privileged_state = jp.hstack([
        state,
        gyro,  # 3
        accelerometer,  # 3
        gravity,  # 3
        linvel,  # 3
        global_angvel,  # 3
        joint_angles - self._default_pose,
        joint_vel,
        root_height,  # 1
        data.actuator_force,  # 29
        contact,  # 2
        feet_vel,  # 4*3
        info["feet_air_time"],  # 2
    ])

    return {
        "state": state,
        "privileged_state": privileged_state,
    }

  def _get_reward(
      self,
      data: mjx.Data,
      action: jax.Array,
      info: dict[str, Any],
      metrics: dict[str, Any],
      done: jax.Array,
      first_contact: jax.Array,
      contact: jax.Array,
  ) -> dict[str, jax.Array]:
    del metrics  # Unused.
    return {
        # Tracking rewards.
        "stance_reward": self._stance_reward(info),
        "swing_reward": self._swing_reward(info),
        "touchdown_reward": self._touchdown_reward(info),
        "touchdown_finegrained_reward": self._touchdown_finegrained_reward(info),
        "new_goal_reward": self._new_goal_reward(info),
        # Base-related rewards.
        "lin_vel_z": self._cost_lin_vel_z(
            self.get_global_linvel(data, "pelvis"),
            self.get_global_linvel(data, "torso"),
        ),
        "ang_vel_xy": self._cost_ang_vel_xy(
            self.get_global_angvel(data, "torso")
        ),
        "orientation": self._cost_orientation(self.get_gravity(data, "torso")),
        "base_height": self._cost_base_height(data.qpos[2]),
        # Energy related rewards.
        "torques": self._cost_torques(data.actuator_force),
        "action_rate": self._cost_action_rate(
            action, info["last_act"], info["last_last_act"]
        ),
        # "energy": self._cost_energy(data.qvel[6:-14], data.actuator_force),
        "dof_acc": self._cost_dof_acc(data.qacc[6:]),
        # Feet related rewards.
        "feet_slip": self._cost_feet_slip(data, contact, info),
        # "feet_clearance": self._cost_feet_clearance(data, info),
        # "feet_height": self._cost_feet_height(
        #     info["swing_peak"], first_contact, info
        # ),
        # "feet_air_time": self._reward_feet_air_time(
        #     info["feet_air_time"], first_contact, info["command"]
        # ),
        # "feet_phase": self._reward_feet_phase(
        #     data,
        #     info["phase"],
        #     self._config.reward_config.max_foot_height,
        #     info["command"],
        # ),
        # Other rewards.
        "alive": self._reward_alive(),
        "termination": self._cost_termination(done),
        # "stand_still": self._cost_stand_still(info["command"], data.qpos[7:]),
        "collision": self._cost_collision(data),
        "contact_force": self._cost_contact_force(data),
        # Pose related rewards.
        # "joint_deviation_hip": self._cost_joint_deviation_hip(
        #     data.qpos[7:], info["command"]
        # ),
        "joint_deviation_knee": self._cost_joint_deviation_knee(data.qpos[7:]),
        # "dof_pos_limits": self._cost_joint_pos_limits(data.qpos[7:-14]),
        "pose": self._cost_pose(data.qpos[7:]),
    }
  
  def _touchdown_reward(self, info: Dict) -> jax.Array:
    contact_status = info["contact_status"]  
    desired_ee_pos = info["desired_ee_positions_w"]  
    current_ee_pos = info["current_ee_positions_w"]
    current_goal_index = info["current_goal_index"]  
    current_plan = info["contact_command"][current_goal_index]  
    next_plan = info["contact_command"][current_goal_index + 1]
    time_left = info["time_left"]

    touchdown_phase = ((time_left < self._config.command_config.touchdown_threshold).astype(bool)) & (~current_plan)
    distance_to_goal = jp.linalg.norm(desired_ee_pos - current_ee_pos, axis=-1)
    transition_factor = jp.clip(time_left / self._config.command_config.touchdown_threshold, 0.0, 1.0) 

    current_plan_reward = jp.where(contact_status == current_plan, 1.0, -0.25)  
    next_plan_reward = jp.where(contact_status == next_plan, 1.0, 0.0)
    contact_reward = transition_factor * current_plan_reward + (1 - transition_factor) * next_plan_reward

    reward = distance_to_goal * contact_reward * touchdown_phase

    return jp.sum(reward)
  
  def _touchdown_finegrained_reward(self, info: Dict, std: float = 0.1) -> jax.Array:
    contact_status = info["contact_status"]  
    desired_ee_pos = info["desired_ee_positions_w"]  
    current_ee_pos = info["current_ee_positions_w"]
    current_goal_index = info["current_goal_index"]  
    current_plan = info["contact_command"][current_goal_index]  
    next_plan = info["contact_command"][current_goal_index + 1] 
    time_left = info["time_left"]

    touchdown_phase = ((time_left < self._config.command_config.touchdown_threshold).astype(bool)) & (~current_plan)
    distance_to_goal = jp.linalg.norm(desired_ee_pos - current_ee_pos, axis=-1)

    # Smooth, fine-grained distance reward using tanh
    fine_grained_distance_reward = 1.0 - jp.tanh(distance_to_goal / std)

    transition_factor = jp.clip(time_left / self._config.command_config.touchdown_threshold, 0.0, 1.0) 

    current_plan_reward = jp.where(contact_status == current_plan, 1.0, -0.25)
    next_plan_reward = jp.where(contact_status == next_plan, 1.0, 0.0)

    contact_reward = transition_factor * current_plan_reward + (1 - transition_factor) * next_plan_reward

    reward = fine_grained_distance_reward * contact_reward * touchdown_phase

    return jp.sum(reward)


  def _swing_reward(self, info: Dict) -> jax.Array:
      contact = info["contact_status"]  # shape: [num_envs, num_feet]
      current_goal_index = info["current_goal_index"]  # int
      planned_contact = info["contact_command"][current_goal_index]  # shape: [num_envs, num_feet]
      
      # Boolean mask for swing phase
      swing_phase = (
          info["time_left"] >= self._config.command_config.touchdown_threshold
      ) & (~planned_contact)

      # +1 for correct swing contact, -5 otherwise
      contact_reward = jp.where(contact == planned_contact, 1.0, -5.0)

      # Apply reward only during swing phase
      swing_reward = contact_reward * swing_phase

      return swing_reward.sum()  # sum across feet



  
  def _stance_reward(self, info: Dict, std: float=0.17) -> jax.Array:
    contact = info["contact_status"]
    current_goal_index = info["current_goal_index"]
    stance_phase = info["contact_command"][current_goal_index]
    is_correct_contact = jp.where(contact == stance_phase, True, False)

    feet_errors = jp.linalg.norm(info["position_command"][current_goal_index] - info["current_ee_positions_w"], axis=-1)

    return (
        (
            (is_correct_contact * jp.exp(-feet_errors / std**2))  
            + (~is_correct_contact * -0.25)
        )
        * stance_phase
    ).sum()
  
  def _new_goal_reward(self, info: Dict) -> jax.Array:
    current_goal_index = info["current_goal_index"]
    previous_goal_index = info["previous_goal_index"]

    # Give a reward of 1.0 when a new goal is reached
    goal_advanced = current_goal_index > previous_goal_index
    reward = jp.where(goal_advanced, 1.0, 0.0)

    return reward

  def _cost_contact_force(self, data: mjx.Data) -> jax.Array:
    l_contact_force = mjx_env.get_sensor_data(
        self.mj_model, data, "left_foot_force"
    )
    r_contact_force = mjx_env.get_sensor_data(
        self.mj_model, data, "right_foot_force"
    )
    cost = jp.clip(
        jp.abs(l_contact_force[2])
        - self._config.reward_config.max_contact_force,
        min=0.0,
    )
    cost += jp.clip(
        jp.abs(r_contact_force[2])
        - self._config.reward_config.max_contact_force,
        min=0.0,
    )
    return cost

  def _cost_collision(self, data: mjx.Data) -> jax.Array:
    c = collision.geoms_colliding(
        data, self._left_hand_geom_id, self._left_thigh_geom_id
    )
    c |= collision.geoms_colliding(
        data, self._right_hand_geom_id, self._right_thigh_geom_id
    )
    return jp.any(c)

  # Tracking rewards.

  def _cost_joint_deviation_hip(
      self, qpos: jax.Array, cmd: jax.Array
  ) -> jax.Array:
    error = qpos[self._hip_indices] - self._default_pose[self._hip_indices]
    # Allow roll deviation when lateral velocity is high.
    weight = jp.where(
        cmd[1] > 0.1,
        jp.array([0.0, 1.0, 0.0, 1.0]),
        jp.array([1.0, 1.0, 1.0, 1.0]),
    )
    cost = jp.sum(jp.abs(error) * weight)
    return cost

  def _cost_joint_deviation_knee(self, qpos: jax.Array) -> jax.Array:
    error = qpos[self._knee_indices] - self._default_pose[self._knee_indices]
    return jp.sum(jp.abs(error))

  def _cost_pose(self, qpos: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qpos - self._default_pose))

  def _cost_joint_pos_limits(self, qpos: jax.Array) -> jax.Array:
    out_of_limits = -jp.clip(qpos - self._soft_lowers, None, 0.0)
    out_of_limits += jp.clip(qpos - self._soft_uppers, 0.0, None)
    return jp.sum(out_of_limits)

  def _reward_tracking_lin_vel(
      self,
      commands: jax.Array,
      local_vel: jax.Array,
  ) -> jax.Array:
    lin_vel_error = jp.sum(jp.square(commands[:2] - local_vel[:2]))
    return jp.exp(-lin_vel_error / self._config.reward_config.tracking_sigma)

  def _reward_tracking_ang_vel(
      self,
      commands: jax.Array,
      ang_vel: jax.Array,
  ) -> jax.Array:
    ang_vel_error = jp.square(commands[2] - ang_vel[2])
    return jp.exp(-ang_vel_error / self._config.reward_config.tracking_sigma)

  # Base-related rewards.

  def _cost_lin_vel_z(
      self,
      global_linvel_torso: jax.Array,
      global_linvel_pelvis: jax.Array,
  ) -> jax.Array:
    torso_cost = jp.square(global_linvel_torso[2])
    pelvis_cost = jp.square(global_linvel_pelvis[2])
    return torso_cost + pelvis_cost

  def _cost_ang_vel_xy(self, global_angvel_torso: jax.Array) -> jax.Array:
    return jp.sum(jp.square(global_angvel_torso[:2]))

  def _cost_orientation(self, torso_zaxis: jax.Array) -> jax.Array:
    return jp.sum(jp.square(torso_zaxis - jp.array([0.073, 0.0, 1.0])))

  def _cost_base_height(self, base_height: jax.Array) -> jax.Array:
    return jp.square(
        base_height - self._config.reward_config.base_height_target
    )

  # Energy related rewards.

  def _cost_torques(self, torques: jax.Array) -> jax.Array:
    return jp.sum(jp.abs(torques))

  def _cost_energy(
      self, qvel: jax.Array, qfrc_actuator: jax.Array
  ) -> jax.Array:
    return jp.sum(jp.abs(qvel) * jp.abs(qfrc_actuator))

  def _cost_action_rate(
      self, act: jax.Array, last_act: jax.Array, last_last_act: jax.Array
  ) -> jax.Array:
    del last_last_act  # Unused.
    return jp.sum(jp.square(act - last_act))

  def _cost_dof_acc(self, qacc: jax.Array) -> jax.Array:
    return jp.sum(jp.square(qacc))

  # Other rewards.

  def _cost_stand_still(
      self, commands: jax.Array, qpos: jax.Array
  ) -> jax.Array:
    cmd_norm = jp.linalg.norm(commands)
    cost = jp.sum(jp.abs(qpos - self._default_pose))
    cost *= cmd_norm < 0.01
    return cost

  def _cost_termination(self, done: jax.Array) -> jax.Array:
    return done

  def _reward_alive(self) -> jax.Array:
    return jp.array(1.0)

  # Feet related rewards.

  def _cost_feet_slip(
      self, data: mjx.Data, contact: jax.Array, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    body_vel = self.get_global_linvel(data, "pelvis")[:2]
    reward = jp.sum(jp.linalg.norm(body_vel, axis=-1) * contact)
    return reward

  def _cost_feet_clearance(
      self, data: mjx.Data, info: dict[str, Any]
  ) -> jax.Array:
    del info  # Unused.
    feet_vel = data.sensordata[self._foot_linvel_sensor_adr]
    vel_xy = feet_vel[..., :2]
    vel_norm = jp.sqrt(jp.linalg.norm(vel_xy, axis=-1))
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    delta = jp.abs(foot_z - self._config.reward_config.max_foot_height)
    return jp.sum(delta * vel_norm)

  def _cost_feet_height(
      self,
      swing_peak: jax.Array,
      first_contact: jax.Array,
      info: dict[str, Any],
  ) -> jax.Array:
    del info  # Unused.
    error = swing_peak / self._config.reward_config.max_foot_height - 1.0
    return jp.sum(jp.square(error) * first_contact)

  def _reward_feet_air_time(
      self,
      air_time: jax.Array,
      first_contact: jax.Array,
      commands: jax.Array,
      threshold_min: float = 0.2,
      threshold_max: float = 0.5,
  ) -> jax.Array:
    del commands  # Unused.
    air_time = (air_time - threshold_min) * first_contact
    air_time = jp.clip(air_time, max=threshold_max - threshold_min)
    reward = jp.sum(air_time)
    return reward

  def _reward_feet_phase(
      self,
      data: mjx.Data,
      phase: jax.Array,
      foot_height: jax.Array,
      command: jax.Array,
  ) -> jax.Array:
    # Reward for tracking the desired foot height.
    foot_pos = data.site_xpos[self._feet_site_id]
    foot_z = foot_pos[..., -1]
    rz = gait.get_rz(phase, swing_height=foot_height)
    error = jp.sum(jp.square(foot_z - rz))
    reward = jp.exp(-error / 0.01)
    body_linvel = self.get_global_linvel(data, "pelvis")[:2]
    body_angvel = self.get_global_angvel(data, "pelvis")[2]
    linvel_mask = jp.logical_or(
        jp.linalg.norm(body_linvel) > 0.1,
        jp.abs(body_angvel) > 0.1,
    )
    mask = jp.logical_or(linvel_mask, jp.linalg.norm(command) > 0.01)
    reward *= mask
    return reward

  # def sample_command(self, rng: jax.Array) -> jax.Array:
  #   rng1, rng2, rng3, rng4 = jax.random.split(rng, 4)

  #   lin_vel_x = jax.random.uniform(
  #       rng1, minval=self._config.lin_vel_x[0], maxval=self._config.lin_vel_x[1]
  #   )
  #   lin_vel_y = jax.random.uniform(
  #       rng2, minval=self._config.lin_vel_y[0], maxval=self._config.lin_vel_y[1]
  #   )
  #   ang_vel_yaw = jax.random.uniform(
  #       rng3,
  #       minval=self._config.ang_vel_yaw[0],
  #       maxval=self._config.ang_vel_yaw[1],
  #   )

  #   # With 10% chance, set everything to zero.
  #   return jp.where(
  #       jax.random.bernoulli(rng4, p=0.1),
  #       jp.zeros(3),
  #       jp.hstack([lin_vel_x, lin_vel_y, ang_vel_yaw]),
  #   )






  



