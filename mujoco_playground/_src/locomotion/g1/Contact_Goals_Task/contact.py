"""Contact tracking task for G1 robot."""


from typing import Any, Dict, Optional, Union

import mujoco

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

from mujoco_playground._src.locomotion.g1.Contact_Goals_Task.contact_goals_generator import ContactGoalsGenerator

def default_config() -> config_dict.ConfigDict:
  return config_dict.create(
      ctrl_dt=0.02,
      sim_dt=0.002,
      episode_length=1000,
      action_repeat=1,
      action_scale=0.0,
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
          goal_time=0.4,
          observation_horizon=2,
          angle_range=[0, 2 * jp.pi],
          stride_length_range=[0.5, 1.5],
          distance_threshold=0.8,
          goal_reached_threshold=2,
          touchdown_threshold=0.2,   
      ),
      contact_goal_config=config_dict.create(
            contact_horizon=10,  # Number of future contact points to plan
            step_length=0.3,     # Forward distance between consecutive steps
            step_width=0.2,      # Lateral distance between feet
            foot_height=0.1,     # Maximum foot height during swing
            base_height=0.5,     # Desired robot base height
            goal_threshold=0.05, # Distance threshold to consider goal achieved
        ),
      lin_vel_x=[-1.0, 1.0],
      lin_vel_y=[-0.5, 0.5],
      ang_vel_yaw=[-1.0, 1.0],
  )


class Contact(g1_base.G1Env):
  """Track Contact goals."""

  def __init__(
      self,
      task: str = "flat_terrain",
      config: config_dict.ConfigDict = default_config(),
      config_overrides: Optional[Dict[str, Union[str, int, list[Any]]]] = None,
  ) -> None:
    super().__init__(
        xml_path=consts.task_to_xml(task).as_posix(),
        config=config,
        config_overrides=config_overrides,
    )
    self._post_init_()
  
  def _post_init_(self) -> None:
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

    self.left_goal_site_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "goal_left_foot")
    self.right_goal_site_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "goal_right_foot")

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

    torso_pos = self._mj_model.body(self._torso_body_id).pos
    left_foot_offset = self._mj_model.site("left_foot").pos - torso_pos
    right_foot_offset = self._mj_model.site("right_foot").pos - torso_pos
    feet_offset = jp.array(
        [left_foot_offset, right_foot_offset]
    )

    self.command_generator = ContactGoalsGenerator(self._config.command_config, self._torso_body_id, feet_offset, self.dt)
  
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
    qpos = qpos.at[7:].set(
        qpos[7:] * jax.random.uniform(key, (29,), minval=0.5, maxval=1.5)
    )

    # d(xyzrpy)=U(-0.5, 0.5)
    rng, key = jax.random.split(rng)
    qvel = qvel.at[0:6].set(
        jax.random.uniform(key, (6,), minval=-0.5, maxval=0.5)
    )

    data = mjx_env.init(self.mjx_model, qpos=qpos, qvel=qvel, ctrl=qpos[7:])

    rng, key = jax.random.split(rng)
    self.command_generator.reset(data, key)
    cmd = self.command_generator.get_command()

    info = {
        "rng": rng,
        "step": 0,
        "command": cmd,
        "last_act": jp.zeros(self.mjx_model.nu),
        "last_last_act": jp.zeros(self.mjx_model.nu),
        "motor_targets": jp.zeros(self.mjx_model.nu),
        "feet_air_time": jp.zeros(2),
        "last_contact": jp.zeros(2, dtype=bool),
        "swing_peak": jp.zeros(2),
    }

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
    motor_targets = self._default_pose + action * self._config.action_scale
    data = mjx_env.step(
        self.mjx_model, state.data, motor_targets, self.n_substeps
    )

    contact = jp.array([
        geoms_colliding(data, geom_id, self._floor_geom_id)
        for geom_id in self._feet_geom_id
    ])
    contact_filt = contact | state.info["last_contact"]
    first_contact = (state.info["feet_air_time"] > 0.0) * contact_filt
    state.info["feet_air_time"] += self.dt

    p_f = data.site_xpos[self._feet_site_id]
    p_fz = p_f[..., -1]
    state.info["swing_peak"] = jp.maximum(state.info["swing_peak"], p_fz)

    obs = self._get_obs(data, state.info, contact)
    done = self._get_termination(data)

    # rewards = self._get_reward(
    #     data, action, state.info, state.metrics, done, first_contact, contact
    # )
    # rewards = {
    #     k: v * self._config.reward_config.scales[k] for k, v in rewards.items()
    # }
    # reward = sum(rewards.values()) * self.dt
    reward = jp.zeros(())

    self.command_generator.step(data)

    state.info["last_last_act"] = state.info["last_act"]
    state.info["last_act"] = action
    state.info["step"] = jp.where(
        done | (state.info["step"] > 500),
        0,
        state.info["step"],
    )
    state.info["command"] = self.command_generator.get_command()
    state.info["feet_air_time"] *= ~contact
    state.info["last_contact"] = contact
    state.info["swing_peak"] *= ~contact
    # for k, v in rewards.items():
    #   state.metrics[f"reward/{k}"] = v
    state.metrics["swing_peak"] = jp.mean(state.info["swing_peak"])

    done = done.astype(reward.dtype)
    state = state.replace(data=data, obs=obs, reward=reward, done=done)
    return state
  
  def _get_termination(self, data: mjx.Data) -> jax.Array:
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
    return (
        fall_termination
        | contact_termination
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


    linvel = self.get_local_linvel(data, "pelvis")
    info["rng"], noise_rng = jax.random.split(info["rng"])
    noisy_linvel = (
        linvel
        + (2 * jax.random.uniform(noise_rng, shape=linvel.shape) - 1)
        * self._config.noise_config.level
        * self._config.noise_config.scales.linvel
    )


    # Flatten command to be able to stack it in obs
    future_ee_positions, contact_plan, time_left = info["command"]

    # Flatten each component
    flat_ee_positions = future_ee_positions.ravel()
    flat_contact_plan = contact_plan.ravel()
    flat_time_left = jp.array([time_left])  # scalar needs to be array-like for stacking

    # Concatenate into one flat array
    flat_command = jp.concatenate([flat_ee_positions, flat_contact_plan, flat_time_left])

    state = jp.hstack([
        noisy_linvel,  # 3
        noisy_gyro,  # 3
        noisy_gravity,  # 3
        flat_command,  # 3
        noisy_joint_angles - self._default_pose,  # 29
        noisy_joint_vel,  # 29
        info["last_act"],  # 29
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
    pass
  
  def update_goal_sites(self, data):
    """
    Updates the two goal sites to follow the latest desired positions.
    
    Args:
        model: MuJoCo model object.
        data: MuJoCo data object.
        desired_ee_positions_w: np.ndarray of shape (2, 3),
            world-frame desired foot positions for [left, right] foot.
    """

    # Find the site IDs once (assuming site names: "goal_left_foot" and "goal_right_foot")
    left_site_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "goal_left_foot")
    right_site_id = mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_SITE, "goal_right_foot")

    desired_ee_positions_w = self.command_generator.get_desired_positions_w()
    
    # Update site positions
    data.site_xpos[left_site_id] = desired_ee_positions_w[0]
    data.site_xpos[right_site_id] = desired_ee_positions_w[1]
  
  

    

