import gymnasium as gym
from gymnasium import spaces
import jax
import jax.numpy as jnp
from mujoco_playground.wrappers import mjx_wrapper  # For MJX compatibility
from mujoco_playground.mjcf import standard_mjcf  # For loading MJCF models

class HumanoidVelocityEnv(gym.Env):
    """
    A Gymnasium environment for training a humanoid to achieve a target velocity.
    """
    def __init__(self, target_velocity=(1.0, 0.0, 0.0), reset_noise_scale=0.1):
        super().__init__()

        self.target_velocity = jnp.array(target_velocity)
        self.reset_noise_scale = reset_noise_scale

        # 1. Load the MJCF model (Humanoid)
        self.mjcf_model = standard_mjcf.Humanoid()
        self.mjx_model = mjx_wrapper.wrap(self.mjcf_model)
        self.mjx_state = self.mjx_model.reset(jax.random.PRNGKey(0)) # Initial MJX state

        # 2. Define Action Space
        # Example: Torques applied to the humanoid's joints
        self.action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(self.mjx_model.num_actuators,), dtype=jnp.float32
        )

        # 3. Define Observation Space
        # Example: Joint angles, joint velocities, torso orientation, velocity
        obs_dim = (
            self.mjx_model.nq + self.mjx_model.nv + 6 # qpos, qvel, torso orientation (e.g., quaternion) + linear velocity
        )
        self.observation_space = spaces.Box(
            low=-jnp.inf, high=jnp.inf, shape=(obs_dim,), dtype=jnp.float32
        )

    def _get_obs(self, mjx_state):
        """Extract relevant observations from the MJX state."""
        qpos = mjx_state.qpos
        qvel = mjx_state.qvel

        # Get torso orientation (example: using rotation matrix to quaternion)
        torso_rot = mjx_state.body_rot[self.mjx_model.body_name2id["torso"]]
        torso_quat = jax.numpy.ravel(jax.scipy.spatial.transform.Rotation.from_matrix(torso_rot).as_quat())

        # Get linear velocity of the torso
        torso_lin_vel = mjx_state.body_velp[self.mjx_model.body_name2id["torso"]]

        return jnp.concatenate([qpos, qvel, torso_quat, torso_lin_vel])

    def _compute_reward(self, mjx_state, action):
        """Compute the reward based on the current state and action."""
        torso_lin_vel = mjx_state.body_velp[self.mjx_model.body_name2id["torso"]][:3] # x, y, z velocity
        velocity_error = jnp.linalg.norm(torso_lin_vel - self.target_velocity)
        # Example reward: Negative of the velocity error
        reward = -velocity_error
        return reward

    def step(self, action):
        """Apply an action, step the simulation, and return the results."""
        # Ensure action is a JAX array
        action = jnp.array(action)

        # Apply the action (assuming direct torque control)
        applied_action = self.mjx_state.actuator_moment.at[:].set(action)
        next_mjx_state = self.mjx_model.step(self.mjx_state.replace(actuator_moment=applied_action))
        self.mjx_state = next_mjx_state

        # Get the observation
        observation = self._get_obs(self.mjx_state)

        # Compute the reward
        reward = self._compute_reward(self.mjx_state, action)

        # Determine if the episode is done (example: time limit)
        terminated = False # Implement your termination condition
        truncated = False # Implement your truncation condition (e.g., episode length)
        info = {}

        return observation, reward, terminated, truncated, info

    def reset(self, seed=None, options=None):
        """Reset the environment to its initial state."""
        super().reset(seed=seed)
        key = jax.random.PRNGKey(seed if seed is not None else 0)
        self.mjx_state = self.mjx_model.reset(key)

        # Add some random noise to the initial state for exploration
        if self.reset_noise_scale > 0:
            noise_qpos = jax.random.normal(key, shape=self.mjx_state.qpos.shape) * self.reset_noise_scale
            noise_qvel = jax.random.normal(key, shape=self.mjx_state.qvel.shape) * self.reset_noise_scale
            self.mjx_state = self.mjx_state.replace(
                qpos=self.mjx_state.qpos + noise_qpos,
                qvel=self.mjx_state.qvel + noise_qvel
            )

        observation = self._get_obs(self.mjx_state)
        info = {}
        return observation, info

    def render(self):
        """Render the current state of the environment (optional)."""
        # Rendering implementation using MJX's rendering capabilities
        raise NotImplementedError("Rendering not yet implemented for this example.")

    def close(self):
        """Clean up resources (if any)."""
        pass