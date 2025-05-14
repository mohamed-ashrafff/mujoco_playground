import jax
import jax.numpy as jnp
from jax import vmap
from mujoco_playground._src import mjx_env
from mujoco_playground._src.locomotion.g1 import g1_constants as consts
from typing import Optional, Sequence, Dict


class ContactGoalsGenerator:
    def __init__(self, config, root_body_id, feet_offset, dt):
        """
        Initializes the contact sequence generator.

        Args:
            config: A dictionary containing configuration parameters for
                contact goal generation. Must contain 'episode_horizon'
                (int, number of contact goals per episode).
            num_envs: The number of parallel environments.
        """
        self.config = config
        self.contact_horizon = config["contact_horizon"]
        self.step_length = config["step_length"]
        self.goal_time = config["goal_time"]
        self.observation_horizon = config["observation_horizon"]
        self.root_body_id = root_body_id 
        self.feet_offset = feet_offset 
        self.angle_range = config["angle_range"] 
        self.stride_length_range = config["stride_length_range"]
        self.distance_threshold = config["distance_threshold"]
        self.goal_reached_threshold = config["goal_reached_threshold"]
        self.touchdown_threshold = config["touchdown_threshold"]
        self.dt = dt

        
        self.contact_plan = jnp.ones((self.contact_horizon, 2), dtype=bool)       
        self.current_ee_positions_w = jnp.zeros((2, 3))  
        self.current_ee_positions_b = jnp.zeros_like(self.current_ee_positions_w)
        self.current_ee_contact = jnp.ones((2,), dtype=bool)  
        self.future_ee_positions_w = jnp.zeros((self.contact_horizon, 2, 3)) 
        self.future_ee_positions_b = jnp.zeros_like(self.future_ee_positions_w) 
        self.desired_ee_positions_w = jnp.zeros((2, 3))
        self.desired_ee_positions_b = jnp.zeros_like(self.desired_ee_positions_w)
        self.heading_command = 0
        self.current_goal_index = 0
        self.previous_goal_index = 0
        self.goal_reached_timer = 0.0
        self.time_left = self.goal_time

        self.gait_patterns = jnp.array(
            [
                [[True, False], [False, True]],  
                [[False, True], [True, False]],  
            ]
        )


    def get_command(self):
        """
        Returns the command for the current episode.
        """
        return (
            self.future_ee_positions_b[self.current_goal_index :: min(self.observation_horizon + self.current_goal_index, self.contact_horizon)], 
            self.contact_plan[self.current_goal_index :: min(self.observation_horizon + self.current_goal_index, self.contact_horizon)],
            self.time_left)
    
    def reset(self, data, rng):
        """
        Handles the reset of the environment.
        """
        rng, key = jax.random.split(rng)
        self.contact_plan = self._generate_contact_plan(key, self.contact_plan)
        rng, key = jax.random.split(rng)
        self._generate_future_ee_positions(data, key)

        self.current_goal_index = 0
        self.time_left = self.goal_time

        self._update_desired_ee_positions()
        self.update_command(data)

    def step(self, data):
        """
        Handles the step of the environment.
        """
        self.time_left -= self.dt
        if self.time_left <= 0.0:
            self.resample_command(data)
            self.time_left = self.goal_time
        self.update_command(data)

    def _generate_contact_plan(self, rng, contact_plan):
        rng, key = jax.random.split(rng)
        mask = jax.random.uniform(key) < 0.5

        def true_branch(_):
            new_plan = contact_plan.at[2::2].set(self.gait_patterns[0, 0])
            new_plan = new_plan.at[1::2].set(self.gait_patterns[0, 1])
            return new_plan

        def false_branch(_):
            new_plan = contact_plan.at[2::2].set(self.gait_patterns[1, 0])
            new_plan = new_plan.at[1::2].set(self.gait_patterns[1, 1])
            return new_plan

        contact_plan = jax.lax.cond(mask, true_branch, false_branch, operand=None)
        return contact_plan
        
    def _generate_future_ee_positions(self, data, rng):
        """
        Generates future end-effector positions for the current episode.
        """
        # Get the current position of the feet in the world frame
        root_pos = data.xpos[self.root_body_id]  
        foot_positions = root_pos + self.feet_offset  

        # Initialize future end effector positions to the current foot positions
        self.future_ee_positions_w = jnp.stack([foot_positions] * self.contact_horizon, axis=0)
        
        # Generate random heading angle and stride length
        rng, key = jax.random.split(rng)
        self.heading_command = jax.random.uniform(key, minval=-self.angle_range[0], maxval=self.angle_range[1])

        direction_x = jnp.cos(self.heading_command)
        direction_y = jnp.sin(self.heading_command)

        rng, key = jax.random.split(rng)
        self.stride_length = jax.random.uniform(key, minval=self.stride_length_range[0], maxval=self.stride_length_range[1])

        # Calculate stride offset for each contact timestep
        self.stride_offsets = jnp.arange(self.contact_horizon) * self.stride_length

        
        x_displacements = self.stride_offsets[:, jnp.newaxis] * direction_x
        y_displacements = self.stride_offsets[:, jnp.newaxis] * direction_y

        # Update future_ee_positions_w
        self.future_ee_positions_w = self.future_ee_positions_w.at[:, :, 0].add(x_displacements)
        self.future_ee_positions_w = self.future_ee_positions_w.at[:, :, 1].add(y_displacements)



    def _convert_ee_to_base_frame(self, data):
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

        contact_horizon = self.future_ee_positions_w.shape[0]
        robot_base_pos_w = data.xpos[self.root_body_id]  # Get the robot base position in world frame
        robot_base_quat_w = data.xquat[self.root_body_id]  # Get the robot base orientation (quaternion) in world frame

        # 1. Translate to Origin
        # Expand robot_base_pos_w to match the shape of future_ee_positions_w for broadcasting
        robot_base_pos_w_expanded = jnp.expand_dims(jnp.expand_dims(robot_base_pos_w, axis=0), axis=0)  # Shape: (1, 1, 3)
        robot_base_pos_w_expanded = jnp.broadcast_to(robot_base_pos_w_expanded, (contact_horizon, 2, 3))
        future_ee_positions_b_translated = self.future_ee_positions_w - robot_base_pos_w_expanded

        # Expand robot_base_pos_w to match the shape of desired_ee_positions_w for broadcasting
        robot_base_pos_w_expanded_desired = jnp.expand_dims(robot_base_pos_w, axis=0) # Shape: (1, 3)
        desired_ee_positions_b_translated = self.desired_ee_positions_w - robot_base_pos_w_expanded_desired

        

        # 2. Rotate to Base Frame
        # Inverse of a quaternion is its conjugate
        robot_base_quat_w_conj = jnp.array([robot_base_quat_w[0], -robot_base_quat_w[1], -robot_base_quat_w[2], -robot_base_quat_w[3]])

        def quat_rotate_batched(q: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
            """
            Rotates each vector `v[i]` by the quaternion `q[i]`.
            Args:
                q: (..., 4)
                v: (..., 3)
            Returns:
                Rotated vectors (..., 3)
            """
            w, x, y, z = jnp.split(q, 4, axis=-1)
            q_vec = jnp.concatenate([x, y, z], axis=-1)

            t = 2.0 * jnp.cross(q_vec, v)
            return v + w * t + jnp.cross(q_vec, t)


        # Expand the quaternion to match the shape of the vectors to be rotated
        quat_expanded = jnp.expand_dims(jnp.expand_dims(robot_base_quat_w_conj, axis=0), axis=0)
        quat_expanded = jnp.broadcast_to(quat_expanded, (contact_horizon, 2, 4))
        self.future_ee_positions_b = quat_rotate_batched(quat_expanded, future_ee_positions_b_translated)

        quat_expanded_desired = jnp.expand_dims(robot_base_quat_w_conj, axis=0)
        quat_expanded_desired = jnp.broadcast_to(quat_expanded_desired, (2, 4))
        self.desired_ee_positions_b = quat_rotate_batched(quat_expanded_desired, desired_ee_positions_b_translated)
    

    def _is_goal_reached(self) -> jnp.ndarray:
        """
        JAX-compatible function to check whether the goal is reached based on end-effector positions
        and contact plan. Returns a boolean JAX array (not Python bool).
        """

        actual_ee_xy = self.current_ee_positions_w[:, :2]  # Shape: (2, 2)
        desired_ee_xy = self.desired_ee_positions_w[:, :2]  # Shape: (2, 2)

        def case_no_contact(_):
            return jnp.array(True)

        def case_both_contact(_):
            distances = jnp.linalg.norm(actual_ee_xy - desired_ee_xy, axis=1)  # Shape: (2,)
            return jnp.all(distances < self.distance_threshold) == True

        def case_one_contact(_):
            # Get index of the foot in contact (0 or 1)
            left_in_contact = self.contact_plan[0]
            dist = jnp.where(
                left_in_contact,
                jnp.linalg.norm(actual_ee_xy[0] - desired_ee_xy[0]),
                jnp.linalg.norm(actual_ee_xy[1] - desired_ee_xy[1]),
            )
            return  jnp.all(dist < self.distance_threshold)

        both_contact = jnp.all(self.contact_plan)
        no_contact = jnp.all(~self.contact_plan)

        return jax.lax.cond(
            no_contact,
            case_no_contact,
            lambda _: jax.lax.cond(both_contact, case_both_contact, case_one_contact, operand=None),
            operand=None
        )

        
    

    def update_command(self, data):
        """
        Called at every step. Checks if the goal has been reached and updates the command.
        This version is JAX-compatible and can be used under vmap or jit.
        """
        self._convert_ee_to_base_frame(data)

        touchdown = (self.goal_time < self.touchdown_threshold) & ~self.contact_plan[self.current_goal_index]
        goal_reached = self._is_goal_reached() & (
            jnp.any(touchdown, axis=-1) | jnp.all(self.contact_plan[self.current_goal_index], axis=-1)
        )

        # Use JAX-compatible conditional update
        self.goal_reached_timer = jax.lax.select(
            goal_reached,
            self.goal_reached_timer + 1.0,
            self.goal_reached_timer
    )


        
        
    def resample_command(self, data):
        """
        Resamples the command if the goal has been reached.
        Meaning pick the next set of goals. 
        """
        if self.goal_reached_timer >= self.goal_reached_threshold:
            self.current_goal_index = (self.current_goal_index + 1) % self.contact_horizon
            self._update_desired_ee_positions()

    def _update_desired_ee_positions(self):
        self.desired_ee_positions_w = self.future_ee_positions_w[self.current_goal_index]

    def get_desired_positions_w(self):
        """
        Returns the desired end-effector positions in world frame.
        """
        return self.desired_ee_positions_w

    

    


        




        

    


    