from typing import Any, Dict, Optional, Union

import jax
import jax.numpy as np
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

class ContactCommandGenerator:
    def __init__(self, command_config, torso_body_id):
        self.config = command_config
        self._torso_body_id = torso_body_id

    def _post_init(self):
        self._generate_contact_plan()
        self._generate_future_ee_positions()
        

    def _generate_contact_plan(self):
        self.contact_plan = np.zeros((self.config.contact_horizon, 2), dtype=bool)
        pattern = np.array([[True, True], [False, True], [True, True], [True, False]])


        # Repeat the pattern enough times to cover the entire horizon
        repeats = (self.config.contact_horizon + pattern.shape[0] - 1) // pattern.shape[0]  # ceil division
        tiled_pattern = np.tile(pattern, (repeats, 1))

        # Slice to match exactly the contact_horizon
        self.contact_plan = tiled_pattern[:self.config.contact_horizon]

        
    
    def _generate_future_ee_positions(self, data):
        """
        Generates future end-effector positions for a robot's feet,
        converted to use NumPy arrays and standard Python loops.

        Args:
            self: An object containing robot configuration (e.g., self.config.contact_horizon,
                self._torso_body_id).
            data: A data object (e.g., MuJoCo mjx.Data-like) containing robot state
                information like xpos (body positions) and xmat (body rotation matrices).
            contact_plan: A NumPy array representing the contact schedule for each foot
                        over the horizon. Shape (horizon, 2) where 2 is for left/right foot.

        Returns:
            A NumPy array of shape (horizon, 2, 3) representing the desired 3D positions
            of the left and right end-effectors over the planning horizon.
        """
        # 1. Get the base of the robot's torso position in world coordinates
        base_pos = data.xpos[self._torso_body_id]

        # 2. Project the base position onto the ground (set Z-component to 0)
        # For NumPy, we create a copy and modify it in-place.
        projected_base_pos = base_pos.copy()
        projected_base_pos[2] = 0.0 # Assuming Z is the up-axis (index 2)

        # 3. Get the robot's right and left vectors relative to its current orientation
        # data.xmat columns are usually [forward (X), left (Y), up (Z)] in world coords.
        robot_left_vector_world = data.xmat[self._torso_body_id, :3, 1]
        robot_right_vector_world = -data.xmat[self._torso_body_id, :3, 1]

        # Define the desired offset magnitude
        offset_magnitude = 0.12

        # 4. Calculate the initial foot positions based on projection and offset
        initial_left_foot_pos = projected_base_pos + robot_left_vector_world * offset_magnitude
        initial_right_foot_pos = projected_base_pos + robot_right_vector_world * offset_magnitude

        # Stack them to form the initial foot_positions array for the loop
        current_foot_positions = np.stack([initial_left_foot_pos, initial_right_foot_pos])  # shape: (2, 3)

        horizon = self.config.contact_horizon

        # Step 2: Heading and stride (these are fixed in your provided code)
        # No random number generation using 'rng' in this snippet, so 'rng' param is omitted.
        heading = np.pi / 2
        direction = np.array([np.cos(heading), np.sin(heading), 0.0])
        # Normalize to a unit vector, handle potential division by zero for zero vectors
        norm_direction = np.linalg.norm(direction)
        if norm_direction > 1e-6: # Check if norm is non-zero
            direction = direction / norm_direction
        else:
            direction = np.array([1.0, 0.0, 0.0]) # Default to forward if no valid direction

        stride_length = 0.4

        # List to collect foot positions at each timestep
        positions_over_time_list = []

        # Replace jax.lax.scan with a standard Python for loop
        # The 'carry' from jax.lax.scan corresponds to 'current_foot_positions' here.
        for t in range(horizon):
            contact = self.contact_plan[t] # Get contact status for this timestep, shape (2,)

            # Create copies to potentially modify, similar to JAX's functional updates
            # This is good practice even though 'new_pos' will be reassigned
            new_left_pos = current_foot_positions[0].copy()
            new_right_pos = current_foot_positions[1].copy()

            # Replace jax.lax.cond with standard Python if/else statements
            # If left foot is in swing (contact=False), move it forward
            if not contact[0]:
                new_left_pos = current_foot_positions[0] + direction * stride_length

            # If right foot is in swing (contact=False), move it forward
            if not contact[1]:
                new_right_pos = current_foot_positions[1] + direction * stride_length

            # Update the foot positions for the next iteration
            current_foot_positions = np.stack([new_left_pos, new_right_pos])

            # Store the current state's positions for the output
            positions_over_time_list.append(current_foot_positions)

        # Convert the list of NumPy arrays into a single NumPy array
        positions_over_time_array = np.array(positions_over_time_list)

        self.future_ee_positions_w = positions_over_time_array

  
    def _compute_command(self, data):
        self.previous_goal_index = np.roll(self.previous_goal_index, shift=1)
        self.previous_goal_index[0] = np.where(self.current_goal_index == -1, 0, self.current_goal_index)
        self.time_left = self.time_left - self.config.ctrl_dt
        if self.time_left <= 0.0:
            self._resample_command()
        self._update_command(data)
          
    def _resample_command(self):
        self.time_left = self.config.resample_time
        goal_reached = self.reach_goal_timer >= self.config.goal_reached_threshold
        if goal_reached:
            self.current_goal_index += 1
            self.desired_ee_positions_w = self.future_ee_positions_w[self.current_goal_index]
            self.reach_goal_timer = 0

    def _update_command(self, data):
        self.convert_ee_to_base_frame(data)

        if self._is_goal_reached(data):
            self.reach_goal_timer += 1
        else:
            self.reach_goal_timer = 0

        start_idx = self.current_goal_index
        slice_len = self.observation_horizon

        # Ensure we don't go out of bounds
        # np.minimum becomes np.minimum
        max_start_idx = self.contact_horizon - slice_len
        start_idx = np.minimum(start_idx, max_start_idx)

        # Dynamic slicing with jax.lax.dynamic_slice translates directly to
        # standard NumPy array slicing.
        # For array[start:end, :, :], the 'end' is (start + slice_len).
        self.position_command = self.future_ee_positions_b[
            start_idx : start_idx + slice_len,
            :, # Take all elements along the second dimension
            :  # Take all elements along the third dimension
        ]

        self.contact_command = self.contact_plan[
            start_idx : start_idx + slice_len,
            : # Take all elements along the second dimension
        ]

    def _is_goal_reached(self, data):
        robot_base_pos_w = data.xpos[self._torso_body_id]
        robot_base_pos_w[2] = 0.0
        robot_base_pos_proj = robot_base_pos_w

        goal_point_1 = self.desired_ee_positions_w[0]
        goal_point_2 = self.desired_ee_positions_w[1]

        dist_to_goal1 = np.linalg.norm(robot_base_pos_proj - goal_point_1)
        dist_to_goal2 = np.linalg.norm(robot_base_pos_proj - goal_point_2)

        
        proximity_condition = (dist_to_goal1 <= 0.18) | \
                                (dist_to_goal2 <= 0.18)
        
        touchdown_phase = self.time_left <= self._config.config.touchdown_threshold
        condition = np.logical_and(proximity_condition, touchdown_phase)
        
        # is_correct_contact = np.all(info["contact_status"] == info["contact_plan"][info["current_goal_index"]])

        # jax.debug.breakpoint()

        return condition

        
        
    def convert_ee_to_base_frame(self, data):
        
        contact_horizon = self._config.contact_horizon
        robot_base_pos_w = data.xpos[self._torso_body_id]  # Get the robot base position in world frame
        robot_base_quat_w = data.xquat[self._torso_body_id] # Get the robot base orientation (quaternion) in world frame

        # 1. Translate to Origin
        # Expand robot_base_pos_w to match the shape of future_ee_positions_w for broadcasting
        robot_base_pos_w_expanded = np.expand_dims(np.expand_dims(robot_base_pos_w, axis=0), axis=0)  # Shape: (1, 1, 3)
        robot_base_pos_w_expanded = np.broadcast_to(robot_base_pos_w_expanded, (contact_horizon, 2, 3))
        future_ee_positions_b_translated = self.future_ee_positions_w - robot_base_pos_w_expanded

        # Expand robot_base_pos_w to match the shape of desired_ee_positions_w for broadcasting
        robot_base_pos_w_expanded_desired = np.expand_dims(robot_base_pos_w, axis=0) # Shape: (1, 3)
        desired_ee_positions_b_translated = self.desired_ee_positions_w - robot_base_pos_w_expanded_desired

        # 2. Rotate to Base Frame
        # Inverse of a quaternion is its conjugate
        robot_base_quat_w_conj = np.array([robot_base_quat_w[0], -robot_base_quat_w[1], -robot_base_quat_w[2], -robot_base_quat_w[3]])

        def quat_rotate_batched(q: np.ndarray, v: np.ndarray) -> np.ndarray:
            """
            Rotates each vector `v[i]` by the quaternion `q[i]`.
            Args:
                q: (..., 4)
                v: (..., 3)
            Returns:
                Rotated vectors (..., 3)
            """
            w, x, y, z = np.split(q, 4, axis=-1)
            q_vec = np.concatenate([x, y, z], axis=-1)

            t = 2.0 * np.cross(q_vec, v)
            return v + w * t + np.cross(q_vec, t)


        # Expand the quaternion to match the shape of the vectors to be rotated
        quat_expanded = np.expand_dims(np.expand_dims(robot_base_quat_w_conj, axis=0), axis=0)
        quat_expanded = np.broadcast_to(quat_expanded, (contact_horizon, 2, 4))
        self.future_ee_positions_b = quat_rotate_batched(quat_expanded, future_ee_positions_b_translated)

        quat_expanded_desired = np.expand_dims(robot_base_quat_w_conj, axis=0)
        quat_expanded_desired = np.broadcast_to(quat_expanded_desired, (2, 4))
        self.desired_ee_positions_b = quat_rotate_batched(quat_expanded_desired, desired_ee_positions_b_translated)

    
