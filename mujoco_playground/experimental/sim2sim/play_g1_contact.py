# Copyright 2024 DeepMind Technologies Limited
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
"""Deploy an MJX policy in ONNX format to C MuJoCo and play with it."""

from brax.training.agents.ppo import networks as ppo_networks
from mujoco_playground.config import locomotion_params, manipulation_params
from mujoco_playground import locomotion, manipulation
import functools
import pickle
import jax.numpy as jp
import jax
import tf2onnx
import tensorflow as tf
from tensorflow.keras import layers
import onnxruntime as rt
from brax.training.acme import running_statistics
from brax.training.agents.ppo import checkpoint
import sys

from etils import epath
import mujoco
import mujoco.viewer as viewer
import numpy as np
import onnxruntime as rt
import argparse

from mujoco_playground._src.locomotion.g1 import g1_constants
from mujoco_playground._src.locomotion.g1 import contact_command_generator
from mujoco_playground._src.locomotion.g1.base import get_assets

import os
os.environ["MUJOCO_GL"] = "egl"
os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"

_HERE = epath.Path(__file__).parent
_ONNX_DIR = _HERE / "onnx"


class OnnxController:
  """ONNX controller for the Go-1 robot."""

  def __init__(
      self,
      policy_path: str,
      default_angles: np.ndarray,
      ctrl_dt: float,
      n_substeps: int,
      torso_body_id: int,
      feet_site_id,
      action_scale: float = 0.5,
  ):
    self._output_names = ["continuous_actions"]
    self._policy = rt.InferenceSession(
        policy_path, providers=["CPUExecutionProvider"]
    )

    self._action_scale = action_scale
    self._default_angles = default_angles
    self._last_action = np.zeros_like(default_angles, dtype=np.float32)

    self._counter = 0
    self._n_substeps = n_substeps
    self.ctrl_dt = ctrl_dt

    self.feet_site_id = feet_site_id

    command_config = {
        "contact_horizon": 1000,
        "observation_horizon": 2,
        "touchdown_threshold": 0.2,
        "resample_time": 0.5,
        "ctrl_dt": self.ctrl_dt,
        "goal_reached_threshold": 2,
    }

    self.command_generator = contact_command_generator.ContactCommandGenerator(command_config, torso_body_id)

    

  def get_obs(self, model, data) -> np.ndarray:
    linvel = data.sensor("local_linvel_pelvis").data
    gyro = data.sensor("gyro_pelvis").data
    imu_xmat = data.site_xmat[model.site("imu_in_pelvis").id].reshape(3, 3)
    gravity = imu_xmat.T @ np.array([0, 0, -1])
    joint_angles = data.qpos[7:] - self._default_angles
    joint_velocities = data.qvel[6:]
    self.command_generator._compute_command(data)
    position_command = self.command_generator.position_command
    contact_command = self.command_generator.contact_command
    time_left = self.command_generator.time_left
    ee_error = data.site_xpos[self._feet_side_id] - self.command_generator.desired_ee_positions_w
    norm_error = np.linalg.norm(ee_error, axis=-1)
    obs = np.hstack([
        linvel,
        gyro,
        gravity,
        np.ravel(contact_command),
        np.ravel(position_command),
        joint_angles,
        joint_velocities,
        self._last_action,
        np.ravel(time_left),
        np.ravel(norm_error)
    ])
    return obs.astype(np.float32)

  def get_control(self, model: mujoco.MjModel, data: mujoco.MjData) -> None:
    self._counter += 1
    if self._counter % self._n_substeps == 0:
      obs = self.get_obs(model, data)
      onnx_input = {"obs": obs.reshape(1, -1)}
      onnx_pred = self._policy.run(self._output_names, onnx_input)[0][0]
      self._last_action = onnx_pred.copy()
      data.ctrl[:] = onnx_pred * self._action_scale + self._default_angles
      phase_tp1 = self._phase + self._phase_dt
      self._phase = np.fmod(phase_tp1 + np.pi, 2 * np.pi) - np.pi


def load_callback(model=None, data=None):
  global onnx_path
  mujoco.set_mjcb_control(None)

  model = mujoco.MjModel.from_xml_path(
      g1_constants.FEET_ONLY_FLAT_TERRAIN_XML.as_posix(),
      assets=get_assets(),
  )
  data = mujoco.MjData(model)

  mujoco.mj_resetDataKeyframe(model, data, 1)

  ctrl_dt = 0.02
  sim_dt = 0.002
  n_substeps = int(round(ctrl_dt / sim_dt))
  model.opt.timestep = sim_dt

  policy = OnnxController(
      policy_path=(_ONNX_DIR / onnx_path).as_posix(),
      default_angles=np.array(model.keyframe("knees_bent").qpos[7:]),
      ctrl_dt=ctrl_dt,
      n_substeps=n_substeps,
      action_scale=0.5,
      torso_body_id=model.body(g1_constants.ROOT_BODY).id,
      feet_site_id=np.array(
        [model.site(name).id for name in g1_constants.FEET_SITES]
    )
  )

  mujoco.set_mjcb_control(policy.get_control)

  return model, data

def convert_brax_to_onnx(ckpt_path, env_name):
    global onnx_path
    ppo_params = locomotion_params.brax_ppo_config(env_name)

    def identity_observation_preprocessor(observation, preprocessor_params):
        del preprocessor_params
        return observation

    network_factory=functools.partial(
        ppo_networks.make_ppo_networks,
        **ppo_params.network_factory,
        # We need to explicitly call the normalization function here since only the brax
        # PPO train.py script creates it if normalize_observations is True.
        preprocess_observations_fn=running_statistics.normalize,
    )

    env_cfg = locomotion.get_default_config(env_name)
    env = locomotion.load(env_name, config=env_cfg)

    obs_size = env.observation_size
    act_size = env.action_size
    print(obs_size, act_size)

    ppo_network = network_factory(obs_size, act_size)

    params = checkpoint.load(ckpt_path)
    params = (params[0], params[1])

    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_inference_fn(params, deterministic=True)

    class MLP(tf.keras.Model):
        def __init__(
            self,
            layer_sizes,
            activation=tf.nn.relu,
            kernel_init="lecun_uniform",
            activate_final=False,
            bias=True,
            layer_norm=False,
            mean_std=None,
        ):
            super().__init__()

            self.layer_sizes = layer_sizes
            self.activation = activation
            self.kernel_init = kernel_init
            self.activate_final = activate_final
            self.bias = bias
            self.layer_norm = layer_norm

            if mean_std is not None:
                self.mean = tf.Variable(mean_std[0], trainable=False, dtype=tf.float32)
                self.std = tf.Variable(mean_std[1], trainable=False, dtype=tf.float32)
            else:
                self.mean = None
                self.std = None

            self.mlp_block = tf.keras.Sequential(name="MLP_0")
            for i, size in enumerate(self.layer_sizes):
                dense_layer = layers.Dense(
                    size,
                    activation=self.activation,
                    kernel_initializer=self.kernel_init,
                    name=f"hidden_{i}",
                    use_bias=self.bias,
                )
                self.mlp_block.add(dense_layer)
                if self.layer_norm:
                    self.mlp_block.add(layers.LayerNormalization(name=f"layer_norm_{i}"))
            if not self.activate_final and self.mlp_block.layers:
                if hasattr(self.mlp_block.layers[-1], 'activation') and self.mlp_block.layers[-1].activation is not None:
                    self.mlp_block.layers[-1].activation = None

            self.submodules = [self.mlp_block]

        def call(self, inputs):
            if isinstance(inputs, list):
                inputs = inputs[0]
            if self.mean is not None and self.std is not None:
                print(self.mean.shape, self.std.shape)
                inputs = (inputs - self.mean) / self.std
            logits = self.mlp_block(inputs)
            loc, _ = tf.split(logits, 2, axis=-1)
            return tf.tanh(loc)

    def make_policy_network(
    param_size,
    mean_std,
    hidden_layer_sizes=[256, 256],
    activation=tf.nn.relu,
    kernel_init="lecun_uniform",
    layer_norm=False,
    ):
        policy_network = MLP(
            layer_sizes=list(hidden_layer_sizes) + [param_size],
            activation=activation,
            kernel_init=kernel_init,
            layer_norm=layer_norm,
            mean_std=mean_std,
        )
        return policy_network
    
    mean = params[0].mean["state"]
    std = params[0].std["state"]

    # Convert mean/std jax arrays to tf tensors.
    mean_std = (tf.convert_to_tensor(mean), tf.convert_to_tensor(std))

    tf_policy_network = make_policy_network(
        param_size=act_size * 2,
        mean_std=mean_std,
        hidden_layer_sizes=ppo_params.network_factory.policy_hidden_layer_sizes,
        activation=tf.nn.swish,
    )

    example_input = tf.zeros((1, obs_size["state"][0]))
    example_output = tf_policy_network(example_input)
    print(example_output.shape)

    def transfer_weights(jax_params, tf_model):
        """
        Transfer weights from a JAX parameter dictionary to the TensorFlow model.

        Parameters:
        - jax_params: dict
        Nested dictionary with structure {block_name: {layer_name: {params}}}.
        For example:
        {
            'CNN_0': {
            'Conv_0': {'kernel': np.ndarray},
            'Conv_1': {'kernel': np.ndarray},
            'Conv_2': {'kernel': np.ndarray},
            },
            'MLP_0': {
            'hidden_0': {'kernel': np.ndarray, 'bias': np.ndarray},
            'hidden_1': {'kernel': np.ndarray, 'bias': np.ndarray},
            'hidden_2': {'kernel': np.ndarray, 'bias': np.ndarray},
            }
        }

        - tf_model: tf.keras.Model
        An instance of the adapted VisionMLP model containing named submodules and layers.
        """
        for layer_name, layer_params in jax_params.items():
            try:
                tf_layer = tf_model.get_layer("MLP_0").get_layer(name=layer_name)
            except ValueError:
                print(f"Layer {layer_name} not found in TensorFlow model.")
                continue
            if isinstance(tf_layer, tf.keras.layers.Dense):
                kernel = np.array(layer_params['kernel'])
                bias = np.array(layer_params['bias'])
                print(f"Transferring Dense layer {layer_name}, kernel shape {kernel.shape}, bias shape {bias.shape}")
                tf_layer.set_weights([kernel, bias])
            else:
                print(f"Unhandled layer type in {layer_name}: {type(tf_layer)}")

        print("Weights transferred successfully.")

    transfer_weights(params[1]['params'], tf_policy_network)

    # Example inputs for the model
    test_input = [np.ones((1, obs_size["state"][0]), dtype=np.float32)]

    spec = [tf.TensorSpec(shape=(1, obs_size["state"][0]), dtype=tf.float32, name="obs")]

    tensorflow_pred = tf_policy_network(test_input)[0]
    # Build the model by calling it with example data
    print(f"Tensorflow prediction: {tensorflow_pred}")

    tf_policy_network.output_names = ['continuous_actions']

    # opset 11 matches isaac lab.
    model_proto, _ = tf2onnx.convert.from_keras(tf_policy_network, input_signature=spec, opset=11, output_path=onnx_path)

onnx_path = None

if __name__ == "__main__":
  parser = argparse.ArgumentParser(
      description="Deploy an ONNX policy for the G1 robot with contact goals."
  )

  parser.add_argument(
      "--checkpoint_path",
      type=str,
      required=True,
      help="Path to the brax checkpoint to use in the play script.",
  )

  parser.add_argument(
      "--onnx_path",
      type=str,
      required=True,
      help="Path to store the resulting onnx policy.",
  )

  parser.add_argument(
      "--env_name",
      type=str,
      required=True,
      help="Name of the environment to use.",
  )

  parsed_args = parser.parse_args()

  onnx_path = parsed_args.onnx_path

  convert_brax_to_onnx(parsed_args.checkpoint_path, parsed_args.env_name)




  viewer.launch(loader=load_callback)
