"""
Copyright 2023 Google LLC

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

     https://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

# pylint: disable=g-bad-todo, abstract-method, consider-using-with, ungrouped-imports
"""Training loop and Decoding of the model."""

# Calling jax.device_count here prevents a "TPU platform already registered" error.
# See github.com/google/maxtext/issues/20 for more

import os
from typing import Sequence

import jax
import tensorflow as tf
from absl import app
from jax import random
from jax.sharding import Mesh

import checkpointing
import max_utils
import optimizers
import pyconfig
from input_pipeline.input_pipeline_interface import create_data_iterator
from layers import models
from layers import quantizations

# Placeholder: internal

# pylint: disable=too-many-positional-arguments

Transformer = models.Transformer
EPS = 1e-8
_DEFAULT_OCDBT_TARGET_DATA_FILE_SIZE = 2 * 1024**3

def setup_mesh_and_model(config):
  """Set up the mesh and the model for training

  Args:
    config

  Returns:
    init_rng: RNG key
    writer: Summary writer for tensorboard
    checkpoint_manager: Orbax checkpointer
    state_mesh_annotations: the mesh annotations for the train state
    model:
    mesh:
    learning_rate_schedule:
    tx:
  """

  init_rng = random.PRNGKey(config.init_weights_seed)
  writer = max_utils.initialize_summary_writer(config)

  # Mesh definition
  devices_array = max_utils.create_device_mesh(config)
  mesh = Mesh(devices_array, config.mesh_axes)

  # Model and Optimizer definition
  quant = quantizations.configure_quantization(config)
  model = Transformer(config, mesh, quant=quant)
  learning_rate_schedule = max_utils.create_learning_rate_schedule(config)
  tx = optimizers.get_optimizer(config, learning_rate_schedule)
  logger = checkpointing.setup_checkpoint_logger(config)
  if config.enable_emergency_checkpoint:
    if config.use_replicator_service:
      checkpoint_manager = checkpointing.create_orbax_emergency_replicator_checkpoint_manager(
          config.local_checkpoint_directory,
          config.local_checkpoint_period,
          mesh,
      )
    else:
      abstract_state, _, _ = max_utils.get_abstract_state(model, tx, config, init_rng, mesh, is_training=True)
      checkpoint_manager = checkpointing.create_orbax_emergency_checkpoint_manager(
          config.local_checkpoint_directory,
          config.checkpoint_dir,
          mesh,
          abstract_state,
          config.local_checkpoint_period,
          config.checkpoint_period,
          logger,
      )
  else:
    # TODO(b/368121306): Remove this once zarr3 support is plumbed on the backend
    use_ocdbt = config.checkpoint_storage_use_ocdbt
    use_zarr3 = config.checkpoint_storage_use_zarr3
    if config.enable_single_controller:
      use_ocdbt, use_zarr3 = False, False
    checkpoint_manager = checkpointing.create_orbax_checkpoint_manager(
        config.checkpoint_dir,
        config.enable_checkpointing,
        config.async_checkpointing,
        config.checkpoint_period,
        config.dataset_type,
        logger,
        use_ocdbt,
        use_zarr3,
    )

  return init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx


def setup_train_loop(config):
  """Set up prerequisites for the training loop -
      checkpoint_manager, PRNG keys, Mesh, Model and optimizer.
      Set up data iterator and tokenizer, initialize the model.

  Args:
    config

  Returns:
    init_rng:
    writer: Summary writer for tensorboard
    checkpoint_manager: Orbax checkpointer
    state_mesh_annotations: the mesh annotations for the train state
    model:
    mesh:
    learning_rate_schedule:
    data_iterator:
    state: the initialized train state
  """
  init_rng, writer, checkpoint_manager, mesh, model, learning_rate_schedule, tx = setup_mesh_and_model(config)
  data_iterator, eval_data_iterator = create_data_iterator(config, mesh)

  state, _, state_mesh_shardings, data_iterator = max_utils.setup_training_state(
      model, data_iterator, tx, config, init_rng, mesh, checkpoint_manager
  )

  return (
      init_rng,
      writer,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      state,
  )


def main(argv: Sequence[str]) -> None:
  jax.config.update("jax_default_prng_impl", "unsafe_rbg")
  jax.config.update("jax_enable_x64", True) # for counting tokens well
  # TF allocates extraneous GPU memory when using TFDS data
  # this leads to CUDA OOMs. WAR for now is to hide GPUs from TF
  tf.config.set_visible_devices([], "GPU")
  os.environ["TF_CPP_MIN_LOG_LEVEL"] = "0"
  if "xla_tpu_spmd_rng_bit_generator_unsafe" not in os.environ.get("LIBTPU_INIT_ARGS", ""):
    os.environ["LIBTPU_INIT_ARGS"] = os.environ.get("LIBTPU_INIT_ARGS", "") + " --xla_tpu_spmd_rng_bit_generator_unsafe=true"
  pyconfig.initialize(argv)
  max_utils.print_system_information()
  config = pyconfig.config
  jax.config.update("jax_debug_nans", config.jax_debug_nans)
  os.environ["TFDS_DATA_DIR"] = config.dataset_path

  (
      init_rng,
      writer,
      checkpoint_manager,
      state_mesh_shardings,
      model,
      mesh,
      learning_rate_schedule,
      data_iterator,
      eval_data_iterator,
      state,
  ) = setup_train_loop(config)

  example_batch = next(data_iterator)
  print("Example batch:", example_batch)


if __name__ == "__main__":
  app.run(main)
