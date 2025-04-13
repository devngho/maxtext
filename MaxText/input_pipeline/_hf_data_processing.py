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
import math
import os
from typing import List, Union

"""Input pipeline using Huggingface datasets."""

import ml_collections
import jax
import datasets
import transformers
import grain.python as grain
import numpy as np

from MaxText.input_pipeline import _input_pipeline_utils
from MaxText import multihost_dataloading
from MaxText import max_logging


def preprocessing_pipeline(
    dataloading_host_index,
    dataloading_host_count,
    global_mesh,
    dataset,
    data_column_names,
    tokenize,
    tokenizer_path,
    hf_access_token,
    global_batch_size,
    max_target_length,
    shuffle,
    data_shuffle_seed,
    add_bos=True,
    add_eos=True,
    packing=True,
    shift=True,
    num_threads=1,
    drop_remainder=False,
    generate_padding_example=False,
    random_access=True,
    epoch=1,
    use_dpo=None,
    use_sft=None,
    sft_train_on_completion_only=True,
    grain_worker_count=1,  # only support 0 or 1
):
  """pipeline for preprocessing HF dataset"""

  assert global_batch_size % global_mesh.size == 0, "Batch size should be divisible number of global devices."

  data_column_names_ = tuple(data_column_names)

  if shuffle:
    dataset = dataset.shuffle(seed=data_shuffle_seed)

  if use_sft:
    dataset = dataset.select_columns(data_column_names)

    supported_columns = [["prompt", "completion"], ["messages"]]
    assert any(
        set(data_column_names) == set(supported) for supported in supported_columns
    ), f"Dataset column names mismatch. Expected columns to match one of {supported_columns}, but got {data_column_names}"
    assert _input_pipeline_utils.is_conversational(
        dataset.features, data_column_names
    ), "Dataset is not in conversational format."

    if len(data_column_names) > 1:
      combined_column_name = "messages"
      dataset_features = datasets.Features(
          {combined_column_name: [{"content": datasets.Value(dtype="string"), "role": datasets.Value(dtype="string")}]}
      )
      dataset = dataset.map(
          _input_pipeline_utils.combine_columns,
          fn_kwargs={"columns": data_column_names, "data_column": combined_column_name},
          remove_columns=data_column_names,
          features=dataset_features,
      )

    data_column_names = list(dataset.features.keys())
    dataset = dataset.map(
        _input_pipeline_utils.extract_messages_and_mask, fn_kwargs={"data_column_name": data_column_names[0]}
    )
  elif tokenize:
    dataset = dataset.select_columns(data_column_names)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_bos_token=add_bos if not use_sft else False,
        add_eos_token=add_eos if not use_sft else False,
        legacy=False,
        token=hf_access_token,
    )
    if tokenizer.pad_token_id is not None:
      pad_id = tokenizer.pad_token_id
    elif tokenizer.unk_token_id is not None:
      pad_id = tokenizer.unk_token_id
    else:
      pad_id = -1

    if not random_access:
      dataset = dataset.map(
          _input_pipeline_utils.tokenization,
          batched=True,
          fn_kwargs={"hf_tokenizer": tokenizer, "truncation": True if not use_sft else False, "max_length": max_target_length - 1, "column_names": data_column_names},
      )
      dataset = dataset.select_columns(data_column_names + ["s_token_count", "s_rows_count"])
    else:
      def transform(x):
          ids = _input_pipeline_utils.tokenization(x, hf_tokenizer=tokenizer, max_length=max_target_length - 1, column_names=data_column_names_, truncation=True if not use_sft else False)

          token_counts = {'s_token_count_' + column_name:  [[len(ids[column_name][i])] for i in range(len(ids[column_name]))] for column_name in data_column_names_}
          rows_counts = {'s_rows_count_' + column_name: [[1] for _ in range(len(ids[column_name]))] for column_name in data_column_names_}

          ids.update(token_counts)
          ids.update(rows_counts)

          return ids

      dataset = dataset.with_transform(transform)
  else:
    dataset = dataset.select_columns(data_column_names + ["s_token_count_"+data_column_names[0], "s_rows_count_"+data_column_names[0]])

  if not random_access:
    dataset = _input_pipeline_utils.HFDataSource(
        dataset,
        dataloading_host_index,
        dataloading_host_count,
        num_threads,
        generate_padding_example,
        max_target_length,
        data_column_names,
    )
  else:
    dataset = _input_pipeline_utils.HFRandomAccessDataSource(
        dataset
    )

  operations = []
  if use_sft:
    operations.append(
        _input_pipeline_utils.SFTPromptMasking(
            text_column_name=data_column_names[0],
            completion_only=sft_train_on_completion_only,
            max_target_length=max_target_length,
            add_bos=add_bos,
            add_eos=add_eos,
            bos_id=tokenizer.bos_token_id,
            eos_id=tokenizer.eos_token_id,
            unk_id=pad_id,
        )
    )
    data_column_names = ("inputs", "targets")
  elif use_dpo:
    lists2array = lambda x: jax.tree.map(np.asarray, x, is_leaf=lambda x: isinstance(x, (list, tuple)))
    operations.append(grain.MapOperation(lists2array))
  else:
    assert len(data_column_names) == 1
    operations.append(_input_pipeline_utils.HFNormalizeFeatures(data_column_names[0]))
    data_column_names = ("inputs", "targets")

  if packing and not use_dpo:
    length_struct = {col: max_target_length for col in (list(data_column_names) + ["s_token_count_"+data_column_names_[0], "s_rows_count_"+data_column_names_[0]])}
    operations.append(
        grain.experimental.PackAndBatchOperation(
            batch_size=global_batch_size // jax.process_count(),
            length_struct=length_struct,
        )
    )
    operations.append(_input_pipeline_utils.ReformatPacking(list(data_column_names) + ["s_token_count_"+column_names for column_names in data_column_names_] + ["s_rows_count_"+column_names for column_names in data_column_names_]))
  else:
    operations.append(_input_pipeline_utils.PadToMaxLength(max_target_length, pad_id))
    operations.append(grain.Batch(batch_size=global_batch_size // jax.process_count(), drop_remainder=drop_remainder))

  if shift and not use_dpo:
    operations.append(_input_pipeline_utils.ShiftData(ignored_ids=[pad_id, tokenizer.bos_token_id], axis=1))

  # Since HuggingFace IterableDataset does not support access through index
  # Indexes generated by dummy_index_sampler is not used.
  # dummy_index_sampler is used as an input place holder for grain.Dataloader
  index_sampler = grain.IndexSampler(
      num_records=len(dataset),
      num_epochs=epoch,
      shard_options=grain.ShardOptions(
          shard_index=dataloading_host_index, shard_count=dataloading_host_count, drop_remainder=False
      ),
      shuffle=shuffle,
      seed=data_shuffle_seed if random_access else None,
  )

  dataloader = grain.DataLoader(
      data_source=dataset,
      operations=operations,
      sampler=index_sampler,
      worker_count=num_threads if random_access else 1,  # only supports one worker for now, more workers results in duplicated data
      worker_buffer_size=1,
      read_options=grain.ReadOptions(num_threads=1 if random_access else num_threads, prefetch_buffer_size=128),
  )

  multihost_gen = multihost_dataloading.MultiHostDataLoadIterator(dataloader, global_mesh)

  # Return multi-host jax.Array prep iterator
  return multihost_gen


def make_hf_train_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices_train,
):
  """Load, preprocess dataset and return iterators"""
  train_ds = datasets.load_from_disk(
      config.hf_path,
      # data_dir=config.hf_data_dir,
      # data_files=config.hf_train_files,
      # split="train",
      # token=config.hf_access_token,
  )

  max_logging.log(f'HF train rows: {len(train_ds)}')

  train_iter = preprocessing_pipeline(
      dataloading_host_index=process_indices_train.index(jax.process_index()),
      dataloading_host_count=len(process_indices_train),
      global_mesh=global_mesh,
      dataset=train_ds,
      data_column_names=config.train_data_columns,
      tokenize=config.tokenize_train_data,
      tokenizer_path=config.tokenizer_path,
      hf_access_token=config.hf_access_token,
      global_batch_size=config.global_batch_size_to_load,
      max_target_length=config.max_target_length,
      shuffle=config.enable_data_shuffling,
      data_shuffle_seed=config.data_shuffle_seed,
      add_bos=config.add_bos,
      add_eos=config.add_eos,
      generate_padding_example=True,
      use_dpo=config.use_dpo,
      random_access=config.hf_random_access,
      num_threads=config.hf_worker_count,
      packing=config.hf_packing,
      drop_remainder=config.drop_last_batch,
      epoch=config.epoch,
      use_sft=config.use_sft,
      sft_train_on_completion_only=config.sft_train_on_completion_only,
  )
  return train_iter


def make_hf_eval_iterator(
    config: ml_collections.ConfigDict,
    global_mesh,
    process_indices_eval,
):
  eval_ds = datasets.load_from_disk(
      config.hf_eval_path,
      # data_dir=config.hf_data_dir,
      # data_files=config.hf_eval_files,
      # split=config.hf_eval_split,
      # streaming=True,
      # token=config.hf_access_token,
  )

  max_logging.log(f'HF eval rows: {len(eval_ds)}')

  if config.eval_steps > 0:
    eval_generate_padding_example = True
  else:
    eval_generate_padding_example = False
  eval_iter = preprocessing_pipeline(
      dataloading_host_index=process_indices_eval.index(jax.process_index()),
      dataloading_host_count=len(process_indices_eval),
      global_mesh=global_mesh,
      dataset=eval_ds,
      data_column_names=config.eval_data_columns,
      tokenize=config.tokenize_eval_data,
      tokenizer_path=config.tokenizer_path,
      hf_access_token=config.hf_access_token,
      global_batch_size=config.global_batch_size_to_load_eval,
      max_target_length=config.max_target_length,
      shuffle=False,
      data_shuffle_seed=config.data_shuffle_seed,
      add_bos=config.add_bos,
      add_eos=config.add_eos,
      generate_padding_example=eval_generate_padding_example,
      use_dpo=config.use_dpo,
      random_access=config.hf_random_access,
      num_threads=config.hf_eval_worker_count,
      packing=config.hf_packing,
      drop_remainder=config.drop_last_batch,
      use_sft=config.use_sft,
      sft_train_on_completion_only=config.sft_train_on_completion_only,
  )
  return eval_iter
