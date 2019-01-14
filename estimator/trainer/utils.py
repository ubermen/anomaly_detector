from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

# Dependency imports
import tensorflow as tf

def get_session_config_from_env_var():

  tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

  if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and 'index' in tf_config['task']):
    # Master should only communicate with itself and ps
    if tf_config['task']['type'] == 'master':
      return tf.ConfigProto(device_filters=['/job:ps', '/job:master'])
    # Worker should only communicate with itself and ps
    elif tf_config['task']['type'] == 'worker':
      return tf.ConfigProto(device_filters=[
        '/job:ps',
        '/job:worker/task:%d' % tf_config['task']['index']
      ])
  return None

def build_input_fns(data_dir, batch_size):

  # Build an iterator over training batches.
  training_dataset = get_dataset_from_file(data_dir, 'train')
  training_dataset = training_dataset.repeat().batch(batch_size)
  train_input_fn = lambda: training_dataset.make_one_shot_iterator().get_next()

  # Build an iterator over the evaluation set.
  eval_dataset = get_dataset_from_file(data_dir, 'eval')
  eval_dataset = eval_dataset.batch(batch_size)
  eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()

  return train_input_fn, eval_input_fn

def get_dataset_from_file(data_dir, file_name):
  dataset = tf.data.TextLineDataset(data_dir + '/' + file_name)
  return dataset

def serving_input_fn():
  string_array = tf.placeholder(tf.string, [None])
  return tf.estimator.export.TensorServingInputReceiver(string_array, string_array)