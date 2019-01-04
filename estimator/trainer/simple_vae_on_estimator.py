# this code is modified version of tensorflow probability example
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os
import numpy as np
import uuid

# Dependency imports
from absl import flags
import argparse
import tensorflow as tf

tfd = tf.contrib.distributions

seq_len = 16
enc_size = 128
IMAGE_SHAPE = [seq_len, enc_size, 1]

kernel_height = max(2, int(seq_len/2))
kernel_width = max(2, int(enc_size/2))
kernel = (kernel_height, kernel_width)

stride_vertical = 1
stride_horizontal = 2
stride = (stride_vertical, stride_horizontal)

flags.DEFINE_float("learning_rate", default=0.0001, help="Initial learning rate.")
flags.DEFINE_integer("max_steps", default=3001, help="Number of training steps to run.")
flags.DEFINE_string("data_dir", default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/data"), help="Directory where data is stored (if using real data).")
flags.DEFINE_string("model_dir", default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"), help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps", default=100, help="Frequency at which to save visualizations.")
flags.DEFINE_integer("batch_size", default=100, help="Batch size.")
flags.DEFINE_string("activation", default="leaky_relu", help="Activation function for all hidden layers.")
flags.DEFINE_string("encoder_id", default="lqad_encoder", help="")
flags.DEFINE_string("decoder_id", default="lqad_decoder", help="")

FLAGS = flags.FLAGS

class VariationalAutoencoder(object) :
  def __init__(self, sequence_length, encoding_size, code_size=2, kernel=None, stride=None, conv1_filter=16, conv2_filter=32) :
    self.code_size = code_size
    self.sequence_length = sequence_length
    self.encoding_size = encoding_size
    self.data_shape = [sequence_length, encoding_size]
    self.is_training = True

    if kernel is not None :
      self.kernel = kernel
    else :
      kernel_height = max(2, int(self.sequence_length/2))
      kernel_width = max(2, int(self.encoding_size/2))
      self.kernel = (kernel_height, kernel_width)

    if stride is not None :
      self.stride = stride
    else :
      stride_vertical = 1
      stride_horizontal = 2
      self.stride = (stride_vertical, stride_horizontal)

    self.conv1_filter = conv1_filter
    self.conv2_filter = conv2_filter
    self.conv_result_height = int(sequence_length / stride_vertical / stride_vertical)
    self.conv_result_width = int(encoding_size / stride_horizontal / stride_horizontal)
    self.final_conv_shape = [self.conv_result_height, self.conv_result_width, self.conv2_filter]

    self.make_encoder = tf.make_template(FLAGS.encoder_id, self.make_encoder)
    self.make_decoder = tf.make_template(FLAGS.decoder_id, self.make_decoder)

  def make_prior(self):
    code_size = self.code_size
    loc = tf.zeros(code_size)
    scale = tf.ones(code_size)
    return tfd.MultivariateNormalDiag(loc, scale)

  def make_encoder(self, data):
    code_size = self.code_size
    sequence_length = self.sequence_length
    encoding_size = self.encoding_size
    conv1_filter = self.conv1_filter
    conv2_filter = self.conv2_filter
    kernel = self.kernel
    stride = self.stride

    # conv
    x = tf.reshape(data, shape=[-1, sequence_length, encoding_size, 1])

    conv1 = tf.layers.conv2d(x, conv1_filter, kernel, stride, activation=tf.nn.relu, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),bias_initializer=tf.contrib.layers.xavier_initializer_conv2d())
    conv2 = tf.layers.conv2d(conv1, conv2_filter, kernel, stride, activation=tf.nn.relu, padding='same', kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),bias_initializer=tf.contrib.layers.xavier_initializer_conv2d())

    # Flatten the data to a 1-D vector for the fully connected layer
    x = tf.contrib.layers.flatten(conv2)
    x = tf.layers.dense(x, encoding_size, tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.contrib.layers.xavier_initializer())
    out = tf.layers.dense(x, code_size)
    loc = tf.layers.dense(x, code_size)
    scale = tf.layers.dense(out, code_size, tf.nn.softplus)
    return tfd.MultivariateNormalDiag(loc, scale)

  def make_decoder(self, code):
    data_shape = self.data_shape
    conv1_filter = self.conv1_filter
    conv2_filter = self.conv2_filter
    encoding_size = self.encoding_size
    final_conv_shape = self.final_conv_shape
    kernel = self.kernel
    stride = self.stride

    # deconv
    x = code
    x = tf.layers.dense(x, encoding_size, tf.nn.relu,kernel_initializer=tf.contrib.layers.xavier_initializer(),bias_initializer=tf.contrib.layers.xavier_initializer())
    x = tf.layers.dense(x, np.prod(final_conv_shape))
    x = tf.reshape(x, shape=[-1] + final_conv_shape)

    conv2 = tf.layers.conv2d_transpose(x, conv1_filter, kernel, stride, padding='same')
    conv1 = tf.layers.conv2d_transpose(conv2, 1, kernel, stride, padding='same')

    logit = tf.reshape(conv1, [-1] + data_shape)
    return tfd.Independent(tfd.Bernoulli(logit), 2)

def model_fn(features, labels, mode, params, config):

  # Define the model
  vae = VariationalAutoencoder(seq_len, enc_size)

  # preprocess data
  data = preprocess(features)

  # make prior
  prior = vae.make_prior()

  # make posterior and sample from data
  posterior = vae.make_encoder(data)
  code = posterior.sample()

  # make decoder and likelihood from data
  decoder = vae.make_decoder(code)
  likelihood = decoder.log_prob(data)
  distortion = -likelihood

  if mode == tf.estimator.ModeKeys.PREDICT :

    # define prediction
    prediction = {
      '_0' : features,
      '_1' : distortion
    }

    export_outputs = {
      'prediction': tf.estimator.export.PredictOutput(prediction)
    }

    loss = None
    train_op = None
    eval_metric_ops = None

  else :

    global_step = tf.train.get_or_create_global_step()

    # Define the loss.
    divergence = tfd.kl_divergence(posterior, prior)
    elbo = tf.reduce_mean(likelihood - divergence)
    learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step, params["max_steps"])
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    loss = -elbo
    train_op = optimizer.minimize(loss, global_step=global_step)

    eval_metric_ops={
      "elbo": tf.metrics.mean(elbo),
      "divergence": tf.metrics.mean(divergence),
      "distortion": tf.metrics.mean(distortion),
    }

    prediction = None
    export_outputs = None

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=eval_metric_ops,
    predictions=prediction,
    export_outputs=export_outputs,
  )

def static_nlog_dataset(data_dir, file_name):
  dataset = tf.data.TextLineDataset(data_dir + '/' + file_name).skip(1)
  return dataset

def build_input_fns(data_dir, batch_size):
  """Builds an Iterator switching between train and heldout data."""

  # Build an iterator over training batches.
  training_dataset = static_nlog_dataset(data_dir, 'globalsignin_devicemodel_train')
  training_dataset = training_dataset.shuffle(10000).repeat().batch(batch_size)
  train_input_fn = lambda: training_dataset.make_one_shot_iterator().get_next()

  # Build an iterator over the heldout set.
  eval_dataset = static_nlog_dataset(data_dir, 'globalsignin_devicemodel_eval')
  eval_dataset = eval_dataset.batch(batch_size)
  eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()

  return train_input_fn, eval_input_fn

def _get_session_config_from_env_var():
  """Returns a tf.ConfigProto instance that has appropriate device_filters set.

  """

  tf_config = json.loads(os.environ.get('TF_CONFIG', '{}'))

  if (tf_config and 'task' in tf_config and 'type' in tf_config['task'] and
          'index' in tf_config['task']):
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

def serving_input_fn():
  string_array = tf.placeholder(tf.string, [None])
  return tf.estimator.export.TensorServingInputReceiver(string_array, string_array)

def preprocess(string_array):
  string_array = tf.strings.substr(string_array,0,seq_len)
  split_stensor = tf.string_split(string_array, delimiter="")
  split_values = split_stensor.values
  unicode_values = tf.map_fn(lambda x: tf.io.decode_raw(x, tf.uint8)[0], split_values, dtype=tf.uint8)
  unicode_tensor = tf.sparse_to_dense(split_stensor.indices, [tf.shape(string_array)[0], seq_len], unicode_values, default_value=-1)
  encoded_tensor = tf.map_fn(lambda x: tf.one_hot(x, enc_size), unicode_tensor, dtype=tf.float32)
  return encoded_tensor

def main(argv):
  del argv  # unused

  params = FLAGS.flag_values_dict()
  params["activation"] = getattr(tf.nn, params["activation"])
  tf.gfile.MakeDirs(FLAGS.model_dir)

  train_input_fn, eval_input_fn = build_input_fns(FLAGS.data_dir, FLAGS.batch_size)

  train_spec = tf.estimator.TrainSpec(
    train_input_fn, max_steps=FLAGS.max_steps)

  exporter = tf.estimator.FinalExporter('exporter', serving_input_fn)

  eval_spec = tf.estimator.EvalSpec(
    eval_input_fn,
    steps=FLAGS.viz_steps,
    exporters=[exporter],
    name='lqad-eval')

  run_config = tf.estimator.RunConfig(session_config=_get_session_config_from_env_var())
  run_config = run_config.replace(model_dir=FLAGS.model_dir)

  estimator = tf.estimator.Estimator(
    model_fn,
    params=params,
    config=run_config
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process data and model path info.')
  parser.add_argument(
    '--job-dir',
    help='GCS location to write checkpoints and export models')
  args = parser.parse_args()
  FLAGS.data_dir = "gs://bigus/data"
  FLAGS.model_dir = args.job_dir
  tf.app.run()