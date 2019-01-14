from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

# Dependency imports
from absl import flags
import argparse
import tensorflow as tf
import tensorflow_probability as tfp
import trainer.utils as utils
import trainer.models as models

tfd = tfp.distributions

seq_len = 16
enc_size = 128
IMAGE_SHAPE = [seq_len, enc_size, 1]

kernel_height = max(2, int(seq_len/4))
kernel_width = max(2, int(enc_size/4))
kernel = (kernel_height, kernel_width)

stride_vertical = 1
stride_horizontal = 2
stride = (stride_vertical, stride_horizontal)

flags.DEFINE_float("learning_rate", default=0.0001, help="Initial learning rate.")
flags.DEFINE_string("data_dir", default="", help="Directory where data is stored (if using real data).")
flags.DEFINE_string("model_dir", default="", help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps", default=100, help="Frequency at which to save visualizations.")
flags.DEFINE_integer("batch_size", default=32, help="Batch size.")
flags.DEFINE_integer("max_steps", default=1000, help="Max steps")
flags.DEFINE_string("encoder_id", default="lqad_encoder", help="")
flags.DEFINE_string("decoder_id", default="lqad_decoder", help="")

FLAGS = flags.FLAGS

def model_fn(features, labels, mode, params, config):

  # Define the model
  vae = models.VariationalAutoencoder(seq_len, enc_size, FLAGS.encoder_id, FLAGS.decoder_id)

  # preprocess data
  data = preprocess(features)

  # make prior!
  prior = vae.make_prior()

  # make posterior and sample from data
  posterior = vae.make_encoder(data)
  code = posterior.sample()

  # make decoder and likelihood from data
  decoder = vae.make_decoder(code)
  likelihood = decoder.log_prob(data)
  distortion = -likelihood

  if mode == tf.estimator.ModeKeys.PREDICT :

    # Define the prediction.
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
    loss = -elbo

    learning_rate = params["learning_rate"]
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
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

def preprocess(string_array):
  string_array = tf.strings.substr(string_array,0,seq_len)
  split_stensor = tf.string_split(string_array, delimiter="")
  split_values = split_stensor.values
  unicode_values = tf.map_fn(lambda x: tf.io.decode_raw(x, tf.uint8)[0], split_values, dtype=tf.uint8)
  #unicode_tensor = tf.sparse.to_dense(split_stensor.indices, [tf.shape(string_array)[0], seq_len], unicode_values, default_value=-1)
  unicode_sparse = tf.sparse.SparseTensor(indices=split_stensor.indices, values=unicode_values, dense_shape=[tf.shape(string_array)[0], seq_len])
  unicode_tensor = tf.sparse.to_dense(unicode_sparse, default_value=-1)
  encoded_tensor = tf.map_fn(lambda x: tf.one_hot(x, enc_size), unicode_tensor, dtype=tf.float32)
  return encoded_tensor

def main(argv):
  del argv  # unused

  params = FLAGS.flag_values_dict()
  tf.gfile.MakeDirs(FLAGS.model_dir)

  train_input_fn, eval_input_fn = utils.build_input_fns(FLAGS.data_dir, FLAGS.batch_size)

  train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=FLAGS.max_steps)

  exporter = tf.estimator.FinalExporter('exporter', utils.serving_input_fn)

  eval_spec = tf.estimator.EvalSpec(
    eval_input_fn,
    steps=FLAGS.viz_steps,
    exporters=[exporter],
    name='lqad-eval')

  #distribution = tf.contrib.distribute.MirroredStrategy()
  run_config = tf.estimator.RunConfig(session_config=utils.get_session_config_from_env_var())
  run_config = run_config.replace(model_dir=FLAGS.model_dir)
  #run_config = run_config.replace(train_distribute=distribution)

  estimator = tf.estimator.Estimator(
    model_fn,
    params=params,
    config=run_config
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == "__main__":
  parser = argparse.ArgumentParser(description='Process data and model path info.')
  parser.add_argument('--job-dir', help='GCS location to write checkpoints and export models')
  parser.add_argument('--data-dir', help='GCS location from which load data')
  args = parser.parse_args()
  FLAGS.model_dir = args.job_dir
  FLAGS.data_dir = args.data_dir
  tf.app.run()