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

tfd = tfp.distributions

flags.DEFINE_float("learning_rate", default=0.0001, help="Initial learning rate.")
flags.DEFINE_string("data_dir", default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/data"), help="Directory where data is stored (if using real data).")
flags.DEFINE_string("model_dir", default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"), help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps", default=100, help="Frequency at which to save visualizations.")
flags.DEFINE_integer("batch_size", default=32, help="Batch size.")
flags.DEFINE_integer("epoch", default=1, help="Epoch count.")
flags.DEFINE_string("activation", default="leaky_relu", help="Activation function for all hidden layers.")

FLAGS = flags.FLAGS

def model_fn(features, labels, mode, params, config):

  if mode == tf.estimator.ModeKeys.PREDICT :

    # Define the prediction.
    prediction_value = None
    prediction = {
      '_0' : features,
      '_1' : prediction_value
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
    loss = None

    learning_rate = params["learning_rate"]
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss, global_step=global_step)

    eval_metric_ops={
      "loss": tf.metrics.mean(loss),
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

  # Build an iterator over training batches.
  training_dataset = static_nlog_dataset(data_dir, 'train')
  training_dataset = training_dataset.repeat(FLAGS.epoch).batch(batch_size)
  train_input_fn = lambda: training_dataset.make_one_shot_iterator().get_next()

  # Build an iterator over the heldout set.
  eval_dataset = static_nlog_dataset(data_dir, 'eval')
  eval_dataset = eval_dataset.batch(batch_size)
  eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()

  return train_input_fn, eval_input_fn

def _get_session_config_from_env_var():

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

def serving_input_fn():
  string_array = tf.placeholder(tf.string, [None])
  return tf.estimator.export.TensorServingInputReceiver(string_array, string_array)

def main(argv):
  del argv  # unused

  params = FLAGS.flag_values_dict()
  params["activation"] = getattr(tf.nn, params["activation"])
  tf.gfile.MakeDirs(FLAGS.model_dir)

  train_input_fn, eval_input_fn = build_input_fns(FLAGS.data_dir, FLAGS.batch_size)

  train_spec = tf.estimator.TrainSpec(train_input_fn)

  exporter = tf.estimator.LatestExporter('exporter', serving_input_fn, exports_to_keep=None)

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
  parser.add_argument('--job-dir', help='GCS location to write checkpoints and export models')
  parser.add_argument('--data-dir', help='GCS location from which load data')
  args = parser.parse_args()
  FLAGS.model_dir = args.job_dir
  FLAGS.data_dir = args.data_dir
  tf.app.run()