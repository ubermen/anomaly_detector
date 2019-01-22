from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
from absl import flags
import argparse
import tensorflow as tf
import tensorflow_probability as tfp
import trainer.utils as utils
import trainer.kde as kde

tfd = tfp.distributions

flags.DEFINE_float("learning_rate", default=0.0001, help="Initial learning rate.")
flags.DEFINE_string("data_dir", default="", help="Directory where data is stored (if using real data).")
flags.DEFINE_string("model_dir", default="", help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps", default=100, help="Frequency at which to save visualizations.")
flags.DEFINE_integer("batch_size", default=32, help="Batch size.")
flags.DEFINE_integer("max_steps", default=1000, help="Max steps")
flags.DEFINE_integer("epoch", default=1, help="Epoch count.")

FLAGS = flags.FLAGS

def model_fn(features, labels, mode, params, config):

  kdeModel = kde.KDE_Model(2.5)
  kdeModel.train(features)
  if mode == tf.estimator.ModeKeys.PREDICT :

    # Define the prediction.
    prediction_value = kdeModel.test(features)
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

  run_config = tf.estimator.RunConfig(session_config=utils.get_session_config_from_env_var())
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
