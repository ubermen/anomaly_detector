from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import argparse
import tensorflow as tf
import tensorflow_probability as tfp
import trainer.engines as engines

tfd = tfp.distributions

def model_fn(features, labels, mode, params, config):
  args = params

  features_float = tf.string_to_number(features, out_type=tf.float32)

  normal_data = tf.Variable([0.], dtype=tf.float32, validate_shape=False)

  if mode == tf.estimator.ModeKeys.PREDICT :

    f = lambda x: tfd.Independent(tfd.Normal(loc=x, scale=1.))
    kde = tfd.MixtureSameFamily(
        mixture_distribution=tfd.Categorical(probs=[1]),
        components_distribution=f(normal_data))

    # Define the prediction.
    prediction = {
      '_0' : features,
      '_1' : kde.log_prob(features_float)
    }

    export_outputs = {
      'lqad_prediction': tf.estimator.export.PredictOutput(prediction)
    }

    loss = None
    train_op = None
    eval_metric_ops = None

  else :

    global_step = tf.train.get_or_create_global_step()

    set_normal_data = tf.assign(normal_data, features_float, validate_shape=False)
    set_step = tf.assign(global_step, params.max_steps)

    # Define the loss.
    loss = tf.Variable(1, dtype=tf.float32)
    train_op = tf.group([set_normal_data, set_step])

    eval_metric_ops=None

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

if __name__ == "__main__":

  keyfile = "/etl/credentials/bi-service-155107.json"

  parser = argparse.ArgumentParser(description='Process data and model path info.')

  # common arguments
  parser.add_argument('--job-dir', help='GCS location to write checkpoints and export models')
  parser.add_argument('--data-dir', help='GCS location from which load data')
  parser.add_argument("--batch-size", help="number of records per batch", type=int, default=100)
  parser.add_argument("--max-steps", help="maximum number of steps", type=int, default=1000)
  parser.add_argument("--viz-steps", type=int, default = 100)
  parser.add_argument("--keyfile", type=str, default=keyfile)
  parser.add_argument("--engine", type=str, default="spark")

  # engine arguments
  parser.add_argument("--cluster-size", help="number of nodes in the cluster", type=int)
  parser.add_argument("--num-ps", help="number of PS nodes in cluster", type=int, default=1)
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
  parser.add_argument("--app-name", help="name of spark application", type=str, default="lqad_spark_application")

  # model arguments
  parser.add_argument("--learning-rate", type=float, default = 0.00005)

  args = parser.parse_args()
  engine = args.engine

  # set engine and run
  if engine == "spark" :
    spark = engines.SparkCluster()
    spark.run(model_fn, args)

  elif engine == "mle" :
    mle = engines.MLEngine()
    mle.run(model_fn, args)