from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import argparse
import tensorflow as tf
import trainer.engines as engines

import trainer.kernel_density as kernel_density
import trainer.kde_normal as kde_normal
import numpy as np


def model_fn(features, labels, mode, params, config):
    features_float = tf.constant([1], dtype=tf.float64)

    try:
        # features_float = tf.to_double(features)
        features_float = tf.string_to_number(features, out_type=tf.float64)
    except TypeError:
        print("## to_double error")
        with tf.Session() as sess:
            print(sess.run(features))




    kde_model = kernel_density.KernelDensity()

    if mode == tf.estimator.ModeKeys.PREDICT:
        # Define the prediction.
        prediction_value = tf.reshape(kde_model.log_sum_pdf(features_float), [-1])
        prediction = {
            '_0': features,
            '_1': prediction_value
        }

        export_outputs = {
            'prediction': tf.estimator.export.PredictOutput(prediction)
        }

        loss = None
        train_op = None
        eval_metric_ops = None

    else:
        # tf.python.training.training_util._increment_global_step(params.max_steps)
        global_step = tf.train.get_or_create_global_step()

        kde_model.set_param(loc=features_float, scale=np.array([3.]), weight=None, kernel_dist=kde_normal.KdeNormalWithSoftPlusScale)

        # Define the loss.
        loss = tf.Variable(1, dtype=tf.float64)

        train_op = tf.assign(global_step, params.max_steps)

        eval_metric_ops = None

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
    parser.add_argument("--max-steps", help="maximum number of steps", type=int, default=1)
    parser.add_argument("--viz-steps", type=int, default=100)
    parser.add_argument("--keyfile", type=str, default=keyfile)
    parser.add_argument("--engine", type=str, default="spark")

    # engine arguments
    parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int)
    parser.add_argument("--num-ps", help="number of PS nodes in cluster", type=int, default=1)
    parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
    parser.add_argument("--app-name", help="name of spark application", type=str, default="lqad_spark_application")

    # model arguments
    parser.add_argument("--learning-rate", type=float, default=0.00005)
    parser.add_argument("--encoder-id", type=str, default="lqad_encoder")
    parser.add_argument("--decoder-id", type=str, default="lqad_decoder")

    args = parser.parse_args()
    engine = args.engine

    # set engine and run
    if engine == "spark":
        spark = engines.SparkCluster()
        spark.run(model_fn, args)

    elif engine == "mle":
        mle = engines.MLEngine()
        mle.run(model_fn, args)
