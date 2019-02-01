from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import argparse
import tensorflow as tf
import tensorflow_probability as tfp
import trainer.engines as engines
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

def model_fn(features, labels, mode, params, config):
  args = params

  # Define the model
  vae = models.VariationalAutoencoder(seq_len, enc_size, args.encoder_id, args.decoder_id)

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
    raw_value = extract_raw_value(features)
    prediction = {
      '_0' : raw_value,
      '_1' : distortion
    }

    export_outputs = {
      'lqad_prediction': tf.estimator.export.PredictOutput(prediction)
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

    learning_rate = args.learning_rate
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

def extract_raw_value(padded):
  split_stensor = tf.string_split(padded, delimiter="\t")
  split_tensor = tf.sparse.to_dense(split_stensor, default_value="")
  raw_value = split_tensor[:,0]
  return raw_value

def preprocess(string_array):
  string_array = tf.strings.substr(string_array, 0, seq_len)
  split_stensor = tf.string_split(string_array, delimiter="")
  split_values = split_stensor.values
  unicode_values = tf.map_fn(lambda x: tf.io.decode_raw(x, tf.uint8)[0], split_values, dtype=tf.uint8)
  unicode_values = tf.map_fn(lambda x: tf.mod(tf.to_int32(x), tf.constant(enc_size, dtype=tf.int32)), unicode_values, dtype=tf.int32)
  unicode_sparse = tf.sparse.SparseTensor(indices=split_stensor.indices, values=unicode_values, dense_shape=[tf.shape(string_array)[0], seq_len])
  unicode_tensor = tf.sparse.to_dense(unicode_sparse, default_value=-1)
  encoded_tensor = tf.map_fn(lambda x: tf.one_hot(x, enc_size), unicode_tensor, dtype=tf.float32)
  return encoded_tensor

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
  parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int)
  parser.add_argument("--num-ps", help="number of PS nodes in cluster", type=int, default=1)
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")
  parser.add_argument("--app-name", help="name of spark application", type=str, default="lqad_spark_application")

  # model arguments
  parser.add_argument("--learning-rate", type=float, default = 0.00005)
  parser.add_argument("--encoder-id", type=str, default = "lqad_encoder")
  parser.add_argument("--decoder-id", type=str, default = "lqad_decoder")

  args = parser.parse_args()
  engine = args.engine
  storage = args.storage

  # set engine and run
  if engine == "spark" :
    spark = engines.SparkCluster()
    spark.run(model_fn, args)

  elif engine == "mle" :
    mle = engines.MLEngine()
    mle.run(model_fn, args)