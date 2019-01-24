from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Dependency imports
import os
import ConfigParser as cp
import argparse
import tensorflow as tf
import tensorflow_probability as tfp
import trainer.utils as utils
import trainer.models as models

tfd = tfp.distributions

from google.cloud import storage

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

def get_context(app_name, project, keyfile, submit_host, python_lib, python_files):

  # generate environment variables
  full_path_list = ['file:{python_lib}/{file}'.format(python_lib=python_lib, file=file) for file in python_files]
  full_paths = ','.join(full_path_list)
  python_path = ':'.join(python_files)

  # cluster execution
  conf = SparkConf() \
    .setMaster(submit_host) \
    .setAppName(app_name) \
    .set('spark.yarn.dist.files','{full_paths}'.format(full_paths=full_paths)) \
    .setExecutorEnv('PYTHONPATH','{python_path}'.format(python_path=python_path)) \

  context = SparkContext(conf=conf)

  # Setup gcs Hadoop Configurations programmatically
  # Require Google Service account
  context._jsc.hadoopConfiguration().set("fs.gs.impl", "com.google.cloud.hadoop.fs.gcs.GoogleHadoopFileSystem")
  context._jsc.hadoopConfiguration().set("fs.gs.project.id", project)
  context._jsc.hadoopConfiguration().set("google.cloud.auth.service.account.enable", "true")
  context._jsc.hadoopConfiguration().set("google.cloud.auth.service.account.json.keyfile", keyfile)

  return context

def get_list(csv):
  raw_list = csv.split(',')
  stripped_list = [v.strip() for v in raw_list]
  return stripped_list

def main(args, ctx):

  client = storage.Client.from_service_account_json(args.keyfile)
  os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = args.keyfile
  tf.gfile.MakeDirs(args.job_dir)

  train_input_fn, eval_input_fn = utils.build_input_fns(args.data_dir, args.batch_size)

  train_spec = tf.estimator.TrainSpec(train_input_fn, max_steps=args.max_steps)

  exporter = tf.estimator.FinalExporter('exporter', utils.serving_input_fn)

  eval_spec = tf.estimator.EvalSpec(
    eval_input_fn,
    steps=args.viz_steps,
    exporters=[exporter],
    name='lqad-eval')

  run_config = tf.estimator.RunConfig(session_config=utils.get_session_config_from_env_var())
  run_config = run_config.replace(model_dir=args.job_dir)

  estimator = tf.estimator.Estimator(
    model_fn,
    params=args,
    config=run_config
  )

  tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)

if __name__ == "__main__":
  from pyspark.context import SparkContext
  from pyspark.conf import SparkConf
  from tensorflowonspark import TFCluster

  config = cp.ConfigParser()
  config.readfp(open('{PROJECT_ROOT}/defaults.cfg'.format(**os.environ)))

  project = config.get('gcp', 'project')
  keyfile = "/etl/credentials/bi-service-155107.json"
  app_name = 'lqad_train_on_spark_test'

  driver = config.get('mysql', 'driver')
  url = config.get('mysql', 'url')
  user = config.get('mysql', 'user')
  password = config.get('mysql', 'password')

  jar_dir = config.get('environment', 'jar_dir')
  submit_host = config.get('environment', 'submit_host')

  python_lib = config.get('environment', 'python_lib')
  python_files = get_list(config.get('environment', 'python_files'))
  sc = get_context(app_name, project, keyfile, submit_host, python_lib, python_files)

  executors = sc._conf.get("spark.executor.instances")
  num_executors = int(executors) if executors is not None else 1

  parser = argparse.ArgumentParser(description='Process data and model path info.')
  parser.add_argument('--job-dir', help='GCS location to write checkpoints and export models')
  parser.add_argument('--data-dir', help='GCS location from which load data')

  parser.add_argument("--batch_size", help="number of records per batch", type=int, default=100)
  parser.add_argument("--cluster_size", help="number of nodes in the cluster", type=int, default=num_executors)
  parser.add_argument("--num_ps", help="number of PS nodes in cluster", type=int, default=1)
  parser.add_argument("--steps", help="maximum number of steps", type=int, default=1000)
  parser.add_argument("--tensorboard", help="launch tensorboard process", action="store_true")

  parser.add_argument("--learning-rate", type=float, default = 0.0001)
  parser.add_argument("--viz-steps", type=int, default = 100)
  parser.add_argument("--batch-size", type=int, default = 32)
  parser.add_argument("--max-steps", type=int, default = 1000)
  parser.add_argument("--encoder-id", type=str, default = "lqad_encoder")
  parser.add_argument("--decoder-id", type=str, default = "lqad_decoder")

  parser.add_argument("--keyfile", type=str, default=keyfile)

  args = parser.parse_args()

  # tf.app.run()
  cluster = TFCluster.run(
    sc, main, args, args.cluster_size, args.num_ps,
    tensorboard=args.tensorboard,
    input_mode=TFCluster.InputMode.TENSORFLOW,
    log_dir=args.job_dir,
    master_node='master'
  )

  cluster.shutdown()