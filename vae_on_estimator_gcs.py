# this code is modified version of tensorflow probability example
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import json
import os

# Dependency imports
from absl import flags
import argparse
import numpy as np
import tensorflow as tf

tfd = tf.contrib.distributions

seq_len = 28
enc_size = 128
IMAGE_SHAPE = [seq_len, enc_size, 1]

flags.DEFINE_float("learning_rate", default=0.0001, help="Initial learning rate.")
flags.DEFINE_integer("max_steps", default=1001, help="Number of training steps to run.")
flags.DEFINE_integer("latent_size", default=16, help="Number of dimensions in the latent code (z).")
flags.DEFINE_integer("base_depth", default=32, help="Base depth for layers.")
flags.DEFINE_string("activation", default="leaky_relu", help="Activation function for all hidden layers.")
flags.DEFINE_integer("batch_size", default=32, help="Batch size.")
flags.DEFINE_integer("n_samples", default=16, help="Number of samples to use in encoding.")
flags.DEFINE_integer("mixture_components", default=100,
                     help="Number of mixture components to use in the prior. Each component is "
                          "a diagonal normal distribution. The parameters of the components are "
                          "intialized randomly, and then learned along with the rest of the "
                          "parameters. If `analytic_kl` is True, `mixture_components` must be "
                          "set to `1`.")
flags.DEFINE_bool("analytic_kl", default=False,
                  help="Whether or not to use the analytic version of the KL. When set to "
                       "False the E_{Z~q(Z|X)}[log p(Z)p(X|Z) - log q(Z|X)] form of the ELBO "
                       "will be used. Otherwise the -KL(q(Z|X) || p(Z)) + "
                       "E_{Z~q(Z|X)}[log p(X|Z)] form will be used. If analytic_kl is True, "
                       "then you must also specify `mixture_components=1`.")
flags.DEFINE_string("data_dir", default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/data"), help="Directory where data is stored (if using real data).")
flags.DEFINE_string("model_dir", default=os.path.join(os.getenv("TEST_TMPDIR", "/tmp"), "vae/"), help="Directory to put the model's fit.")
flags.DEFINE_integer("viz_steps", default=100, help="Frequency at which to save visualizations.")
flags.DEFINE_bool("fake_data", default=False, help="If true, uses fake data instead of MNIST.")
flags.DEFINE_bool("delete_existing", default=False, help="If true, deletes existing `model_dir` directory.")

FLAGS = flags.FLAGS

def _softplus_inverse(x):
  """Helper which computes the function inverse of `tf.nn.softplus`."""
  return tf.log(tf.math.expm1(x))

def make_encoder(activation, latent_size, base_depth):
  """Creates the encoder function.
  Args:
    activation: Activation function in hidden layers.
    latent_size: The dimensionality of the encoding.
    base_depth: The lowest depth for a layer.
  Returns:
    encoder: A `callable` mapping a `Tensor` of images to a
      `tfd.Distribution` instance over encodings.
  """
  conv = functools.partial(
    tf.keras.layers.Conv2D, padding="SAME", activation=activation)

  encoder_net = tf.keras.Sequential([
    conv(base_depth, 5, 1),
    conv(base_depth, 5, 2),
    conv(2 * base_depth, 5, 1),
    conv(2 * base_depth, 5, 2),
    conv(4 * latent_size, (int(seq_len/4), int(enc_size/4)), padding="VALID"),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(2 * latent_size, activation=None),
  ])

  def encoder(images):
    images = 2 * tf.cast(images, dtype=tf.float32) - 1
    try :
      net = encoder_net(images)
    except IndexError :
      raise Exception("[debug] tensor shape = ", tf.shape(images))

    return tfd.MultivariateNormalDiag(
      loc=net[..., :latent_size],
      scale_diag=tf.nn.softplus(net[..., latent_size:] +
                                _softplus_inverse(1.0)),
      name="code")

  return encoder

def make_decoder(activation, latent_size, output_shape, base_depth):
  """Creates the decoder function.
  Args:
    activation: Activation function in hidden layers.
    latent_size: Dimensionality of the encoding.
    output_shape: The output image shape.
    base_depth: Smallest depth for a layer.
  Returns:
    decoder: A `callable` mapping a `Tensor` of encodings to a
      `tfd.Distribution` instance over images.
  """
  deconv = functools.partial(
    tf.keras.layers.Conv2DTranspose, padding="SAME", activation=activation)
  conv = functools.partial(
    tf.keras.layers.Conv2D, padding="SAME", activation=activation)

  decoder_net = tf.keras.Sequential([
    deconv(2 * base_depth, (int(seq_len/4), int(enc_size/4)), padding="VALID"),
    deconv(2 * base_depth, 5),
    deconv(2 * base_depth, 5, 2),
    deconv(base_depth, 5),
    deconv(base_depth, 5, 2),
    deconv(base_depth, 5),
    conv(output_shape[-1], 5, activation=None),
  ])

  def decoder(codes):
    original_shape = tf.shape(codes)
    # Collapse the sample and batch dimension and convert to rank-4 tensor for
    # use with a convolutional decoder network.
    codes = tf.reshape(codes, (-1, 1, 1, latent_size))
    logits = decoder_net(codes)
    try :
      logits = tf.reshape(
        logits, shape=tf.concat([original_shape[:-1], output_shape], axis=0))
    except ValueError :
      raise Exception('original_shape=' + str(original_shape) +
                      '\noutput_shape=' + str(output_shape)
                      )
    return tfd.Independent(tfd.Bernoulli(logits=logits),
                           reinterpreted_batch_ndims=len(output_shape),
                           name="image")

  return decoder

def make_mixture_prior(latent_size, mixture_components):
  """Creates the mixture of Gaussians prior distribution.
  Args:
    latent_size: The dimensionality of the latent representation.
    mixture_components: Number of elements of the mixture.
  Returns:
    random_prior: A `tfd.Distribution` instance representing the distribution
      over encodings in the absence of any evidence.
  """
  if mixture_components == 1:
    # See the module docstring for why we don't learn the parameters here.
    return tfd.MultivariateNormalDiag(
      loc=tf.zeros([latent_size]),
      scale_identity_multiplier=1.0)

  loc = tf.get_variable(name="loc", shape=[mixture_components, latent_size])
  raw_scale_diag = tf.get_variable(
    name="raw_scale_diag", shape=[mixture_components, latent_size])
  mixture_logits = tf.get_variable(
    name="mixture_logits", shape=[mixture_components])

  return tfd.MixtureSameFamily(
    components_distribution=tfd.MultivariateNormalDiag(
      loc=loc,
      scale_diag=tf.nn.softplus(raw_scale_diag)),
    mixture_distribution=tfd.Categorical(logits=mixture_logits),
    name="prior")

def model_fn(features, labels, mode, params, config):
  """Builds the model function for use in an estimator.
  Arguments:
    features: The input features for the estimator.
    labels: The labels, unused here.
    mode: Signifies whether it is train or test or predict.
    params: Some hyperparameters as a dictionary.
    config: The RunConfig, unused here.
  Returns:
    EstimatorSpec: A tf.estimator.EstimatorSpec instance.
  """
  del labels, config

  if params["analytic_kl"] and params["mixture_components"] != 1:
    raise NotImplementedError(
      "Using `analytic_kl` is only supported when `mixture_components = 1` "
      "since there's no closed form otherwise.")

  encoder = make_encoder(params["activation"],
                         params["latent_size"],
                         params["base_depth"])
  decoder = make_decoder(params["activation"],
                         params["latent_size"],
                         IMAGE_SHAPE,
                         params["base_depth"])
  latent_prior = make_mixture_prior(params["latent_size"],
                                    params["mixture_components"])


  approx_posterior = encoder(features)
  approx_posterior_sample = approx_posterior.sample(params["n_samples"])
  decoder_likelihood = decoder(approx_posterior_sample)

  # `distortion` is just the negative log likelihood.
  distortion = -decoder_likelihood.log_prob(features)
  avg_distortion = tf.reduce_mean(distortion)
  tf.summary.scalar("distortion", avg_distortion)

  if params["analytic_kl"]:
    rate = tfd.kl_divergence(approx_posterior, latent_prior)
  else:
    rate = (approx_posterior.log_prob(approx_posterior_sample)
            - latent_prior.log_prob(approx_posterior_sample))
  avg_rate = tf.reduce_mean(rate)
  tf.summary.scalar("rate", avg_rate)

  elbo_local = -(rate + distortion)

  elbo = tf.reduce_mean(elbo_local)

  tf.summary.scalar("elbo", elbo)

  importance_weighted_elbo = tf.reduce_mean(
    tf.reduce_logsumexp(elbo_local, axis=0) -
    tf.log(tf.to_float(params["n_samples"])))
  tf.summary.scalar("elbo/importance_weighted", importance_weighted_elbo)

  # Decode samples from the prior for visualization.
  random_image = decoder(latent_prior.sample(16))

  # Perform variational inference by minimizing the -ELBO.
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step,
                                        params["max_steps"])
  tf.summary.scalar("learning_rate", learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)

  loss = -elbo
  train_op = optimizer.minimize(loss, global_step=global_step)
  eval_metric_ops={
    "elbo": tf.metrics.mean(elbo),
    "elbo/importance_weighted": tf.metrics.mean(importance_weighted_elbo),
    "rate": tf.metrics.mean(avg_rate),
    "distortion": tf.metrics.mean(avg_distortion),
  }

  prediction = {
    'value' : features,
    'anomaly_score' : distortion
  }

  export_outputs = {
    'prediction': tf.estimator.export.PredictOutput(prediction)
  }

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    eval_metric_ops=eval_metric_ops,
    predictions=prediction,
    export_outputs=export_outputs,
  )

def static_nlog_dataset(data_dir, file_name):
  dataset = tf.data.TextLineDataset(data_dir + '/' + file_name)
  return dataset

def build_input_fns(data_dir, batch_size):
  """Builds an Iterator switching between train and heldout data."""

  # Build an iterator over training batches.
  training_dataset = static_nlog_dataset(data_dir, 'globalsignin_devicemodel_train')
  training_dataset = training_dataset.shuffle(1000).repeat().batch(batch_size)
  train_input_fn = lambda: extract_feature(training_dataset.make_one_shot_iterator().get_next())

  # Build an iterator over the heldout set.
  eval_dataset = static_nlog_dataset(data_dir, 'globalsignin_devicemodel_eval')
  eval_dataset = eval_dataset.batch(batch_size)
  eval_input_fn = lambda: extract_feature(eval_dataset.make_one_shot_iterator().get_next())

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
  feature = extract_feature(string_array)
  return tf.estimator.export.TensorServingInputReceiver(feature, string_array)

def extract_feature(string_array):
  string_array = tf.strings.substr(string_array,0,seq_len)
  split_stensor = tf.string_split(string_array, delimiter="")
  split_values = split_stensor.values
  unicode_values = tf.map_fn(lambda x: tf.io.decode_raw(x, tf.uint8)[0], split_values, dtype=tf.uint8)
  unicode_tensor = tf.sparse_to_dense(split_stensor.indices, [split_stensor.dense_shape[0], seq_len], unicode_values, default_value=-1)
  encoded_tensor = tf.map_fn(lambda x: tf.one_hot(x, enc_size), unicode_tensor, dtype=tf.float32)
  reshaped_tensor = tf.map_fn(lambda x: tf.reshape(x, IMAGE_SHAPE), encoded_tensor)
  return reshaped_tensor

def main(argv):
  del argv  # unused

  params = FLAGS.flag_values_dict()
  params["activation"] = getattr(tf.nn, params["activation"])
  if FLAGS.delete_existing and tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
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
  FLAGS.max_steps = 101
  tf.app.run()