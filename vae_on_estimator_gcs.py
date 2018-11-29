# this code is modified version of tensorflow probability example
# https://github.com/tensorflow/probability/blob/master/tensorflow_probability/examples/vae.py

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools
import os

# Dependency imports
from absl import flags
import numpy as np
import pandas as pd
from six.moves import urllib
import tensorflow as tf

tfd = tf.contrib.distributions

seq_len = 28
enc_size = 128
IMAGE_SHAPE = [seq_len, enc_size, 1]

flags.DEFINE_float("learning_rate", default=0.001, help="Initial learning rate.")
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
flags.DEFINE_integer("viz_steps", default=500, help="Frequency at which to save visualizations.")
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
    net = encoder_net(images)
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


def pack_images(images, rows, cols):
  """Helper utility to make a field of images."""
  shape = tf.shape(images)
  width = shape[-3]
  height = shape[-2]
  depth = shape[-1]
  images = tf.reshape(images, (-1, width, height, depth))
  batch = tf.shape(images)[0]
  rows = tf.minimum(rows, batch)
  cols = tf.minimum(batch // rows, cols)
  images = images[:rows * cols]
  images = tf.reshape(images, (rows, cols, width, height, depth))
  images = tf.transpose(images, [0, 2, 1, 3, 4])
  images = tf.reshape(images, [1, rows * width, cols * height, depth])
  return images


def image_tile_summary(name, tensor, rows=8, cols=8):
  tf.summary.image(name, pack_images(tensor, rows, cols), max_outputs=1)


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

  image_tile_summary("input", tf.to_float(features), rows=1, cols=16)

  approx_posterior = encoder(features)
  approx_posterior_sample = approx_posterior.sample(params["n_samples"])
  decoder_likelihood = decoder(approx_posterior_sample)
  image_tile_summary(
    "recon/sample",
    tf.to_float(decoder_likelihood.sample()[:3, :16]),
    rows=3,
    cols=16)
  image_tile_summary(
    "recon/mean",
    decoder_likelihood.mean()[:3, :16],
    rows=3,
    cols=16)

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
  loss = -elbo
  tf.summary.scalar("elbo", elbo)

  importance_weighted_elbo = tf.reduce_mean(
    tf.reduce_logsumexp(elbo_local, axis=0) -
    tf.log(tf.to_float(params["n_samples"])))
  tf.summary.scalar("elbo/importance_weighted", importance_weighted_elbo)

  # Decode samples from the prior for visualization.
  random_image = decoder(latent_prior.sample(16))
  image_tile_summary(
    "random/sample", tf.to_float(random_image.sample()), rows=4, cols=4)
  image_tile_summary("random/mean", random_image.mean(), rows=4, cols=4)

  # Perform variational inference by minimizing the -ELBO.
  global_step = tf.train.get_or_create_global_step()
  learning_rate = tf.train.cosine_decay(params["learning_rate"], global_step,
                                        params["max_steps"])
  tf.summary.scalar("learning_rate", learning_rate)
  optimizer = tf.train.AdamOptimizer(learning_rate)
  train_op = optimizer.minimize(loss, global_step=global_step)

  return tf.estimator.EstimatorSpec(
    mode=mode,
    loss=loss,
    train_op=train_op,
    eval_metric_ops={
      "elbo": tf.metrics.mean(elbo),
      "elbo/importance_weighted": tf.metrics.mean(importance_weighted_elbo),
      "rate": tf.metrics.mean(avg_rate),
      "distortion": tf.metrics.mean(avg_distortion),
    },
  )

def convert_string_to_onehot(value):
  instance_length = min(seq_len, len(value))
  bool_arr = np.zeros((seq_len, enc_size), dtype=bool)
  for i in range(instance_length) :
    unicode = value[i]
    if unicode < enc_size :
      bool_arr[i][unicode] = True
  return bool_arr

def static_nlog_dataset(data_dir, file_name):
  dataset = tf.data.TextLineDataset(data_dir + '/' + file_name)
  str_to_arr = lambda string: convert_string_to_onehot(string)

  def _parser(s):
    booltensor = tf.py_func(str_to_arr, [s], tf.bool)
    reshaped = tf.reshape(booltensor, [seq_len, enc_size, 1])
    return tf.to_float(reshaped), tf.constant(0, tf.int32)

  return dataset.map(_parser)

def build_fake_input_fns(batch_size):
  """Builds fake MNIST-style data for unit testing."""
  dataset = tf.data.Dataset.from_tensor_slices(
    np.random.rand(batch_size, *IMAGE_SHAPE).astype("float32")).map(
    lambda row: (row, 0)).batch(batch_size)

  train_input_fn = lambda: dataset.repeat().make_one_shot_iterator().get_next()
  eval_input_fn = lambda: dataset.make_one_shot_iterator().get_next()
  return train_input_fn, eval_input_fn


def build_input_fns(data_dir, batch_size):
  """Builds an Iterator switching between train and heldout data."""

  # Build an iterator over training batches.
  training_dataset = static_nlog_dataset(data_dir, 'globalsignin_devicemodel_train')
  training_dataset = training_dataset.shuffle(1000).repeat().batch(batch_size)
  train_input_fn = lambda: training_dataset.make_one_shot_iterator().get_next()

  # Build an iterator over the heldout set.
  eval_dataset = static_nlog_dataset(data_dir, 'globalsignin_devicemodel_eval')
  eval_dataset = eval_dataset.batch(batch_size)
  eval_input_fn = lambda: eval_dataset.make_one_shot_iterator().get_next()

  return train_input_fn, eval_input_fn

def main(argv):
  del argv  # unused

  params = FLAGS.flag_values_dict()
  params["activation"] = getattr(tf.nn, params["activation"])
  if FLAGS.delete_existing and tf.gfile.Exists(FLAGS.model_dir):
    tf.logging.warn("Deleting old log directory at {}".format(FLAGS.model_dir))
    tf.gfile.DeleteRecursively(FLAGS.model_dir)
  tf.gfile.MakeDirs(FLAGS.model_dir)

  if FLAGS.fake_data:
    train_input_fn, eval_input_fn = build_fake_input_fns(FLAGS.batch_size)
  else:
    train_input_fn, eval_input_fn = build_input_fns(FLAGS.data_dir,
                                                    FLAGS.batch_size)

  estimator = tf.estimator.Estimator(
    model_fn,
    params=params,
    config=tf.estimator.RunConfig(
      model_dir=FLAGS.model_dir,
      save_checkpoints_steps=FLAGS.viz_steps,
    ),
  )

  for _ in range(FLAGS.max_steps // FLAGS.viz_steps):
    estimator.train(train_input_fn, steps=FLAGS.viz_steps)
    eval_results = estimator.evaluate(eval_input_fn)
    print("Evaluation_results:\n\t%s\n" % eval_results)


if __name__ == "__main__":
  FLAGS.data_dir = "gs://bigus/lqad/data"
  FLAGS.model_dir = "gs://bigus/lqad/model"
  FLAGS.delete_existing = True
  tf.app.run()