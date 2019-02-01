from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import numpy as np

from tensorflow.python.framework import constant_op
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import check_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn
from tensorflow.python.ops import random_ops
from tensorflow.python.ops.distributions import distribution
from tensorflow.python.ops.distributions import kullback_leibler
from tensorflow.python.ops.distributions import special_math
from tensorflow.python.util.tf_export import tf_export


__all__ = [
    "KdeNormal",
    "KdeNormalWithSoftplusScale",
]


class KdeNormal(distribution.Distribution):

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="KdeNormal"):
    parameters = dict(locals())
    with ops.name_scope(name, values=[loc, scale]) as name:
      with ops.control_dependencies([check_ops.assert_positive(scale)] if
                                    validate_args else []):
        self._loc = array_ops.identity(loc, name="loc")
        self._log_size = tf.to_double(tf.size(self.loc))
        self._scale = array_ops.identity(scale, name="scale")
        self._log_normalization = 0.5 * math.log(2. * math.pi) + math_ops.log(self.scale)
        self._log_normalization_cnt_table()

        check_ops.assert_same_float_dtype([self._loc, self._scale])
    super(KdeNormal, self).__init__(
        dtype=self._scale.dtype,
        reparameterization_type=distribution.FULLY_REPARAMETERIZED,
        validate_args=validate_args,
        allow_nan_stats=allow_nan_stats,
        parameters=parameters,
        graph_parents=[self._loc, self._scale],
        name=name)

  @staticmethod
  def _param_shapes(sample_shape):
    return dict(
        zip(("loc", "scale"), ([ops.convert_to_tensor(
            sample_shape, dtype=dtypes.int32)] * 2)))

  @property
  def loc(self):
    """Distribution parameter for the mean."""
    return self._loc

  @property
  def scale(self):
    """Distribution parameter for standard deviation."""
    return self._scale

  @property
  def normalization(self):
    return self._log_normalization

  @property
  def normalization_cnt_table(self):
    return self._log_normalization_cnt_table


  def _log_normalization_cnt_table(self):
    y, idx, cnts = tf.unique_with_counts(self.loc)
    yy = tf.as_string(y)

    default_value = tf.constant(0,  dtype=tf.int32)
    self._log_normalization_cnt_table = tf.contrib.lookup.HashTable(
        tf.contrib.lookup.KeyValueTensorInitializer(yy, cnts), default_value)


  def _batch_shape_tensor(self):
    return array_ops.broadcast_dynamic_shape(
        array_ops.shape(self.loc),
        array_ops.shape(self.scale))

  def _batch_shape(self):
    return array_ops.broadcast_static_shape(
        self.loc.get_shape(),
        self.scale.get_shape())

  def _event_shape_tensor(self):
    return constant_op.constant([], dtype=dtypes.int32)

  def _event_shape(self):
    return tensor_shape.scalar()

  def _sample_n(self, n, seed=None):
        shape = array_ops.concat([[n], self.batch_shape_tensor()], 0)
        sampled = random_ops.random_normal(
            shape=shape, mean=0., stddev=1., dtype=self.loc.dtype, seed=seed)
        return sampled * self.scale + self.loc

  def _log_prob(self, x):
#         print("_log_prob: " + str(x))
#         print("_log_unnormalized_prob: " + str(self._log_unnormalized_prob(x).eval()))
#         print(tf.size(self.loc).eval())

#         return self._log_unnormalized_prob(x) / self._log_size
        return self._log_unnormalized_prob(x) - self.normalization


  def _log_unnormalized_prob(self, x):
#         return self._z(x) / self.scale
    return -0.5 * math_ops.square(self._z(x))

#   def _log_normalization(self):
#     return 0.5 * math.log(2. * math.pi) + math_ops.log(self.scale)

  def _entropy(self):
    # Use broadcasting rules to calculate the full broadcast scale.
        scale = self.scale * array_ops.ones_like(self.loc)
        return 0.5 * math.log(2. * math.pi * math.e) + math_ops.log(scale)

  def _mean(self):
        return self.loc * array_ops.ones_like(self.scale)

  def _quantile(self, p):
        return self._inv_z(special_math.ndtri(p))

  def _stddev(self):
        return self.scale * array_ops.ones_like(self.loc)

  def _mode(self):
        return self._mean()

  def _z(self, x):
    """Standardize input `x` to a unit normal."""
    with ops.name_scope("standardize", values=[x]):
#         data = tf.abs(x - self.loc)

#         dataShape = tf.square(data)
#         condition = tf.less(data, self.scale)
#         print(self.loc.eval())
#         print(data.eval())
#         print()
#         print((tf.where(condition, tf.ones_like(dataShape), tf.zeros_like(dataShape))).eval())
#         print((tf.math.reduce_sum((tf.where(condition, tf.ones_like(dataShape), tf.zeros_like(dataShape))), [-1], keep_dims=True)).eval())
#         return tf.math.reduce_sum(tf.where(condition, tf.ones_like(dataShape), tf.zeros_like(dataShape)))
        return (x - self.loc) / self.scale


  def _inv_z(self, z):
    """Reconstruct input `x` from a its normalized version."""
    with ops.name_scope("reconstruct", values=[z]):
        return z * self.scale + self.loc



class KdeNormalWithSoftplusScale(KdeNormal):
  """Normal with softplus applied to `scale`."""

  def __init__(self,
               loc,
               scale,
               validate_args=False,
               allow_nan_stats=True,
               name="KdeNormalWithSoftplusScale"):
    parameters = dict(locals())
    with ops.name_scope(name, values=[scale]) as name:
      super(KdeNormalWithSoftplusScale, self).__init__(
          loc=loc,
          scale=nn.softplus(scale, name="softplus_scale"),
          validate_args=validate_args,
          allow_nan_stats=allow_nan_stats,
          name=name)
    self._parameters = parameters



# @kullback_leibler.RegisterKL(Normal, Normal)
# def _kl_normal_normal(n_a, n_b, name=None):
#   """Calculate the batched KL divergence KL(n_a || n_b) with n_a and n_b Normal.
#   Args:
#     n_a: instance of a Normal distribution object.
#     n_b: instance of a Normal distribution object.
#     name: (optional) Name to use for created operations.
#       default is "kl_normal_normal".
#   Returns:
#     Batchwise KL(n_a || n_b)
#   """
#   with ops.name_scope(name, "kl_normal_normal", [n_a.loc, n_b.loc]):
#     one = constant_op.constant(1, dtype=n_a.dtype)
#     two = constant_op.constant(2, dtype=n_a.dtype)
#     half = constant_op.constant(0.5, dtype=n_a.dtype)
#     s_a_squared = math_ops.square(n_a.scale)
#     s_b_squared = math_ops.square(n_b.scale)
#     ratio = s_a_squared / s_b_squared
#     return (math_ops.square(n_a.loc - n_b.loc) / (two * s_b_squared) +
#             half * (ratio - one - math_ops.log(ratio)))