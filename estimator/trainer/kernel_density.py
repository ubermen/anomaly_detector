from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops.distributions import distribution
# from tensorflow.python.ops.distributions import normal
from tensorflow.python.framework import ops

import trainer.kde_normal as kde_normal
import tensorflow as tf
import numpy as np




class KernelDensity(distribution.Distribution):

    def __init__(self):
        self._kernel = kde_normal.KdeNormalWithSoftPlusScale(tf.constant([1], dtype=tf.float64), np.array([3.]))
        print("## kernelDensity init")

    def set_param(self, loc, scale, weight=None, kernel_dist=kde_normal.KdeNormalWithSoftPlusScale,
                     validate_args=False, allow_nan_stats=True, name="KernelDensity"):
            parameters = locals()
            with ops.name_scope(name, values=[loc, scale, weight]):
                self._kernel = kernel_dist(loc, scale)

            super(KernelDensity, self).__init__(
                dtype=self._kernel._scale.dtype,
                reparameterization_type=distribution.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                graph_parents=[self._kernel._loc, self._kernel._scale],
                name=name)




    def log_sum_pdf(self, x):
      return tf.map_fn(lambda xi: tf.reduce_logsumexp(self._kernel._log_prob(xi), [-1], keep_dims=True), x)
    #         return tf.math.reduce_sum(self._kernel._log_prob(x))
#         return tf.math.reduce_logsumexp(self._kernel._log_prob(x), [-1], keep_dims=True)
