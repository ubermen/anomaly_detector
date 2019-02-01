from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.ops import math_ops
from tensorflow.python.ops.distributions import distribution
# from tensorflow.python.ops.distributions import normal
from tensorflow.python.framework import ops

import trainer.kdeNormal as kdeNormal
import tensorflow as tf
import math



class KernelDensity(distribution.Distribution):

    def __init__(self, loc, scale, weight=None, kernel_dist=kdeNormal.KdeNormalWithSoftplusScale,
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
#         return tf.math.reduce_sum(self._kernel._log_prob(x))
#         return tf.reduce_logsumexp(self._kernel._log_prob(x), [-1], keep_dims=True)
        return self._kernel._log_prob(x)
