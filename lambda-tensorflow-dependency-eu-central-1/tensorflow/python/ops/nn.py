# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# pylint: disable=unused-import,g-bad-import-order
"""## Activation Functions

The activation ops provide different types of nonlinearities for use in neural
networks.  These include smooth nonlinearities (`sigmoid`, `tanh`, `elu`,
`softplus`, and `softsign`), continuous but not everywhere differentiable
functions (`relu`, `relu6`, and `relu_x`), and random regularization
(`dropout`).

All activation ops apply componentwise, and produce a tensor of the same
shape as the input tensor.

@@relu
@@relu6
@@elu
@@softplus
@@softsign
@@dropout
@@bias_add
@@sigmoid
@@tanh

## Convolution

The convolution ops sweep a 2-D filter over a batch of images, applying the
filter to each window of each image of the appropriate size.  The different
ops trade off between generic vs. specific filters:

* `conv2d`: Arbitrary filters that can mix channels together.
* `depthwise_conv2d`: Filters that operate on each channel independently.
* `separable_conv2d`: A depthwise spatial filter followed by a pointwise filter.

Note that although these ops are called "convolution", they are strictly
speaking "cross-correlation" since the filter is combined with an input window
without reversing the filter.  For details, see [the properties of
cross-correlation](https://en.wikipedia.org/wiki/Cross-correlation#Properties).

The filter is applied to image patches of the same size as the filter and
strided according to the `strides` argument.  `strides = [1, 1, 1, 1]` applies
the filter to a patch at every offset, `strides = [1, 2, 2, 1]` applies the
filter to every other image patch in each dimension, etc.

Ignoring channels for the moment, and assume that the 4-D `input` has shape
`[batch, in_height, in_width, ...]` and the 4-D `filter` has shape
`[filter_height, filter_width, ...]`, then the spatial semantics of the
convolution ops are as follows: first, according to the padding scheme chosen
as `'SAME'` or `'VALID'`, the output size and the padding pixels are computed.
For the `'SAME'` padding, the output height and width are computed as:

    out_height = ceil(float(in_height) / float(strides[1]))
    out_width  = ceil(float(in_width) / float(strides[2]))

and the padding on the top and left are computed as:

    pad_along_height = ((out_height - 1) * strides[1] +
                        filter_height - in_height)
    pad_along_width = ((out_width - 1) * strides[2] +
                       filter_width - in_width)
    pad_top = pad_along_height / 2
    pad_left = pad_along_width / 2

Note that the division by 2 means that there might be cases when the padding on
both sides (top vs bottom, right vs left) are off by one. In this case, the
bottom and right sides always get the one additional padded pixel. For example,
when `pad_along_height` is 5, we pad 2 pixels at the top and 3 pixels at the
bottom. Note that this is different from existing libraries such as cuDNN and
Caffe, which explicitly specify the number of padded pixels and always pad the
same number of pixels on both sides.

For the `'VALID`' padding, the output height and width are computed as:

    out_height = ceil(float(in_height - filter_height + 1) / float(strides[1]))
    out_width  = ceil(float(in_width - filter_width + 1) / float(strides[2]))

and the padding values are always zero. The output is then computed as

    output[b, i, j, :] =
        sum_{di, dj} input[b, strides[1] * i + di - pad_top,
                           strides[2] * j + dj - pad_left, ...] *
                     filter[di, dj, ...]

where any value outside the original input image region are considered zero (
i.e. we pad zero values around the border of the image).

Since `input` is 4-D, each `input[b, i, j, :]` is a vector.  For `conv2d`, these
vectors are multiplied by the `filter[di, dj, :, :]` matrices to produce new
vectors.  For `depthwise_conv_2d`, each scalar component `input[b, i, j, k]`
is multiplied by a vector `filter[di, dj, k]`, and all the vectors are
concatenated.

@@conv2d
@@depthwise_conv2d
@@separable_conv2d
@@atrous_conv2d
@@conv2d_transpose
@@conv3d

## Pooling

The pooling ops sweep a rectangular window over the input tensor, computing a
reduction operation for each window (average, max, or max with argmax).  Each
pooling op uses rectangular windows of size `ksize` separated by offset
`strides`.  For example, if `strides` is all ones every window is used, if
`strides` is all twos every other window is used in each dimension, etc.

In detail, the output is

    output[i] = reduce(value[strides * i:strides * i + ksize])

where the indices also take into consideration the padding values. Please refer
to the `Convolution` section for details about the padding calculation.

@@avg_pool
@@max_pool
@@max_pool_with_argmax
@@avg_pool3d
@@max_pool3d

## Morphological filtering

Morphological operators are non-linear filters used in image processing.

[Greyscale morphological dilation]
(https://en.wikipedia.org/wiki/Dilation_(morphology)) is the max-sum counterpart
of standard sum-product convolution:

    output[b, y, x, c] =
        max_{dy, dx} input[b,
                           strides[1] * y + rates[1] * dy,
                           strides[2] * x + rates[2] * dx,
                           c] +
                     filter[dy, dx, c]

The `filter` is usually called structuring function. Max-pooling is a special
case of greyscale morphological dilation when the filter assumes all-zero
values (a.k.a. flat structuring function).

[Greyscale morphological erosion]
(https://en.wikipedia.org/wiki/Erosion_(morphology)) is the min-sum counterpart
of standard sum-product convolution:

    output[b, y, x, c] =
        min_{dy, dx} input[b,
                           strides[1] * y - rates[1] * dy,
                           strides[2] * x - rates[2] * dx,
                           c] -
                     filter[dy, dx, c]

Dilation and erosion are dual to each other. The dilation of the input signal
`f` by the structuring signal `g` is equal to the negation of the erosion of
`-f` by the reflected `g`, and vice versa.

Striding and padding is carried out in exactly the same way as in standard
convolution. Please refer to the `Convolution` section for details.

@@dilation2d
@@erosion2d

## Normalization

Normalization is useful to prevent neurons from saturating when inputs may
have varying scale, and to aid generalization.

@@l2_normalize
@@local_response_normalization
@@sufficient_statistics
@@normalize_moments
@@moments

## Losses

The loss ops measure error between two tensors, or between a tensor and zero.
These can be used for measuring accuracy of a network in a regression task
or for regularization purposes (weight decay).

@@l2_loss

## Classification

TensorFlow provides several operations that help you perform classification.

@@sigmoid_cross_entropy_with_logits
@@softmax
@@log_softmax
@@softmax_cross_entropy_with_logits
@@sparse_softmax_cross_entropy_with_logits
@@weighted_cross_entropy_with_logits

## Embeddings

TensorFlow provides library support for looking up values in embedding
tensors.

@@embedding_lookup
@@embedding_lookup_sparse

## Recurrent Neural Networks

TensorFlow provides a number of methods for constructing Recurrent
Neural Networks.  Most accept an `RNNCell`-subclassed object
(see the documentation for `tf.nn.rnn_cell`).

@@dynamic_rnn
@@rnn
@@state_saving_rnn
@@bidirectional_rnn

## Evaluation

The evaluation ops are useful for measuring the performance of a network.
Since they are nondifferentiable, they are typically used at evaluation time.

@@top_k
@@in_top_k

## Candidate Sampling

Do you want to train a multiclass or multilabel model with thousands
or millions of output classes (for example, a language model with a
large vocabulary)?  Training with a full Softmax is slow in this case,
since all of the classes are evaluated for every training example.
Candidate Sampling training algorithms can speed up your step times by
only considering a small randomly-chosen subset of contrastive classes
(called candidates) for each batch of training examples.

See our [Candidate Sampling Algorithms Reference]
(../../extras/candidate_sampling.pdf)

### Sampled Loss Functions

TensorFlow provides the following sampled loss functions for faster training.

@@nce_loss
@@sampled_softmax_loss

### Candidate Samplers

TensorFlow provides the following samplers for randomly sampling candidate
classes when using one of the sampled loss functions above.

@@uniform_candidate_sampler
@@log_uniform_candidate_sampler
@@learned_unigram_candidate_sampler
@@fixed_unigram_candidate_sampler

### Miscellaneous candidate sampling utilities

@@compute_accidental_hits

"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import candidate_sampling_ops
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import embedding_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import nn_grad
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import numerics
from tensorflow.python.ops import random_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import seq2seq
from tensorflow.python.ops import sparse_ops
from tensorflow.python.ops import variable_scope as vs
from tensorflow.python.ops.math_ops import sigmoid
from tensorflow.python.ops.math_ops import tanh
from tensorflow.python.util.all_util import make_all

# Bring more nn-associated functionality into this package.
# go/tf-wildcard-import
# pylint: disable=wildcard-import
from tensorflow.python.ops.nn_ops import *
from tensorflow.python.ops.candidate_sampling_ops import *
from tensorflow.python.ops.embedding_ops import *
from tensorflow.python.ops.rnn import *
# pylint: enable=wildcard-import


def sigmoid_cross_entropy_with_logits(logits, targets, name=None):
  """Computes sigmoid cross entropy given `logits`.

  Measures the probability error in discrete classification tasks in which each
  class is independent and not mutually exclusive.  For instance, one could
  perform multilabel classification where a picture can contain both an elephant
  and a dog at the same time.

  For brevity, let `x = logits`, `z = targets`.  The logistic loss is

        z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = z * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = z * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + log(1 + exp(-x))
      = x - x * z + log(1 + exp(-x))

  For x < 0, to avoid overflow in exp(-x), we reformulate the above

        x - x * z + log(1 + exp(-x))
      = log(exp(x)) - x * z + log(1 + exp(-x))
      = - x * z + log(1 + exp(x))

  Hence, to ensure stability and avoid overflow, the implementation uses this
  equivalent formulation

      max(x, 0) - x * z + log(1 + exp(-abs(x)))

  `logits` and `targets` must have the same type and shape.

  Args:
    logits: A `Tensor` of type `float32` or `float64`.
    targets: A `Tensor` of the same type and shape as `logits`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    logistic losses.

  Raises:
    ValueError: If `logits` and `targets` do not have the same shape.
  """
  with ops.op_scope([logits, targets], name, "logistic_loss") as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    targets = ops.convert_to_tensor(targets, name="targets")
    try:
      targets.get_shape().merge_with(logits.get_shape())
    except ValueError:
      raise ValueError(
          "logits and targets must have the same shape (%s vs %s)"
          % (logits.get_shape(), targets.get_shape()))

    # The logistic loss formula from above is
    #   x - x * z + log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   -x * z + log(1 + exp(x))
    # Note that these two expressions can be combined into the following:
    #   max(x, 0) - x * z + log(1 + exp(-abs(x)))
    # To allow computing gradients at zero, we define custom versions of max and
    # abs functions.
    zeros = array_ops.zeros_like(logits, dtype=logits.dtype)
    cond = (logits >= zeros)
    relu_logits = math_ops.select(cond, logits, zeros)
    neg_abs_logits = math_ops.select(cond, -logits, logits)
    return math_ops.add(relu_logits - logits * targets,
                        math_ops.log(1 + math_ops.exp(neg_abs_logits)),
                        name=name)


def weighted_cross_entropy_with_logits(logits, targets, pos_weight,
                                       name=None):
  """Computes a weighted cross entropy.

  This is like `sigmoid_cross_entropy_with_logits()` except that `pos_weight`,
  allows one to trade off recall and precision by up- or down-weighting the
  cost of a positive error relative to a negative error.

  The usual cross-entropy cost is defined as:

    targets * -log(sigmoid(logits)) + (1 - targets) * -log(1 - sigmoid(logits))

  The argument `pos_weight` is used as a multiplier for the positive targets:

    targets * -log(sigmoid(logits)) * pos_weight +
        (1 - targets) * -log(1 - sigmoid(logits))

  For brevity, let `x = logits`, `z = targets`, `q = pos_weight`.
  The loss is:

        qz * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
      = qz * -log(1 / (1 + exp(-x))) + (1 - z) * -log(exp(-x) / (1 + exp(-x)))
      = qz * log(1 + exp(-x)) + (1 - z) * (-log(exp(-x)) + log(1 + exp(-x)))
      = qz * log(1 + exp(-x)) + (1 - z) * (x + log(1 + exp(-x))
      = (1 - z) * x + (qz +  1 - z) * log(1 + exp(-x))
      = (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))

  Setting `l = (1 + (q - 1) * z)`, to ensure stability and avoid overflow,
  the implementation uses

      (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))

  `logits` and `targets` must have the same type and shape.

  Args:
    logits: A `Tensor` of type `float32` or `float64`.
    targets: A `Tensor` of the same type and shape as `logits`.
    pos_weight: A coefficient to use on the positive examples.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of the same shape as `logits` with the componentwise
    weightedlogistic losses.

  Raises:
    ValueError: If `logits` and `targets` do not have the same shape.
  """
  with ops.op_scope([logits, targets], name, "logistic_loss") as name:
    logits = ops.convert_to_tensor(logits, name="logits")
    targets = ops.convert_to_tensor(targets, name="targets")
    try:
      targets.get_shape().merge_with(logits.get_shape())
    except ValueError:
      raise ValueError(
          "logits and targets must have the same shape (%s vs %s)"
          % (logits.get_shape(), targets.get_shape()))

    # The logistic loss formula from above is
    #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(-x))
    # For x < 0, a more numerically stable formula is
    #   (1 - z) * x + (1 + (q - 1) * z) * log(1 + exp(x)) - l * x
    # To avoid branching, we use the combined version
    #   (1 - z) * x + l * (log(1 + exp(-abs(x))) + max(-x, 0))
    log_weight = 1 + (pos_weight - 1) * targets
    return math_ops.add(
        (1 - targets) * logits,
        log_weight * (math_ops.log(1 + math_ops.exp(-math_ops.abs(logits))) +
                      nn_ops.relu(-logits)),
        name=name)


def relu_layer(x, weights, biases, name=None):
  """Computes Relu(x * weight + biases).

  Args:
    x: a 2D tensor.  Dimensions typically: batch, in_units
    weights: a 2D tensor.  Dimensions typically: in_units, out_units
    biases: a 1D tensor.  Dimensions: out_units
    name: A name for the operation (optional).  If not specified
      "nn_relu_layer" is used.

  Returns:
    A 2-D Tensor computing relu(matmul(x, weights) + biases).
    Dimensions typically: batch, out_units.
  """
  with ops.op_scope([x, weights, biases], name, "relu_layer") as name:
    x = ops.convert_to_tensor(x, name="x")
    weights = ops.convert_to_tensor(weights, name="weights")
    biases = ops.convert_to_tensor(biases, name="biases")
    xw_plus_b = nn_ops.bias_add(math_ops.matmul(x, weights), biases)
    return nn_ops.relu(xw_plus_b, name=name)


def l2_normalize(x, dim, epsilon=1e-12, name=None):
  """Normalizes along dimension `dim` using an L2 norm.

  For a 1-D tensor with `dim = 0`, computes

      output = x / sqrt(max(sum(x**2), epsilon))

  For `x` with more dimensions, independently normalizes each 1-D slice along
  dimension `dim`.

  Args:
    x: A `Tensor`.
    dim: Dimension along which to normalize.
    epsilon: A lower bound value for the norm. Will use `sqrt(epsilon)` as the
      divisor if `norm < sqrt(epsilon)`.
    name: A name for this operation (optional).

  Returns:
    A `Tensor` with the same shape as `x`.
  """
  with ops.op_scope([x], name, "l2_normalize") as name:
    x = ops.convert_to_tensor(x, name="x")
    square_sum = math_ops.reduce_sum(math_ops.square(x), [dim], keep_dims=True)
    x_inv_norm = math_ops.rsqrt(math_ops.maximum(square_sum, epsilon))
    return math_ops.mul(x, x_inv_norm, name=name)


def zero_fraction(value, name=None):
  """Returns the fraction of zeros in `value`.

  If `value` is empty, the result is `nan`.

  This is useful in summaries to measure and report sparsity.  For example,

      z = tf.Relu(...)
      summ = tf.scalar_summary('sparsity', tf.nn.zero_fraction(z))

  Args:
    value: A tensor of numeric type.
    name: A name for the operation (optional).

  Returns:
    The fraction of zeros in `value`, with type `float32`.
  """
  with ops.op_scope([value], name, "zero_fraction"):
    value = ops.convert_to_tensor(value, name="value")
    zero = constant_op.constant(0, dtype=value.dtype, name="zero")
    return math_ops.reduce_mean(math_ops.cast(math_ops.equal(value, zero),
                                              dtypes.float32))


def depthwise_conv2d(input, filter, strides, padding, name=None):
  """Depthwise 2-D convolution.

  Given an input tensor of shape `[batch, in_height, in_width, in_channels]`
  and a filter tensor of shape
  `[filter_height, filter_width, in_channels, channel_multiplier]`
  containing `in_channels` convolutional filters of depth 1, `depthwise_conv2d`
  applies a different filter to each input channel (expanding from 1 channel
  to `channel_multiplier` channels for each), then concatenates the results
  together.  The output has `in_channels * channel_multiplier` channels.

  In detail,

      output[b, i, j, k * channel_multiplier + q] =
          sum_{di, dj} input[b, strides[1] * i + di, strides[2] * j + dj, k] *
                       filter[di, dj, k, q]

  Must have `strides[0] = strides[3] = 1`.  For the most common case of the
  same horizontal and vertical strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: 4-D with shape `[batch, in_height, in_width, in_channels]`.
    filter: 4-D with shape
      `[filter_height, filter_width, in_channels, channel_multiplier]`.
    strides: 1-D of size 4.  The stride of the sliding window for each
      dimension of `input`.
    padding: A string, either `'VALID'` or `'SAME'`.  The padding algorithm.
      See the [comment here](https://www.tensorflow.org/api_docs/python/nn.html#convolution)
    name: A name for this operation (optional).

  Returns:
    A 4-D `Tensor` of shape
    `[batch, out_height, out_width, in_channels * channel_multiplier].`
  """
  with ops.op_scope([input, filter], name, "depthwise") as name:
    input = ops.convert_to_tensor(input, name="tensor_in")
    filter = ops.convert_to_tensor(filter, name="filter_in")
    # A shape is required to statically compute the number of separable filters.
    if filter.get_shape().ndims is not None:
      assert len(filter.get_shape()) == 4
      in_channels = filter.get_shape()[2]
      # Sanity checks, if shape information is available for the inputs.
      if input.get_shape().ndims is not None:
        assert len(input.get_shape()) == 4
        assert input.get_shape()[3] == in_channels, (
            "Mismatched input depth %d and number of depthwise filters %d." % (
                input.get_shape()[3].value, in_channels))
    else:
      assert input.get_shape().ndims is not None, (
          "Either tensor must provide static shape information.")
      assert input.get_shape().ndims == 4
      in_channels = input.get_shape()[3]

    if in_channels == 1:
      return nn_ops.conv2d(input, filter, strides, padding, name=name)
    else:
      return nn_ops.depthwise_conv2d_native(input, filter, strides, padding,
                                            name=name)


def separable_conv2d(input, depthwise_filter, pointwise_filter, strides,
                     padding,
                     name=None):
  """2-D convolution with separable filters.

  Performs a depthwise convolution that acts separately on channels followed by
  a pointwise convolution that mixes channels.  Note that this is separability
  between dimensions `[1, 2]` and `3`, not spatial separability between
  dimensions `1` and `2`.

  In detail,

      output[b, i, j, k] = sum_{di, dj, q, r]
          input[b, strides[1] * i + di, strides[2] * j + dj, q] *
          depthwise_filter[di, dj, q, r] *
          pointwise_filter[0, 0, q * channel_multiplier + r, k]

  `strides` controls the strides for the depthwise convolution only, since
  the pointwise convolution has implicit strides of `[1, 1, 1, 1]`.  Must have
  `strides[0] = strides[3] = 1`.  For the most common case of the same
  horizontal and vertical strides, `strides = [1, stride, stride, 1]`.

  Args:
    input: 4-D `Tensor` with shape `[batch, in_height, in_width, in_channels]`.
    depthwise_filter: 4-D `Tensor` with shape
      `[filter_height, filter_width, in_channels, channel_multiplier]`.
      Contains `in_channels` convolutional filters of depth 1.
    pointwise_filter: 4-D `Tensor` with shape
      `[1, 1, channel_multiplier * in_channels, out_channels]`.  Pointwise
      filter to mix channels after `depthwise_filter` has convolved spatially.
    strides: 1-D of size 4.  The strides for the depthwise convolution for
      each dimension of `input`.
    padding: A string, either `'VALID'` or `'SAME'`.  The padding algorithm.
      See the [comment here](https://www.tensorflow.org/api_docs/python/nn.html#convolution)
    name: A name for this operation (optional).

  Returns:
    A 4-D `Tensor` of shape `[batch, out_height, out_width, out_channels]`.

  Raises:
    ValueError: If channel_multiplier * in_channels > out_channels,
      which means that the separable convolution is overparameterized.
  """
  with ops.op_scope([input, depthwise_filter, pointwise_filter],
                   name, "separable_conv2d") as name:
    input = ops.convert_to_tensor(input, name="tensor_in")
    depthwise_filter = ops.convert_to_tensor(depthwise_filter,
                                             name="depthwise_filter")
    pointwise_filter = ops.convert_to_tensor(pointwise_filter,
                                             name="pointwise_filter")

    if pointwise_filter.get_shape().ndims is not None:
      assert len(pointwise_filter.get_shape()) == 4
      assert pointwise_filter.get_shape()[0] == 1
      assert pointwise_filter.get_shape()[1] == 1
      if depthwise_filter.get_shape().ndims and input.get_shape().ndims:
        channel_multiplier = depthwise_filter.get_shape()[3]
        in_channels = input.get_shape()[3]
        out_channels = pointwise_filter.get_shape()[3]
        if channel_multiplier * in_channels > out_channels:
          raise ValueError(
              ("Refusing to perform an overparameterized separable "
               "convolution: channel_multiplier * in_channels = "
               "%d * %d = %d > %d = out_channels" %
               (channel_multiplier, in_channels,
                channel_multiplier * in_channels, out_channels)))
    # The layout of the ops in the graph are expected to be as follows:
    # depthwise_conv2d  // Conv2D op corresponding to native deptwise conv.
    # separable_conv2d  // Conv2D op corresponding to the pointwise conv.
    depthwise = nn_ops.depthwise_conv2d_native(input, depthwise_filter, strides,
                                               padding, name="depthwise")
    return nn_ops.conv2d(depthwise, pointwise_filter, [1, 1, 1, 1],
                         padding="VALID", name=name)


def sufficient_statistics(x, axes, shift=None, keep_dims=False, name=None):
  """Calculate the sufficient statistics for the mean and variance of `x`.

  These sufficient statistics are computed using the one pass algorithm on
  an input that's optionally shifted. See:
  https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Computing_shifted_data

  Args:
    x: A `Tensor`.
    axes: Array of ints. Axes along which to compute mean and variance.
    shift: A `Tensor` containing the value by which to shift the data for
      numerical stability, or `None` if no shift is to be performed. A shift
      close to the true mean provides the most numerically stable results.
    keep_dims: produce statistics with the same dimensionality as the input.
    name: Name used to scope the operations that compute the sufficient stats.

  Returns:
    Four `Tensor` objects of the same type as `x`:
    * the count (number of elements to average over).
    * the (possibly shifted) sum of the elements in the array.
    * the (possibly shifted) sum of squares of the elements in the array.
    * the shift by which the mean must be corrected or None if `shift` is None.
  """
  with ops.op_scope([x, axes, shift], name, "sufficient_statistics"):
    x = ops.convert_to_tensor(x, name="x")
    x_shape = x.get_shape()
    if x_shape.is_fully_defined():
      counts = 1
      m_shape = []
      for d in xrange(x_shape.ndims):
        dim = x_shape[d].value
        if d in set(axes):
          counts *= dim
          dim = 1
        m_shape.append(dim)
      counts = constant_op.constant(counts, dtype=x.dtype)
    else:  # shape needs to be inferred at runtime.
      x_shape = array_ops.shape(x)
      select_axes = sparse_ops.sparse_to_dense(axes, array_ops.shape(x_shape),
                                               True, False)
      m_shape = math_ops.select(select_axes, array_ops.ones_like(x_shape),
                                x_shape)
      counts = math_ops.cast(
          math_ops.reduce_prod(x_shape / m_shape),
          x.dtype,
          name="count")
    if shift is not None:
      shift = ops.convert_to_tensor(shift, name="shift")
      m_ss = math_ops.sub(x, shift)
      v_ss = math_ops.squared_difference(x, shift)
    else:  # no shift.
      m_ss = x
      v_ss = math_ops.square(x)
    m_ss = math_ops.reduce_sum(m_ss, axes, keep_dims=keep_dims, name="mean_ss")
    v_ss = math_ops.reduce_sum(v_ss, axes, keep_dims=keep_dims, name="var_ss")
  return counts, m_ss, v_ss, shift


def normalize_moments(counts, mean_ss, variance_ss, shift, name=None):
  """Calculate the mean and variance of based on the sufficient statistics.

  Args:
    counts: A `Tensor` containing a the total count of the data (one value).
    mean_ss: A `Tensor` containing the mean sufficient statistics: the (possibly
      shifted) sum of the elements to average over.
    variance_ss: A `Tensor` containing the variance sufficient statistics: the
      (possibly shifted) squared sum of the data to compute the variance over.
    shift: A `Tensor` containing the value by which the data is shifted for
      numerical stability, or `None` if no shift was performed.
    name: Name used to scope the operations that compute the moments.

  Returns:
    Two `Tensor` objects: `mean` and `variance`.
  """
  with ops.op_scope([counts, mean_ss, variance_ss, shift], name, "normalize"):
    divisor = math_ops.inv(counts, name="divisor")
    if shift is not None:
      shifted_mean = math_ops.mul(mean_ss, divisor, name="shifted_mean")
      mean = math_ops.add(shifted_mean, shift, name="mean")
    else:  # no shift.
      shifted_mean = math_ops.mul(mean_ss, divisor, name="mean")
      mean = shifted_mean
    variance = math_ops.sub(
        math_ops.mul(variance_ss, divisor),
        math_ops.square(shifted_mean),
        name="variance")
  return (mean, variance)


def moments(x, axes, shift=None, name=None, keep_dims=False):
  """Calculate the mean and variance of `x`.

  The mean and variance are calculated by aggregating the contents of `x`
  across `axes`.  If `x` is 1-D and `axes = [0]` this is just the mean
  and variance of a vector.

  When using these moments for batch normalization (see
  `tf.nn.batch_normalization`):
    * for so-called "global normalization", used with convolutional filters with
      shape `[batch, height, width, depth]`, pass `axes=[0, 1, 2]`.
    * for simple batch normalization pass `axes=[0]` (batch only).

  Args:
    x: A `Tensor`.
    axes: array of ints.  Axes along which to compute mean and
      variance.
    shift: A `Tensor` containing the value by which to shift the data for
      numerical stability, or `None` if no shift is to be performed. A shift
      close to the true mean provides the most numerically stable results.
    keep_dims: produce moments with the same dimensionality as the input.
    name: Name used to scope the operations that compute the moments.

  Returns:
    Two `Tensor` objects: `mean` and `variance`.
  """
  with ops.op_scope([x, axes, shift], name, "moments"):
    counts, m_ss, v_ss, shift = sufficient_statistics(x,
                                                      axes,
                                                      shift=shift,
                                                      keep_dims=keep_dims,
                                                      name=name)
    return normalize_moments(counts, m_ss, v_ss, shift, name=name)


def batch_normalization(x,
                        mean,
                        variance,
                        offset,
                        scale,
                        variance_epsilon,
                        name=None):
  """Batch normalization.

  As described in http://arxiv.org/abs/1502.03167.
  Normalizes a tensor by `mean` and `variance`, and applies (optionally) a
  `scale` \\\\(\gamma\\\\) to it, as well as an `offset` \\\\(\\beta\\\\):

  \\\\(\\frac{\gamma(x-\mu)}{\sigma}+\\beta\\\\)

  `mean`, `variance`, `offset` and `scale` are all expected to be of one of two
  shapes:
    * In all generality, they can have the same number of dimensions as the
      input `x`, with identical sizes as `x` for the dimensions that are not
      normalized over (the 'depth' dimension(s)), and dimension 1 for the
      others which are being normalized over.
      `mean` and `variance` in this case would typically be the outputs of
      `tf.nn.moments(..., keep_dims=True)` during training, or running averages
      thereof during inference.
    * In the common case where the 'depth' dimension is the last dimension in
      the input tensor `x`, they may be one dimensional tensors of the same
      size as the 'depth' dimension.
      This is the case for example for the common `[batch, depth]` layout of
      fully-connected layers, and `[batch, height, width, depth]` for
      convolutions.
      `mean` and `variance` in this case would typically be the outputs of
      `tf.nn.moments(..., keep_dims=False)` during training, or running averages
      thereof during inference.

  Args:
    x: Input `Tensor` of arbitrary dimensionality.
    mean: A mean `Tensor`.
    variance: A variance `Tensor`.
    offset: An offset `Tensor`, often denoted \\\\(\\beta\\\\) in equations, or
      None. If present, will be added to the normalized tensor.
    scale: A scale `Tensor`, often denoted \\\\(\gamma\\\\) in equations, or
      `None`. If present, the scale is applied to the normalized tensor.
    variance_epsilon: A small float number to avoid dividing by 0.
    name: A name for this operation (optional).

  Returns:
    the normalized, scaled, offset tensor.
  """
  with ops.op_scope([x, mean, variance, scale, offset], name, "batchnorm"):
    inv = math_ops.rsqrt(variance + variance_epsilon)
    if scale is not None:
      inv *= scale
    return x * inv + (
        offset - mean * inv if offset is not None else -mean * inv)


def batch_norm_with_global_normalization(t,
                                         m,
                                         v,
                                         beta,
                                         gamma,
                                         variance_epsilon,
                                         scale_after_normalization,
                                         name=None):
  """Batch normalization.

  This op is deprecated. See `tf.nn.batch_normalization`.

  Args:
    t: A 4D input Tensor.
    m: A 1D mean Tensor with size matching the last dimension of t.
      This is the first output from tf.nn.moments,
      or a saved moving average thereof.
    v: A 1D variance Tensor with size matching the last dimension of t.
      This is the second output from tf.nn.moments,
      or a saved moving average thereof.
    beta: A 1D beta Tensor with size matching the last dimension of t.
      An offset to be added to the normalized tensor.
    gamma: A 1D gamma Tensor with size matching the last dimension of t.
      If "scale_after_normalization" is true, this tensor will be multiplied
      with the normalized tensor.
    variance_epsilon: A small float number to avoid dividing by 0.
    scale_after_normalization: A bool indicating whether the resulted tensor
      needs to be multiplied with gamma.
    name: A name for this operation (optional).

   Returns:
     A batch-normalized `t`.
  """
  return batch_normalization(t, m, v, beta, gamma if scale_after_normalization
                             else None, variance_epsilon, name)


def _sum_rows(x):
  """Returns a vector summing up each row of the matrix x."""
  # _sum_rows(x) is equivalent to math_ops.reduce_sum(x, 1) when x is
  # a matrix.  The gradient of _sum_rows(x) is more efficient than
  # reduce_sum(x, 1)'s gradient in today's implementation. Therefore,
  # we use _sum_rows(x) in the nce_loss() computation since the loss
  # is mostly used for training.
  cols = array_ops.shape(x)[1]
  ones_shape = array_ops.pack([cols, 1])
  ones = array_ops.ones(ones_shape, x.dtype)
  return array_ops.reshape(math_ops.matmul(x, ones), [-1])


def _compute_sampled_logits(weights, biases, inputs, labels, num_sampled,
                            num_classes, num_true=1,
                            sampled_values=None,
                            subtract_log_q=True,
                            remove_accidental_hits=False,
                            partition_strategy="mod",
                            name=None):
  """Helper function for nce_loss and sampled_softmax_loss functions.

  Computes sampled output training logits and labels suitable for implementing
  e.g. noise-contrastive estimation (see nce_loss) or sampled softmax (see
  sampled_softmax_loss).

  Note: In the case where num_true > 1, we assign to each target class
  the target probability 1 / num_true so that the target probabilities
  sum to 1 per-example.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        `[num_classes, dim]`.  The (possibly-partitioned) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    subtract_log_q: A `bool`.  whether to subtract the log expected count of
        the labels in the sample to get the logits of the true labels.
        Default is True.  Turn off for Negative Sampling.
    remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  Default is
        False.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).
  Returns:
    out_logits, out_labels: `Tensor` objects each with shape
        `[batch_size, num_true + num_sampled]`, for passing to either
        `nn.sigmoid_cross_entropy_with_logits` (NCE) or
        `nn.softmax_cross_entropy_with_logits` (sampled softmax).
  """

  if not isinstance(weights, list):
    weights = [weights]

  with ops.op_scope(
      weights + [biases, inputs, labels], name, "compute_sampled_logits"):
    if labels.dtype != dtypes.int64:
      labels = math_ops.cast(labels, dtypes.int64)
    labels_flat = array_ops.reshape(labels, [-1])

    # Sample the negative labels.
    #   sampled shape: [num_sampled] tensor
    #   true_expected_count shape = [batch_size, 1] tensor
    #   sampled_expected_count shape = [num_sampled] tensor
    if sampled_values is None:
      sampled_values = candidate_sampling_ops.log_uniform_candidate_sampler(
          true_classes=labels,
          num_true=num_true,
          num_sampled=num_sampled,
          unique=True,
          range_max=num_classes)
    # NOTE: pylint cannot tell that 'sampled_values' is a sequence
    # pylint: disable=unpacking-non-sequence
    sampled, true_expected_count, sampled_expected_count = sampled_values
    # pylint: enable=unpacking-non-sequence

    # labels_flat is a [batch_size * num_true] tensor
    # sampled is a [num_sampled] int tensor
    all_ids = array_ops.concat(0, [labels_flat, sampled])

    # weights shape is [num_classes, dim]
    all_w = embedding_ops.embedding_lookup(
        weights, all_ids, partition_strategy=partition_strategy)
    all_b = embedding_ops.embedding_lookup(biases, all_ids)
    # true_w shape is [batch_size * num_true, dim]
    # true_b is a [batch_size * num_true] tensor
    true_w = array_ops.slice(
        all_w, [0, 0], array_ops.pack([array_ops.shape(labels_flat)[0], -1]))
    true_b = array_ops.slice(all_b, [0], array_ops.shape(labels_flat))

    # inputs shape is [batch_size, dim]
    # true_w shape is [batch_size * num_true, dim]
    # row_wise_dots is [batch_size, num_true, dim]
    dim = array_ops.shape(true_w)[1:2]
    new_true_w_shape = array_ops.concat(0, [[-1, num_true], dim])
    row_wise_dots = math_ops.mul(
        array_ops.expand_dims(inputs, 1),
        array_ops.reshape(true_w, new_true_w_shape))
    # We want the row-wise dot plus biases which yields a
    # [batch_size, num_true] tensor of true_logits.
    dots_as_matrix = array_ops.reshape(row_wise_dots,
                                       array_ops.concat(0, [[-1], dim]))
    true_logits = array_ops.reshape(_sum_rows(dots_as_matrix), [-1, num_true])
    true_b = array_ops.reshape(true_b, [-1, num_true])
    true_logits += true_b

    # Lookup weights and biases for sampled labels.
    #   sampled_w shape is [num_sampled, dim]
    #   sampled_b is a [num_sampled] float tensor
    sampled_w = array_ops.slice(
        all_w, array_ops.pack([array_ops.shape(labels_flat)[0], 0]), [-1, -1])
    sampled_b = array_ops.slice(all_b, array_ops.shape(labels_flat), [-1])

    # inputs has shape [batch_size, dim]
    # sampled_w has shape [num_sampled, dim]
    # sampled_b has shape [num_sampled]
    # Apply X*W'+B, which yields [batch_size, num_sampled]
    sampled_logits = math_ops.matmul(inputs,
                                     sampled_w,
                                     transpose_b=True) + sampled_b

    if remove_accidental_hits:
      acc_hits = candidate_sampling_ops.compute_accidental_hits(
          labels, sampled, num_true=num_true)
      acc_indices, acc_ids, acc_weights = acc_hits

      # This is how SparseToDense expects the indices.
      acc_indices_2d = array_ops.reshape(acc_indices, [-1, 1])
      acc_ids_2d_int32 = array_ops.reshape(math_ops.cast(
          acc_ids, dtypes.int32), [-1, 1])
      sparse_indices = array_ops.concat(
          1, [acc_indices_2d, acc_ids_2d_int32], "sparse_indices")
      # Create sampled_logits_shape = [batch_size, num_sampled]
      sampled_logits_shape = array_ops.concat(
          0,
          [array_ops.shape(labels)[:1], array_ops.expand_dims(num_sampled, 0)])
      if sampled_logits.dtype != acc_weights.dtype:
        acc_weights = math_ops.cast(acc_weights, sampled_logits.dtype)
      sampled_logits += sparse_ops.sparse_to_dense(
          sparse_indices, sampled_logits_shape, acc_weights,
          default_value=0.0, validate_indices=False)

    if subtract_log_q:
      # Subtract log of Q(l), prior probability that l appears in sampled.
      true_logits -= math_ops.log(true_expected_count)
      sampled_logits -= math_ops.log(sampled_expected_count)

    # Construct output logits and labels. The true labels/logits start at col 0.
    out_logits = array_ops.concat(1, [true_logits, sampled_logits])
    # true_logits is a float tensor, ones_like(true_logits) is a float tensor
    # of ones. We then divide by num_true to ensure the per-example labels sum
    # to 1.0, i.e. form a proper probability distribution.
    out_labels = array_ops.concat(
        1, [array_ops.ones_like(true_logits) / num_true,
            array_ops.zeros_like(sampled_logits)])

  return out_logits, out_labels


def nce_loss(weights, biases, inputs, labels, num_sampled, num_classes,
             num_true=1,
             sampled_values=None,
             remove_accidental_hits=False,
             partition_strategy="mod",
             name="nce_loss"):
  """Computes and returns the noise-contrastive estimation training loss.

  See [Noise-contrastive estimation: A new estimation principle for
  unnormalized statistical models]
  (http://www.jmlr.org/proceedings/papers/v9/gutmann10a/gutmann10a.pdf).
  Also see our [Candidate Sampling Algorithms Reference]
  (../../extras/candidate_sampling.pdf)

  Note: In the case where `num_true` > 1, we assign to each target class
  the target probability 1 / `num_true` so that the target probabilities
  sum to 1 per-example.

  Note: It would be useful to allow a variable number of target classes per
  example.  We hope to provide this functionality in a future release.
  For now, if you have a variable number of target classes, you can pad them
  out to a constant number by either repeating them or by padding
  with an otherwise unused class.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        [num_classes, dim].  The (possibly-partitioned) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    remove_accidental_hits:  A `bool`.  Whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  If set to
        `True`, this is a "Sampled Logistic" loss instead of NCE, and we are
        learning to generate log-odds instead of log probabilities.  See
        our [Candidate Sampling Algorithms Reference]
        (../../extras/candidate_sampling.pdf).
        Default is False.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).

  Returns:
    A `batch_size` 1-D tensor of per-example NCE losses.
  """
  logits, labels = _compute_sampled_logits(
      weights, biases, inputs, labels, num_sampled, num_classes,
      num_true=num_true,
      sampled_values=sampled_values,
      subtract_log_q=True,
      remove_accidental_hits=remove_accidental_hits,
      partition_strategy=partition_strategy,
      name=name)
  sampled_losses = sigmoid_cross_entropy_with_logits(logits,
                                                     labels,
                                                     name="sampled_losses")
  # sampled_losses is batch_size x {true_loss, sampled_losses...}
  # We sum out true and sampled losses.
  return _sum_rows(sampled_losses)


def sampled_softmax_loss(weights, biases, inputs, labels, num_sampled,
                         num_classes, num_true=1,
                         sampled_values=None,
                         remove_accidental_hits=True,
                         partition_strategy="mod",
                         name="sampled_softmax_loss"):
  """Computes and returns the sampled softmax training loss.

  This is a faster way to train a softmax classifier over a huge number of
  classes.

  This operation is for training only.  It is generally an underestimate of
  the full softmax loss.

  At inference time, you can compute full softmax probabilities with the
  expression `tf.nn.softmax(tf.matmul(inputs, tf.transpose(weights)) + biases)`.

  See our [Candidate Sampling Algorithms Reference]
  (../../extras/candidate_sampling.pdf)

  Also see Section 3 of [Jean et al., 2014](http://arxiv.org/abs/1412.2007)
  ([pdf](http://arxiv.org/pdf/1412.2007.pdf)) for the math.

  Args:
    weights: A `Tensor` of shape `[num_classes, dim]`, or a list of `Tensor`
        objects whose concatenation along dimension 0 has shape
        [num_classes, dim].  The (possibly-sharded) class embeddings.
    biases: A `Tensor` of shape `[num_classes]`.  The class biases.
    inputs: A `Tensor` of shape `[batch_size, dim]`.  The forward
        activations of the input network.
    labels: A `Tensor` of type `int64` and shape `[batch_size,
        num_true]`. The target classes.  Note that this format differs from
        the `labels` argument of `nn.softmax_cross_entropy_with_logits`.
    num_sampled: An `int`.  The number of classes to randomly sample per batch.
    num_classes: An `int`. The number of possible classes.
    num_true: An `int`.  The number of target classes per training example.
    sampled_values: a tuple of (`sampled_candidates`, `true_expected_count`,
        `sampled_expected_count`) returned by a `*_candidate_sampler` function.
        (if None, we default to `log_uniform_candidate_sampler`)
    remove_accidental_hits:  A `bool`.  whether to remove "accidental hits"
        where a sampled class equals one of the target classes.  Default is
        True.
    partition_strategy: A string specifying the partitioning strategy, relevant
        if `len(weights) > 1`. Currently `"div"` and `"mod"` are supported.
        Default is `"mod"`. See `tf.nn.embedding_lookup` for more details.
    name: A name for the operation (optional).

  Returns:
    A `batch_size` 1-D tensor of per-example sampled softmax losses.

  """
  logits, labels = _compute_sampled_logits(
      weights, biases, inputs, labels, num_sampled, num_classes,
      num_true=num_true,
      sampled_values=sampled_values,
      subtract_log_q=True,
      remove_accidental_hits=remove_accidental_hits,
      partition_strategy=partition_strategy,
      name=name)
  sampled_losses = nn_ops.softmax_cross_entropy_with_logits(logits, labels)
  # sampled_losses is a [batch_size] tensor.
  return sampled_losses


# TODO(cwhipkey): sigmoid and tanh should not be exposed from tf.nn.
__all__ = make_all(__name__)
__all__.append("zero_fraction")  # documented in training.py

# Modules whitelisted for reference through tf.nn.
# TODO(cwhipkey): migrate callers to use the submodule directly.
__all__.extend(["nn_ops", "rnn_cell", "seq2seq"])

# Symbols whitelisted for export without documentation.
# TODO(cwhipkey): review these and move to contrib or expose through
# documentation.
__all__.extend([
    "all_candidate_sampler",
    "batch_norm_with_global_normalization",
    "batch_normalization",
    "bidirectional_rnn",
    "conv2d_backprop_filter",
    "conv2d_backprop_input",
    "depthwise_conv2d_native",
    "dynamic_rnn",
    "lrn",
    "relu_layer",
    "rnn",
    "state_saving_rnn",
    "xw_plus_b",
])
