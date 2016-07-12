# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

"""TensorFlow ops for array / tensor manipulation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.ops import array_ops as array_ops_
from tensorflow.python.ops import math_ops


def split_squeeze(dim, num_split, tensor_in):
  """Splits input on given dimension and then squeezes that dimension.

  Args:
    dim: Dimension to split and squeeze on.
    num_split: integer, the number of ways to split.
    tensor_in: Input tensor of shape [N1, N2, .. Ndim, .. Nx].

  Returns:
    List of tensors [N1, N2, .. Ndim-1, Ndim+1, .. Nx].
  """
  return [array_ops_.squeeze(t, squeeze_dims=[dim])
          for t in array_ops_.split(dim, num_split, tensor_in)]


def expand_concat(dim, inputs):
  """Expands inputs on given dimension and then concatenates them.

  Args:
    dim: Dimension to expand and concatenate on.
    inputs: List of tensors of the same shape [N1, ... Nx].

  Returns:
    A tensor of shape [N1, .. Ndim, ... Nx]
  """
  return array_ops_.concat(dim, [array_ops_.expand_dims(t, dim)
                                 for t in inputs])


def one_hot_matrix(tensor_in, num_classes, on_value=1.0, off_value=0.0):
  """Encodes indices from given tensor as one-hot tensor.

  TODO(ilblackdragon): Ideally implementation should be
  part of TensorFlow with Eigen-native operation.

  Args:
    tensor_in: Input tensor of shape [N1, N2].
    num_classes: Number of classes to expand index into.
    on_value: Tensor or float, value to fill-in given index.
    off_value: Tensor or float, value to fill-in everything else.
  Returns:
    Tensor of shape [N1, N2, num_classes] with 1.0 for each id in original
    tensor.
  """
  return array_ops_.one_hot(
      math_ops.cast(tensor_in, dtypes.int64), num_classes, on_value, off_value)
