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
"""## Script Language Operators.

TensorFlow provides allows you to wrap python/numpy functions as
TensorFlow operators.

"""

# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np

from tensorflow.python import pywrap_tensorflow
from tensorflow.python.framework import ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import gen_script_ops


class FuncRegistry(object):
  """A helper class to keep track of registered py functions.

  FuncRegistry keeps a map from unique tokens (string) to python
  functions, which takes numpy arrays and outputs numpy arrays.
  """

  def __init__(self):
    self._unique_id = 0
    self._funcs = {}

  def insert(self, func):
    """Registers `func` and returns a unique token for this entry."""
    token = self._next_unique_token()
    self._funcs[token] = func
    return token

  def remove(self, token):
    """Removes the registered function corresponding to `token`."""
    self._funcs.pop(token, None)

  def __call__(self, token, args):
    """Calls the registered function for `token` with args."""
    func = self._funcs[token]
    if func is None:
      raise ValueError("callback %s is not found" % token)
    ret = func(*args)
    # Ensures that we return either a single numpy array or a list of numpy
    # arrays.
    if isinstance(ret, (tuple, list)):
      ret = [np.array(x, order="C") for x in ret]
    else:
      ret = np.array(ret, order="C")
    return ret

  def size(self):
    """Returns how many functions are currently registered."""
    return len(self._funcs)

  def _next_unique_token(self):
    """Returns a unique token."""
    uid = self._unique_id
    self._unique_id += 1
    return "pyfunc_%d" % uid

# Global registry for py functions.
_py_funcs = FuncRegistry()

pywrap_tensorflow.InitializePyTrampoline(_py_funcs)


class CleanupFunc(object):
  """A helper class to remove a registered function from _py_funcs."""

  def __init__(self, token):
    self._token = token

  def __del__(self):
    _py_funcs.remove(self._token)


def py_func(func, inp, Tout, name=None):
  """Wraps a python function and uses it as a tensorflow op.

  Given a python function `func`, which takes numpy arrays as its
  inputs and returns numpy arrays as its outputs. E.g.,

  ```python
  def my_func(x):
    # x will be a numpy array with the contents of the placeholder below
    return np.sinh(x)
  inp = tf.placeholder(tf.float32, [...])
  y = py_func(my_func, [inp], [tf.float32])
  ```

  The above snippet constructs a tf graph which invokes a numpy
  sinh(x) as an op in the graph.

  Args:
    func: A python function.
    inp: A list of `Tensor`.
    Tout: A list of tensorflow data types indicating what `func`
          returns.
    name: A name for the operation (optional).

  Returns:
    A list of `Tensor` which `func` computes.
  """
  token = _py_funcs.insert(func)
  # We tie the registered function's life-time with the current
  # default graph. I.e., when the current graph is destroyed, we
  # should remove its py funcs.
  cleanup = CleanupFunc(token)
  g = ops.get_default_graph()
  # pylint: disable=protected-access
  #
  # TODO(zhifengc): Consider adding a Graph method to collect
  # `cleanup` objects in one of its member.
  if not hasattr(g, "_cleanup_py_funcs_used_in_graph"):
    g._cleanup_py_funcs_used_in_graph = []

  # When g is destroyed, elements in _cleanup_py_funcs_used_in_graph
  # will be destroyed and their __del__ will remove the 'token' from
  # the funcs registry.
  g._cleanup_py_funcs_used_in_graph.append(cleanup)

  return gen_script_ops._py_func(input=inp, token=token, Tout=Tout, name=name)
  # pylint: enable=protected-access


ops.RegisterShape("PyFunc")(common_shapes.unknown_shape)

ops.NoGradient("PyFunc")
