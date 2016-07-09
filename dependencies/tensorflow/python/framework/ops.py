# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Classes and functions used to construct graphs."""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import contextlib
import copy
import linecache
import re
import sys
import threading
import weakref

import six
from tensorflow.core.framework import attr_value_pb2
from tensorflow.core.framework import graph_pb2
from tensorflow.core.framework import versions_pb2
from tensorflow.python.framework import device as pydev
from tensorflow.python.framework import dtypes
from tensorflow.python.framework import registry
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import versions
from tensorflow.python.util import compat
from tensorflow.python.platform import logging


def _convert_stack(stack):
  """Converts a stack extracted using _extract_stack() to a traceback stack.

  Args:
    stack: A list of n 4-tuples, (filename, lineno, name, frame_globals).

  Returns:
    A list of n 4-tuples (filename, lineno, name, code), where the code tuple
    element is calculated from the corresponding elements of the input tuple.
  """
  ret = []
  for filename, lineno, name, frame_globals in stack:
    linecache.checkcache(filename)
    line = linecache.getline(filename, lineno, frame_globals)
    if line:
      line = line.strip()
    else:
      line = None
    ret.append((filename, lineno, name, line))
  return ret


# pylint: disable=line-too-long
def _extract_stack():
  """A lightweight re-implementation of traceback.extract_stack.

  NOTE(mrry): traceback.extract_stack eagerly retrieves the line of code for
    each stack frame using linecache, which results in an abundance of stat()
    calls. This implementation does not retrieve the code, and any consumer
    should apply _convert_stack to the result to obtain a traceback that can
    be formatted etc. using traceback methods.

  Returns:
    A list of 4-tuples (filename, lineno, name, frame_globals) corresponding to
    the call stack of the current thread.
  """
  # pylint: enable=line-too-long
  try:
    raise ZeroDivisionError
  except ZeroDivisionError:
    f = sys.exc_info()[2].tb_frame.f_back
  ret = []
  while f is not None:
    lineno = f.f_lineno
    co = f.f_code
    filename = co.co_filename
    name = co.co_name
    frame_globals = f.f_globals
    ret.append((filename, lineno, name, frame_globals))
    f = f.f_back
  ret.reverse()
  return ret


def _as_graph_element(obj):
  """Convert `obj` to a graph element if possible, otherwise return `None`.

  Args:
    obj: Object to convert.

  Returns:
    The result of `obj._as_graph_element()` if that method is available;
        otherwise `None`.
  """
  conv_fn = getattr(obj, "_as_graph_element", None)
  if conv_fn and callable(conv_fn):
    return conv_fn()
  return None


class Tensor(object):
  """Represents a value produced by an `Operation`.

  A `Tensor` is a symbolic handle to one of the outputs of an
  `Operation`. It does not hold the values of that operation's output,
  but instead provides a means of computing those values in a
  TensorFlow [`Session`](../../api_docs/python/client.md#Session).

  This class has two primary purposes:

  1. A `Tensor` can be passed as an input to another `Operation`.
     This builds a dataflow connection between operations, which
     enables TensorFlow to execute an entire `Graph` that represents a
     large, multi-step computation.

  2. After the graph has been launched in a session, the value of the
     `Tensor` can be computed by passing it to
     [`Session.run()`](../../api_docs/python/client.md#Session.run).
     `t.eval()` is a shortcut for calling
     `tf.get_default_session().run(t)`.

  In the following example, `c`, `d`, and `e` are symbolic `Tensor`
  objects, whereas `result` is a numpy array that stores a concrete
  value:

  ```python
  # Build a dataflow graph.
  c = tf.constant([[1.0, 2.0], [3.0, 4.0]])
  d = tf.constant([[1.0, 1.0], [0.0, 1.0]])
  e = tf.matmul(c, d)

  # Construct a `Session` to execute the graph.
  sess = tf.Session()

  # Execute the graph and store the value that `e` represents in `result`.
  result = sess.run(e)
  ```

  @@dtype
  @@name
  @@value_index
  @@graph
  @@op
  @@consumers

  @@eval

  @@get_shape
  @@set_shape

  """

  # List of Python operators that we allow to override.
  OVERLOADABLE_OPERATORS = {
      # Binary.
      "__add__",
      "__radd__",
      "__sub__",
      "__rsub__",
      "__mul__",
      "__rmul__",
      "__div__",
      "__rdiv__",
      "__truediv__",
      "__rtruediv__",
      "__floordiv__",
      "__rfloordiv__",
      "__mod__",
      "__rmod__",
      "__lt__",
      "__le__",
      "__gt__",
      "__ge__",
      "__and__",
      "__rand__",
      "__or__",
      "__ror__",
      "__xor__",
      "__rxor__",
      "__getitem__",
      "__pow__",
      "__rpow__",
      # Unary.
      "__invert__",
      "__neg__",
      "__abs__"
  }

  def __init__(self, op, value_index, dtype):
    """Creates a new `Tensor`.

    Args:
      op: An `Operation`. `Operation` that computes this tensor.
      value_index: An `int`. Index of the operation's endpoint that produces
        this tensor.
      dtype: A `DType`. Type of elements stored in this tensor.

    Raises:
      TypeError: If the op is not an `Operation`.
    """
    if not isinstance(op, Operation):
      raise TypeError("op needs to be an Operation: %s" % op)
    self._op = op
    self._value_index = value_index
    self._dtype = dtypes.as_dtype(dtype)
    self._shape = tensor_shape.unknown_shape()
    # List of operations that use this Tensor as input.  We maintain this list
    # to easily navigate a computation graph.
    self._consumers = []

  @property
  def op(self):
    """The `Operation` that produces this tensor as an output."""
    return self._op

  @property
  def dtype(self):
    """The `DType` of elements in this tensor."""
    return self._dtype

  @property
  def graph(self):
    """The `Graph` that contains this tensor."""
    return self._op.graph

  @property
  def name(self):
    """The string name of this tensor."""
    if not self._op.name:
      raise ValueError("Operation was not named: %s" % self._op)
    return "%s:%d" % (self._op.name, self._value_index)

  @property
  def device(self):
    """The name of the device on which this tensor will be produced, or None."""
    return self._op.device

  def _shape_as_list(self):
    if self._shape.ndims is not None:
      return [dim.value for dim in self._shape.dims]
    else:
      return None

  def get_shape(self):
    """Returns the `TensorShape` that represents the shape of this tensor.

    The shape is computed using shape inference functions that are
    registered for each `Operation` type using `tf.RegisterShape`.
    See [`TensorShape`](../../api_docs/python/framework.md#TensorShape) for more
    details of what a shape represents.

    The inferred shape of a tensor is used to provide shape
    information without having to launch the graph in a session. This
    can be used for debugging, and providing early error messages. For
    example:

    ```python
    c = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])

    print(c.get_shape())
    ==> TensorShape([Dimension(2), Dimension(3)])

    d = tf.constant([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])

    print(d.get_shape())
    ==> TensorShape([Dimension(4), Dimension(2)])

    # Raises a ValueError, because `c` and `d` do not have compatible
    # inner dimensions.
    e = tf.matmul(c, d)

    f = tf.matmul(c, d, transpose_a=True, transpose_b=True)

    print(f.get_shape())
    ==> TensorShape([Dimension(3), Dimension(4)])
    ```

    In some cases, the inferred shape may have unknown dimensions. If
    the caller has additional information about the values of these
    dimensions, `Tensor.set_shape()` can be used to augment the
    inferred shape.

    Returns:
      A `TensorShape` representing the shape of this tensor.
    """
    return self._shape

  def set_shape(self, shape):
    """Updates the shape of this tensor.

    This method can be called multiple times, and will merge the given
    `shape` with the current shape of this tensor. It can be used to
    provide additional information about the shape of this tensor that
    cannot be inferred from the graph alone. For example, this can be used
    to provide additional information about the shapes of images:

    ```python
    _, image_data = tf.TFRecordReader(...).read(...)
    image = tf.image.decode_png(image_data, channels=3)

    # The height and width dimensions of `image` are data dependent, and
    # cannot be computed without executing the op.
    print(image.get_shape())
    ==> TensorShape([Dimension(None), Dimension(None), Dimension(3)])

    # We know that each image in this dataset is 28 x 28 pixels.
    image.set_shape([28, 28, 3])
    print(image.get_shape())
    ==> TensorShape([Dimension(28), Dimension(28), Dimension(3)])
    ```

    Args:
      shape: A `TensorShape` representing the shape of this tensor.

    Raises:
      ValueError: If `shape` is not compatible with the current shape of
        this tensor.
    """
    self._shape = self._shape.merge_with(shape)

  @property
  def value_index(self):
    """The index of this tensor in the outputs of its `Operation`."""
    return self._value_index

  def consumers(self):
    """Returns a list of `Operation`s that consume this tensor.

    Returns:
      A list of `Operation`s.
    """
    return self._consumers

  def _add_consumer(self, consumer):
    """Add a consumer to this tensor.

    Args:
      consumer: an Operation.

    Raises:
      TypeError: if the consumer is not an Operation.
    """
    if not isinstance(consumer, Operation):
      raise TypeError("Consumer must be an Operation: %s" % consumer)
    self._consumers.append(consumer)

  def _as_node_def_input(self):
    """Return a value to use for the NodeDef "input" attribute.

    The returned string can be used in a NodeDef "input" attribute
    to indicate that the NodeDef uses this Tensor as input.

    Raises:
      ValueError: if this Tensor's Operation does not have a name.

    Returns:
      a string.
    """
    if not self._op.name:
      raise ValueError("Operation was not named: %s" % self._op)
    if self._value_index == 0:
      return self._op.name
    else:
      return "%s:%d" % (self._op.name, self._value_index)

  def __str__(self):
    return "Tensor(\"%s\"%s%s%s)" % (
        self.name,
        (", shape=%s" % self.get_shape())
        if self.get_shape().ndims is not None else "",
        (", dtype=%s" % self._dtype.name) if self._dtype else "",
        (", device=%s" % self.device) if self.device else "")

  def __repr__(self):
    return "<tf.Tensor '%s' shape=%s dtype=%s>" % (
        self.name, self.get_shape(), self._dtype.name)

  def __hash__(self):
    # Necessary to support Python's collection membership operators
    return id(self)

  def __eq__(self, other):
    # Necessary to support Python's collection membership operators
    return id(self) == id(other)

  # NOTE(mrry): This enables the Tensor's overloaded "right" binary
  # operators to run when the left operand is an ndarray, because it
  # accords the Tensor class higher priority than an ndarray, or a
  # numpy matrix.
  # TODO(mrry): Convert this to using numpy's __numpy_ufunc__
  # mechanism, which allows more control over how Tensors interact
  # with ndarrays.
  __array_priority__ = 100

  @staticmethod
  def _override_operator(operator, func):
    """Overrides (string) operator on Tensors to call func.

    Args:
      operator: the string name of the operator to override.
      func: the function that replaces the overriden operator.

    Raises:
      ValueError: If operator has already been overwritten,
        or if operator is not allowed to be overwritten.
    """
    existing = getattr(Tensor, operator, None)
    if existing is not None:
      # Check to see if this is a default method-wrapper or slot wrapper which
      # will be true for the comparison operators.
      if not isinstance(existing, type(object.__lt__)):
        raise ValueError("operator %s cannot be overwritten again." % operator)
    if operator not in Tensor.OVERLOADABLE_OPERATORS:
      raise ValueError("Overriding %s is disallowed" % operator)
    setattr(Tensor, operator, func)

  def __iter__(self):
    """Dummy method to prevent iteration. Do not call.

    NOTE(mrry): If we register __getitem__ as an overloaded operator,
    Python will valiantly attempt to iterate over the Tensor from 0 to
    infinity.  Declaring this method prevents this unintended
    behavior.

    Raises:
      TypeError: when invoked.
    """
    raise TypeError("'Tensor' object is not iterable")

  def eval(self, feed_dict=None, session=None):
    """Evaluates this tensor in a `Session`.

    Calling this method will execute all preceding operations that
    produce the inputs needed for the operation that produces this
    tensor.

    *N.B.* Before invoking `Tensor.eval()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.

    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values.
        See [`Session.run()`](../../api_docs/python/client.md#Session.run) for a
        description of the valid feed values.
      session: (Optional.) The `Session` to be used to evaluate this tensor. If
        none, the default session will be used.

    Returns:
      A numpy array corresponding to the value of this tensor.

    """
    return _eval_using_default_session(self, feed_dict, self.graph, session)


def _TensorTensorConversionFunction(t, dtype=None, name=None, as_ref=False):
  _ = name, as_ref
  if dtype and not dtype.is_compatible_with(t.dtype):
    raise ValueError(
        "Tensor conversion requested dtype %s for Tensor with dtype %s: %r"
        % (dtype.name, t.dtype.name, str(t)))
  return t


_tensor_conversion_func_registry = {
    0: [(Tensor, _TensorTensorConversionFunction)]}


def convert_to_tensor(value, dtype=None, name=None, as_ref=False):
  """Converts the given `value` to a `Tensor`.

  This function converts Python objects of various types to `Tensor`
  objects. It accepts `Tensor` objects, numpy arrays, Python lists,
  and Python scalars. For example:

  ```python
  import numpy as np
  array = np.random.rand(32, 100, 100)

  def my_func(arg):
    arg = tf.convert_to_tensor(arg, dtype=tf.float32)
    return tf.matmul(arg, arg) + arg

  # The following calls are equivalent.
  value_1 = my_func(tf.constant([[1.0, 2.0], [3.0, 4.0]]))
  value_2 = my_func([[1.0, 2.0], [3.0, 4.0]])
  value_3 = my_func(np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32))
  ```

  This function can be useful when composing a new operation in Python
  (such as `my_func` in the example above). All standard Python op
  constructors apply this function to each of their Tensor-valued
  inputs, which allows those ops to accept numpy arrays, Python lists,
  and scalars in addition to `Tensor` objects.

  Args:
    value: An object whose type has a registered `Tensor` conversion function.
    dtype: Optional element type for the returned tensor. If missing, the
      type is inferred from the type of `value`.
    name: Optional name to use if a new `Tensor` is created.
    as_ref: True if we want the result as a ref tensor.

  Returns:
    A `Tensor` based on `value`.

  Raises:
    TypeError: If no conversion function is registered for `value`.
    RuntimeError: If a registered conversion function returns an invalid value.

  """
  error_prefix = "" if name is None else "%s: " % name
  if dtype is not None:
    dtype = dtypes.as_dtype(dtype)
  for _, funcs_at_priority in sorted(_tensor_conversion_func_registry.items()):
    for base_type, conversion_func in funcs_at_priority:
      if isinstance(value, base_type):
        ret = conversion_func(value, dtype=dtype, name=name, as_ref=as_ref)
        if not isinstance(ret, Tensor):
          raise RuntimeError(
              "%sConversion function %r for type %s returned non-Tensor: %r"
              % (error_prefix, conversion_func, base_type, ret))
        if dtype and not dtype.is_compatible_with(ret.dtype):
          raise RuntimeError(
              "%sConversion function %r for type %s returned incompatible "
              "dtype: requested = %s, actual = %s"
              % (error_prefix, conversion_func, base_type,
                 dtype.name, ret.dtype.name))
        return ret
  raise TypeError("%sCannot convert %r with type %s to Tensor: "
                  "no conversion function registered."
                  % (error_prefix, value, type(value)))


def convert_n_to_tensor(values, dtype=None, name=None, as_ref=False):
  """Converts `values` to a list of `Tensor` objects.

  Args:
    values: A list of objects that can be consumed by `tf.convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` objects.
    name: (Optional.) A name prefix to used when a new `Tensor` is
      created, in which case element `i` will be given the name `name
      + '_' + i`.
    as_ref: True if the caller wants the results as ref tensors.

  Returns:
    A list of `Tensor` and/or `IndexedSlices` objects.

  Raises:
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  if not isinstance(values, collections.Sequence):
    raise TypeError("values must be a list.")
  ret = []
  for i, value in enumerate(values):
    n = None if name is None else "%s_%d" % (name, i)
    ret.append(convert_to_tensor(value, dtype=dtype, name=n, as_ref=as_ref))
  return ret


def convert_to_tensor_or_indexed_slices(value, dtype=None, name=None,
                                        as_ref=False):
  """Converts the given object to a `Tensor` or an `IndexedSlices`.

  If `value` is an `IndexedSlices` it is returned
  unmodified. Otherwise, it is converted to a `Tensor` using
  `convert_to_tensor()`.

  Args:
    value: An `IndexedSlices` or an object that can be consumed by
      `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor` or
      `IndexedSlices`.
    name: (Optional.) A name to use if a new `Tensor` is created.
    as_ref: True if the caller wants the results as ref tensors.

  Returns:
    An `Tensor` or an `IndexedSlices` based on `value`.

  Raises:
    ValueError: If `dtype` does not match the element type of `value`.
  """
  if isinstance(value, IndexedSlices):
    if dtype and not dtypes.as_dtype(dtype).is_compatible_with(value.dtype):
      raise ValueError(
          "Tensor conversion requested dtype %s for Tensor with dtype %s: %r"
          % (dtypes.as_dtype(dtype).name, value.dtype.name, str(value)))
    return value
  else:
    return convert_to_tensor(value, dtype=dtype, name=name, as_ref=as_ref)


def convert_n_to_tensor_or_indexed_slices(values, dtype=None, name=None,
                                          as_ref=False):
  """Converts `values` to a list of `Tensor` or `IndexedSlices` objects.

  Args:
    values: A list of `None`, `IndexedSlices`, or objects that can be consumed
      by `convert_to_tensor()`.
    dtype: (Optional.) The required `DType` of the returned `Tensor`
      `IndexedSlices`.
    name: (Optional.) A name prefix to used when a new `Tensor` is
      created, in which case element `i` will be given the name `name
      + '_' + i`.
    as_ref: True if the caller wants the results as ref tensors.

  Returns:
    A list of `Tensor` and/or `IndexedSlices` objects.

  Raises:
    TypeError: If no conversion function is registered for an element in
      `values`.
    RuntimeError: If a registered conversion function returns an invalid
      value.
  """
  if not isinstance(values, collections.Sequence):
    raise TypeError("values must be a list.")
  ret = []
  for i, value in enumerate(values):
    if value is None:
      ret.append(value)
    else:
      n = None if name is None else "%s_%d" % (name, i)
      ret.append(
          convert_to_tensor_or_indexed_slices(value, dtype=dtype, name=n,
                                              as_ref=as_ref))
  return ret


def register_tensor_conversion_function(base_type, conversion_func,
                                        priority=100):
  """Registers a function for converting objects of `base_type` to `Tensor`.

  The conversion function must have the following signature:

      def conversion_func(value, dtype=None, name=None, as_ref=False):
        # ...

  It must return a `Tensor` with the given `dtype` if specified. If the
  conversion function creates a new `Tensor`, it should use the given
  `name` if specified. All exceptions will be propagated to the caller.

  If `as_ref` is true, the function must return a `Tensor` reference,
  such as a `Variable`.

  NOTE: The conversion functions will execute in order of priority,
  followed by order of registration. To ensure that a conversion function
  `F` runs before another conversion function `G`, ensure that `F` is
  registered with a smaller priority than `G`.

  Args:
    base_type: The base type or tuple of base types for all objects that
      `conversion_func` accepts.
    conversion_func: A function that converts instances of `base_type` to
      `Tensor`.
    priority: Optional integer that indicates the priority for applying this
      conversion function. Conversion functions with smaller priority values
      run earlier than conversion functions with larger priority values.
      Defaults to 100.

  Raises:
    TypeError: If the arguments do not have the appropriate type.

  """
  if not (isinstance(base_type, type) or
          (isinstance(base_type, tuple)
           and all(isinstance(x, type) for x in base_type))):
    raise TypeError("base_type must be a type or a tuple of types.")
  if not callable(conversion_func):
    raise TypeError("conversion_func must be callable.")

  try:
    funcs_at_priority = _tensor_conversion_func_registry[priority]
  except KeyError:
    funcs_at_priority = []
    _tensor_conversion_func_registry[priority] = funcs_at_priority
  funcs_at_priority.append((base_type, conversion_func))


class IndexedSlices(object):
  """A sparse representation of a set of tensor slices at given indices.

  This class is a simple wrapper for a pair of `Tensor` objects:

  * `values`: A `Tensor` of any dtype with shape `[D0, D1, ..., Dn]`.
  * `indices`: A 1-D integer `Tensor` with shape `[D0]`.

  An `IndexedSlices` is typically used to represent a subset of a larger
  tensor `dense` of shape `[LARGE0, D1, .. , DN]` where `LARGE0 >> D0`.
  The values in `indices` are the indices in the first dimension of
  the slices that have been extracted from the larger tensor.

  The dense tensor `dense` represented by an `IndexedSlices` `slices` has

  ```python
  dense[slices.indices[i], :, :, :, ...] = slices.values[i, :, :, :, ...]
  ```

  The `IndexedSlices` class is used principally in the definition of
  gradients for operations that have sparse gradients
  (e.g. [`tf.gather`](../../api_docs/python/array_ops.md#gather)).

  Contrast this representation with
  [`SparseTensor`](../../api_docs/python/sparse_ops.md#SparseTensor),
  which uses multi-dimensional indices and scalar values.

  @@__init__

  @@values
  @@indices
  @@dense_shape

  @@name
  @@dtype
  @@device
  @@op
  """

  def __init__(self, values, indices, dense_shape=None):
    """Creates an `IndexedSlices`."""
    _get_graph_from_inputs([values, indices, dense_shape])
    self._values = values
    self._indices = indices
    self._dense_shape = dense_shape

  @property
  def values(self):
    """A `Tensor` containing the values of the slices."""
    return self._values

  @property
  def indices(self):
    """A 1-D `Tensor` containing the indices of the slices."""
    return self._indices

  @property
  def dense_shape(self):
    """A 1-D `Tensor` containing the shape of the corresponding dense tensor."""
    return self._dense_shape

  @property
  def name(self):
    """The name of this `IndexedSlices`."""
    return self.values.name

  @property
  def device(self):
    """The name of the device on which `values` will be produced, or `None`."""
    return self.values.device

  @property
  def op(self):
    """The `Operation` that produces `values` as an output."""
    return self.values.op

  @property
  def dtype(self):
    """The `DType` of elements in this tensor."""
    return self.values.dtype

  @property
  def graph(self):
    """The `Graph` that contains the values, indices, and shape tensors."""
    return self._values.graph

  def __str__(self):
    return "IndexedSlices(indices=%s, values=%s%s)" % (
        self._indices, self._values,
        (", dense_shape=%s" % self._dense_shape) if self._dense_shape else "")

  def __neg__(self):
    return IndexedSlices(-self.values, self.indices, self.dense_shape)


IndexedSlicesValue = collections.namedtuple(
    "IndexedSlicesValue", ["values", "indices", "dense_shape"])


class SparseTensor(object):
  """Represents a sparse tensor.

  Tensorflow represents a sparse tensor as three separate dense tensors:
  `indices`, `values`, and `shape`.  In Python, the three tensors are
  collected into a `SparseTensor` class for ease of use.  If you have separate
  `indices`, `values`, and `shape` tensors, wrap them in a `SparseTensor`
  object before passing to the ops below.

  Concretely, the sparse tensor `SparseTensor(indices, values, shape)` is

  * `indices`: A 2-D int64 tensor of shape `[N, ndims]`.
  * `values`: A 1-D tensor of any type and shape `[N]`.
  * `shape`: A 1-D int64 tensor of shape `[ndims]`.

  where `N` and `ndims` are the number of values, and number of dimensions in
  the `SparseTensor` respectively.

  The corresponding dense tensor satisfies

  ```python
  dense.shape = shape
  dense[tuple(indices[i])] = values[i]
  ```

  By convention, `indices` should be sorted in row-major order (or equivalently
  lexicographic order on the tuples `indices[i]`).  This is not enforced when
  `SparseTensor` objects are constructed, but most ops assume correct ordering.
  If the ordering of sparse tensor `st` is wrong, a fixed version can be
  obtained by calling `tf.sparse_reorder(st)`.

  Example: The sparse tensor

  ```python
  SparseTensor(indices=[[0, 0], [1, 2]], values=[1, 2], shape=[3, 4])
  ```

  represents the dense tensor

  ```python
  [[1, 0, 0, 0]
   [0, 0, 2, 0]
   [0, 0, 0, 0]]
  ```

  @@__init__
  @@indices
  @@values
  @@dtype
  @@shape
  @@graph
  """

  def __init__(self, indices, values, shape):
    """Creates a `SparseTensor`.

    Args:
      indices: A 2-D int64 tensor of shape `[N, ndims]`.
      values: A 1-D tensor of any type and shape `[N]`.
      shape: A 1-D int64 tensor of shape `[ndims]`.

    Returns:
      A `SparseTensor`
    """
    with op_scope([indices, values, shape], None, "SparseTensor"):
      indices = convert_to_tensor(indices, name="indices", dtype=dtypes.int64)
      # Always pass as_ref=True because we want to be able to update
      # values later if it is a VariableOp.
      # TODO(touts): Consider adding mutable_values() when 'values'
      # is a VariableOp and updating users of SparseTensor.
      values = convert_to_tensor(values, name="values", as_ref=True)
      shape = convert_to_tensor(shape, name="shape", dtype=dtypes.int64)
    self._indices = indices
    self._values = values
    self._shape = shape

    indices_shape = indices.get_shape().with_rank(2)
    values_shape = values.get_shape().with_rank(1)
    shape_shape = shape.get_shape().with_rank(1)

    # Assert number of rows in indices match the number of elements in values.
    indices_shape[0].merge_with(values_shape[0])
    # Assert number of columns in indices matches the number of elements in
    # shape.
    indices_shape[1].merge_with(shape_shape[0])

  @property
  def indices(self):
    """The indices of non-zero values in the represented dense tensor.

    Returns:
      A 2-D Tensor of int64 with shape `[N, ndims]`, where `N` is the
        number of non-zero values in the tensor, and `ndims` is the rank.
    """
    return self._indices

  @property
  def values(self):
    """The non-zero values in the represented dense tensor.

    Returns:
      A 1-D Tensor of any data type.
    """
    return self._values

  @property
  def dtype(self):
    """The `DType` of elements in this tensor."""
    return self._values.dtype

  @property
  def shape(self):
    """A 1-D Tensor of int64 representing the shape of the dense tensor."""
    return self._shape

  @property
  def graph(self):
    """The `Graph` that contains the index, value, and shape tensors."""
    return self._indices.graph

  def __str__(self):
    return "SparseTensor(indices=%s, values=%s, shape=%s)" % (
        self._indices, self._values, self._shape)


SparseTensorValue = collections.namedtuple("SparseTensorValue",
                                           ["indices", "values", "shape"])


def _device_string(dev_spec):
  if isinstance(dev_spec, pydev.Device):
    return dev_spec.to_string()
  else:
    return dev_spec


def _NodeDef(op_type, name, device=None, attrs=None):
  """Create a NodeDef proto.

  Args:
    op_type: Value for the "op" attribute of the NodeDef proto.
    name: Value for the "name" attribute of the NodeDef proto.
    device: string, device, or function from NodeDef to string.
      Value for the "device" attribute of the NodeDef proto.
    attrs: Optional dictionary where the key is the attribute name (a string)
      and the value is the respective "attr" attribute of the NodeDef proto (an
      AttrValue).

  Returns:
    A graph_pb2.NodeDef protocol buffer.
  """
  node_def = graph_pb2.NodeDef()
  node_def.op = compat.as_bytes(op_type)
  node_def.name = compat.as_bytes(name)
  if attrs is not None:
    for k, v in six.iteritems(attrs):
      node_def.attr[k].CopyFrom(v)
  if device is not None:
    if callable(device):
      node_def.device = device(node_def)
    else:
      node_def.device = _device_string(device)
  return node_def


# Copied from core/framework/node_def_util.cc
# TODO(mrry,josh11b): Consolidate this validation in C++ code.
_VALID_OP_NAME_REGEX = re.compile("[A-Za-z0-9.][A-Za-z0-9_.\\-/]*")


class Operation(object):
  """Represents a graph node that performs computation on tensors.

  An `Operation` is a node in a TensorFlow `Graph` that takes zero or
  more `Tensor` objects as input, and produces zero or more `Tensor`
  objects as output. Objects of type `Operation` are created by
  calling a Python op constructor (such as
  [`tf.matmul()`](../../api_docs/python/math_ops.md#matmul))
  or [`Graph.create_op()`](../../api_docs/python/framework.md#Graph.create_op).

  For example `c = tf.matmul(a, b)` creates an `Operation` of type
  "MatMul" that takes tensors `a` and `b` as input, and produces `c`
  as output.

  After the graph has been launched in a session, an `Operation` can
  be executed by passing it to
  [`Session.run()`](../../api_docs/python/client.md#Session.run).
  `op.run()` is a shortcut for calling `tf.get_default_session().run(op)`.

  @@name
  @@type
  @@inputs
  @@control_inputs
  @@outputs
  @@device
  @@graph

  @@run

  @@get_attr
  @@traceback
  """

  def __init__(self, node_def, g, inputs=None, output_types=None,
               control_inputs=None, input_types=None, original_op=None,
               op_def=None):
    """Creates an `Operation`.

    NOTE: This constructor validates the name of the `Operation` (passed
    as `node_def.name`). Valid `Operation` names match the following
    regular expression:

        [A-Za-z0-9.][A-Za-z0-9_.\\-/]*

    Args:
      node_def: `graph_pb2.NodeDef`.  `NodeDef` for the `Operation`.
        Used for attributes of `graph_pb2.NodeDef`, typically `name`,
        `op`, and `device`.  The `input` attribute is irrelevant here
        as it will be computed when generating the model.
      g: `Graph`. The parent graph.
      inputs: list of `Tensor` objects. The inputs to this `Operation`.
      output_types: list of `DType` objects.  List of the types of the
        `Tensors` computed by this operation.  The length of this list indicates
        the number of output endpoints of the `Operation`.
      control_inputs: list of operations or tensors from which to have a
        control dependency.
      input_types: List of `DType` objects representing the
        types of the tensors accepted by the `Operation`.  By default
        uses `[x.dtype.base_dtype for x in inputs]`.  Operations that expect
        reference-typed inputs must specify these explicitly.
      original_op: Optional. Used to associate the new `Operation` with an
        existing `Operation` (for example, a replica with the op that was
        replicated).
      op_def: Optional. The `op_def_pb2.OpDef` proto that describes the
        op type that this `Operation` represents.

    Raises:
      TypeError: if control inputs are not Operations or Tensors,
        or if `node_def` is not a `NodeDef`,
        or if `g` is not a `Graph`,
        or if `inputs` are not tensors,
        or if `inputs` and `input_types` are incompatible.
      ValueError: if the `node_def` name is not valid.
    """
    if not isinstance(node_def, graph_pb2.NodeDef):
      raise TypeError("node_def needs to be a NodeDef: %s" % node_def)
    if node_def.ByteSize() >= (1 << 31) or node_def.ByteSize() < 0:
      raise ValueError(
          "Cannot create an Operation with a NodeDef larger than 2GB.")
    if not _VALID_OP_NAME_REGEX.match(node_def.name):
      raise ValueError("'%s' is not a valid node name" % node_def.name)
    if not isinstance(g, Graph):
      raise TypeError("g needs to be a Graph: %s" % g)
    self._node_def = copy.deepcopy(node_def)
    self._graph = g
    if inputs is None:
      inputs = []
    elif not isinstance(inputs, list):
      raise TypeError("inputs needs to be a list of Tensors: %s" % inputs)
    self._inputs = list(inputs)  # Defensive copy.
    for a in self._inputs:
      if not isinstance(a, Tensor):
        raise TypeError("input needs to be a Tensor: %s" % a)
      # Mark that we consume the inputs.
      a._add_consumer(self)  # pylint: disable=protected-access
    if output_types is None:
      output_types = []
    self._output_types = output_types
    self._outputs = [Tensor(self, i, output_type)
                     for i, output_type in enumerate(output_types)]
    if input_types is None:
      input_types = [i.dtype.base_dtype for i in self._inputs]
    else:
      if not all(x.is_compatible_with(i.dtype)
                 for i, x in zip(self._inputs, input_types)):
        raise TypeError("Inputs are not compatible with input types")
    self._input_types = input_types

    # Build the list of control inputs.
    self._control_inputs = []
    if control_inputs:
      for c in control_inputs:
        c_op = None
        if isinstance(c, Operation):
          c_op = c
        elif isinstance(c, (Tensor, IndexedSlices)):
          c_op = c.op
        else:
          raise TypeError("Control input must be an Operation, "
                          "a Tensor, or IndexedSlices: %s" % c)
        self._control_inputs.append(c_op)

    self._original_op = original_op
    self._op_def = op_def
    self._traceback = _extract_stack()
    # Add this op to the current control flow context:
    self._control_flow_context = g._get_control_flow_context()
    if self._control_flow_context is not None:
      self._control_flow_context.AddOp(self)
    # NOTE(keveman): Control flow context's AddOp could be creating new ops and
    # setting op.inputs[index] = new_op. Thus the new ops' id could be larger
    # than this op's id even though this op depend on them. Therefore, delaying
    # assigning id to this op until all ops this could be dependent on are
    # created.
    self._id_value = self._graph._next_id()  # pylint: disable=protected-access
    self._recompute_node_def()

  def values(self):
    """DEPRECATED: Use outputs."""
    return tuple(self.outputs)

  def _get_control_flow_context(self):
    """Returns the control flow context of this op.

    Returns:
      A context object.
    """
    return self._control_flow_context

  def _set_control_flow_context(self, context):
    """Sets the current control flow context of this op.

    Args:
      context: a context object.
    """
    self._control_flow_context = context

  @property
  def name(self):
    """The full name of this operation."""
    return self._node_def.name

  @property
  def _id(self):
    """The unique integer id of this operation."""
    return self._id_value

  @property
  def device(self):
    """The name of the device to which this op has been assigned, if any.

    Returns:
      The string name of the device to which this op has been
      assigned, or an empty string if it has not been assigned to a
      device.
    """
    return self._node_def.device

  def _set_device(self, device):
    """Set the device of this operation.

    Args:
      device: string or device..  The device to set.
    """
    self._node_def.device = _device_string(device)

  def _add_input(self, tensor, dtype=None):
    """Add a new input to this operation.

    Args:
      tensor: the Tensor to add as an input.
      dtype: tf.DType: type of the input; defaults to
        the tensor's dtype.

    Raises:
      TypeError: if tensor is not a Tensor,
        or if input tensor type is not convertible to dtype.
      ValueError: if the Tensor is from a different graph.
    """
    if not isinstance(tensor, Tensor):
      raise TypeError("tensor must be a Tensor: %s" % tensor)
    _assert_same_graph(self, tensor)
    if dtype is None:
      dtype = tensor.dtype
    else:
      dtype = dtypes.as_dtype(dtype)
      if not dtype.is_compatible_with(tensor.dtype):
        raise TypeError(
            "Cannot convert a tensor of type %s to an input of type %s"
            % (tensor.dtype.name, dtype.name))
    self._inputs.append(tensor)
    self._input_types.append(dtype)
    tensor._add_consumer(self)  # pylint: disable=protected-access
    self._recompute_node_def()

  def _update_input(self, index, tensor, dtype=None):
    """Update the input to this operation at the given index.

    NOTE: This is for TF internal use only. Please don't use it.

    Args:
      index: the index of the input to update.
      tensor: the Tensor to be used as the input at the given index.
      dtype: tf.DType: type of the input; defaults to
        the tensor's dtype.

    Raises:
      TypeError: if tensor is not a Tensor,
        or if input tensor type is not convertible to dtype.
      ValueError: if the Tensor is from a different graph.
    """
    if not isinstance(tensor, Tensor):
      raise TypeError("tensor must be a Tensor: %s" % tensor)
    _assert_same_graph(self, tensor)
    if dtype is None:
      dtype = tensor.dtype
    else:
      dtype = dtypes.as_dtype(dtype)
      if not dtype.is_compatible_with(tensor.dtype):
        raise TypeError(
            "Cannot convert a tensor of type %s to an input of type %s"
            % (tensor.dtype.name, dtype.name))

    self._inputs[index].consumers().remove(self)
    self._inputs[index] = tensor
    self._input_types[index] = dtype
    tensor._add_consumer(self)  # pylint: disable=protected-access
    self._recompute_node_def()

  def _add_control_input(self, op):
    """Add a new control input to this operation.

    Args:
      op: the Operation to add as control input.

    Raises:
      TypeError: if op is not an Operation.
      ValueError: if op is from a different graph.
    """
    if not isinstance(op, Operation):
      raise TypeError("op must be an Operation: %s" % op)
    _assert_same_graph(self, op)
    self._control_inputs.append(op)
    self._recompute_node_def()

  # Methods below are used when building the NodeDef and Graph proto.
  def _recompute_node_def(self):
    del self._node_def.input[:]
    self._node_def.input.extend([t._as_node_def_input() for t in self._inputs])
    if self._control_inputs:
      self._node_def.input.extend(["^%s" % op.name for op in
                                   self._control_inputs])

  def __str__(self):
    return str(self._node_def)

  @property
  def outputs(self):
    """The list of `Tensor` objects representing the outputs of this op."""
    return self._outputs

# pylint: disable=protected-access
  class _InputList(object):
    """Immutable input list wrapper."""

    def __init__(self, op):
      self._op = op

    def __iter__(self):
      return iter(self._op._inputs)

    def __len__(self):
      return len(self._op._inputs)

    def __bool__(self):
      return bool(self._op._inputs)

    # Python 3 wants __bool__, Python 2.7 wants __nonzero__
    __nonzero__ = __bool__

    def __getitem__(self, i):
      return self._op._inputs[i]
# pylint: enable=protected-access

  @property
  def inputs(self):
    """The list of `Tensor` objects representing the data inputs of this op."""
    return Operation._InputList(self)

  @property
  def _input_dtypes(self):
    return self._input_types

  @property
  def control_inputs(self):
    """The `Operation` objects on which this op has a control dependency.

    Before this op is executed, TensorFlow will ensure that the
    operations in `self.control_inputs` have finished executing. This
    mechanism can be used to run ops sequentially for performance
    reasons, or to ensure that the side effects of an op are observed
    in the correct order.

    Returns:
      A list of `Operation` objects.

    """
    return self._control_inputs

  @property
  def type(self):
    """The type of the op (e.g. `"MatMul"`)."""
    return self._node_def.op

  @property
  def graph(self):
    """The `Graph` that contains this operation."""
    return self._graph

  @property
  def node_def(self):
    """Returns a serialized `NodeDef` representation of this operation.

    Returns:
      A
      [`NodeDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
      protocol buffer.
    """
    return self._node_def

  @property
  def op_def(self):
    """Returns the `OpDef` proto that represents the type of this op.

    Returns:
      An
      [`OpDef`](https://www.tensorflow.org/code/tensorflow/core/framework/op_def.proto)
      protocol buffer.
    """
    return self._op_def

  @property
  def traceback(self):
    """Returns the call stack from when this operation was constructed."""
    return _convert_stack(self._traceback)

  def get_attr(self, name):
    """Returns the value of the attr of this op with the given `name`.

    Args:
      name: The name of the attr to fetch.

    Returns:
      The value of the attr, as a Python object.

    Raises:
      ValueError: If this op does not have an attr with the given `name`.
    """
    fields = ["s", "i", "f", "b", "type", "shape", "tensor"]
    if name not in self._node_def.attr:
      raise ValueError("No attr named '" + name + "' in " +
                       str(self._node_def))
    x = self._node_def.attr[name]
    # Treat an empty oneof value as an empty list.
    if not x.WhichOneof("value"):
      return []
    if x.HasField("list"):
      for f in fields:
        if getattr(x.list, f):
          return list(getattr(x.list, f))
      return []
    else:
      for f in fields:
        if x.HasField(f):
          return getattr(x, f)
      assert False, "Unsupported field type in " + str(x)

  def run(self, feed_dict=None, session=None):
    """Runs this operation in a `Session`.

    Calling this method will execute all preceding operations that
    produce the inputs needed for this operation.

    *N.B.* Before invoking `Operation.run()`, its graph must have been
    launched in a session, and either a default session must be
    available, or `session` must be specified explicitly.

    Args:
      feed_dict: A dictionary that maps `Tensor` objects to feed values.
        See [`Session.run()`](../../api_docs/python/client.md#Session.run)
        for a description of the valid feed values.
      session: (Optional.) The `Session` to be used to run to this operation. If
        none, the default session will be used.
    """
    _run_using_default_session(self, feed_dict, self.graph, session)


_gradient_registry = registry.Registry("gradient")


class RegisterGradient(object):
  """A decorator for registering the gradient function for an op type.

  This decorator is only used when defining a new op type. For an op
  with `m` inputs and `n` outputs, the gradient function is a function
  that takes the original `Operation` and `n` `Tensor` objects
  (representing the gradients with respect to each output of the op),
  and returns `m` `Tensor` objects (representing the partial gradients
  with respect to each input of the op).

  For example, assuming that operations of type `"Sub"` take two
  inputs `x` and `y`, and return a single output `x - y`, the
  following gradient function would be registered:

  ```python
  @tf.RegisterGradient("Sub")
  def _sub_grad(unused_op, grad):
    return grad, tf.neg(grad)
  ```

  The decorator argument `op_type` is the string type of an
  operation. This corresponds to the `OpDef.name` field for the proto
  that defines the operation.

  @@__init__
  """

  def __init__(self, op_type):
    """Creates a new decorator with `op_type` as the Operation type.

    Args:
      op_type: The string type of an operation. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
    """
    if not isinstance(op_type, six.string_types):
      raise TypeError("op_type must be a string")
    self._op_type = op_type

  def __call__(self, f):
    """Registers the function `f` as gradient function for `op_type`."""
    _gradient_registry.register(f, self._op_type)
    return f


def NoGradient(op_type):
  """Specifies that ops of type `op_type` do not have a defined gradient.

  This function is only used when defining a new op type. It may be
  used for ops such as `tf.size()` that are not differentiable.  For
  example:

  ```python
  tf.NoGradient("Size")
  ```

  Args:
    op_type: The string type of an operation. This corresponds to the
      `OpDef.name` field for the proto that defines the operation.

  Raises:
    TypeError: If `op_type` is not a string.

  """
  if not isinstance(op_type, six.string_types):
    raise TypeError("op_type must be a string")
  _gradient_registry.register(None, op_type)


def get_gradient_function(op):
  """Returns the function that computes gradients for "op"."""
  if not op.inputs: return None
  try:
    op_type = op.get_attr("_gradient_op_type")
  except ValueError:
    op_type = op.type
  return _gradient_registry.lookup(op_type)


_shape_registry = registry.Registry("shape functions")
_default_shape_function_registry = registry.Registry("default shape functions")


class RegisterShape(object):
  """A decorator for registering the shape function for an op type.

  This decorator is only used when defining a new op type. A shape
  function is a function from an `Operation` object to a list of
  `TensorShape` objects, with one `TensorShape` for each output of the
  operation.

  For example, assuming that operations of type `"Sub"` take two
  inputs `x` and `y`, and return a single output `x - y`, all with the
  same shape, the following shape function would be registered:

  ```python
  @tf.RegisterShape("Sub")
  def _sub_shape(op):
    return [op.inputs[0].get_shape().merge_with(op.inputs[1].get_shape())]
  ```

  The decorator argument `op_type` is the string type of an
  operation. This corresponds to the `OpDef.name` field for the proto
  that defines the operation.

  """

  def __init__(self, op_type):
    """Saves the `op_type` as the `Operation` type."""
    if not isinstance(op_type, six.string_types):
      raise TypeError("op_type must be a string")
    self._op_type = op_type

  def __call__(self, f):
    """Registers "f" as the shape function for "op_type"."""
    if f is None:
      # None is a special "weak" value that provides a default shape function,
      # and can be overridden by a non-None registration.
      try:
        _default_shape_function_registry.register(_no_shape_function,
                                                  self._op_type)
      except KeyError:
        # Ignore duplicate registrations of the weak value. This can
        # occur if the op library input to wrapper generation
        # inadvertently links in one or more of the standard op
        # libraries.
        pass
    else:
      _shape_registry.register(f, self._op_type)
    return f


def _no_shape_function(op):
  return [tensor_shape.unknown_shape() for _ in op.outputs]


def set_shapes_for_outputs(op):
  """Uses the registered shape functions to set the shapes for op's outputs."""
  try:
    shape_func = _shape_registry.lookup(op.type)
  except LookupError:
    try:
      shape_func = _default_shape_function_registry.lookup(op.type)
    except LookupError:
      raise RuntimeError("No shape function registered for standard op: %s"
                         % op.type)
  shapes = shape_func(op)
  if len(op.outputs) != len(shapes):
    raise RuntimeError(
        "Shape function for op %s returned %d shapes but expected %d" %
        (op, len(shapes), len(op.outputs)))
  for output, s in zip(op.outputs, shapes):
    output.set_shape(s)


class OpStats(object):
  """A holder for statistics about an operator.

  This class holds information about the resource requirements for an op,
  including the size of its weight parameters on-disk and how many FLOPS it
  requires to execute forward inference.

  If you define a new operation, you can create a function that will return a
  set of information about its usage of the CPU and disk space when serialized.
  The function itself takes a Graph object that's been set up so you can call
  methods like get_tensor_by_name to help calculate the results, and a NodeDef
  argument.

  """

  def __init__(self, statistic_type, value=None):
    """Sets up the initial placeholders for the statistics."""
    self.statistic_type = statistic_type
    self.value = value

  @property
  def statistic_type(self):
    return self._statistic_type

  @statistic_type.setter
  def statistic_type(self, statistic_type):
    self._statistic_type = statistic_type

  @property
  def value(self):
    return self._value

  @value.setter
  def value(self, value):
    self._value = value

  def __iadd__(self, other):
    if other.statistic_type != self.statistic_type:
      raise ValueError("Can't add an OpStat of type %s to one of %s.",
                       self.statistic_type, other.statistic_type)
    if self.value is None:
      self.value = other.value
    elif other.value is not None:
      self._value += other.value
    return self

_stats_registry = registry.Registry("statistical functions")


class RegisterStatistics(object):
  """A decorator for registering the statistics function for an op type.

  This decorator is very similar to the RegisterShapes class, and can be defined
  for an op type so that it gives a report on the resources used by an instance
  of an operator, in the form of an OpStats object.

  Well-known types of statistics include these so far:

  - weight_parameters: For operations like MatMul, Conv, and BiasAdd that take
    learned weights as inputs, this statistic captures how many numerical values
    are used. This is good to know because the weights take up most of the size
    of a typical serialized graph on disk.

  - flops: When running a graph, the bulk of the computation happens doing
    numerical calculations like matrix multiplications. This type allows a node
    to return how many floating-point operations it takes to complete. The
    total number of FLOPs for a graph is a good guide to its expected latency.

  You can add your own statistics just by picking a new type string, registering
  functions for the ops you care about, and then calling something like
  python/tools/graph_metrics.py with the new type as an argument.

  If a statistic for an op is registered multiple times, a KeyError will be
  raised.

  For example, you can define a new metric called doohickey for a Foo operation
  by placing this in your code:

  ```python
  @ops.RegisterStatistics("Foo", "doohickey")
  def _calc_foo_bojangles(unused_graph, unused_node_def):
    return ops.OpStats("doohickey", 20)
  ```

  Then in client code you can retrieve the value by making this call:

  ```python
  doohickey = ops.get_stats_for_node_def(graph, node_def, "doohickey")
  ```

  If the NodeDef is for an op with a registered doohickey function, you'll get
  back the calculated amount in doohickey.value, or None if it's not defined.

  """

  def __init__(self, op_type, statistic_type):
    """Saves the `op_type` as the `Operation` type."""
    if not isinstance(op_type, six.string_types):
      raise TypeError("op_type must be a string.")
    if "," in op_type:
      raise TypeError("op_type must not contain a comma.")
    self._op_type = op_type
    if not isinstance(statistic_type, six.string_types):
      raise TypeError("statistic_type must be a string.")
    if "," in statistic_type:
      raise TypeError("statistic_type must not contain a comma.")
    self._statistic_type = statistic_type

  def __call__(self, f):
    """Registers "f" as the statistics function for "op_type"."""
    _stats_registry.register(f, self._op_type + "," + self._statistic_type)
    return f


def get_stats_for_node_def(graph, node, statistic_type):
  """Looks up the node's statistics function in the registry and calls it.

  This function takes a Graph object and a NodeDef from a GraphDef, and if
  there's an associated statistics method, calls it and returns a result. If no
  function has been registered for the particular node type, it returns an empty
  statistics object.

  Args:
    graph: A Graph object that's been set up with the node's graph.
    node: A NodeDef describing the operator.
    statistic_type: A string identifying the statistic we're interested in.
  Returns:
    An OpStats object containing information about resource usage.
  """

  try:
    stats_func = _stats_registry.lookup(node.op + "," + statistic_type)
    result = stats_func(graph, node)
  except LookupError:
    result = OpStats(statistic_type)
  return result


class Graph(object):
  """A TensorFlow computation, represented as a dataflow graph.

  A `Graph` contains a set of
  [`Operation`](../../api_docs/python/framework.md#Operation) objects,
  which represent units of computation; and
  [`Tensor`](../../api_docs/python/framework.md#Tensor) objects, which represent
  the units of data that flow between operations.

  A default `Graph` is always registered, and accessible by calling
  [`tf.get_default_graph()`](../../api_docs/python/framework.md#get_default_graph).
  To add an operation to the default graph, simply call one of the functions
  that defines a new `Operation`:

  ```
  c = tf.constant(4.0)
  assert c.graph is tf.get_default_graph()
  ```

  Another typical usage involves the
  [`Graph.as_default()`](../../api_docs/python/framework.md#Graph.as_default)
  context manager, which overrides the current default graph for the
  lifetime of the context:

  ```python
  g = tf.Graph()
  with g.as_default():
    # Define operations and tensors in `g`.
    c = tf.constant(30.0)
    assert c.graph is g
  ```

  Important note: This class *is not* thread-safe for graph construction. All
  operations should be created from a single thread, or external
  synchronization must be provided. Unless otherwise specified, all methods
  are not thread-safe.

  @@__init__
  @@as_default
  @@as_graph_def
  @@finalize
  @@finalized

  @@control_dependencies
  @@device
  @@name_scope

  A `Graph` instance supports an arbitrary number of "collections"
  that are identified by name. For convenience when building a large
  graph, collections can store groups of related objects: for
  example, the `tf.Variable` uses a collection (named
  [`tf.GraphKeys.VARIABLES`](../../api_docs/python/framework.md#GraphKeys)) for
  all variables that are created during the construction of a graph. The caller
  may define additional collections by specifying a new name.

  @@add_to_collection
  @@get_collection

  @@as_graph_element
  @@get_operation_by_name
  @@get_tensor_by_name
  @@get_operations

  @@seed
  @@unique_name
  @@version
  @@graph_def_versions

  @@create_op
  @@gradient_override_map
  """

  def __init__(self):
    """Creates a new, empty Graph."""
    self._nodes_by_id = dict()
    self._next_node_id = [dict()]
    self._next_id_counter = 0
    self._nodes_by_name = dict()
    # Current name stack: a pair of uniquified names and plain names.
    self._name_stack = ("", "")
    # Maps a name used in the graph to the next id to use for that name.
    self._names_in_use = {}
    # Functions that will be applied to choose a device if none is specified.
    self._device_function_stack = []
    # Default original_op applied to new ops.
    self._default_original_op = None
    # Current control flow context. It could be either CondContext or
    # WhileContext defined in ops/control_flow_ops.py
    self._control_flow_context = None
    # A new node will depend of the union of all of the nodes in the stack.
    self._control_dependencies_stack = []
    # Arbritrary collections of objects.
    self._collections = {}
    # The graph-level random seed
    self._seed = None
    # A map from op type to the kernel label that should be used.
    self._op_to_kernel_label_map = {}
    # A map from op type to an alternative op type that should be used when
    # computing gradients.
    self._gradient_override_map = {}
    # True if the graph is considered "finalized".  In that case no
    # new operations can be added.
    self._finalized = False
    # Functions defined in the graph
    self._functions = collections.OrderedDict()
    # Default GraphDef versions
    self._graph_def_versions = versions_pb2.VersionDef(
        producer=versions.GRAPH_DEF_VERSION,
        min_consumer=versions.GRAPH_DEF_VERSION_MIN_CONSUMER)

  def _check_not_finalized(self):
    """Check if the graph is finalized.

    Raises:
      RuntimeError: If the graph finalized.
    """
    if self._finalized:
      raise RuntimeError("Graph is finalized and cannot be modified.")

  def _add_op(self, op):
    """Adds 'op' to the graph.

    Args:
      op: the Operator or Tensor to add.

    Raises:
      TypeError: if op is not an Operation or Tensor.
      ValueError: if the op.name or op._id are already used.
    """
    self._check_not_finalized()
    if not isinstance(op, (Tensor, Operation)):
      raise TypeError("op must be a Tensor or Operation: %s" % op)

    if op._id in self._nodes_by_id:
      raise ValueError("cannot add an op with id %d as it already "
                       "exists in the graph" % op._id)
    if op.name in self._nodes_by_name:
      raise ValueError("cannot add op with name %s as that name "
                       "is already used" % op.name)
    self._nodes_by_id[op._id] = op
    self._nodes_by_name[op.name] = op

  @property
  def version(self):
    """Returns a version number that increases as ops are added to the graph.

    Note that this is unrelated to the
    [GraphDef version](#Graph.graph_def_version).
    """
    return self._next_id_counter

  @property
  def graph_def_versions(self):
    """The GraphDef version information of this graph.

    For details on the meaning of each version, see [`GraphDef`]
    (https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto).

    Returns:
      A `VersionDef`.
    """
    return self._graph_def_versions

  @property
  def seed(self):
    return self._seed

  @seed.setter
  def seed(self, seed):
    self._seed = seed

  @property
  def finalized(self):
    """True if this graph has been finalized."""
    return self._finalized

  def finalize(self):
    """Finalizes this graph, making it read-only.

    After calling `g.finalize()`, no new operations can be added to
    `g`.  This method is used to ensure that no operations are added
    to a graph when it is shared between multiple threads, for example
    when using a [`QueueRunner`](../../api_docs/python/train.md#QueueRunner).
    """
    self._finalized = True

  def _get_control_flow_context(self):
    """Returns the current control flow context.

    Returns:
      A context object.
    """
    return self._control_flow_context

  def _set_control_flow_context(self, context):
    """Sets the current control flow context.

    Args:
      context: a context object.
    """
    self._control_flow_context = context

  def as_graph_def(self, from_version=None, add_shapes=False):
    """Returns a serialized `GraphDef` representation of this graph.

    The serialized `GraphDef` can be imported into another `Graph`
    (using [`import_graph_def()`](#import_graph_def)) or used with the
    [C++ Session API](../../api_docs/cc/index.md).

    This method is thread-safe.

    Args:
      from_version: Optional.  If this is set, returns a `GraphDef`
        containing only the nodes that were added to this graph since
        its `version` property had the given value.
      add_shapes: If true, adds an "_output_shapes" list attr to each
        node with the inferred shapes of each of its outputs.

    Returns:
      A [`GraphDef`](https://www.tensorflow.org/code/tensorflow/core/framework/graph.proto)
      protocol buffer.

    Raises:
      ValueError: If the `graph_def` would be too large.
    """
    graph = graph_pb2.GraphDef()
    graph.versions.CopyFrom(self._graph_def_versions)
    bytesize = 0
    for op_id in sorted(self._nodes_by_id):
      op = self._nodes_by_id[op_id]
      if from_version is None or op_id > from_version:
        graph.node.extend([op.node_def])
        if op.outputs and add_shapes:
          graph.node[-1].attr["_output_shapes"].list.shape.extend([
              output.get_shape().as_proto() for output in op.outputs])
        bytesize += op.node_def.ByteSize()
        if bytesize >= (1 << 31) or bytesize < 0:
          raise ValueError("GraphDef cannot be larger than 2GB.")
    if self._functions:
      for f in self._functions.values():
        bytesize += f.ByteSize()
        if bytesize >= (1 << 31) or bytesize < 0:
          raise ValueError("GraphDef cannot be larger than 2GB.")
      graph.library.function.extend(self._functions.values())
    return graph

  def _is_function(self, name):
    """Tests whether 'name' is registered in this graph's function library.

    Args:
      name: string op name.
    Returns:
      bool indicating whether or not 'name' is registered in function library.
    """
    return name in self._functions

  def _get_function(self, name):
    """Returns the function definition for 'name'.

    Args:
      name: string function name.
    Returns:
      The function def proto.
    """
    return self._functions[name]

  def _add_function(self, function_def):
    """Adds a function to the graph.

    The function is specified as a [`FunctionDef`]
    (https://www.tensorflow.org/code/tensorflow/core/framework/function.proto)
    protocol buffer.

    After the function has been added, you can call to the function by
    passing the function name in place of an op name to
    `Graph.create_op()`.

    Args:
      function_def: A `FunctionDef` protocol buffer.
    """
    previous_def = self._functions.get(function_def.signature.name, None)
    if previous_def:
      if previous_def != function_def:
        raise ValueError("Another function is already defined with that name")
      else:
        # No need to add again.
        return
    self._functions[function_def.signature.name] = function_def

  # Helper functions to create operations.
  def create_op(self, op_type, inputs, dtypes,
                input_types=None, name=None, attrs=None, op_def=None,
                compute_shapes=True, compute_device=True):
    """Creates an `Operation` in this graph.

    This is a low-level interface for creating an `Operation`. Most
    programs will not call this method directly, and instead use the
    Python op constructors, such as `tf.constant()`, which add ops to
    the default graph.

    Args:
      op_type: The `Operation` type to create. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
      inputs: A list of `Tensor` objects that will be inputs to the `Operation`.
      dtypes: A list of `DType` objects that will be the types of the tensors
        that the operation produces.
      input_types: (Optional.) A list of `DType`s that will be the types of
        the tensors that the operation consumes. By default, uses the base
        `DType` of each input in `inputs`. Operations that expect
        reference-typed inputs must specify `input_types` explicitly.
      name: (Optional.) A string name for the operation. If not specified, a
        name is generated based on `op_type`.
      attrs: (Optional.) A dictionary where the key is the attribute name (a
        string) and the value is the respective `attr` attribute of the
        `NodeDef` proto that will represent the operation (an `AttrValue`
        proto).
      op_def: (Optional.) The `OpDef` proto that describes the `op_type` that
        the operation will have.
      compute_shapes: (Optional.) If True, shape inference will be performed
        to compute the shapes of the outputs.
      compute_device: (Optional.) If True, device functions will be executed
        to compute the device property of the Operation.

    Raises:
      TypeError: if any of the inputs is not a `Tensor`.

    Returns:
      An `Operation` object.

    """
    self._check_not_finalized()
    for idx, a in enumerate(inputs):
      if not isinstance(a, Tensor):
        raise TypeError("Input #%d is not a tensor: %s" % (idx, a))
    if name is None:
      name = op_type
    # If a names ends with a '/' it is a "name scope" and we use it as-is,
    # after removing the trailing '/'.
    if name and name[-1] == "/":
      name = name[:-1]
    else:
      name = self.unique_name(name)

    node_def = _NodeDef(op_type, name, device=None, attrs=attrs)

    # Apply a kernel label if one has been specified for this op_type.
    try:
      kernel_label = self._op_to_kernel_label_map[op_type]
      node_def.attr["_kernel"].CopyFrom(
          attr_value_pb2.AttrValue(s=compat.as_bytes(kernel_label)))
    except KeyError:
      pass

    # Apply the overriding op_type for gradients if one has been
    # specified for this op_type.
    try:
      mapped_op_type = self._gradient_override_map[op_type]
      node_def.attr["_gradient_op_type"].CopyFrom(
          attr_value_pb2.AttrValue(s=compat.as_bytes(mapped_op_type)))
    except KeyError:
      pass

    control_inputs = self._control_dependencies_for_inputs(inputs)
    ret = Operation(node_def, self, inputs=inputs, output_types=dtypes,
                    control_inputs=control_inputs, input_types=input_types,
                    original_op=self._default_original_op, op_def=op_def)
    if compute_shapes:
      set_shapes_for_outputs(ret)
    self._add_op(ret)
    self._record_op_seen_by_control_dependencies(ret)
    if compute_device:
      self._apply_device_functions(ret)
    return ret

  def as_graph_element(self, obj, allow_tensor=True, allow_operation=True):
    """Returns the object referred to by `obj`, as an `Operation` or `Tensor`.

    This function validates that `obj` represents an element of this
    graph, and gives an informative error message if it is not.

    This function is the canonical way to get/validate an object of
    one of the allowed types from an external argument reference in the
    Session API.

    This method may be called concurrently from multiple threads.

    Args:
      obj: A `Tensor`, an `Operation`, or the name of a tensor or operation.
        Can also be any object with an `_as_graph_element()` method that returns
        a value of one of these types.
      allow_tensor: If true, `obj` may refer to a `Tensor`.
      allow_operation: If true, `obj` may refer to an `Operation`.

    Returns:
      The `Tensor` or `Operation` in the Graph corresponding to `obj`.

    Raises:
      TypeError: If `obj` is not a type we support attempting to convert
        to types.
      ValueError: If `obj` is of an appropriate type but invalid. For
        example, an invalid string.
      KeyError: If `obj` is not an object in the graph.
    """

    # The vast majority of this function is figuring
    # out what an API user might be doing wrong, so
    # that we can give helpful error messages.
    #
    # Ideally, it would be nice to split it up, but we
    # need context to generate nice error messages.

    if allow_tensor and allow_operation:
      types_str = "Tensor or Operation"
    elif allow_tensor:
      types_str = "Tensor"
    elif allow_operation:
      types_str = "Operation"
    else:
      raise ValueError("allow_tensor and allow_operation can't both be False.")

    obj = _as_graph_element(obj) or obj

    # If obj appears to be a name...
    if isinstance(obj, compat.bytes_or_text_types):
      name = compat.as_str(obj)

      if ":" in name and allow_tensor:
        # Looks like a Tensor name and can be a Tensor.
        try:
          op_name, out_n = name.split(":")
          out_n = int(out_n)
        except:
          raise ValueError("The name %s looks a like a Tensor name, but is "
                           "not a valid one. Tensor names must be of the "
                           "form \"<op_name>:<output_index>\"." % repr(name))
        if op_name in self._nodes_by_name:
          op = self._nodes_by_name[op_name]
        else:
          raise KeyError("The name %s refers to a Tensor which does not "
                         "exist. The operation, %s, does not exist in the "
                         "graph." % (repr(name), repr(op_name)))
        try:
          return op.outputs[out_n]
        except:
          raise KeyError("The name %s refers to a Tensor which does not "
                         "exist. The operation, %s, exists but only has "
                         "%s outputs."
                         % (repr(name), repr(op_name), len(op.outputs)))

      elif ":" in name and not allow_tensor:
        # Looks like a Tensor name but can't be a Tensor.
        raise ValueError("Name %s appears to refer to a Tensor, not a %s."
                         % (repr(name), types_str))

      elif ":" not in name and allow_operation:
        # Looks like an Operation name and can be an Operation.
        if name not in self._nodes_by_name:
          raise KeyError("The name %s refers to an Operation not in the "
                         "graph." % repr(name))
        return self._nodes_by_name[name]

      elif ":" not in name and not allow_operation:
        # Looks like an Operation name but can't be an Operation.
        if name in self._nodes_by_name:
          # Yep, it's an Operation name
          err_msg = ("The name %s refers to an Operation, not a %s."
                     % (repr(name), types_str))
        else:
          err_msg = ("The name %s looks like an (invalid) Operation name, "
                     "not a %s." % (repr(name), types_str))
        err_msg += (" Tensor names must be of the form "
                    "\"<op_name>:<output_index>\".")
        raise ValueError(err_msg)

    elif isinstance(obj, Tensor) and allow_tensor:
      # Actually obj is just the object it's referring to.
      if obj.graph is not self:
        raise ValueError("Tensor %s is not an element of this graph." % obj)
      return obj
    elif isinstance(obj, Operation) and allow_operation:
      # Actually obj is just the object it's referring to.
      if obj.graph is not self:
        raise ValueError("Operation %s is not an element of this graph." % obj)
      return obj
    else:
      # We give up!
      raise TypeError("Can not convert a %s into a %s."
                      % (type(obj).__name__, types_str))

  def get_operations(self):
    """Return the list of operations in the graph.

    You can modify the operations in place, but modifications
    to the list such as inserts/delete have no effect on the
    list of operations known to the graph.

    This method may be called concurrently from multiple threads.

    Returns:
      A list of Operations.
    """
    return list(self._nodes_by_id.values())

  def get_operation_by_name(self, name):
    """Returns the `Operation` with the given `name`.

    This method may be called concurrently from multiple threads.

    Args:
      name: The name of the `Operation` to return.

    Returns:
      The `Operation` with the given `name`.

    Raises:
      TypeError: If `name` is not a string.
      KeyError: If `name` does not correspond to an operation in this graph.
    """

    if not isinstance(name, six.string_types):
      raise TypeError("Operation names are strings (or similar), not %s."
                      % type(name).__name__)
    return self.as_graph_element(name, allow_tensor=False, allow_operation=True)

  def get_tensor_by_name(self, name):
    """Returns the `Tensor` with the given `name`.

    This method may be called concurrently from multiple threads.

    Args:
      name: The name of the `Tensor` to return.

    Returns:
      The `Tensor` with the given `name`.

    Raises:
      TypeError: If `name` is not a string.
      KeyError: If `name` does not correspond to a tensor in this graph.
    """
    # Names should be strings.
    if not isinstance(name, six.string_types):
      raise TypeError("Tensor names are strings (or similar), not %s."
                      % type(name).__name__)
    return self.as_graph_element(name, allow_tensor=True, allow_operation=False)

  def _next_id(self):
    """Id for next Operation instance. Also increments the internal id."""
    self._check_not_finalized()
    self._next_id_counter += 1
    return self._next_id_counter

  @property
  def _last_id(self):
    return self._next_id_counter

  def as_default(self):
    """Returns a context manager that makes this `Graph` the default graph.

    This method should be used if you want to create multiple graphs
    in the same process. For convenience, a global default graph is
    provided, and all ops will be added to this graph if you do not
    create a new graph explicitly. Use this method the `with` keyword
    to specify that ops created within the scope of a block should be
    added to this graph.

    The default graph is a property of the current thread. If you
    create a new thread, and wish to use the default graph in that
    thread, you must explicitly add a `with g.as_default():` in that
    thread's function.

    The following code examples are equivalent:

    ```python
    # 1. Using Graph.as_default():
    g = tf.Graph()
    with g.as_default():
      c = tf.constant(5.0)
      assert c.graph is g

    # 2. Constructing and making default:
    with tf.Graph().as_default() as g:
      c = tf.constant(5.0)
      assert c.graph is g
    ```

    Returns:
      A context manager for using this graph as the default graph.
    """
    return _default_graph_stack.get_controller(self)

  def add_to_collection(self, name, value):
    """Stores `value` in the collection with the given `name`.

    Note that collections are not sets, so it is possible to add a value to
    a collection several times.

    Args:
      name: The key for the collection. The `GraphKeys` class
        contains many standard names for collections.
      value: The value to add to the collection.
    """
    self._check_not_finalized()
    if name not in self._collections:
      self._collections[name] = [value]
    else:
      self._collections[name].append(value)

  def add_to_collections(self, names, value):
    """Stores `value` in the collections given by `names`.

    Note that collections are not sets, so it is possible to add a value to
    a collection several times. This function makes sure that duplicates in
    `names` are ignored, but it will not check for pre-existing membership of
    `value` in any of the collections in `names`.

    Args:
      names: The keys for the collections to add to. The `GraphKeys` class
        contains many standard names for collections.
      value: The value to add to the collections.
    """
    for name in set(names):
      self.add_to_collection(name, value)

  def get_collection(self, name, scope=None):
    """Returns a list of values in the collection with the given `name`.

    Args:
      name: The key for the collection. For example, the `GraphKeys` class
        contains many standard names for collections.
      scope: (Optional.) If supplied, the resulting list is filtered to include
        only items whose name begins with this string.

    Returns:
      The list of values in the collection with the given `name`, or
      an empty list if no value has been added to that collection. The
      list contains the values in the order under which they were
      collected.
    """
    if scope is None:
      return self._collections.get(name, list())
    else:
      c = []
      for item in self._collections.get(name, list()):
        if hasattr(item, "name") and item.name.startswith(scope):
          c.append(item)
      return c

  def get_all_collection_keys(self):
    """Returns a list of collections used in this graph."""
    return [x for x in self._collections if isinstance(x, six.string_types)]

  @contextlib.contextmanager
  def _original_op(self, op):
    """Python 'with' handler to help annotate ops with their originator.

    An op may have an 'original_op' property that indicates the op on which
    it was based. For example a replica op is based on the op that was
    replicated and a gradient op is based on the op that was differentiated.

    All ops created in the scope of this 'with' handler will have
    the given 'op' as their original op.

    Args:
      op: The Operation that all ops created in this scope will have as their
        original op.

    Yields:
      Nothing.
    """
    old_original_op = self._default_original_op
    try:
      self._default_original_op = op
      yield
    finally:
      self._default_original_op = old_original_op

  # pylint: disable=g-doc-return-or-yield
  @contextlib.contextmanager
  def name_scope(self, name):
    """Returns a context manager that creates hierarchical names for operations.

    A graph maintains a stack of name scopes. A `with name_scope(...):`
    statement pushes a new name onto the stack for the lifetime of the context.

    The `name` argument will be interpreted as follows:

    * A string (not ending with '/') will create a new name scope, in which
      `name` is appended to the prefix of all operations created in the
      context. If `name` has been used before, it will be made unique by
      calling `self.unique_name(name)`.
    * A scope previously captured from a `with g.name_scope(...) as
      scope:` statement will be treated as an "absolute" name scope, which
      makes it possible to re-enter existing scopes.
    * A value of `None` or the empty string will reset the current name scope
      to the top-level (empty) name scope.

    For example:

    ```python
    with tf.Graph().as_default() as g:
      c = tf.constant(5.0, name="c")
      assert c.op.name == "c"
      c_1 = tf.constant(6.0, name="c")
      assert c_1.op.name == "c_1"

      # Creates a scope called "nested"
      with g.name_scope("nested") as scope:
        nested_c = tf.constant(10.0, name="c")
        assert nested_c.op.name == "nested/c"

        # Creates a nested scope called "inner".
        with g.name_scope("inner"):
          nested_inner_c = tf.constant(20.0, name="c")
          assert nested_inner_c.op.name == "nested/inner/c"

        # Create a nested scope called "inner_1".
        with g.name_scope("inner"):
          nested_inner_1_c = tf.constant(30.0, name="c")
          assert nested_inner_1_c.op.name == "nested/inner_1/c"

          # Treats `scope` as an absolute name scope, and
          # switches to the "nested/" scope.
          with g.name_scope(scope):
            nested_d = tf.constant(40.0, name="d")
            assert nested_d.op.name == "nested/d"

            with g.name_scope(""):
              e = tf.constant(50.0, name="e")
              assert e.op.name == "e"
    ```

    The name of the scope itself can be captured by `with
    g.name_scope(...) as scope:`, which stores the name of the scope
    in the variable `scope`. This value can be used to name an
    operation that represents the overall result of executing the ops
    in a scope. For example:

    ```python
    inputs = tf.constant(...)
    with g.name_scope('my_layer') as scope:
      weights = tf.Variable(..., name="weights")
      biases = tf.Variable(..., name="biases")
      affine = tf.matmul(inputs, weights) + biases
      output = tf.nn.relu(affine, name=scope)
    ```

    Args:
      name: A name for the scope.

    Returns:
      A context manager that installs `name` as a new name scope.
    """
    try:
      old_stack = self._name_stack
      if not name:  # Both for name=None and name="" we re-set to empty scope.
        new_stack = (None, None)
      elif name and name[-1] == "/":
        new_stack = (name[:-1], name[:-1])
      else:
        new_stack = (self.unique_name(name), self._plain_name(name))
      self._name_stack = new_stack
      yield "" if new_stack[0] is None else new_stack[0] + "/"
    finally:
      self._name_stack = old_stack
  # pylint: enable=g-doc-return-or-yield

  def unique_name(self, name):
    """Return a unique operation name for `name`.

    Note: You rarely need to call `unique_name()` directly.  Most of
    the time you just need to create `with g.name_scope()` blocks to
    generate structured names.

    `unique_name` is used to generate structured names, separated by
    `"/"`, to help identify operations when debugging a graph.
    Operation names are displayed in error messages reported by the
    TensorFlow runtime, and in various visualization tools such as
    TensorBoard.

    Args:
      name: The name for an operation.

    Returns:
      A string to be passed to `create_op()` that will be used
      to name the operation being created.
    """
    if self._name_stack[0]:
      name = self._name_stack[0] + "/" + name
    i = self._names_in_use.get(name, 0)
    # Increment the number for "name".
    self._names_in_use[name] = i + 1
    if i > 0:
      base_name = name
      # Make sure the composed name is not already used.
      while name in self._names_in_use:
        name = "%s_%d" % (base_name, i)
        i += 1
      # Mark the composed name as used in case someone wants
      # to call unique_name("name_1").
      self._names_in_use[name] = 1
    return name

  # TODO(touts): remove
  def _plain_name(self, name):
    """Return the fully scoped 'name'.

    Args:
      name: a string.

    Returns:
      'name' scoped in the current name stack, without any uniquified
      elements.
    """
    if self._name_stack[1]:
      return self._name_stack[1] + "/" + name
    else:
      return name

  @contextlib.contextmanager
  def device(self, device_name_or_function):
    """Returns a context manager that specifies the default device to use.

    The `device_name_or_function` argument may either be a device name
    string, a device function, or None:

    * If it is a device name string, all operations constructed in
      this context will be assigned to the device with that name, unless
      overridden by a nested `device()` context.
    * If it is a function, it will be treated as function from
      Operation objects to device name strings, and invoked each time
      a new Operation is created. The Operation will be assigned to
      the device with the returned name.
    * If it is None, all `device()` invocations from the enclosing context
      will be ignored.

    For example:

    ```python
    with g.device('/gpu:0'):
      # All operations constructed in this context will be placed
      # on GPU 0.
      with g.device(None):
        # All operations constructed in this context will have no
        # assigned device.

    # Defines a function from `Operation` to device string.
    def matmul_on_gpu(n):
      if n.type == "MatMul":
        return "/gpu:0"
      else:
        return "/cpu:0"

    with g.device(matmul_on_gpu):
      # All operations of type "MatMul" constructed in this context
      # will be placed on GPU 0; all other operations will be placed
      # on CPU 0.
    ```

    Args:
      device_name_or_function: The device name or function to use in
        the context.

    Returns:
      A context manager that specifies the default device to use for newly
      created ops.
    """
    if (device_name_or_function is not None
        and not callable(device_name_or_function)):
      device_function = pydev.merge_device(device_name_or_function)
    else:
      device_function = device_name_or_function

    try:
      self._device_function_stack.append(device_function)
      yield
    finally:
      self._device_function_stack.pop()

  def _apply_device_functions(self, op):
    """Applies the current device function stack to the given operation."""
    # Apply any device functions in reverse order, so that the most recently
    # pushed function has the first chance to apply a device to the op.
    # We apply here because the result can depend on the Operation's
    # signature, which is computed in the Operation constructor.
    for device_function in reversed(self._device_function_stack):
      if device_function is None:
        break
      op._set_device(device_function(op))

  class _ControlDependenciesController(object):
    """Context manager for `control_dependencies()`."""

    def __init__(self, graph, control_inputs):
      """Create a new `_ControlDependenciesController`.

      A `_ControlDependenciesController` is the context manager for
      `with tf.control_dependencies()` blocks.  These normally nest,
      as described in the documentation for `control_dependencies()`.

      The `control_inputs` argument list control dependencies that must be
      added to the current set of control dependencies.  Because of
      uniquification the set can be empty even if the caller passed a list of
      ops.  The special value `None` indicates that we want to start a new
      empty set of control dependencies instead of extending the current set.

      In that case we also clear the current control flow context, which is an
      additional mechanism to add control dependencies.

      Args:
        graph: The graph that this controller is  managing.
        control_inputs: List of ops to use as control inputs in addition
          to the current control dependencies.  None to indicate that
          the dependencies should be cleared.
      """
      self._graph = graph
      if control_inputs is None:
        self._control_inputs = []
        self._new_stack = True
      else:
        self._control_inputs = control_inputs
        self._new_stack = False
      self._seen_nodes = set()
      self._old_stack = None
      self._old_control_flow_context = None

# pylint: disable=protected-access
    def __enter__(self):
      if self._new_stack:
        # Clear the control_dependencies graph.
        self._old_stack = self._graph._control_dependencies_stack
        self._graph._control_dependencies_stack = []
        # Clear the control_flow_context too.
        self._old_control_flow_context = self._graph._get_control_flow_context()
        self._graph._set_control_flow_context(None)
      self._graph._push_control_dependencies_controller(self)

    def __exit__(self, unused_type, unused_value, unused_traceback):
      self._graph._pop_control_dependencies_controller(self)
      if self._new_stack:
        self._graph._control_dependencies_stack = self._old_stack
        self._graph._set_control_flow_context(self._old_control_flow_context)
# pylint: enable=protected-access

    @property
    def control_inputs(self):
      return self._control_inputs

    def add_op(self, op):
      self._seen_nodes.add(op)

    def op_in_group(self, op):
      return op in self._seen_nodes

  def _push_control_dependencies_controller(self, controller):
    self._control_dependencies_stack.append(controller)

  def _pop_control_dependencies_controller(self, controller):
    assert self._control_dependencies_stack[-1] is controller
    self._control_dependencies_stack.pop()

  def _current_control_dependencies(self):
    ret = set()
    for controller in self._control_dependencies_stack:
      for op in controller.control_inputs:
        ret.add(op)
    return ret

  def _control_dependencies_for_inputs(self, input_tensors):
    """For an op that takes `input_tensors` as inputs, compute control inputs.

    The returned control dependencies should yield an execution that
    is equivalent to adding all control inputs in
    self._control_dependencies_stack to a newly created op. However,
    this function attempts to prune the returned control dependencies
    by observing that nodes created within the same `with
    control_dependencies(...):` block may have data dependencies that make
    the explicit approach redundant.

    Args:
      input_tensors: The direct data dependencies for an op to be created.

    Returns:
      A list of control inputs for the op to be created.
    """
    ret = []
    input_ops = set([t.op for t in input_tensors])
    for controller in self._control_dependencies_stack:
      # If any of the input_ops already depends on the inputs from controller,
      # we say that the new op is dominated (by that input), and we therefore
      # do not need to add control dependences for this controller's inputs.
      dominated = False
      for op in input_ops:
        if controller.op_in_group(op):
          dominated = True
          break
      if not dominated:
        # Don't add a control input if we already have a data dependency on i.
        # NOTE(mrry): We do not currently track transitive data dependencies,
        #   so we may add redundant control inputs.
        ret.extend([c for c in controller.control_inputs if c not in input_ops])
    return ret

  def _record_op_seen_by_control_dependencies(self, op):
    """Record that the given op depends on all registered control dependencies.

    Args:
      op: An Operation.
    """
    for controller in self._control_dependencies_stack:
      controller.add_op(op)

  def control_dependencies(self, control_inputs):
    """Returns a context manager that specifies control dependencies.

    Use with the `with` keyword to specify that all operations constructed
    within the context should have control dependencies on
    `control_inputs`. For example:

    ```python
    with g.control_dependencies([a, b, c]):
      # `d` and `e` will only run after `a`, `b`, and `c` have executed.
      d = ...
      e = ...
    ```

    Multiple calls to `control_dependencies()` can be nested, and in
    that case a new `Operation` will have control dependencies on the union
    of `control_inputs` from all active contexts.

    ```python
    with g.control_dependencies([a, b]):
      # Ops constructed here run after `a` and `b`.
      with g.control_dependencies([c, d]):
        # Ops constructed here run after `a`, `b`, `c`, and `d`.
    ```

    You can pass None to clear the control dependencies:

    ```python
    with g.control_dependencies([a, b]):
      # Ops constructed here run after `a` and `b`.
      with g.control_dependencies(None):
        # Ops constructed here run normally, not waiting for either `a` or `b`.
        with g.control_dependencies([c, d]):
          # Ops constructed here run after `c` and `d`, also not waiting
          # for either `a` or `b`.
    ```

    *N.B.* The control dependencies context applies *only* to ops that
    are constructed within the context. Merely using an op or tensor
    in the context does not add a control dependency. The following
    example illustrates this point:

    ```python
    # WRONG
    def my_func(pred, tensor):
      t = tf.matmul(tensor, tensor)
      with tf.control_dependencies([pred]):
        # The matmul op is created outside the context, so no control
        # dependency will be added.
        return t

    # RIGHT
    def my_func(pred, tensor):
      with tf.control_dependencies([pred]):
        # The matmul op is created in the context, so a control dependency
        # will be added.
        return tf.matmul(tensor, tensor)
    ```

    Args:
      control_inputs: A list of `Operation` or `Tensor` objects which
        must be executed or computed before running the operations
        defined in the context.  Can also be `None` to clear the control
        dependencies.

    Returns:
     A context manager that specifies control dependencies for all
     operations constructed within the context.

    Raises:
      TypeError: If `control_inputs` is not a list of `Operation` or
        `Tensor` objects.
    """
    if control_inputs is None:
      return self._ControlDependenciesController(self, None)
    # First convert the inputs to ops, and deduplicate them.
    # NOTE(mrry): Other than deduplication, we do not currently track direct
    #   or indirect dependencies between control_inputs, which may result in
    #   redundant control inputs.
    control_ops = []
    current = self._current_control_dependencies()
    for c in control_inputs:
      c = self.as_graph_element(c)
      if isinstance(c, Tensor):
        c = c.op
      elif not isinstance(c, Operation):
        raise TypeError("Control input must be Operation or Tensor: %s" % c)
      if c not in current:
        control_ops.append(c)
        current.add(c)
    return self._ControlDependenciesController(self, control_ops)

  # pylint: disable=g-doc-return-or-yield
  @contextlib.contextmanager
  def _kernel_label_map(self, op_to_kernel_label_map):
    """EXPERIMENTAL: A context manager for setting kernel labels.

    This context manager can be used to select particular
    implementations of kernels within the scope of the context.

    For example:

        with ops.Graph().as_default() as g:
          f_1 = Foo()  # Uses the default registered kernel for the Foo op.
          with g.kernel_label_map({"Foo": "v_2"}):
            f_2 = Foo()  # Uses the registered kernel with label "v_2"
                         # for the Foo op.
            with g.kernel_label_map({"Foo": "v_3"}):
              f_3 = Foo()  # Uses the registered kernel with label "v_3"
                           # for the Foo op.
              with g.kernel_label_map({"Foo": ""}):
                f_4 = Foo()  # Uses the default registered kernel
                             # for the Foo op.

    Args:
      op_to_kernel_label_map: A dictionary mapping op type strings to
        kernel label strings.

    Returns:
      A context manager that sets the kernel label to be used for one or more
      ops created in that context.

    Raises:
      TypeError: If op_to_kernel_label_map is not a dictionary mapping
        strings to strings.
    """
    if not isinstance(op_to_kernel_label_map, dict):
      raise TypeError("op_to_kernel_label_map must be a dictionary mapping "
                      "strings to strings")
    # The saved_labels dictionary stores any currently-set labels that
    # will be overridden by this context manager.
    saved_labels = {}
    # Install the given label
    for op_type, label in op_to_kernel_label_map.items():
      if not (isinstance(op_type, six.string_types)
              and isinstance(label, six.string_types)):
        raise TypeError("op_to_kernel_label_map must be a dictionary mapping "
                        "strings to strings")
      try:
        saved_labels[op_type] = self._op_to_kernel_label_map[op_type]
      except KeyError:
        pass
      self._op_to_kernel_label_map[op_type] = label
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the labels set for this context, and restore any saved labels.
      for op_type, label in op_to_kernel_label_map.items():
        try:
          self._op_to_kernel_label_map[op_type] = saved_labels[op_type]
        except KeyError:
          del self._op_to_kernel_label_map[op_type]
  # pylint: enable=g-doc-return-or-yield

  # pylint: disable=g-doc-return-or-yield
  @contextlib.contextmanager
  def gradient_override_map(self, op_type_map):
    """EXPERIMENTAL: A context manager for overriding gradient functions.

    This context manager can be used to override the gradient function
    that will be used for ops within the scope of the context.

    For example:

    ```python
    @tf.RegisterGradient("CustomSquare")
    def _custom_square_grad(op, inputs):
      # ...

    with tf.Graph().as_default() as g:
      c = tf.constant(5.0)
      s_1 = tf.square(c)  # Uses the default gradient for tf.square.
      with g.gradient_override_map({"Square": "CustomSquare"}):
        s_2 = tf.square(s_2)  # Uses _custom_square_grad to compute the
                              # gradient of s_2.
    ```

    Args:
      op_type_map: A dictionary mapping op type strings to alternative op
        type strings.

    Returns:
      A context manager that sets the alternative op type to be used for one
      or more ops created in that context.

    Raises:
      TypeError: If `op_type_map` is not a dictionary mapping strings to
        strings.
    """
    if not isinstance(op_type_map, dict):
      raise TypeError("op_type_map must be a dictionary mapping "
                      "strings to strings")
    # The saved_mappings dictionary stores any currently-set mappings that
    # will be overridden by this context manager.
    saved_mappings = {}
    # Install the given label
    for op_type, mapped_op_type in op_type_map.items():
      if not (isinstance(op_type, six.string_types)
              and isinstance(mapped_op_type, six.string_types)):
        raise TypeError("op_type_map must be a dictionary mapping "
                        "strings to strings")
      try:
        saved_mappings[op_type] = self._gradient_override_map[op_type]
      except KeyError:
        pass
      self._gradient_override_map[op_type] = mapped_op_type
    try:
      yield  # The code within the context runs here.
    finally:
      # Remove the labels set for this context, and restore any saved labels.
      for op_type, mapped_op_type in op_type_map.items():
        try:
          self._gradient_override_map[op_type] = saved_mappings[op_type]
        except KeyError:
          del self._gradient_override_map[op_type]
  # pylint: enable=g-doc-return-or-yield


def device(dev):
  """Wrapper for `Graph.device()` using the default graph.

  See
  [`Graph.device()`](../../api_docs/python/framework.md#Graph.device)
  for more details.

  Args:
    device_name_or_function: The device name or function to use in
      the context.

  Returns:
    A context manager that specifies the default device to use for newly
    created ops.
  """
  return get_default_graph().device(dev)


def name_scope(name):
  """Wrapper for `Graph.name_scope()` using the default graph.

  See
  [`Graph.name_scope()`](../../api_docs/python/framework.md#Graph.name_scope)
  for more details.

  Args:
    name: A name for the scope.

  Returns:
    A context manager that installs `name` as a new name scope in the
    default graph.
  """
  return get_default_graph().name_scope(name)


def control_dependencies(control_inputs):
  """Wrapper for `Graph.control_dependencies()` using the default graph.

  See [`Graph.control_dependencies()`](../../api_docs/python/framework.md#Graph.control_dependencies)
  for more details.

  Args:
    control_inputs: A list of `Operation` or `Tensor` objects which
      must be executed or computed before running the operations
      defined in the context.  Can also be `None` to clear the control
      dependencies.

  Returns:
   A context manager that specifies control dependencies for all
   operations constructed within the context.
  """
  return get_default_graph().control_dependencies(control_inputs)


class _DefaultStack(threading.local):
  """A thread-local stack of objects for providing implicit defaults."""

  def __init__(self):
    super(_DefaultStack, self).__init__()
    self.stack = []

  def get_default(self):
    return self.stack[-1] if len(self.stack) >= 1 else None

  def reset(self):
    self.stack = []

  @contextlib.contextmanager
  def get_controller(self, default):
    """A context manager for manipulating a default stack."""
    try:
      self.stack.append(default)
      yield default
    finally:
      assert self.stack[-1] is default
      self.stack.pop()


_default_session_stack = _DefaultStack()


def default_session(session):
  """Python "with" handler for defining a default session.

  This function provides a means of registering a session for handling
  Tensor.eval() and Operation.run() calls. It is primarily intended for use
  by session.Session, but can be used with any object that implements
  the Session.run() interface.

  Use with the "with" keyword to specify that Tensor.eval() and Operation.run()
  invocations within the scope of a block should be executed by a particular
  session.

  The default session applies to the current thread only, so it is always
  possible to inspect the call stack and determine the scope of a default
  session. If you create a new thread, and wish to use the default session
  in that thread, you must explicitly add a "with ops.default_session(sess):"
  block in that thread's function.

  Example:
    The following code examples are equivalent:

    # 1. Using the Session object directly:
    sess = ...
    c = tf.constant(5.0)
    sess.run(c)

    # 2. Using default_session():
    sess = ...
    with ops.default_session(sess):
      c = tf.constant(5.0)
      result = c.eval()

    # 3. Overriding default_session():
    sess = ...
    with ops.default_session(sess):
      c = tf.constant(5.0)
      with ops.default_session(...):
        c.eval(session=sess)

  Args:
    session: The session to be installed as the default session.

  Returns:
    A context manager for the default session.
  """
  return _default_session_stack.get_controller(weakref.ref(session))


def get_default_session():
  """Returns the default session for the current thread.

  The returned `Session` will be the innermost session on which a
  `Session` or `Session.as_default()` context has been entered.

  NOTE: The default session is a property of the current thread. If you
  create a new thread, and wish to use the default session in that
  thread, you must explicitly add a `with sess.as_default():` in that
  thread's function.

  Returns:
    The default `Session` being used in the current thread.
  """
  ref = _default_session_stack.get_default()
  if ref is None:
    # No default session has been registered.
    return None
  else:
    # De-reference ref.
    ret = ref()
    if ret is None:
      # This should never happen with the current session implementations.
      raise RuntimeError("Default session has been garbage collected.")
  return ret


def _eval_using_default_session(tensors, feed_dict, graph, session=None):
  """Uses the default session to evaluate one or more tensors.

  Args:
    tensors: A single Tensor, or a list of Tensor objects.
    feed_dict: A dictionary that maps Tensor objects (or tensor names) to lists,
      numpy ndarrays, TensorProtos, or strings.
    graph: The graph in which the tensors are defined.
    session: (Optional) A different session to use to evaluate "tensors".

  Returns:
    Either a single numpy ndarray if "tensors" is a single tensor; or a list
    of numpy ndarrays that each correspond to the respective element in
    "tensors".

  Raises:
    ValueError: If no default session is available; the default session
      does not have "graph" as its graph; or if "session" is specified,
      and it does not have "graph" as its graph.
  """
  if session is None:
    session = get_default_session()
    if session is None:
      raise ValueError("Cannot evaluate tensor using eval(): No default "
                       "session is registered. Use `with "
                       "sess.as_default()` or pass an explicit session to "
                       "eval(session=sess)")
    if session.graph is not graph:
      raise ValueError("Cannot use the default session to evaluate tensor: "
                       "the tensor's graph is different from the session's "
                       "graph. Pass an explicit session to "
                       "eval(session=sess).")
  else:
    if session.graph is not graph:
      raise ValueError("Cannot use the given session to evaluate tensor: "
                       "the tensor's graph is different from the session's "
                       "graph.")
  return session.run(tensors, feed_dict)


def _run_using_default_session(operation, feed_dict, graph, session=None):
  """Uses the default session to run "operation".

  Args:
    operation: The Operation to be run.
    feed_dict: A dictionary that maps Tensor objects (or tensor names) to lists,
      numpy ndarrays, TensorProtos, or strings.
    graph: The graph in which "operation" is defined.
    session: (Optional) A different session to use to run "operation".

  Raises:
    ValueError: If no default session is available; the default session
      does not have "graph" as its graph; or if "session" is specified,
      and it does not have "graph" as its graph.
  """
  if session is None:
    session = get_default_session()
    if session is None:
      raise ValueError("Cannot execute operation using Run(): No default "
                       "session is registered. Use 'with "
                       "default_session(sess)' or pass an explicit session to "
                       "Run(session=sess)")
    if session.graph is not graph:
      raise ValueError("Cannot use the default session to execute operation: "
                       "the operation's graph is different from the "
                       "session's graph. Pass an explicit session to "
                       "Run(session=sess).")
  else:
    if session.graph is not graph:
      raise ValueError("Cannot use the given session to execute operation: "
                       "the operation's graph is different from the session's "
                       "graph.")
  session.run(operation, feed_dict)


class _DefaultGraphStack(_DefaultStack):
  """A thread-local stack of objects for providing an implicit default graph."""

  def __init__(self):
    super(_DefaultGraphStack, self).__init__()
    self._global_default_graph = None

  def get_default(self):
    """Override that returns a global default if the stack is empty."""
    ret = super(_DefaultGraphStack, self).get_default()
    if ret is None:
      ret = self._GetGlobalDefaultGraph()
    return ret

  def _GetGlobalDefaultGraph(self):
    if self._global_default_graph is None:
      # TODO(mrry): Perhaps log that the default graph is being used, or set
      #   provide some other feedback to prevent confusion when a mixture of
      #   the global default graph and an explicit graph are combined in the
      #   same process.
      self._global_default_graph = Graph()
    return self._global_default_graph

  def reset(self):
    super(_DefaultGraphStack, self).reset()
    self._global_default_graph = None

_default_graph_stack = _DefaultGraphStack()


def reset_default_graph():
  """Clears the default graph stack and resets the global default graph.

  NOTE: The default graph is a property of the current thread. This
  function applies only to the current thread.  Calling this function while
  a `tf.Session` or `tf.InteractiveSession` is active will result in undefined
  behavior. Using any previously created `tf.Operation` or `tf.Tensor` objects
  after calling this function will result in undefined behavior.
  """
  _default_graph_stack.reset()


def get_default_graph():
  """Returns the default graph for the current thread.

  The returned graph will be the innermost graph on which a
  `Graph.as_default()` context has been entered, or a global default
  graph if none has been explicitly created.

  NOTE: The default graph is a property of the current thread. If you
  create a new thread, and wish to use the default graph in that
  thread, you must explicitly add a `with g.as_default():` in that
  thread's function.

  Returns:
    The default `Graph` being used in the current thread.
  """
  return _default_graph_stack.get_default()


def _assert_same_graph(original_item, item):
  """Fail if the 2 items are from different graphs.

  Args:
    original_item: Original item to check against.
    item: Item to check.

  Raises:
    ValueError: if graphs do not match.
  """
  if original_item.graph is not item.graph:
    raise ValueError(
        "%s must be from the same graph as %s." % (item, original_item))


def _get_graph_from_inputs(op_input_list, graph=None):
  """Returns the appropriate graph to use for the given inputs.

  This library method provides a consistent algorithm for choosing the graph
  in which an Operation should be constructed:

  1. If the "graph" is specified explicitly, we validate that all of the inputs
     in "op_input_list" are compatible with that graph.
  2. Otherwise, we attempt to select a graph from the first Operation-
     or Tensor-valued input in "op_input_list", and validate that all other
     such inputs are in the same graph.
  3. If the graph was not specified and it could not be inferred from
     "op_input_list", we attempt to use the default graph.

  Args:
    op_input_list: A list of inputs to an operation, which may include `Tensor`,
      `Operation`, and other objects that may be converted to a graph element.
    graph: (Optional) The explicit graph to use.

  Raises:
    TypeError: If op_input_list is not a list or tuple, or if graph is not a
      Graph.
    ValueError: If a graph is explicitly passed and not all inputs are from it,
      or if the inputs are from multiple graphs, or we could not find a graph
      and there was no default graph.

  Returns:
    The appropriate graph to use for the given inputs.
  """
  op_input_list = tuple(op_input_list)  # Handle generators correctly
  if graph and not isinstance(graph, Graph):
    raise TypeError("Input graph needs to be a Graph: %s" % graph)

  # 1. We validate that all of the inputs are from the same graph. This is
  #    either the supplied graph parameter, or the first one selected from one
  #    the graph-element-valued inputs. In the latter case, we hold onto
  #    that input in original_graph_element so we can provide a more
  #    informative error if a mismatch is found.
  original_graph_element = None
  for op_input in op_input_list:
    # Determine if this is a valid graph_element.
    graph_element = None
    if isinstance(op_input, (Operation, Tensor, SparseTensor, IndexedSlices)):
      graph_element = op_input
    else:
      graph_element = _as_graph_element(op_input)

    if graph_element:
      if not graph:
        original_graph_element = graph_element
        graph = graph_element.graph
      elif original_graph_element:
        _assert_same_graph(original_graph_element, graph_element)
      elif graph_element.graph is not graph:
        raise ValueError(
            "%s is not from the passed-in graph." % graph_element)

  # 2. If all else fails, we use the default graph, which is always there.
  return graph or get_default_graph()


class GraphKeys(object):
  """Standard names to use for graph collections.

  The standard library uses various well-known names to collect and
  retrieve values associated with a graph. For example, the
  `tf.Optimizer` subclasses default to optimizing the variables
  collected under `tf.GraphKeys.TRAINABLE_VARIABLES` if none is
  specified, but it is also possible to pass an explicit list of
  variables.

  The following standard keys are defined:

  * `VARIABLES`: the `Variable` objects that comprise a model, and
    must be saved and restored together. See
    [`tf.all_variables()`](../../api_docs/python/state_ops.md#all_variables)
    for more details.
  * `TRAINABLE_VARIABLES`: the subset of `Variable` objects that will
    be trained by an optimizer. See
    [`tf.trainable_variables()`](../../api_docs/python/state_ops.md#trainable_variables)
    for more details.
  * `SUMMARIES`: the summary `Tensor` objects that have been created in the
    graph. See
    [`tf.merge_all_summaries()`](../../api_docs/python/train.md#merge_all_summaries)
    for more details.
  * `QUEUE_RUNNERS`: the `QueueRunner` objects that are used to
    produce input for a computation. See
    [`tf.start_queue_runners()`](../../api_docs/python/train.md#start_queue_runners)
    for more details.
  * `MOVING_AVERAGE_VARIABLES`: the subset of `Variable` objects that will also
    keep moving averages.  See
    [`tf.moving_average_variables()`](../../api_docs/python/state_ops.md#moving_average_variables)
    for more details.
  * `REGULARIZATION_LOSSES`: regularization losses collected during graph
    construction.
  * `WEIGHTS`: weights inside neural network layers
  * `BIASES`: biases inside neural network layers
  * `ACTIVATIONS`: activations of neural network layers
  """

  # Key to collect Variable objects that must be saved and restored
  # by the model.
  VARIABLES = "variables"
  # Key to collect Variable objects that will be trained by the
  # optimizers.
  TRAINABLE_VARIABLES = "trainable_variables"
  # Key to collect summaries.
  SUMMARIES = "summaries"
  # Key to collect QueueRunners.
  QUEUE_RUNNERS = "queue_runners"
  # Key to collect table initializers.
  TABLE_INITIALIZERS = "table_initializer"
  # Key to collect asset filepaths. An asset represents an external resource
  # like a vocabulary file.
  ASSET_FILEPATHS = "asset_filepaths"
  # Key to collect Variable objects that keep moving averages.
  MOVING_AVERAGE_VARIABLES = "moving_average_variables"
  # Key to collect regularization losses at graph construction.
  REGULARIZATION_LOSSES = "regularization_losses"
  # Key to collect concatenated sharded variables.
  CONCATENATED_VARIABLES = "concatenated_variables"
  # Key to collect savers.
  SAVERS = "savers"
  # Key to collect weights
  WEIGHTS = "weights"
  # Key to collect biases
  BIASES = "biases"
  # Key to collect activations
  ACTIVATIONS = "activations"

  # Key to indicate various ops.
  INIT_OP = "init_op"
  READY_OP = "ready_op"
  GLOBAL_STEP = "global_step"


def add_to_collection(name, value):
  """Wrapper for `Graph.add_to_collection()` using the default graph.

  See [`Graph.add_to_collection()`](../../api_docs/python/framework.md#Graph.add_to_collection)
  for more details.

  Args:
    name: The key for the collection. For example, the `GraphKeys` class
      contains many standard names for collections.
    value: The value to add to the collection.
  """
  get_default_graph().add_to_collection(name, value)


def add_to_collections(names, value):
  """Wrapper for `Graph.add_to_collections()` using the default graph.

  See [`Graph.add_to_collections()`](../../api_docs/python/framework.md#Graph.add_to_collections)
  for more details.

  Args:
    names: The key for the collections. The `GraphKeys` class
      contains many standard names for collections.
    value: The value to add to the collections.
  """
  get_default_graph().add_to_collections(names, value)


def get_collection(key, scope=None):
  """Wrapper for `Graph.get_collection()` using the default graph.

  See [`Graph.get_collection()`](../../api_docs/python/framework.md#Graph.get_collection)
  for more details.

  Args:
    key: The key for the collection. For example, the `GraphKeys` class
      contains many standard names for collections.
    scope: (Optional.) If supplied, the resulting list is filtered to include
      only items whose name begins with this string.

  Returns:
    The list of values in the collection with the given `name`, or
    an empty list if no value has been added to that collection. The
    list contains the values in the order under which they were
    collected.
  """
  return get_default_graph().get_collection(key, scope)


def get_all_collection_keys():
  """Returns a list of collections used in the default graph."""
  return get_default_graph().get_all_collection_keys()


# pylint: disable=g-doc-return-or-yield
@contextlib.contextmanager
def op_scope(values, name, default_name=None):
  """Returns a context manager for use when defining a Python op.

  This context manager validates that the given `values` are from the
  same graph, ensures that that graph is the default graph, and pushes a
  name scope.

  For example, to define a new Python op called `my_op`:

  ```python
  def my_op(a, b, c, name=None):
    with tf.op_scope([a, b, c], name, "MyOp") as scope:
      a = tf.convert_to_tensor(a, name="a")
      b = tf.convert_to_tensor(b, name="b")
      c = tf.convert_to_tensor(c, name="c")
      # Define some computation that uses `a`, `b`, and `c`.
      return foo_op(..., name=scope)
  ```

  Args:
    values: The list of `Tensor` arguments that are passed to the op function.
    name: The name argument that is passed to the op function.
    default_name: The default name to use if the `name` argument is `None`.

  Returns:
    A context manager for use in defining Python ops. Yields the name scope.

  Raises:
    ValueError: if neither `name` nor `default_name` is provided.
  """
  g = _get_graph_from_inputs(values)
  n = default_name if name is None else name
  if n is None:
    raise ValueError(
        "At least one of name (%s) and default_name (%s) must be provided." % (
            name, default_name))
  with g.as_default(), g.name_scope(n) as scope:
    yield scope
# pylint: enable=g-doc-return-or-yield


_proto_function_registry = registry.Registry("proto functions")


def register_proto_function(collection_name, proto_type=None, to_proto=None,
                            from_proto=None):
  """Registers `to_proto` and `from_proto` functions for collection_name.

  `to_proto` function converts a Python object to the corresponding protocol
  buffer, and returns the protocol buffer.

  `from_proto` function converts protocol buffer into a Python object, and
  returns the object..

  Args:
    collection_name: Name of the collection.
    proto_type: Protobuf type, such as `saver_pb2.SaverDef`,
      `variable_pb2.VariableDef`, `queue_runner_pb2.QueueRunnerDef`..
    to_proto: Function that implements Python object to protobuf conversion.
    from_proto: Function that implements protobuf to Python object conversion.
  """
  if to_proto and not callable(to_proto):
    raise TypeError("to_proto must be callable.")
  if from_proto and not callable(from_proto):
    raise TypeError("from_proto must be callable.")

  _proto_function_registry.register((proto_type, to_proto, from_proto),
                                    collection_name)


def get_collection_proto_type(collection_name):
  """Returns the proto_type for collection_name."""
  try:
    return _proto_function_registry.lookup(collection_name)[0]
  except LookupError:
    return None


def get_to_proto_function(collection_name):
  """Returns the to_proto function for collection_name."""
  try:
    return _proto_function_registry.lookup(collection_name)[1]
  except LookupError:
    return None


def get_from_proto_function(collection_name):
  """Returns the from_proto function for collection_name."""
  try:
    return _proto_function_registry.lookup(collection_name)[2]
  except LookupError:
    return None
