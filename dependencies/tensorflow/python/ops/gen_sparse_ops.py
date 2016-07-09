"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


def _deserialize_many_sparse(serialized_sparse, dtype, name=None):
  r"""Deserialize and concatenate `SparseTensors` from a serialized minibatch.

  The input `serialized_sparse` must be a string matrix of shape `[N x 3]` where
  `N` is the minibatch size and the rows correspond to packed outputs of
  `SerializeSparse`.  The ranks of the original `SparseTensor` objects
  must all match.  When the final `SparseTensor` is created, it has rank one
  higher than the ranks of the incoming `SparseTensor` objects
  (they have been concatenated along a new row dimension).

  The output `SparseTensor` object's shape values for all dimensions but the
  first are the max across the input `SparseTensor` objects' shape values
  for the corresponding dimensions.  Its first shape value is `N`, the minibatch
  size.

  The input `SparseTensor` objects' indices are assumed ordered in
  standard lexicographic order.  If this is not the case, after this
  step run `SparseReorder` to restore index ordering.

  For example, if the serialized input is a `[2 x 3]` matrix representing two
  original `SparseTensor` objects:

      index = [ 0]
              [10]
              [20]
      values = [1, 2, 3]
      shape = [50]

  and

      index = [ 2]
              [10]
      values = [4, 5]
      shape = [30]

  then the final deserialized `SparseTensor` will be:

      index = [0  0]
              [0 10]
              [0 20]
              [1  2]
              [1 10]
      values = [1, 2, 3, 4, 5]
      shape = [2 50]

  Args:
    serialized_sparse: A `Tensor` of type `string`.
      2-D, The `N` serialized `SparseTensor` objects.
      Must have 3 columns.
    dtype: A `tf.DType`. The `dtype` of the serialized `SparseTensor` objects.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (sparse_indices, sparse_values, sparse_shape).
    sparse_indices: A `Tensor` of type `int64`.
    sparse_values: A `Tensor` of type `dtype`.
    sparse_shape: A `Tensor` of type `int64`.
  """
  return _op_def_lib.apply_op("DeserializeManySparse",
                              serialized_sparse=serialized_sparse,
                              dtype=dtype, name=name)


def _serialize_many_sparse(sparse_indices, sparse_values, sparse_shape,
                           name=None):
  r"""Serialize an `N`-minibatch `SparseTensor` into an `[N, 3]` string `Tensor`.

  The `SparseTensor` must have rank `R` greater than 1, and the first dimension
  is treated as the minibatch dimension.  Elements of the `SparseTensor`
  must be sorted in increasing order of this first dimension.  The serialized
  `SparseTensor` objects going into each row of `serialized_sparse` will have
  rank `R-1`.

  The minibatch size `N` is extracted from `sparse_shape[0]`.

  Args:
    sparse_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the minibatch `SparseTensor`.
    sparse_values: A `Tensor`.
      1-D.  The `values` of the minibatch `SparseTensor`.
    sparse_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the minibatch `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  return _op_def_lib.apply_op("SerializeManySparse",
                              sparse_indices=sparse_indices,
                              sparse_values=sparse_values,
                              sparse_shape=sparse_shape, name=name)


def _serialize_sparse(sparse_indices, sparse_values, sparse_shape, name=None):
  r"""Serialize a `SparseTensor` into a string 3-vector (1-D `Tensor`) object.

  Args:
    sparse_indices: A `Tensor` of type `int64`.
      2-D.  The `indices` of the `SparseTensor`.
    sparse_values: A `Tensor`. 1-D.  The `values` of the `SparseTensor`.
    sparse_shape: A `Tensor` of type `int64`.
      1-D.  The `shape` of the `SparseTensor`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `string`.
  """
  return _op_def_lib.apply_op("SerializeSparse",
                              sparse_indices=sparse_indices,
                              sparse_values=sparse_values,
                              sparse_shape=sparse_shape, name=name)


def _sparse_concat(indices, values, shapes, concat_dim, name=None):
  r"""Concatenates a list of `SparseTensor` along the specified dimension.

  Concatenation is with respect to the dense versions of these sparse tensors.
  It is assumed that each input is a `SparseTensor` whose elements are ordered
  along increasing dimension number.

  All inputs' shapes must match, except for the concat dimension.  The
  `indices`, `values`, and `shapes` lists must have the same length.

  The output shape is identical to the inputs', except along the concat
  dimension, where it is the sum of the inputs' sizes along that dimension.

  The output elements will be resorted to preserve the sort order along
  increasing dimension number.

  This op runs in `O(M log M)` time, where `M` is the total number of non-empty
  values across all inputs. This is due to the need for an internal sort in
  order to concatenate efficiently across an arbitrary dimension.

  For example, if `concat_dim = 1` and the inputs are

      sp_inputs[0]: shape = [2, 3]
      [0, 2]: "a"
      [1, 0]: "b"
      [1, 1]: "c"

      sp_inputs[1]: shape = [2, 4]
      [0, 1]: "d"
      [0, 2]: "e"

  then the output will be

      shape = [2, 7]
      [0, 2]: "a"
      [0, 4]: "d"
      [0, 5]: "e"
      [1, 0]: "b"
      [1, 1]: "c"

  Graphically this is equivalent to doing

      [    a] concat [  d e  ] = [    a   d e  ]
      [b c  ]        [       ]   [b c          ]

  Args:
    indices: A list of at least 2 `Tensor` objects of type `int64`.
      2-D.  Indices of each input `SparseTensor`.
    values: A list with the same number of `Tensor` objects as `indices` of `Tensor` objects of the same type.
      1-D.  Non-empty values of each `SparseTensor`.
    shapes: A list with the same number of `Tensor` objects as `indices` of `Tensor` objects of type `int64`.
      1-D.  Shapes of each `SparseTensor`.
    concat_dim: An `int` that is `>= 0`. Dimension to concatenate along.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).
    output_indices: A `Tensor` of type `int64`. 2-D.  Indices of the concatenated `SparseTensor`.
    output_values: A `Tensor`. Has the same type as `values`. 1-D.  Non-empty values of the concatenated `SparseTensor`.
    output_shape: A `Tensor` of type `int64`. 1-D.  Shape of the concatenated `SparseTensor`.
  """
  return _op_def_lib.apply_op("SparseConcat", indices=indices, values=values,
                              shapes=shapes, concat_dim=concat_dim, name=name)


def _sparse_reorder(input_indices, input_values, input_shape, name=None):
  r"""Reorders a SparseTensor into the canonical, row-major ordering.

  Note that by convention, all sparse ops preserve the canonical ordering along
  increasing dimension number. The only time ordering can be violated is during
  manual manipulation of the indices and values vectors to add entries.

  Reordering does not affect the shape of the SparseTensor.

  If the tensor has rank `R` and `N` non-empty values, `input_indices` has
  shape `[N, R]`, input_values has length `N`, and input_shape has length `R`.

  Args:
    input_indices: A `Tensor` of type `int64`.
      2-D.  `N x R` matrix with the indices of non-empty values in a
      SparseTensor, possibly not in canonical ordering.
    input_values: A `Tensor`.
      1-D.  `N` non-empty values corresponding to `input_indices`.
    input_shape: A `Tensor` of type `int64`.
      1-D.  Shape of the input SparseTensor.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values).
    output_indices: A `Tensor` of type `int64`. 2-D.  `N x R` matrix with the same indices as input_indices, but
      in canonical row-major ordering.
    output_values: A `Tensor`. Has the same type as `input_values`. 1-D.  `N` non-empty values corresponding to `output_indices`.
  """
  return _op_def_lib.apply_op("SparseReorder", input_indices=input_indices,
                              input_values=input_values,
                              input_shape=input_shape, name=name)


def _sparse_split(split_dim, indices, values, shape, num_split, name=None):
  r"""Split a `SparseTensor` into `num_split` tensors along one dimension.

  If the `shape[split_dim]` is not an integer multiple of `num_split`. Slices
  `[0 : shape[split_dim] % num_split]` gets one extra dimension.
  For example, if `split_dim = 1` and `num_split = 2` and the input is

      input_tensor = shape = [2, 7]
      [    a   d e  ]
      [b c          ]

  Graphically the output tensors are:

      output_tensor[0] = shape = [2, 4]
      [    a  ]
      [b c    ]

      output_tensor[1] = shape = [2, 3]
      [ d e  ]
      [      ]

  Args:
    split_dim: A `Tensor` of type `int64`.
      0-D.  The dimension along which to split.  Must be in the range
      `[0, rank(shape))`.
    indices: A `Tensor` of type `int64`.
      2-D tensor represents the indices of the sparse tensor.
    values: A `Tensor`. 1-D tensor represents the values of the sparse tensor.
    shape: A `Tensor` of type `int64`.
      1-D. tensor represents the shape of the sparse tensor.
      output indices: A list of 1-D tensors represents the indices of the output
      sparse tensors.
    num_split: An `int` that is `>= 1`. The number of ways to split.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_indices, output_values, output_shape).
    output_indices: A list of `num_split` `Tensor` objects of type `int64`.
    output_values: A list of `num_split` `Tensor` objects of the same type as values. A list of 1-D tensors represents the values of the output sparse
      tensors.
    output_shape: A list of `num_split` `Tensor` objects of type `int64`. A list of 1-D tensors represents the shape of the output sparse
      tensors.
  """
  return _op_def_lib.apply_op("SparseSplit", split_dim=split_dim,
                              indices=indices, values=values, shape=shape,
                              num_split=num_split, name=name)


def _sparse_to_dense(sparse_indices, output_shape, sparse_values,
                     default_value, validate_indices=None, name=None):
  r"""Converts a sparse representation into a dense tensor.

  Builds an array `dense` with shape `output_shape` such that

  ```prettyprint
  # If sparse_indices is scalar
  dense[i] = (i == sparse_indices ? sparse_values : default_value)

  # If sparse_indices is a vector, then for each i
  dense[sparse_indices[i]] = sparse_values[i]

  # If sparse_indices is an n by d matrix, then for each i in [0, n)
  dense[sparse_indices[i][0], ..., sparse_indices[i][d-1]] = sparse_values[i]
  ```

  All other values in `dense` are set to `default_value`.  If `sparse_values` is a
  scalar, all sparse indices are set to this single value.

  Indices should be sorted in lexicographic order, and indices must not
  contain any repeats. If `validate_indices` is true, these properties
  are checked during execution.

  Args:
    sparse_indices: A `Tensor`. Must be one of the following types: `int32`, `int64`.
      0-D, 1-D, or 2-D.  `sparse_indices[i]` contains the complete
      index where `sparse_values[i]` will be placed.
    output_shape: A `Tensor`. Must have the same type as `sparse_indices`.
      1-D.  Shape of the dense output tensor.
    sparse_values: A `Tensor`.
      1-D.  Values corresponding to each row of `sparse_indices`,
      or a scalar value to be used for all sparse indices.
    default_value: A `Tensor`. Must have the same type as `sparse_values`.
      Scalar value to set for indices not specified in
      `sparse_indices`.
    validate_indices: An optional `bool`. Defaults to `True`.
      If true, indices are checked to make sure they are sorted in
      lexicographic order and that there are no repeats.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `sparse_values`.
    Dense output tensor of shape `output_shape`.
  """
  return _op_def_lib.apply_op("SparseToDense", sparse_indices=sparse_indices,
                              output_shape=output_shape,
                              sparse_values=sparse_values,
                              default_value=default_value,
                              validate_indices=validate_indices, name=name)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "DeserializeManySparse"
  input_arg {
    name: "serialized_sparse"
    type: DT_STRING
  }
  output_arg {
    name: "sparse_indices"
    type: DT_INT64
  }
  output_arg {
    name: "sparse_values"
    type_attr: "dtype"
  }
  output_arg {
    name: "sparse_shape"
    type: DT_INT64
  }
  attr {
    name: "dtype"
    type: "type"
  }
}
op {
  name: "SerializeManySparse"
  input_arg {
    name: "sparse_indices"
    type: DT_INT64
  }
  input_arg {
    name: "sparse_values"
    type_attr: "T"
  }
  input_arg {
    name: "sparse_shape"
    type: DT_INT64
  }
  output_arg {
    name: "serialized_sparse"
    type: DT_STRING
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "SerializeSparse"
  input_arg {
    name: "sparse_indices"
    type: DT_INT64
  }
  input_arg {
    name: "sparse_values"
    type_attr: "T"
  }
  input_arg {
    name: "sparse_shape"
    type: DT_INT64
  }
  output_arg {
    name: "serialized_sparse"
    type: DT_STRING
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "SparseConcat"
  input_arg {
    name: "indices"
    type: DT_INT64
    number_attr: "N"
  }
  input_arg {
    name: "values"
    type_attr: "T"
    number_attr: "N"
  }
  input_arg {
    name: "shapes"
    type: DT_INT64
    number_attr: "N"
  }
  output_arg {
    name: "output_indices"
    type: DT_INT64
  }
  output_arg {
    name: "output_values"
    type_attr: "T"
  }
  output_arg {
    name: "output_shape"
    type: DT_INT64
  }
  attr {
    name: "concat_dim"
    type: "int"
    has_minimum: true
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 2
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "SparseReorder"
  input_arg {
    name: "input_indices"
    type: DT_INT64
  }
  input_arg {
    name: "input_values"
    type_attr: "T"
  }
  input_arg {
    name: "input_shape"
    type: DT_INT64
  }
  output_arg {
    name: "output_indices"
    type: DT_INT64
  }
  output_arg {
    name: "output_values"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "SparseSplit"
  input_arg {
    name: "split_dim"
    type: DT_INT64
  }
  input_arg {
    name: "indices"
    type: DT_INT64
  }
  input_arg {
    name: "values"
    type_attr: "T"
  }
  input_arg {
    name: "shape"
    type: DT_INT64
  }
  output_arg {
    name: "output_indices"
    type: DT_INT64
    number_attr: "num_split"
  }
  output_arg {
    name: "output_values"
    type_attr: "T"
    number_attr: "num_split"
  }
  output_arg {
    name: "output_shape"
    type: DT_INT64
    number_attr: "num_split"
  }
  attr {
    name: "num_split"
    type: "int"
    has_minimum: true
    minimum: 1
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "SparseToDense"
  input_arg {
    name: "sparse_indices"
    type_attr: "Tindices"
  }
  input_arg {
    name: "output_shape"
    type_attr: "Tindices"
  }
  input_arg {
    name: "sparse_values"
    type_attr: "T"
  }
  input_arg {
    name: "default_value"
    type_attr: "T"
  }
  output_arg {
    name: "dense"
    type_attr: "T"
  }
  attr {
    name: "validate_indices"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "Tindices"
    type: "type"
    allowed_values {
      list {
        type: DT_INT32
        type: DT_INT64
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
