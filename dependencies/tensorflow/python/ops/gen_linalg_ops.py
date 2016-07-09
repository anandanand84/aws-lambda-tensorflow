"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


def batch_cholesky(input, name=None):
  r"""Calculates the Cholesky decomposition of a batch of square matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices, with the same constraints as the single matrix Cholesky
  decomposition above. The output is a tensor of the same shape as the input
  containing the Cholesky decompositions for all input submatrices `[..., :, :]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[..., M, M]`.
  """
  return _op_def_lib.apply_op("BatchCholesky", input=input, name=name)


def batch_matrix_determinant(input, name=None):
  r"""Calculates the determinants for a batch of square matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a 1-D tensor containing the determinants
  for all input submatrices `[..., :, :]`.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[...]`.
  """
  return _op_def_lib.apply_op("BatchMatrixDeterminant", input=input,
                              name=name)


def batch_matrix_inverse(input, name=None):
  r"""Calculates the inverse of square invertible matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. The output is a tensor of the same shape as the input
  containing the inverse for all input submatrices `[..., :, :]`.

  The op uses the Cholesky decomposition if the matrices are symmetric positive
  definite and LU decomposition with partial pivoting otherwise.

  If a matrix is not invertible there is no guarantee what the op does. It
  may detect the condition and raise an exception or it may simply return a
  garbage result.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[..., M, M]`.
  """
  return _op_def_lib.apply_op("BatchMatrixInverse", input=input, name=name)


def batch_matrix_solve(matrix, rhs, name=None):
  r"""Solves systems of linear equations. Checks for invertibility.

  Matrix is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices. Rhs is a tensor of shape
  `[..., M, K]`. The output is a tensor shape `[..., M, K]` where each output
  matrix satisfies matrix[..., :, :] * output[..., :, :] = rhs[..., :, :].

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[..., M, M]`.
    rhs: A `Tensor`. Must have the same type as `matrix`.
      Shape is `[..., M, K]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`. Shape is `[..., M, K]`.
  """
  return _op_def_lib.apply_op("BatchMatrixSolve", matrix=matrix, rhs=rhs,
                              name=name)


def batch_matrix_solve_ls(matrix, rhs, l2_regularizer, fast=None, name=None):
  r"""Solves multiple linear least-squares problems.

  `matrix` is a tensor of shape `[..., M, N]` whose inner-most 2 dimensions
  form square matrices. Rhs is a tensor of shape `[..., M, K]`. The output
  is a tensor shape `[..., N, K]` where each output matrix solves each of
  the equations matrix[..., :, :] * output[..., :, :] = rhs[..., :, :] in the
  least squares sense.

  Below we will use the following notation for each pair of
  matrix and right-hand sides in the batch:

  `matrix`=\\(A \in \Re^{m \times n}\\),
  `rhs`=\\(B  \in \Re^{m \times k}\\),
  `output`=\\(X  \in \Re^{n \times k}\\),
  `l2_regularizer`=\\(\lambda\\).

  If `fast` is `True`, then the solution is computed by solving the normal
  equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
  \\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the least-squares
  problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||A Z - B||_F^2 +
  \lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is computed as
  \\(X = A^T (A A^T + \lambda I)^{-1} B\\), which (for \\(\lambda = 0\\)) is the
  minimum-norm solution to the under-determined linear system, i.e.
  \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||Z||_F^2 \\), subject to
  \\(A Z = B\\). Notice that the fast path is only numerically stable when
  \\(A\\) is numerically full rank and has a condition number
  \\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach}}}\\) or\\(\lambda\\) is
  sufficiently large.

  If `fast` is `False` then the solution is computed using the rank revealing QR
  decomposition with column pivoting. This will always compute a least-squares
  solution that minimizes the residual norm \\(||A X - B||_F^2\\), even when
  \\(A\\) is rank deficient or ill-conditioned. Notice: The current version does
  not compute a minimum norm solution. If `fast` is `False` then `l2_regularizer`
  is ignored.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[..., M, N]`.
    rhs: A `Tensor`. Must have the same type as `matrix`.
      Shape is `[..., M, K]`.
    l2_regularizer: A `Tensor` of type `float64`.
    fast: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`. Shape is `[..., N, K]`.
  """
  return _op_def_lib.apply_op("BatchMatrixSolveLs", matrix=matrix, rhs=rhs,
                              l2_regularizer=l2_regularizer, fast=fast,
                              name=name)


def batch_matrix_triangular_solve(matrix, rhs, lower=None, name=None):
  r"""Solves systems of linear equations with upper or lower triangular matrices by

  backsubstitution.

  `matrix` is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions form
  square matrices. If `lower` is `True` then the strictly upper triangular part
  of each inner-most matrix is ignored. If `lower` is False then the strictly
  lower triangular part of each inner-most matrix is ignored. `rhs` is a tensor
  of shape [..., M, K]`.

  The output is a tensor of shape `[..., M, K]`. If `lower` is `True` then the
  output satisfies
  \\(\sum_{k=0}^{i}\\) matrix[..., i, k] * output[..., k, j] = rhs[..., i, j].
  If `lower` is false then the strictly then the output satisfies
  \\(sum_{k=i}^{K-1}\\) matrix[..., i, k] * output[..., k, j] = rhs[..., i, j].

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[..., M, M]`.
    rhs: A `Tensor`. Must have the same type as `matrix`.
      Shape is `[..., M, K]`.
    lower: An optional `bool`. Defaults to `True`.
      Boolean indicating whether matrix is lower or upper triangular.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`. Shape is `[..., M, K]`.
  """
  return _op_def_lib.apply_op("BatchMatrixTriangularSolve", matrix=matrix,
                              rhs=rhs, lower=lower, name=name)


def batch_self_adjoint_eig(input, name=None):
  r"""Calculates the Eigen Decomposition of a batch of square self-adjoint matrices.

  The input is a tensor of shape `[..., M, M]` whose inner-most 2 dimensions
  form square matrices, with the same constraints as the single matrix
  SelfAdjointEig.

  The result is a '[..., M+1, M] matrix with [..., 0,:] containing the
  eigenvalues, and subsequent [...,1:, :] containing the eigenvectors.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[..., M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[..., M+1, M]`.
  """
  return _op_def_lib.apply_op("BatchSelfAdjointEig", input=input, name=name)


def cholesky(input, name=None):
  r"""Calculates the Cholesky decomposition of a square matrix.

  The input has to be symmetric and positive definite. Only the lower-triangular
  part of the input will be used for this operation. The upper-triangular part
  will not be read.

  The result is the lower-triangular matrix of the Cholesky decomposition of the
  input.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[M, M]`.
  """
  return _op_def_lib.apply_op("Cholesky", input=input, name=name)


def matrix_determinant(input, name=None):
  r"""Calculates the determinant of a square matrix.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      A tensor of shape `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    A scalar, equal to the determinant of the input.
  """
  return _op_def_lib.apply_op("MatrixDeterminant", input=input, name=name)


def matrix_inverse(input, name=None):
  r"""Calculates the inverse of a square invertible matrix.

  The op uses the Cholesky decomposition if the matrix is symmetric positive
  definite and LU decomposition with partial pivoting otherwise.

  If the matrix is not invertible there is no guarantee what the op does. It
  may detect the condition and raise an exception or it may simply return a
  garbage result.

  Args:
    input: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`.
    Shape is `[M, M]` containing the matrix inverse of the input.
  """
  return _op_def_lib.apply_op("MatrixInverse", input=input, name=name)


def matrix_solve(matrix, rhs, name=None):
  r"""Solves a system of linear equations. Checks for invertibility.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[M, M]`.
    rhs: A `Tensor`. Must have the same type as `matrix`. Shape is `[M, K]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`.
    Shape is `[M, K]` containing the tensor that solves
    matrix * output = rhs.
  """
  return _op_def_lib.apply_op("MatrixSolve", matrix=matrix, rhs=rhs,
                              name=name)


def matrix_solve_ls(matrix, rhs, l2_regularizer, fast=None, name=None):
  r"""Solves a linear least-squares problem.

  Below we will use the following notation
  `matrix`=\\(A \in \Re^{m \times n}\\),
  `rhs`=\\(B  \in \Re^{m \times k}\\),
  `output`=\\(X  \in \Re^{n \times k}\\),
  `l2_regularizer`=\\(\lambda\\).

  If `fast` is `True`, then the solution is computed by solving the normal
  equations using Cholesky decomposition. Specifically, if \\(m \ge n\\) then
  \\(X = (A^T A + \lambda I)^{-1} A^T B\\), which solves the least-squares
  problem \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||A Z - B||_F^2 +
  \lambda ||Z||_F^2\\). If \\(m \lt n\\) then `output` is computed as
  \\(X = A^T (A A^T + \lambda I)^{-1} B\\),
  which (for \\(\lambda = 0\\)) is the minimum-norm solution to the
  under-determined linear system, i.e.
  \\(X = \mathrm{argmin}_{Z \in \Re^{n \times k}} ||Z||_F^2 \\),
  subject to \\(A Z = B\\).
  Notice that the fast path is only numerically stable when \\(A\\) is
  numerically full rank and has a condition number
  \\(\mathrm{cond}(A) \lt \frac{1}{\sqrt{\epsilon_{mach}}}\\)
  or \\(\lambda\\) is sufficiently large.

  If `fast` is `False` then the solution is computed using the rank revealing QR
  decomposition with column pivoting. This will always compute a least-squares
  solution that minimizes the residual norm \\(||A X - B||_F^2 \\), even when
  \\( A \\) is rank deficient or ill-conditioned. Notice: The current version
  does not compute a minimum norm solution. If `fast` is `False` then
  `l2_regularizer` is ignored.

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[M, N]`.
    rhs: A `Tensor`. Must have the same type as `matrix`. Shape is `[M, K]`.
    l2_regularizer: A `Tensor` of type `float64`.
    fast: An optional `bool`. Defaults to `True`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`.
    Shape is `[N, K]` containing the tensor that solves
    `matrix * output = rhs` in the least-squares sense.
  """
  return _op_def_lib.apply_op("MatrixSolveLs", matrix=matrix, rhs=rhs,
                              l2_regularizer=l2_regularizer, fast=fast,
                              name=name)


def matrix_triangular_solve(matrix, rhs, lower=None, name=None):
  r"""Solves a system of linear equations with an upper or lower triangular matrix by

  backsubstitution.

  `matrix` is a matrix of shape `[M, M]`. If `lower` is `True` then the strictly
  upper triangular part of `matrix` is ignored. If `lower` is False then the
  strictly lower triangular part of `matrix` is ignored. `rhs` is a matrix of
  shape [M, K]`.

  The output is a matrix of shape `[M, K]`. If `lower` is `True` then the output
  satisfies \\(\sum_{k=0}^{i}\\) matrix[i, k] * output[k, j] = rhs[i, j].
  If `lower` is false then output satisfies
  \\(\sum_{k=i}^{K-1}\\) matrix[i, k] * output[k, j] = rhs[i, j].

  Args:
    matrix: A `Tensor`. Must be one of the following types: `float32`, `float64`.
      Shape is `[M, M]`.
    rhs: A `Tensor`. Must have the same type as `matrix`. Shape is `[M, K]`.
    lower: An optional `bool`. Defaults to `True`.
      Boolean indicating whether matrix is lower or upper triangular.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `matrix`. Shape is `[M, K]`.
  """
  return _op_def_lib.apply_op("MatrixTriangularSolve", matrix=matrix, rhs=rhs,
                              lower=lower, name=name)


def self_adjoint_eig(input, name=None):
  r"""Calculates the Eigen Decomposition of a square Self-Adjoint matrix.

  Only the lower-triangular part of the input will be used in this case. The
  upper-triangular part will not be read.

  The result is a M+1 x M matrix whose first row is the eigenvalues, and
  subsequent rows are eigenvectors.

  Args:
    input: A `Tensor`. Must be one of the following types: `float64`, `float32`.
      Shape is `[M, M]`.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `input`. Shape is `[M+1, M]`.
  """
  return _op_def_lib.apply_op("SelfAdjointEig", input=input, name=name)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "BatchCholesky"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
}
op {
  name: "BatchMatrixDeterminant"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "BatchMatrixInverse"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "BatchMatrixSolve"
  input_arg {
    name: "matrix"
    type_attr: "T"
  }
  input_arg {
    name: "rhs"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "BatchMatrixSolveLs"
  input_arg {
    name: "matrix"
    type_attr: "T"
  }
  input_arg {
    name: "rhs"
    type_attr: "T"
  }
  input_arg {
    name: "l2_regularizer"
    type: DT_DOUBLE
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "fast"
    type: "bool"
    default_value {
      b: true
    }
  }
}
op {
  name: "BatchMatrixTriangularSolve"
  input_arg {
    name: "matrix"
    type_attr: "T"
  }
  input_arg {
    name: "rhs"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "lower"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "BatchSelfAdjointEig"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
}
op {
  name: "Cholesky"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
}
op {
  name: "MatrixDeterminant"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "MatrixInverse"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "MatrixSolve"
  input_arg {
    name: "matrix"
    type_attr: "T"
  }
  input_arg {
    name: "rhs"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "MatrixSolveLs"
  input_arg {
    name: "matrix"
    type_attr: "T"
  }
  input_arg {
    name: "rhs"
    type_attr: "T"
  }
  input_arg {
    name: "l2_regularizer"
    type: DT_DOUBLE
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
  attr {
    name: "fast"
    type: "bool"
    default_value {
      b: true
    }
  }
}
op {
  name: "MatrixTriangularSolve"
  input_arg {
    name: "matrix"
    type_attr: "T"
  }
  input_arg {
    name: "rhs"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "lower"
    type: "bool"
    default_value {
      b: true
    }
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_FLOAT
        type: DT_DOUBLE
      }
    }
  }
}
op {
  name: "SelfAdjointEig"
  input_arg {
    name: "input"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
    allowed_values {
      list {
        type: DT_DOUBLE
        type: DT_FLOAT
      }
    }
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
