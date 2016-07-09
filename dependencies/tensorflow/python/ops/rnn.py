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

"""RNN helpers for TensorFlow models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.framework import tensor_util
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import control_flow_ops
from tensorflow.python.ops import logging_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops import tensor_array_ops
from tensorflow.python.ops import variable_scope as vs


def rnn(cell, inputs, initial_state=None, dtype=None,
        sequence_length=None, scope=None):
  """Creates a recurrent neural network specified by RNNCell "cell".

  The simplest form of RNN network generated is:
    state = cell.zero_state(...)
    outputs = []
    for input_ in inputs:
      output, state = cell(input_, state)
      outputs.append(output)
    return (outputs, state)

  However, a few other options are available:

  An initial state can be provided.
  If the sequence_length vector is provided, dynamic calculation is performed.
  This method of calculation does not compute the RNN steps past the maximum
  sequence length of the minibatch (thus saving computational time),
  and properly propagates the state at an example's sequence length
  to the final state output.

  The dynamic calculation performed is, at time t for batch row b,
    (output, state)(b, t) =
      (t >= sequence_length(b))
        ? (zeros(cell.output_size), states(b, sequence_length(b) - 1))
        : cell(input(b, t), state(b, t - 1))

  Args:
    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, cell.input_size].
    initial_state: (optional) An initial state for the RNN.  This must be
      a tensor of appropriate type and shape [batch_size x cell.state_size].
    dtype: (optional) The data type for the initial state.  Required if
      initial_state is not provided.
    sequence_length: Specifies the length of each sequence in inputs.
      An int32 or int64 vector (tensor) size [batch_size].  Values in [0, T).
    scope: VariableScope for the created subgraph; defaults to "RNN".

  Returns:
    A pair (outputs, state) where:
      outputs is a length T list of outputs (one for each input)
      state is the final state

  Raises:
    TypeError: If "cell" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """

  if not isinstance(cell, rnn_cell.RNNCell):
    raise TypeError("cell must be an instance of RNNCell")
  if not isinstance(inputs, list):
    raise TypeError("inputs must be a list")
  if not inputs:
    raise ValueError("inputs must not be empty")

  outputs = []
  with vs.variable_scope(scope or "RNN"):
    fixed_batch_size = inputs[0].get_shape().with_rank_at_least(1)[0]
    if fixed_batch_size.value:
      batch_size = fixed_batch_size.value
    else:
      batch_size = array_ops.shape(inputs[0])[0]
    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, dtype must be.")
      state = cell.zero_state(batch_size, dtype)

    if sequence_length is not None:
      sequence_length = math_ops.to_int32(sequence_length)

    if sequence_length:  # Prepare variables
      zero_output = array_ops.zeros(
          array_ops.pack([batch_size, cell.output_size]), inputs[0].dtype)
      zero_output.set_shape(
          tensor_shape.TensorShape([fixed_batch_size.value, cell.output_size]))
      min_sequence_length = math_ops.reduce_min(sequence_length)
      max_sequence_length = math_ops.reduce_max(sequence_length)

    for time, input_ in enumerate(inputs):
      if time > 0: vs.get_variable_scope().reuse_variables()
      # pylint: disable=cell-var-from-loop
      call_cell = lambda: cell(input_, state)
      # pylint: enable=cell-var-from-loop
      if sequence_length:
        (output, state) = _rnn_step(
            time, sequence_length, min_sequence_length, max_sequence_length,
            zero_output, state, call_cell)
      else:
        (output, state) = call_cell()

      outputs.append(output)

    return (outputs, state)


def state_saving_rnn(cell, inputs, state_saver, state_name,
                     sequence_length=None, scope=None):
  """RNN that accepts a state saver for time-truncated RNN calculation.

  Args:
    cell: An instance of RNNCell.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, cell.input_size].
    state_saver: A state saver object with methods `state` and `save_state`.
    state_name: The name to use with the state_saver.
    sequence_length: (optional) An int32/int64 vector size [batch_size].
      See the documentation for rnn() for more details about sequence_length.
    scope: VariableScope for the created subgraph; defaults to "RNN".

  Returns:
    A pair (outputs, state) where:
      outputs is a length T list of outputs (one for each input)
      states is the final state

  Raises:
    TypeError: If "cell" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """
  initial_state = state_saver.state(state_name)
  (outputs, state) = rnn(cell, inputs, initial_state=initial_state,
                         sequence_length=sequence_length, scope=scope)
  save_state = state_saver.save_state(state_name, state)
  with ops.control_dependencies([save_state]):
    outputs[-1] = array_ops.identity(outputs[-1])

  return (outputs, state)


def _rnn_step(
    time, sequence_length, min_sequence_length, max_sequence_length,
    zero_output, state, call_cell):
  """Calculate one step of a dynamic RNN minibatch.

  Returns an (output, state) pair conditioned on the sequence_lengths.
  The pseudocode is something like:

  if t >= max_sequence_length:
    return (zero_output, state)
  if t < min_sequence_length:
    return call_cell()

  # Selectively output zeros or output, old state or new state depending
  # on if we've finished calculating each row.
  new_output, new_state = call_cell()
  final_output = np.vstack([
    zero_output if time >= sequence_lengths[r] else new_output_r
    for r, new_output_r in enumerate(new_output)
  ])
  final_state = np.vstack([
    state[r] if time >= sequence_lengths[r] else new_state_r
    for r, new_state_r in enumerate(new_state)
  ])
  return (final_output, final_state)

  Args:
    time: Python int, the current time step
    sequence_length: int32 `Tensor` vector of size [batch_size]
    min_sequence_length: int32 `Tensor` scalar, min of sequence_length
    max_sequence_length: int32 `Tensor` scalar, max of sequence_length
    zero_output: `Tensor` vector of shape [output_size]
    state: `Tensor` matrix of shape [batch_size, state_size]
    call_cell: lambda returning tuple of (new_output, new_state) where
      new_output is a `Tensor` matrix of shape [batch_size, output_size]
      new_state is a `Tensor` matrix of shape [batch_size, state_size]

  Returns:
    A tuple of (final_output, final_state) as given by the pseudocode above:
      final_output is a `Tensor` matrix of shape [batch_size, output_size]
      final_state is a `Tensor` matrix of shape [batch_size, state_size]
  """
  # Step 1: determine whether we need to call_cell or not
  empty_update = lambda: (zero_output, state)
  state_shape = state.get_shape()
  output, new_state = control_flow_ops.cond(
      time < max_sequence_length, call_cell, empty_update)

  # Step 2: determine whether we need to copy through state and/or outputs
  existing_output_state = lambda: (output, new_state)

  def copy_through():
    # Use broadcasting select to determine which values should get
    # the previous state & zero output, and which values should get
    # a calculated state & output.
    copy_cond = (time >= sequence_length)
    return (math_ops.select(copy_cond, zero_output, output),
            math_ops.select(copy_cond, state, new_state))

  (output, state) = control_flow_ops.cond(
      time < min_sequence_length, existing_output_state, copy_through)
  output.set_shape(zero_output.get_shape())
  state.set_shape(state_shape)
  return (output, state)


def _reverse_seq(input_seq, lengths):
  """Reverse a list of Tensors up to specified lengths.

  Args:
    input_seq: Sequence of seq_len tensors of dimension (batch_size, depth)
    lengths:   A tensor of dimension batch_size, containing lengths for each
               sequence in the batch. If "None" is specified, simply reverses
               the list.

  Returns:
    time-reversed sequence
  """
  if lengths is None:
    return list(reversed(input_seq))

  for input_ in input_seq:
    input_.set_shape(input_.get_shape().with_rank(2))

  # Join into (time, batch_size, depth)
  s_joined = array_ops.pack(input_seq)

  # Reverse along dimension 0
  s_reversed = array_ops.reverse_sequence(s_joined, lengths, 0, 1)
  # Split again into list
  result = array_ops.unpack(s_reversed)
  return result


def bidirectional_rnn(cell_fw, cell_bw, inputs,
                      initial_state_fw=None, initial_state_bw=None,
                      dtype=None, sequence_length=None, scope=None):
  """Creates a bidirectional recurrent neural network.

  Similar to the unidirectional case above (rnn) but takes input and builds
  independent forward and backward RNNs with the final forward and backward
  outputs depth-concatenated, such that the output will have the format
  [time][batch][cell_fw.output_size + cell_bw.output_size]. The input_size of
  forward and backward cell must match. The initial state for both directions
  is zero by default (but can be set optionally) and no intermediate states are
  ever returned -- the network is fully unrolled for the given (passed in)
  length(s) of the sequence(s) or completely unrolled if length(s) is not given.

  Args:
    cell_fw: An instance of RNNCell, to be used for forward direction.
    cell_bw: An instance of RNNCell, to be used for backward direction.
    inputs: A length T list of inputs, each a tensor of shape
      [batch_size, cell.input_size].
    initial_state_fw: (optional) An initial state for the forward RNN.
      This must be a tensor of appropriate type and shape
      [batch_size x cell.state_size].
    initial_state_bw: (optional) Same as for initial_state_fw.
    dtype: (optional) The data type for the initial state.  Required if either
      of the initial states are not provided.
    sequence_length: (optional) An int32/int64 vector, size [batch_size],
      containing the actual lengths for each of the sequences.
    scope: VariableScope for the created subgraph; defaults to "BiRNN"

  Returns:
    A set of output `Tensors` where:
      outputs is a length T list of outputs (one for each input), which
      are depth-concatenated forward and backward outputs

  Raises:
    TypeError: If "cell_fw" or "cell_bw" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """

  if not isinstance(cell_fw, rnn_cell.RNNCell):
    raise TypeError("cell_fw must be an instance of RNNCell")
  if not isinstance(cell_bw, rnn_cell.RNNCell):
    raise TypeError("cell_bw must be an instance of RNNCell")
  if not isinstance(inputs, list):
    raise TypeError("inputs must be a list")
  if not inputs:
    raise ValueError("inputs must not be empty")

  name = scope or "BiRNN"
  # Forward direction
  with vs.variable_scope(name + "_FW"):
    output_fw, _ = rnn(cell_fw, inputs, initial_state_fw, dtype,
                       sequence_length)

  # Backward direction
  with vs.variable_scope(name + "_BW"):
    tmp, _ = rnn(cell_bw, _reverse_seq(inputs, sequence_length),
                 initial_state_bw, dtype, sequence_length)
  output_bw = _reverse_seq(tmp, sequence_length)
  # Concat each of the forward/backward outputs
  outputs = [array_ops.concat(1, [fw, bw])
             for fw, bw in zip(output_fw, output_bw)]

  return outputs


def dynamic_rnn(cell, inputs, sequence_length, initial_state=None, dtype=None,
                parallel_iterations=None, time_major=False, scope=None):
  """Creates a recurrent neural network specified by RNNCell "cell".

  This function is functionally identical to the function `rnn` above, but
  performs fully dynamic unrolling of `inputs`.

  Unlike `rnn`, the input `inputs` is not a Python list of `Tensors`.  Instead,
  it is a single `Tensor` where the maximum time is either the first or second
  dimension (see the parameter `time_major`).  The corresponding output is
  a single `Tensor` having the same number of time steps and batch size.

  The parameter `sequence_length` is required and dynamic calculation is
  automatically performed.

  Args:
    cell: An instance of RNNCell.
    inputs: The RNN inputs.
      If time_major == False (default), this must be a tensor of shape:
        `[batch_size, max_time, cell.input_size]`.
      If time_major == True, this must be a tensor of shape:
        `[max_time, batch_size, cell.input_size]`.
    sequence_length: An int32/int64 vector (tensor) size [batch_size].
    initial_state: (optional) An initial state for the RNN.  This must be
      a tensor of appropriate type and shape [batch_size x cell.state_size].
    dtype: (optional) The data type for the initial state.  Required if
      initial_state is not provided.
    parallel_iterations: (Default: 32).  The number of iterations to run in
      parallel.  Those operations which do not have any temporal dependency
      and can be run in parallel, will be.  This parameter trades off
      time for space.  Values >> 1 use more memory but take less time,
      while smaller values use less memory but computations take longer.
    time_major: The shape format of the `inputs` and `outputs` Tensors.
      If true, these `Tensors` must be shaped `[max_time, batch_size, depth]`.
      If false, these `Tensors` must be shaped `[batch_size, max_time, depth]`.
      Using time_major = False is a bit more efficient because it avoids
      transposes at the beginning and end of the RNN calculation.  However,
      most TensorFlow data is batch-major, so by default this function
      accepts input and emits output in batch-major form.
    scope: VariableScope for the created subgraph; defaults to "RNN".

  Returns:
    A pair (outputs, state) where:
      outputs: The RNN output `Tensor`.
        If time_major == False (default), this will be a `Tensor` shaped:
          `[batch_size, max_time, cell.output_size]`.
        If time_major == True, this will be a `Tensor` shaped:
          `[max_time, batch_size, cell.output_size]`.
      state: The final state, shaped:
        `[batch_size, cell.state_size]`.

  Raises:
    TypeError: If "cell" is not an instance of RNNCell.
    ValueError: If inputs is None or an empty list.
  """

  if not isinstance(cell, rnn_cell.RNNCell):
    raise TypeError("cell must be an instance of RNNCell")

  # By default, time_major==False and inputs are batch-major: shaped
  #   [batch, time, depth]
  # For internal calculations, we transpose to [time, batch, depth]
  if not time_major:
    inputs = array_ops.transpose(inputs, [1, 0, 2])  # (B,T,D) => (T,B,D)

  parallel_iterations = parallel_iterations or 32
  sequence_length = math_ops.to_int32(sequence_length)
  sequence_length = array_ops.identity(sequence_length, name="sequence_length")

  with vs.variable_scope(scope or "RNN"):
    input_shape = array_ops.shape(inputs)
    batch_size = input_shape[1]

    if initial_state is not None:
      state = initial_state
    else:
      if not dtype:
        raise ValueError("If no initial_state is provided, dtype must be.")
      state = cell.zero_state(batch_size, dtype)

    def _assert_has_shape(x, shape):
      x_shape = array_ops.shape(x)
      packed_shape = array_ops.pack(shape)
      return logging_ops.Assert(
          math_ops.reduce_all(math_ops.equal(x_shape, packed_shape)),
          ["Expected shape for Tensor %s is " % x.name,
           packed_shape, " but saw shape: ", x_shape])

    # Perform some shape validation
    with ops.control_dependencies(
        [_assert_has_shape(sequence_length, [batch_size])]):
      sequence_length = array_ops.identity(sequence_length, name="CheckSeqLen")

    (outputs, final_state) = _dynamic_rnn_loop(
        cell, inputs, state, sequence_length,
        parallel_iterations=parallel_iterations)

    # Outputs of _dynamic_rnn_loop are always shaped [time, batch, depth].
    # If we are performing batch-major calculations, transpose output back
    # to shape [batch, time, depth]
    if not time_major:
      outputs = array_ops.transpose(outputs, [1, 0, 2])  # (T,B,D) => (B,T,D)

    return (outputs, final_state)


def _dynamic_rnn_loop(cell, inputs, initial_state, sequence_length,
                      parallel_iterations):
  """Internal implementation of Dynamic RNN.

  Args:
    cell: An instance of RNNCell.
    inputs: A `Tensor` of shape [time, batch_size, depth].
    initial_state: A `Tensor` of shape [batch_size, depth].
    sequence_length: An `int32` `Tensor` of shape [batch_size].
    parallel_iterations: Positive Python int.

  Returns:
    Tuple (final_outputs, final_state).
    final_outputs:
      A `Tensor` of shape [time, batch_size, depth]`.
    final_state:
      A `Tensor` of shape [batch_size, depth].
  """
  state = initial_state
  assert isinstance(parallel_iterations, int), "parallel_iterations must be int"

  # Construct an initial output
  input_shape = array_ops.shape(inputs)
  (time_steps, batch_size, unused_depth) = array_ops.unpack(input_shape, 3)

  # Prepare dynamic conditional copying of state & output
  zero_output = array_ops.zeros(
      array_ops.pack([batch_size, cell.output_size]), inputs.dtype)
  min_sequence_length = math_ops.reduce_min(sequence_length)
  max_sequence_length = math_ops.reduce_max(sequence_length)

  time = array_ops.constant(0, dtype=dtypes.int32, name="time")

  output_ta = tensor_array_ops.TensorArray(
      dtype=inputs.dtype, size=time_steps,
      tensor_array_name="dynamic_rnn_output")

  input_ta = tensor_array_ops.TensorArray(
      dtype=inputs.dtype, size=time_steps,
      tensor_array_name="dynamic_rnn_input")

  input_ta = input_ta.unpack(inputs)

  def _time_step(time, state, output_ta_t):
    input_t = input_ta.read(time)

    (output, new_state) = _rnn_step(
        time, sequence_length, min_sequence_length, max_sequence_length,
        zero_output, state, lambda: cell(input_t, state))

    output_ta_t = output_ta_t.write(time, output)

    return (time + 1, new_state, output_ta_t)

  (unused_final_time, final_state, output_final_ta) = control_flow_ops.While(
      cond=lambda time, _1, _2: time < time_steps,
      body=_time_step,
      loop_vars=(time, state, output_ta),
      parallel_iterations=parallel_iterations)

  final_outputs = output_final_ta.pack()

  return (final_outputs, final_state)
