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

"""## Control Flow Operations

TensorFlow provides several operations and classes that you can use to control
the execution of operations and add conditional dependencies to your graph.

@@identity
@@tuple
@@group
@@no_op
@@count_up_to
@@cond

## Logical Operators

TensorFlow provides several operations that you can use to add logical operators
to your graph.

@@logical_and
@@logical_not
@@logical_or
@@logical_xor

## Comparison Operators

TensorFlow provides several operations that you can use to add comparison
operators to your graph.

@@equal
@@not_equal
@@less
@@less_equal
@@greater
@@greater_equal
@@select
@@where

## Debugging Operations

TensorFlow provides several operations that you can use to validate values and
debug your graph.

@@is_finite
@@is_inf
@@is_nan
@@verify_tensor_all_finite
@@check_numerics
@@add_check_numerics_ops
@@Assert
@@Print
"""
# pylint: disable=g-bad-name
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

import six
from six.moves import xrange  # pylint: disable=redefined-builtin

from tensorflow.python.framework import dtypes
from tensorflow.python.framework import ops
from tensorflow.python.framework import tensor_shape
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import common_shapes
from tensorflow.python.ops import constant_op
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import gen_control_flow_ops
from tensorflow.python.ops import gen_data_flow_ops
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import tensor_array_ops
# pylint: disable=wildcard-import,undefined-variable
from tensorflow.python.ops.gen_control_flow_ops import *
# pylint: enable=wildcard-import
from tensorflow.python.platform import logging


# We override the 'tuple' for a control flow op, so we keep python's
# existing 'tuple' for later use in this module.
_basetuple = tuple


# pylint: disable=protected-access
def _Identity(data, name=None):
  """Return a tensor with the same shape and contents as the input tensor.

  Args:
    data: A Tensor.
    name: A name for this operation (optional).

  Returns:
    A Tensor with the same type and value as the input Tensor.
  """
  if not data.dtype.is_ref_dtype:
    return array_ops.identity(data, name=name)
  else:
    return gen_array_ops._ref_identity(data, name=name)


def _NextIteration(data, name=None):
  if not data.dtype.is_ref_dtype:
    return next_iteration(data, name=name)
  else:
    return ref_next_iteration(data, name=name)


def _Merge(values, name=None):
  if all([v.dtype.is_ref_dtype for v in values]):
    return gen_control_flow_ops._ref_merge(values, name)
  else:
    return gen_control_flow_ops._merge(values, name)


def _Enter(data, frame_name, is_constant=False, parallel_iterations=10,
           use_ref=True, name=None):
  """Creates or finds a child frame, and makes `data` available to it.

  The unique `frame_name` is used by the `Executor` to identify frames. If
  `is_constant` is true, `output` is a constant in the child frame; otherwise
  it may be changed in the child frame. At most `parallel_iterations` iterations
  are run in parallel in the child frame.

  Args:
    data: The tensor to be made available to the child frame.
    frame_name: The name of the child frame.
    is_constant: If true, the output is constant within the child frame.
    parallel_iterations: The number of iterations allowed to run in parallel.
    use_ref: If true, use ref_enter if data is of ref type.
    name: A name for this operation (optional).

  Returns:
    The same tensor as `data`.
  """
  if data.dtype.is_ref_dtype and use_ref:
    return ref_enter(data, frame_name, is_constant, parallel_iterations,
                     name=name)
  else:
    return enter(data, frame_name, is_constant, parallel_iterations,
                 name=name)


def exit(data, name=None):
  """Exits the current frame to its parent frame.

  Exit makes its input `data` available to the parent frame.

  Args:
    data: The tensor to be made available to the parent frame.
    name: A name for this operation (optional).

  Returns:
    The same tensor as `data`.
  """
  if data.dtype.is_ref_dtype:
    return gen_control_flow_ops._ref_exit(data, name)
  else:
    return gen_control_flow_ops._exit(data, name)


def switch(data, pred, dtype=None, name=None):
  """Forwards `data` to an output determined by `pred`.

  If `pred` is true, the `data` input is forwared to the first output.
  Otherwise, the data goes to the second output.

  This op handles `Tensor`s and `IndexedSlices`.

  Args:
    data: The tensor to be forwarded to the appropriate output.
    pred: A scalar that specifies which output port will receive data.
    dtype: Optional element type for the returned tensor. If missing,
           the type is inferred from the type of `value`.
    name: A name for this operation (optional).

  Returns:
    `(output_false, output_true)`: If `pred` is true, data will be forwarded to
    `output_true`, otherwise it goes to `output_false`.
  """
  with ops.op_scope([data, pred], name, "Switch") as name:
    data = ops.convert_to_tensor_or_indexed_slices(data, dtype=dtype,
                                                   name="data")
    pred = ops.convert_to_tensor(pred, name="pred")
    if isinstance(data, ops.Tensor):
      return gen_control_flow_ops._switch(data, pred, name=name)
    else:
      val, ind, dense_shape = data.values, data.indices, data.dense_shape
      val_f, val_t = gen_control_flow_ops._switch(val, pred, name=name)
      ind_f, ind_t = gen_control_flow_ops._switch(ind, pred, name="indices")
      if dense_shape:
        dense_shape_f, dense_shape_t = gen_control_flow_ops._switch(
            dense_shape, pred, name="dense_shape")
      else:
        dense_shape_f, dense_shape_t = None, None
      return (ops.IndexedSlices(val_f, ind_f, dense_shape_f),
              ops.IndexedSlices(val_t, ind_t, dense_shape_t))


def merge(inputs, name=None):
  """Returns the value of an available element of `inputs`.

  This op tests each of the tensors in `inputs` in turn to determine if any of
  them is available. If it finds an available tensor, it returns it and its
  index in `inputs`.

  It is an error if more than one tensor in `inputs` is available. If no tensor
  in `inputs` is available, the returned tensor and index are not set.

  This op handles both `Tensor`s and `IndexedSlices`. If inputs has a mix of
  `Tensor`s and `IndexedSlices`, all inputs are converted to IndexedSlices
  before merging.

  Args:
    inputs: The input tensors, at most one of which is available.
    name: A name for this operation (optional).

  Returns:
    A tuple containing the chosen input tensor and its index in `inputs`.

  Raises:
    ValueError: If inputs are IndexedSlices and some but not all have a
      dense_shape property.
  """
  with ops.op_scope(inputs, name, "Merge") as name:
    inputs = [ops.convert_to_tensor_or_indexed_slices(inp)
              for inp in inputs]
    if all([isinstance(inp, ops.Tensor) for inp in inputs]):
      return _Merge(inputs, name=name)
    else:
      inputs = math_ops._as_indexed_slices_list(inputs)
      values, _ = _Merge([inp.values for inp in inputs], name=name)
      indices, chosen_index = _Merge(
          [inp.indices for inp in inputs], name="indices")
      if any(inp.dense_shape for inp in inputs):
        if not all(inp.dense_shape for inp in inputs):
          raise ValueError("Either all merged IndexedSlices must have a "
                           "dense_shape, or none must have a dense_shape.")
        dense_shape, _ = _Merge(
            [inp.dense_shape for inp in inputs], name="dense_shape")
      else:
        dense_shape = None
      return ops.IndexedSlices(values, indices, dense_shape), chosen_index


def _SwitchRefOrTensor(data, pred, name="Switch"):
  """Forwards `data` to an output determined by `pred`.

  If `pred` is true, the `data` input is forwared to the first output.
  Otherwise, the data goes to the second output.

  This op handles `Tensor`s and `IndexedSlices`.

  Args:
    data: The tensor to be forwarded to the appropriate output.
    pred: A scalar that specifies which output port will receive data.
    name: A name for this operation (optional).

  Returns:
    `(output_false, output_false)`: If `pred` is true, data will be forwarded to
    `output_true`, otherwise it goes to `output_false`.

  Raises:
    TypeError: if data is not a Tensor or IndexedSlices
  """
  data = ops.convert_to_tensor_or_indexed_slices(data, name="data")
  with ops.device(data.device):
    if isinstance(data, ops.Tensor):
      if not data.dtype.is_ref_dtype:
        return switch(data, pred, name=name)
      else:
        return ref_switch(data, pred, name=name)
    else:
      return switch(data, pred, name=name)


def _convert_tensorarrays_to_flows(tensors_or_tensor_arrays):
  return [ta.flow if isinstance(ta, tensor_array_ops.TensorArray)
          else ta
          for ta in tensors_or_tensor_arrays]


def _convert_flows_to_tensorarrays(tensors_or_tensorarrays, tensors_or_flows):
  if len(tensors_or_tensorarrays) != len(tensors_or_flows):
    raise ValueError(
        "Lengths of original Tensor list and new list do not match: %d vs. %d"
        % (len(tensors_or_tensorarrays), len(tensors_or_flows)))
  return [
      tensor_array_ops.TensorArray(
          dtype=ta.dtype, handle=ta.handle, flow=t_or_flow)
      if isinstance(ta, tensor_array_ops.TensorArray)
      else t_or_flow
      for (ta, t_or_flow) in zip(tensors_or_tensorarrays, tensors_or_flows)]


class ControlFlowOpWrapper(object):
  """A wrapper class for Operation.

  A wrapped op allows us to capture the uses of its inputs and outputs. In
  gradients(), right before calling the gradient function of an op, we wrap
  the op by calling MakeWrapper. So during the exection of the gradient
  function of an op , any time when one of its inputs/outputs is used, we
  generate code to remember its values for all iterations.
  """

  class _ControlFlowOpInputs(object):
    """An indirection to capture the input tensors needed in backprop."""

    def __init__(self, op, grad_state):
      self._op = op
      self._grad_state = grad_state
      self._inputs = None

    def __len__(self):
      return len(self._op._inputs)

    def __getitem__(self, index):
      if self._inputs is None:
        self._inputs = [None for _ in self._op.inputs]
      if isinstance(index, int):
        val = self._inputs[index]
        if val is None:
          f_val = self._op.inputs[index]
          val = self._grad_state.GetRealValue(f_val)
          self._inputs[index] = val
        return val
      elif isinstance(index, slice):
        start, stop, step = index.indices(len(self))
        vals = [self[i] for i in xrange(start, stop, step)]
        return vals
      else:
        raise TypeError("index must be an integer or slice")

  class _ControlFlowOpOutputs(object):
    """An indirection to capture the output tensors needed in backprop."""

    def __init__(self, op, grad_state):
      self._op = op
      self._grad_state = grad_state
      self._outputs = None

    def __len__(self):
      return len(self._op._outputs)

    def __getitem__(self, index):
      if self._outputs is None:
        self._outputs = [None for _ in self._op.outputs]
      if isinstance(index, int):
        val = self._outputs[index]
        if val is None:
          f_val = self._op.outputs[index]
          val = self._grad_state.GetRealValue(f_val)
          self._outputs[index] = val
        return val
      elif isinstance(index, slice):
        start, stop, step = index.indices(len(self))
        vals = [self[i] for i in xrange(start, stop, step)]
        return vals
      else:
        raise TypeError("index must be an integer or slice")

  def __init__(self, op, grad_state):
    self._grad_state = grad_state   # The GradLoopState this op belongs to.
    self._op = op
    self._inputs = None
    self._outputs = None

  @property
  def grad_state(self):
    return self._grad_state

  @property
  def inputs(self):
    if self._inputs is None:
      self._inputs = self._ControlFlowOpInputs(self._op, self._grad_state)
    return self._inputs

  @property
  def outputs(self):
    if self._outputs is None:
      self._outputs = self._ControlFlowOpOutputs(self._op, self._grad_state)
    return self._outputs

  @property
  def op(self):
    return self._op

  @property
  def name(self):
    """Returns the name of this instance of op."""
    return self._op.name

  @property
  def _id(self):
    """Returns the unique id of this operation."""
    return self._op._id

  @property
  def device(self):
    """Returns the device of this operation.

    Returns:
      a string or None if the device was not set.
    """
    return self._op.device

  @property
  def type(self):
    """Returns the type of the op."""
    return self._op.type

  @property
  def graph(self):
    """The `Graph` that contains this operation."""
    return self._op.graph

  def get_attr(self, name):
    """Returns the value of the attr of this op with the given `name`."""
    return self._op.get_attr(name)

  def _get_control_flow_context(self):
    """Returns the control flow context of this op."""
    return self._op._get_control_flow_context()


def _IsLoopConstantEnter(op):
  """Returns true iff op is a loop invariant."""
  is_enter = (op.type == "Enter" or op.type == "RefEnter")
  return is_enter and op.get_attr("is_constant")


def _IsLoopExit(op):
  return op.type == "Exit" or op.type == "RefExit"


class GradLoopState(object):
  """The state used for constructing the gradient graph for a while loop.

  We create a GradLoopState for each while loop in forward and its
  corresponding while loop in backprop. This gives us access to both
  the forward and the backprop WhileContexts.

  During the construction of gradient graph, any time when we detect
  a forward value that is needed for backprop, we create a history
  accumulator and add it to `history_map`. Any time when we backprop
  a loop switch op (in _SwitchGrad), we add the grad merge op in
  `switch_map`.
  """

  def __init__(self, forward_ctxt, outer_grad_state):
    # The grad loop state for the outer while loop.
    self._outer_grad_state = None

    # The while loop context for forward.
    self._forward_context = None

    # The loop counter added by AddForwardCounter. It is the value
    # of the loop counter for the next iteration.
    self._forward_index = None

    # A sync op for forward.
    self._forward_sync = None

    # The while loop context for backprop.
    self._grad_context = None

    # The loop counter added by AddBackPropCounter. It is the value
    # of the loop counter for the current iteration.
    self._grad_index = None

    # A sync op for backprop.
    self._grad_sync = None

    # Information needed by backprop.
    self._history_map = {}
    self._switch_map = {}

    self._outer_grad_state = outer_grad_state
    if outer_grad_state:
      outer_forward_ctxt = outer_grad_state.forward_context
    else:
      outer_forward_ctxt = forward_ctxt.outer_context

    # Add the forward loop counter.
    if outer_forward_ctxt: outer_forward_ctxt.Enter()
    cnt, forward_index = forward_ctxt.AddForwardCounter()
    if outer_forward_ctxt: outer_forward_ctxt.Exit()
    self._forward_context = forward_ctxt
    self._forward_index = forward_index

    # Add the backprop WhileContext, and the backprop loop counter.
    if outer_grad_state:
      # This is a nested loop. Remember the iteration counts for each
      # execution of this inner loop.
      outer_forward_ctxt.AddName(cnt.name)
      history_cnt = outer_grad_state.AddForwardAccumulator(cnt)

      outer_grad_ctxt = outer_grad_state.grad_context
      outer_grad_ctxt.Enter()
      self._grad_context = WhileContext(forward_ctxt.parallel_iterations,
                                        forward_ctxt.back_prop,
                                        forward_ctxt.name)
      real_cnt = outer_grad_state.AddBackPropAccumulatedValue(history_cnt, cnt)
      self._grad_index = self._grad_context.AddBackPropCounter(real_cnt)
      outer_grad_ctxt.Exit()
    else:
      if outer_forward_ctxt: outer_forward_ctxt.Enter()
      self._grad_context = WhileContext(forward_ctxt.parallel_iterations,
                                        forward_ctxt.back_prop,
                                        forward_ctxt.name)
      self._grad_index = self._grad_context.AddBackPropCounter(cnt)
      if outer_forward_ctxt: outer_forward_ctxt.Exit()

  @property
  def outer_grad_state(self):
    """The grad loop state for outer loop."""
    return self._outer_grad_state

  @property
  def forward_context(self):
    """The while loop context for forward."""
    return self._forward_context

  @property
  def forward_index(self):
    """The loop index of forward loop."""
    return self._forward_index

  @property
  def forward_sync(self):
    """A control trigger node for synchronization in the forward loop.

    One main use is to keep the push ops of a stack executed in the
    iteration order.
    """
    if self._forward_sync is None:
      with ops.control_dependencies(None):
        self._forward_sync = control_trigger(name="f_sync")
      self._forward_sync._set_control_flow_context(self._forward_context)
      self._forward_index.op._add_control_input(self._forward_sync)
    return self._forward_sync

  @property
  def grad_context(self):
    """The corresponding WhileContext for gradient."""
    return self._grad_context

  @property
  def grad_index(self):
    """The loop index of backprop loop."""
    return self._grad_index

  @property
  def grad_sync(self):
    """A control trigger node for synchronization in the grad loop.

    One main use is to keep the pop ops of a stack executed in the
    iteration order.
    """
    if self._grad_sync is None:
      with ops.control_dependencies(None):
        self._grad_sync = control_trigger(name="b_sync")
      self._grad_sync._set_control_flow_context(self._grad_context)
      self._grad_index.op._add_control_input(self._grad_sync)
    return self._grad_sync

  @property
  def history_map(self):
    """The map that records all the tensors needed for backprop."""
    return self._history_map

  @property
  def switch_map(self):
    """The map that records all the Switch ops for the While loop."""
    return self._switch_map

  def AddForwardAccumulator(self, value, dead_branch=False):
    """Add an accumulator for each forward tensor that is needed in backprop.

    This is added to the forward loop at the first time when a tensor
    in the forward loop is used by backprop gradient computation loop.
    We create an accumulator that accumulates the value of tensor at each
    iteration. Called in the control flow context where gradients() is called.

    The pseudocode is:
    ```
      acc = stack();
      while (_pivot) {
        acc = stack_push(acc, value);
      }
    ```

    We make sure that the stack push op in one iteration is executed before
    next iteration. This is achieved by adding a control edge from
    `forward_index.op.inputs[0].op` to the push op, and another control
    edge from the push op to either `forward_index.op` or `forward_sync`.

    Args:
      value: The tensor that is to be accumulated.
      dead_branch: True iff the tensor is on a dead branch of a cond.

    Returns:
      The stack that contains the accumulated history of the tensor.
    """
    # TODO(yuanbyu): Make sure the colocation of stack ops and value.
    # pylint: disable=protected-access
    acc = gen_data_flow_ops._stack(value.dtype.base_dtype, name="f_acc")
    # pylint: enable=protected-access

    # Make acc available in the forward context.
    enter_acc = self.forward_context.AddValue(acc)

    # Add the stack_push op in the context of value.op.
    value_ctxt = value.op._get_control_flow_context()
    if _IsLoopExit(value.op):
      value_ctxt = value_ctxt.outer_context
    if value_ctxt == self.forward_context:
      # value is not nested in the forward context.
      self.forward_context.Enter()
      push = gen_data_flow_ops._stack_push(enter_acc, value)
      self.forward_context.Exit()
      # Protect stack push and order it before forward_index.
      self.forward_index.op._add_control_input(push.op)
    else:
      # value is in a cond context within the forward context.
      assert isinstance(value_ctxt, CondContext)
      if dead_branch:
        # The special case for creating a zero tensor for a dead
        # branch of a switch. See ControlFlowState.ZerosLike().
        value_ctxt.outer_context.Enter()
        push = gen_data_flow_ops._stack_push(enter_acc, value)
        value_ctxt.outer_context.Exit()
        push.op._set_control_flow_context(value_ctxt)
      else:
        value_ctxt.Enter()
        push = gen_data_flow_ops._stack_push(enter_acc, value)
        value_ctxt.Exit()
      # Protect stack push and order it before forward_sync.
      self.forward_sync._add_control_input(push.op)
    # Order stack push after the successor of forward_index
    add_op = self.forward_index.op.inputs[0].op
    push.op._add_control_input(add_op)
    return acc

  def AddBackPropAccumulatedValue(self, history_value, value,
                                  dead_branch=False):
    """Add the getter for an accumulated value in the grad context.

    This is added to the backprop loop. Called in the grad context to
    get the value of an accumulated value. The stack pop op must be guarded
    by the pred of the controlling cond.

    Args:
      history_value: The history (a stack) of a value.
      value: The value that is pushed onto the stack.
      dead_branch: True iff the tensor is on a dead branch of a cond.

    Returns:
      The current value (the top of the stack).
    """
    history_ctxt = history_value.op._get_control_flow_context()
    # Find the cond context that controls history_value.
    cond_ctxt = None
    value_ctxt = value.op._get_control_flow_context()
    while value_ctxt and value_ctxt != history_ctxt:
      if isinstance(value_ctxt, CondContext):
        cond_ctxt = value_ctxt
        break
      value_ctxt = value_ctxt.outer_context
    if cond_ctxt:
      # Guard stack pop with a switch if it is controlled by a cond
      grad_state = self
      pred = None
      while not pred and grad_state:
        pred = grad_state.history_map.get(cond_ctxt.pred.name)
        grad_state = grad_state.outer_grad_state
      branch = (1 - cond_ctxt.branch) if dead_branch else cond_ctxt.branch
      history_value = _SwitchRefOrTensor(history_value, pred)[branch]
    pop = gen_data_flow_ops._stack_pop(history_value, value.dtype.base_dtype)
    if self.grad_context.parallel_iterations > 1:
      # All pops are ordered after pivot_for_body and before grad_sync.
      self.grad_sync._add_control_input(pop.op)
    return pop

  def GetRealValue(self, value):
    """Get the real value.

    If backprop "uses" a value produced by forward inference, an
    accumulator is added in the forward loop to accumulate its values.
    We use the accumulated value.

    Args:
      value: A tensor to be captured.

    Returns:
      The same tensor value from the saved history.
    """
    assert value.op.type != "Variable"
    real_value = self._history_map.get(value.name)
    if real_value is None:
      if _IsLoopConstantEnter(value.op):
        # Special case for loop invariant.
        if self._outer_grad_state:
          # This is a nested loop so we record the history of this
          # value in outer_forward_ctxt.
          self._grad_context.Exit()
          outer_value = value.op.inputs[0]
          history_value = self._outer_grad_state.AddForwardAccumulator(
              outer_value)
          self._grad_context.Enter()
        else:
          # Just use the input value of this Enter node.
          real_value = GetRealOp(value.op).inputs[0]
      else:
        # Record the history of this value in forward_ctxt.
        # NOTE(yuanbyu): Don't record for constants.
        self._grad_context.Exit()
        history_value = self.AddForwardAccumulator(value)
        self._grad_context.Enter()

      if real_value is None:
        # Add the stack pop op in the grad context.
        real_value = self.AddBackPropAccumulatedValue(history_value, value)
      self._history_map[value.name] = real_value
    return real_value


def _GetWhileContext(op):
  """Get the WhileContext to which this op belongs."""
  ctxt = op._get_control_flow_context()
  if ctxt:
    ctxt = ctxt.GetWhileContext()
  return ctxt


class ControlFlowState(object):
  """Maintain the mapping from the loops to their grad states."""

  def __init__(self):
    self._map = {}   # maps forward loop context to GradLoopState

  def _GetGradState(self, op):
    forward_ctxt = _GetWhileContext(op)
    if forward_ctxt is None:
      return None
    return self._map.get(forward_ctxt)

  def MakeWrapper(self, op):
    """Make a wrapper for op if it is in a WhileContext."""
    grad_state = self._GetGradState(op)
    if grad_state:
      return ControlFlowOpWrapper(op, grad_state)
    return op

  def GetAllLoopExits(self):
    """Return a list containing the exits of all the loops."""
    loop_exits = []
    for forward_ctxt in self._map:
      for loop_exit in forward_ctxt.loop_exits:
        loop_exits.append(loop_exit)
    return loop_exits

  def EnterGradWhileContext(self, op):
    """Enter the WhileContext for gradient computation."""
    grad_state = self._GetGradState(op)
    if grad_state:
      grad_state.grad_context.Enter()

  def ExitGradWhileContext(self, op):
    """Exit the WhileContext for gradient computation."""
    grad_state = self._GetGradState(op)
    if grad_state:
      grad_state.grad_context.Exit()

  def AddWhileContext(self, op, between_op_list, between_ops):
    """Add the grad state for the while loop that op belongs to.

    Note that op is an Exit, and this method must be called in
    the control flow context where gradients() is called.

    Note that this method modifies `between_op_list` and `between_ops`.
    """
    forward_ctxt = _GetWhileContext(op)
    grad_state = self._map.get(forward_ctxt)
    if grad_state is None:
      # This is a new while loop so create a grad state for it.
      outer_forward_ctxt = forward_ctxt.outer_context
      if outer_forward_ctxt:
        outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
      outer_grad_state = None
      if outer_forward_ctxt:
        outer_grad_state = self._map.get(outer_forward_ctxt)
      grad_state = GradLoopState(forward_ctxt, outer_grad_state)
      self._map[forward_ctxt] = grad_state

      # We need to include all exits of a loop for backprop.
      for loop_exit in forward_ctxt.loop_exits:
        if not between_ops[loop_exit.op._id]:
          between_ops[loop_exit.op._id] = True
          between_op_list.append(loop_exit.op)

  def ZerosLikeForExit(self, val):
    """Create zeros_like gradient for a loop exit.

    If the result of a loop variable is not used but is involved in
    computing the result of some needed loop variable, we create a
    zero-valued tensor that is fed as gradient for the Exit node of that
    loop variable. Note that val.op is an Exit, and this method must be
    called in the control flow context where gradients() is called.

    Args:
      val: The output tensor of an Exit op.

    Returns:
      A zero tensor of the same shape of val.
    """
    val_shape = val.get_shape()
    forward_ctxt = val.op._get_control_flow_context()
    outer_forward_ctxt = forward_ctxt.outer_context
    if outer_forward_ctxt:
      outer_forward_ctxt = outer_forward_ctxt.GetWhileContext()
    outer_grad_state = None
    if outer_forward_ctxt:
      outer_grad_state = self._map.get(outer_forward_ctxt)
    if outer_grad_state:
      # This is a nested loop.
      if val_shape.is_fully_defined():
        # If the shape is known statically, just create a zero tensor
        # with the right shape in the right context.
        outer_grad_state.grad_context.Enter()
        result = array_ops.zeros(val_shape.dims, val.dtype)
        outer_grad_state.grad_context.Exit()
      else:
        history_val = outer_grad_state.AddForwardAccumulator(val)
        outer_grad_ctxt = outer_grad_state.grad_context
        outer_grad_ctxt.Enter()
        real_val = outer_grad_state.AddBackPropAccumulatedValue(
            history_val, val)
        result = array_ops.zeros_like(real_val)
        outer_grad_ctxt.Exit()
    else:
      # This is not a nested loop.
      if val_shape.is_fully_defined():
        # If the shape is known statically, just create a zero tensor
        # with the right shape.
        result = array_ops.zeros(val_shape.dims, val.dtype)
      else:
        result = array_ops.zeros_like(val)
    return result

  def ZerosLike(self, op, index):
    """Create zeros_like for the specified output of an op.

    This method must be called in the grad loop context.

    Args:
      op: A tensorflow operation.
      index: the index for a specific output of the op.

    Returns:
      A zero tensor of the same shape of op.outputs[index].
    """
    if IsLoopSwitch(op): return None
    dead_branch = op.type in {"Switch", "RefSwitch"}
    forward_ctxt = _GetWhileContext(op)
    if forward_ctxt is None:
      return array_ops.zeros_like(op.outputs[index])
    op_ctxt = op._get_control_flow_context()
    grad_state = self._map.get(forward_ctxt)
    val = ops.convert_to_tensor(op.outputs[index], name="tensor")
    shape = val.get_shape()
    if shape.is_fully_defined():
      # If the shape is known statically, just create a zero tensor with
      # the right shape in the grad loop context.
      result = constant_op.constant(0, shape=shape.dims, dtype=val.dtype)
      if dead_branch:
        # op is a cond switch. Guard the zero tensor with a switch.
        pred = grad_state.history_map.get(op_ctxt.pred.name)
        branch = op_ctxt.branch
        result = _SwitchRefOrTensor(result, pred)[1 - branch]
    else:
      # Unknown shape so keep a history of the shape at runtime.
      if dead_branch:
        # Need to add a special switch to guard the value.
        pred = op_ctxt.pred
        branch = op_ctxt.branch
        op_ctxt.outer_context.Enter()
        val = _SwitchRefOrTensor(op.inputs[0], pred)[1 - branch]
        zeros_shape = array_ops.shape(val)
        op_ctxt.outer_context.Exit()
        val.op._set_control_flow_context(op_ctxt)
        zeros_shape.op._set_control_flow_context(op_ctxt)
      else:
        op_ctxt.Enter()
        zeros_shape = array_ops.shape(val)
        op_ctxt.Exit()

      # Add forward accumulator for shape.
      grad_state.grad_context.Exit()
      history_shape = grad_state.AddForwardAccumulator(zeros_shape, dead_branch)
      grad_state.grad_context.Enter()

      # Create a zero tensor with the right shape.
      shape = grad_state.AddBackPropAccumulatedValue(
          history_shape, zeros_shape, dead_branch)
      result = array_ops.zeros(shape, val.dtype)
    return result


def GetRealOp(op):
  """Get the real op by removing the wrapper."""
  while isinstance(op, ControlFlowOpWrapper):
    op = op.op
  return op


def MaybeCreateControlFlowState(between_op_list, between_ops):
  """Create the state for all the while loops involved in one gradients().

  We create a ControlFlowState when there are while loops involved in
  gradients(). In gradients(), control flow logic is only invoked when
  the ControlFlowState is not None.

  Note that this method modifies `between_op_list` and `between_ops`.
  """
  loop_state = None
  for op in between_op_list:
    if _IsLoopExit(op):
      if loop_state is None:
        loop_state = ControlFlowState()
      loop_state.AddWhileContext(op, between_op_list, between_ops)
  return loop_state


def IsLoopSwitch(op):
  """Return true if `op` is the Switch for a While loop."""
  if op.type == "Switch" or op.type == "RefSwitch":
    ctxt = op._get_control_flow_context()
    return ctxt and isinstance(ctxt, WhileContext)
  return False


class ControlFlowContext(object):
  """The base class for control flow context.

  The usage pattern is a sequence of (Enter, Exit) followed by a final
  ExitResult.

  We maintain the following state for control flow contexts during graph
  construction:
   1. graph has _control_flow_context: the current context used to
      construct new nodes. Changed by ctxt.Enter() and ctxt.Exit()
   2. op has _control_flow_context: the context to which the op belongs.
      Set at the time the op is created. Immutable.
   3. A ControlFlowContext has _outer_context: the context in which this
      context is created. Set at the time a context is created. Immutable.
   4. A ControlFlowContext has _context_stack.
      Pushed and popped by ctxt.Enter() and ctxt.Exit()
  """

  def __init__(self):
    self._outer_context = ops.get_default_graph()._get_control_flow_context()
    self._context_stack = []
    # Values that have been already seen in this context.
    self._values = set()
    # Values referenced by but external to this context.
    self._external_values = {}

  @property
  def outer_context(self):
    """Return the context containing this context."""
    return self._outer_context

  def AddName(self, name):
    self._values.add(name)

  # pylint: disable=protected-access
  def Enter(self):
    """Enter this control flow context."""
    graph = ops.get_default_graph()
    self._context_stack.append(graph._get_control_flow_context())
    graph._set_control_flow_context(self)

  def Exit(self):
    """Exit this control flow context."""
    graph = ops.get_default_graph()
    last_context = self._context_stack.pop()
    graph._set_control_flow_context(last_context)

  def ExitResult(self, result):
    """Make a list of tensors available in the outer context."""
    if self._outer_context:
      for x in result:
        self._outer_context.AddName(x.name)

  def GetWhileContext(self):
    """Return the while context containing this context."""
    if self._outer_context:
      return self._outer_context.GetWhileContext()
    return None

  def MaybeAddToWhileContext(self, op):
    """Add a control dependency to the containing WhileContext.

    The added control dependency ensures that the outputs of this op
    belong to the WhileContext. Do nothing if the op is not contained
    in a WhileContext.

    Args:
      op: An operation.
    """
    while_ctxt = self.GetWhileContext()
    if while_ctxt is not None:
      # pylint: disable=protected-access
      op._add_control_input(while_ctxt.GetControlPivot().op)
      # pylint: enable=protected-access


class CondContext(ControlFlowContext):
  """The context for the conditional construct."""

  def __init__(self, pred, pivot, branch):
    ControlFlowContext.__init__(self)
    self._pred = pred         # The boolean tensor for the cond predicate
    self._pivot = pivot       # The predicate tensor in this branch
    self._branch = branch     # 0 or 1 representing this branch

    # Values considered to have been already seen in this context.
    self._values.add(pred.name)
    self._values.add(pivot.name)

  @property
  def pred(self):
    return self._pred

  @property
  def pivot(self):
    return self._pivot

  @property
  def branch(self):
    return self._branch

  def AddValue(self, val):
    """Add `val` to the current context and its outer context recursively."""
    result = val
    if val.name not in self._values:
      self._values.add(val.name)
      if self._outer_context:
        result = self._outer_context.AddValue(val)
        self._values.add(result.name)
      with ops.control_dependencies(None):
        result = _SwitchRefOrTensor(result, self._pred)[self._branch]
      # pylint: disable=protected-access
      result.op._set_control_flow_context(self)
      # pylint: enable=protected-access

      self._values.add(result.name)
      self._external_values[val.name] = result
    return result

  def AddOp(self, op):
    """Add `op` to the current context."""
    if not op.inputs:
      # Add this op to the enclosing while context
      self.MaybeAddToWhileContext(op)
      # pylint: disable=protected-access
      op._add_control_input(self._pivot.op)
      # pylint: enable=protected-access
      for x in op.outputs:
        self._values.add(x.name)
    else:
      for index in range(len(op.inputs)):
        x = op.inputs[index]
        if x.name not in self._values:
          self._values.add(x.name)
          # Add this value to the parent contexts up to the context that
          # creates this value.
          real_x = x
          if self._outer_context:
            real_x = self._outer_context.AddValue(x)
            self._values.add(real_x.name)
          real_x = _SwitchRefOrTensor(real_x, self._pred)[self._branch]
          self._external_values[x.name] = real_x
        x = self._external_values.get(x.name)
        if x is not None:
          op._update_input(index, x)
      for x in op.outputs:
        self._values.add(x.name)

  def BuildCondBranch(self, fn):
    """Add the subgraph defined by fn() to the graph."""
    r = fn()
    result = []
    if r is not None:
      if not isinstance(r, list) and not isinstance(r, _basetuple):
        r = [r]
      for v in r:
        real_v = v
        if isinstance(v, ops.Operation):
          # Use pivot as the proxy for this op.
          real_v = with_dependencies([v], self._pivot)
        elif v.name not in self._values:
          # Handle the special case of lambda: x
          self._values.add(v.name)
          if self._outer_context:
            real_v = self._outer_context.AddValue(v)
            self._values.add(real_v.name)
          real_v = _SwitchRefOrTensor(real_v, self._pred)[self._branch]
          self._external_values[v.name] = real_v
        else:
          external_v = self._external_values.get(v.name)
          if external_v is not None:
            real_v = external_v
        result.append(real_v)
    return result


def cond(pred, fn1, fn2, name=None):
  """Return either fn1() or fn2() based on the boolean predicate `pred`.

  `fn1` and `fn2` both return lists of output tensors. `fn1` and `fn2` must have
  the same non-zero number and type of outputs.

  Args:
    pred: A scalar determining whether to return the result of `fn1` or `fn2`.
    fn1: The function to be performed if pred is true.
    fn2: The function to be performed if pref is false.
    name: Optional name prefix for the returned tensors.

  Returns:
    Tensors returned by the call to either `fn1` or `fn2`. If the functions
    return a singleton list, the element is extracted from the list.

  Raises:
    TypeError: if `fn1` or `fn2` is not callable.
    ValueError: if `fn1` and `fn2` do not return the same number of tensors, or
                return tensors of different types.

  Example:

  ```python
    x = tf.constant(2)
    y = tf.constant(5)
    def f1(): return tf.mul(x, 17)
    def f2(): return tf.add(y, 23)
    r = cond(math_ops.less(x, y), f1, f2)
    # r is set to f1().
    # Operations in f2 (e.g., tf.add) are not executed.
  ```

  """
  with ops.op_scope([pred], name, "cond") as name:
    if not callable(fn1):
      raise TypeError("fn1 must be callable.")
    if not callable(fn2):
      raise TypeError("fn2 must be callable.")

    # Add the Switch to the graph.
    if isinstance(pred, bool):
      raise TypeError("pred must not be a Python bool")
    p_2, p_1 = switch(pred, pred)
    pivot_1 = array_ops.identity(p_1, name="switch_t")
    pivot_2 = array_ops.identity(p_2, name="switch_f")
    pred = array_ops.identity(pred, name="pred_id")

    # Build the graph for the true branch in a new context.
    context_t = CondContext(pred, pivot_1, 1)
    context_t.Enter()
    res_t = context_t.BuildCondBranch(fn1)
    context_t.ExitResult(res_t)
    context_t.Exit()

    # Build the graph for the false branch in a new context.
    context_f = CondContext(pred, pivot_2, 0)
    context_f.Enter()
    res_f = context_f.BuildCondBranch(fn2)
    context_f.ExitResult(res_f)
    context_f.Exit()

    # Add the final merge to the graph.
    if len(res_t) != len(res_f):
      raise ValueError("fn1 and fn2 must return the same number of results.")
    if not res_t:
      raise ValueError("fn1 and fn2 must return at least one result.")
    for x, y in zip(res_f, res_t):
      assert ((isinstance(x, ops.IndexedSlices) and
               isinstance(y, ops.IndexedSlices)) or
              (isinstance(x, ops.Tensor) and isinstance(y, ops.Tensor)))
      val_x = x if isinstance(x, ops.Tensor) else x.values
      val_y = y if isinstance(y, ops.Tensor) else y.values
      if val_x.dtype.base_dtype != val_y.dtype.base_dtype:
        raise ValueError("Outputs of fn1 and fn2 must have the same type: "
                         "%s, %s" % (val_x.dtype.name, val_y.dtype.name))
    merges = [merge([x[0], x[1]])[0] for x in zip(res_f, res_t)]
    return merges[0] if len(merges) == 1 else merges


# TODO(yuanbyu): Consider having a unified notion of context for
# not only conditionals and loops but also control dependency and
# subgraphs.
class WhileContext(ControlFlowContext):
  """The context for the loop construct."""

  def __init__(self, parallel_iterations, back_prop, name):
    ControlFlowContext.__init__(self)
    self._name = ops.get_default_graph().unique_name(name)
    self._parallel_iterations = parallel_iterations
    self._back_prop = back_prop
    # We use this node to control constants created by the pred lambda.
    self._pivot_for_pred = None
    # We use this node to control constants created by the body lambda.
    self._pivot_for_body = None
    # The boolean tensor for loop termination condition. Used in code
    # generation for gradient computation
    self._pivot = None
    # The list of exit tensors for loop variables.
    self._loop_exits = None

  @property
  def name(self):
    return self._name

  @property
  def parallel_iterations(self):
    """The number of iterations allowed to run in parallel."""
    return self._parallel_iterations

  @property
  def back_prop(self):
    """True iff backprop is enabled for this While loop."""
    return self._back_prop

  @property
  def pivot(self):
    """The boolean tensor representing the loop termination condition."""
    return self._pivot

  @property
  def loop_exits(self):
    """The list of exit tensors for loop variables."""
    return self._loop_exits

  def GetWhileContext(self):
    return self

  def GetControlPivot(self):
    if self._pivot_for_body:
      return self._pivot_for_body
    return self._pivot_for_pred

  def AddValue(self, val):
    """Add `val` to the current context and its outer context recursively."""
    result = val
    if val.name not in self._values:
      self._values.add(val.name)
      if self._outer_context is not None:
        result = self._outer_context.AddValue(val)
      # Create an Enter to make `result` known to this loop context.
      with ops.control_dependencies(None):
        enter = _Enter(result, self._name, is_constant=True,
                       parallel_iterations=self._parallel_iterations)
      # pylint: disable=protected-access
      enter.op._set_control_flow_context(self)
      # pylint: enable=protected-access

      # Add `enter` in this context.
      self._values.add(enter.name)
      self._external_values[val.name] = enter
      result = enter
    else:
      actual_val = self._external_values.get(val.name)
      if actual_val is not None:
        result = actual_val
    return result

  def AddOp(self, op):
    """Adds `op` to the current context."""
    if not op.inputs:
      if not op.control_inputs:
        # Add a control edge from the control pivot to this op.
        # pylint: disable=protected-access
        op._add_control_input(self.GetControlPivot().op)
        # pylint: enable=protected-access
      else:
        # Control edges must be in the same context.
        for x in op.control_inputs:
          assert x._get_control_flow_context() == self, (
              "Control inputs must come from Operations in the same while "
              "loop context (not an outer context).")
      for x in op.outputs:
        self._values.add(x.name)
    else:
      for index in range(len(op.inputs)):
        x = op.inputs[index]
        self.AddValue(x)
        real_x = self._external_values.get(x.name)
        if real_x is not None:
          op._update_input(index, real_x)
          # Add a control dependency to prevent loop invariants from
          # enabling ops that should not be executed.
          if real_x.op.type == "RefEnter" and real_x.op.get_attr("is_constant"):
            # pylint: disable=protected-access
            op._add_control_input(self.GetControlPivot().op)
            # pylint: enable=protected-access
      for x in op.outputs:
        self._values.add(x.name)

  def AddForwardCounter(self):
    """Adds a loop that counts the number of iterations.

    This is added to the forward loop at the time when we start to
    create the loop for backprop gradient computation. Called in
    the outer context of this forward context.

    The pseudocode is:
      `n = 0; while (_pivot) { n++; }`

    Returns:
      The number of iterations taken by the forward loop and the loop index.
    """
    n = constant_op.constant(0, name="f_count")
    assert n.op._get_control_flow_context() == self.outer_context

    self.Enter()
    self.AddName(n.name)
    enter_n = _Enter(n, self._name, is_constant=False,
                     parallel_iterations=self._parallel_iterations,
                     name="f_count")
    merge_n = merge([enter_n, enter_n])[0]
    switch_n = switch(merge_n, self._pivot)

    index = math_ops.add(switch_n[1], 1)
    next_n = _NextIteration(index)
    merge_n.op._update_input(1, next_n)

    total_iterations = exit(switch_n[0], name="f_count")
    self.ExitResult([total_iterations])
    self.Exit()
    return total_iterations, next_n

  def AddBackPropCounter(self, count):
    """Add the backprop loop that controls the iterations.

    This is added to the backprop loop. It is used to control the loop
    termination of the backprop loop. Called in the outer context of
    this grad context.

    The pseudocode is:
      `n = count; while (n >= 1) { n--; }`

    Args:
      count: The number of iterations for backprop.

    Returns:
      The loop index.
    """
    one = constant_op.constant(1, name="b_count")

    self.Enter()
    self.AddName(count.name)
    enter_count = _Enter(count, self._name, is_constant=False,
                         parallel_iterations=self._parallel_iterations,
                         name="b_count")
    merge_count = merge([enter_count, enter_count])[0]
    self._pivot_for_pred = merge_count

    cond = math_ops.greater_equal(merge_count, one)
    self._pivot = loop_cond(cond, name="b_count")
    switch_count = switch(merge_count, self._pivot)

    index = math_ops.sub(switch_count[1], one)
    self._pivot_for_body = index
    next_count = _NextIteration(index)
    merge_count.op._update_input(1, next_count)

    self.Exit()
    return next_count

  def AddBackPropAccumulator(self, value):
    """Add an accumulation loop for every loop invariant.

    This is added to the backprop loop. It is used to accumulate
    partial gradients within each loop iteration. Called when in the
    gradient while context.

    The pseudocode is:
      ```
      acc = 0.0;
      while (_pivot) {
        acc += value;
      }
      ```

    Args:
      value: The partial gradient of an iteration for a loop invariant.

    Returns:
      The gradient for a loop invariant.
    """
    self.Exit()
    if self.outer_context: self.outer_context.Enter()
    acc = constant_op.constant(0, value.dtype, name="b_acc")
    if self.outer_context: self.outer_context.Exit()
    self.Enter()
    self.AddName(acc.name)
    enter_acc = _Enter(acc, self._name, is_constant=False,
                       parallel_iterations=self._parallel_iterations,
                       name="b_acc")
    merge_acc = merge([enter_acc, enter_acc], name="b_acc")[0]
    switch_acc = switch(merge_acc, self._pivot)

    add_acc = math_ops.add(switch_acc[1], value)
    next_acc = _NextIteration(add_acc)
    merge_acc.op._update_input(1, next_acc)

    acc_result = exit(switch_acc[0], name="b_acc")
    self.ExitResult([acc_result])
    return acc_result

  def BuildLoop(self, pred, body, loop_vars):
    """Add the loop termination condition and body to the graph."""

    # Keep original_loop_vars to identify which are TensorArrays
    original_loop_vars = loop_vars
    # Connvert TensorArrays to their flow variables
    loop_vars = _convert_tensorarrays_to_flows(loop_vars)
    loop_vars = ops.convert_n_to_tensor_or_indexed_slices(loop_vars)
    # Let the context know the loop variabes so the loop variables
    # would be added in the outer contexts properly.
    self._values = set([x.name for x in loop_vars])
    real_vars = loop_vars
    if self._outer_context:
      real_vars = [self._outer_context.AddValue(x) for x in loop_vars]
    with ops.control_dependencies(None):
      enter_vars = [_Enter(x, self._name, is_constant=False,
                           parallel_iterations=self._parallel_iterations)
                    for x in real_vars]
    for x in enter_vars:
      x.op._set_control_flow_context(self)  # pylint: disable=protected-access
    self._values = set([x.name for x in enter_vars])

    merge_vars = [merge([x, x])[0] for x in enter_vars]
    self._pivot_for_pred = merge_vars[0]

    # Build the graph for pred.
    merge_vars_with_tensor_arrays = (
        _convert_flows_to_tensorarrays(original_loop_vars, merge_vars))
    c = ops.convert_to_tensor(pred(*merge_vars_with_tensor_arrays))
    self._pivot = loop_cond(c, name="LoopCond")
    switch_vars = [_SwitchRefOrTensor(x, self._pivot) for x in merge_vars]

    # Build the graph for body.
    vars_for_body = [_Identity(x[1]) for x in switch_vars]
    self._pivot_for_body = vars_for_body[0]
    # Convert TensorArray flow variables inside the context back into
    # their associated TensorArrays for calling the body.
    vars_for_body_with_tensor_arrays = (
        _convert_flows_to_tensorarrays(original_loop_vars, vars_for_body))

    body_result = body(*vars_for_body_with_tensor_arrays)
    if not isinstance(body_result, collections.Sequence):
      body_result = [body_result]
    # Store body_result to keep track of TensorArrays returned by body
    original_body_result = body_result
    # Convert TensorArrays returned by body into their flow variables
    result = _convert_tensorarrays_to_flows(body_result)
    result = ops.convert_n_to_tensor_or_indexed_slices(result)
    next_vars = [_NextIteration(x) for x in result]

    # Add the back edges to complete the loop.
    assert len(merge_vars) == len(next_vars)
    for x in zip(merge_vars, next_vars):
      x[0].op._update_input(1, x[1])

    # Add the exit ops.
    exit_vars = [exit(x[0]) for x in switch_vars]
    self._loop_exits = exit_vars

    for m_var, n_var, e_var in zip(merge_vars, next_vars, exit_vars):
      if m_var.get_shape().is_compatible_with(n_var.get_shape()):
        e_var.set_shape(m_var.get_shape().merge_with(n_var.get_shape()))

    # Exit the loop.
    self.ExitResult(exit_vars)

    # Convert TensorArray flow variables outside the context back into
    # their associated TensorArrays for returning to caller.
    exit_vars_with_tensor_arrays = (
        _convert_flows_to_tensorarrays(original_body_result, exit_vars))
    return (exit_vars_with_tensor_arrays[0]
            if len(exit_vars) == 1
            else exit_vars_with_tensor_arrays)


def While(cond, body, loop_vars, parallel_iterations=10, back_prop=True,
          name=None):
  """Repeat `body` while the condition `cond` is true.

  `cond` is a function taking a list of tensors and returning a boolean scalar
  tensor. `body` is a function taking a list of tensors and returning a list of
  tensors of the same length and with the same types as the input. `loop_vars`
  is a list of tensors that is passed to both `cond` and `body`.

  In addition to regular Tensors or IndexedSlices, the body may accept and
  return TensorArray objects.  The flows of the TensorArray objects will
  be appropriately forwarded between loops and during gradient calculations.

  While `cond` evaluates to true, `body` is executed.

  Args:
    cond: The termination condition of the loop.
    body: A function that represents the loop body.
    loop_vars: The list of variable input tensors.
    parallel_iterations: The number of iterations allowed to run in parallel.
    back_prop: Whether backprop is enabled for this while loop.
    name: Optional name prefix for the returned tensors.

  Returns:
    The output tensors for the loop variables after the loop.

  Raises:
    TypeError: if `cond` or `body` is not callable.
    ValueError: if `loop_var` is empty.

  Example:
    ```python
    i = constant(0)
    c = lambda i: math_ops.less(i, 10)
    b = lambda i: math_ops.add(i, 1)
    r = While(c, b, [i])
    ```
  """
  with ops.op_scope(loop_vars, name, "While") as name:
    if not loop_vars:
      raise ValueError("No loop variables provided")
    if not callable(cond):
      raise TypeError("cond must be callable.")
    if not callable(body):
      raise TypeError("body must be callable.")

    context = WhileContext(parallel_iterations, back_prop, name)
    context.Enter()
    result = context.BuildLoop(cond, body, loop_vars)
    context.Exit()
    return result


def _AsTensorList(x, p):
  """Return x as a list of Tensors or IndexedSlices.

  For entries of `x` that are Operations, this returns an Identity of `p`
  with a dependency on the operation.

  Args:
    x: A Tensor/IndexedSlices/Operation or a list or tuple of them.
    p: A Tensor to return for entries in `x` that are Operations.

  Returns:
    A list of Tensors or IndexedSlices.
  """
  if not isinstance(x, (list, _basetuple)):
    x = [x]

  l = []
  for v in x:
    if isinstance(v, ops.Operation):
      v = with_dependencies([v], p)
    v = ops.convert_to_tensor_or_indexed_slices(v)
    if isinstance(v, ops.Tensor):
      l.append(array_ops.identity(v))
    else:
      l.append(ops.IndexedSlices(array_ops.identity(v.values),
                                 array_ops.identity(v.indices)))
  return l


def _CheckResults(a, b):
  assert len(a) == len(b), (
      "Values returned by a() and b() must have the same length.")
  for x, y in zip(a, b):
    assert x.dtype == y.dtype, (
        "Values returned by a() [%s] and b() [%s] must have "
        "the same type: %s, %s." %
        (x.name, y.name, x.dtype.name, y.dtype.name))


def with_dependencies(dependencies, output_tensor, name=None):
  """Produces the content of `output_tensor` only after `dependencies`.

  In some cases, a user may want the output of an operation to be
  consumed externally only after some other dependencies have run
  first. This function ensures returns `output_tensor`, but only after all
  operations in `dependencies` have run. Note that this means that there is
  no guarantee that `output_tensor` will be evaluated after any `dependencies`
  have run.

  See also `tuple` and `group`.

  Args:
    dependencies: A list of operations to run before this op finishes.
    output_tensor: A `Tensor` or `IndexedSlices` that will be returned.
    name: (Optional) A name for this operation.

  Returns:
    Same as `output_tensor`.

  Raises:
    TypeError: if `output_tensor` is not a `Tensor` or `IndexedSlices`.
  """
  with ops.op_scope(dependencies + [output_tensor], name,
                    "control_dependency") as name:
    with ops.device(output_tensor.device):
      with ops.control_dependencies(dependencies):
        output_tensor = ops.convert_to_tensor_or_indexed_slices(output_tensor)
        if isinstance(output_tensor, ops.Tensor):
          return _Identity(output_tensor, name=name)
        else:
          return ops.IndexedSlices(_Identity(output_tensor.values, name=name),
                                   output_tensor.indices,
                                   output_tensor.dense_shape)


def _GroupControlDeps(dev, deps, name=None):
  with ops.control_dependencies(deps):
    if dev is None:
      return no_op(name=name)
    else:
      with ops.device(dev):
        return no_op(name=name)


# TODO(touts): Accept "inputs" as a list.
def group(*inputs, **kwargs):
  """Create an op that groups multiple operations.

  When this op finishes, all ops in `input` have finished. This op has no
  output.

  See also `tuple` and `with_dependencies`.

  Args:
    *inputs: One or more tensors to group.
    **kwargs: Optional parameters to pass when constructing the NodeDef.
    name: A name for this operation (optional).

  Returns:
    An Operation that executes all its inputs.

  Raises:
    ValueError: If an unknown keyword argument is provided, or if there are
                no inputs.
  """
  name = kwargs.pop("name", None)
  if kwargs:
    raise ValueError("Unknown keyword arguments: " + ", ".join(kwargs.keys()))
  if not inputs:
    # TODO(touts): Would make sense to return a NoOp.
    raise ValueError("No inputs provided")
  with ops.op_scope(inputs, name, "group_deps") as name:
    # Sorts *inputs according to their devices.
    ops_on_device = {}  # device -> operations specified on the device.
    for inp in inputs:
      dev = inp.device
      if dev in ops_on_device:
        ops_on_device[dev].append(inp)
      else:
        ops_on_device[dev] = [inp]
    if len(ops_on_device) == 1:
      # 1-level tree. The root node is the returned NoOp node.
      (dev, deps), = ops_on_device.items()
      return _GroupControlDeps(dev, deps, name=name)
    # 2-level tree. The root node is the returned NoOp node.
    # deps contains 1 NoOp node for each device.
    deps = []

    def device_key(dev):
      """A sort key that allows None to be compared to strings."""
      return "" if dev is None else dev
    for dev in sorted(six.iterkeys(ops_on_device), key=device_key):
      deps.append(_GroupControlDeps(dev, ops_on_device[dev]))
    return _GroupControlDeps(None, deps, name=name)


def tuple(tensors, name=None, control_inputs=None):
  """Group tensors together.

  This creates a tuple of tensors with the same values as the `tensors`
  argument, except that the value of each tensor is only returned after the
  values of all tensors have been computed.

  `control_inputs` contains additional ops that have to finish before this op
  finishes, but whose outputs are not returned.

  This can be used as a "join" mechanism for parallel computations: all the
  argument tensors can be computed in parallel, but the values of any tensor
  returned by `tuple` are only available after all the parallel computations
  are done.

  See also `group` and `with_dependencies`.

  Args:
    tensors: A list of `Tensor`s or `IndexedSlices`, some entries can be `None`.
    name: (optional) A name to use as a `name_scope` for the operation.
    control_inputs: List of additional ops to finish before returning.

  Returns:
    Same as `tensors`.

  Raises:
    ValueError: If `tensors` does not contain any `Tensor` or `IndexedSlices`.
    TypeError: If `control_inputs` is not a list of `Operation` or `Tensor`
      objects.

  """
  with ops.op_scope(tensors, name, "tuple") as name:
    gating_ops = [t.op for t in tensors if t]
    if control_inputs:
      for c in control_inputs:
        if isinstance(c, ops.Tensor):
          c = c.op
        elif not isinstance(c, ops.Operation):
          raise TypeError("Control input must be Operation or Tensor: %s" % c)
        gating_ops.append(c)
    # Note that in order to ensure ordering in the pbtxt, we must take care to
    # ensure the order here.
    gating_ops = sorted(set(gating_ops), key=lambda op: op._id)  # Uniquify ops.
    if not gating_ops:
      raise ValueError("Must have at least one Tensor: %s" % tensors)
    gate = group(*gating_ops)
    tpl = []
    for t in tensors:
      if t:
        tpl.append(with_dependencies([gate], t))
      else:
        tpl.append(None)
    return tpl


# TODO(yuanbyu, mrry): Handle stride to support sliding windows.
def foldl(fn, elems, initializer=None, name=None):
  """The foldl operator on the unpacked tensors of a tensor.

  This foldl operator applies the function `fn` to a sequence of elements
  from left to right. The elements are made of the tensors unpacked from
  `elems`. If `initializer` is None, `elems` must contain at least one
  element.

  Args:
    fn: The function to be performed.
    elems: A tensor to be unpacked.
    initializer: (optional) The initial value for the accumulator.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor resulting from applying `fn` consecutively on each
    element/slice of `elems`, from left to right.

  Raises:
    TypeError: if `fn` is not callable.

  Example:
    ```python
    elems = [1, 2, 3, 4, 5, 6]
    sum = foldl(lambda a, x: a + x, elems)
    ```
  """
  with ops.op_scope([elems], name, "foldl") as name:
    if not callable(fn):
      raise TypeError("fn must be callable.")

    # Convert elems to tensor array.
    n = array_ops.shape(elems)[0]
    elems_ta = tensor_array_ops.TensorArray(dtype=elems.dtype, size=n,
                                            dynamic_size=False)
    elems_ta = elems_ta.unpack(elems)

    if initializer is None:
      a = elems_ta.read(0)
      i = constant_op.constant(1)
    else:
      a = ops.convert_to_tensor(initializer)
      i = constant_op.constant(0)

    def compute(i, a):
      a = fn(a, elems_ta.read(i))
      return [i + 1, a]
    _, r_a = While(lambda i, a: i < n, compute, [i, a])
    return r_a


def foldr(fn, elems, initializer=None, name=None):
  """The foldr operator operator on the unpacked tensors of a tensor.

  This foldr operator applies the function `fn` to a sequence of elements
  from right to left. The elements are made of the tensors unpacked from
  `elems`. If `initializer` is None, `elems` must contain at least one
  element.

  Args:
    fn: The function to be performed.
    elems: A tensor that is unpacked into a sequence of tensors to apply `fn`.
    initializer: (optional) The initial value for the accumulator.
    use_tensor_array: (optional) use tensor_array if true.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor resulting from applying `fn` consecutively on each
    element/slice of `elems`, from right to left.

  Raises:
    TypeError: if `fn` is not callable.

  Example:
    ```python
    elems = [1, 2, 3, 4, 5, 6]
    sum = foldr(lambda a, x: a + x, elems)
    ```
  """
  with ops.op_scope([elems], name, "foldr") as name:
    if not callable(fn):
      raise TypeError("fn must be callable.")

    # Convert elems to tensor array.
    n = array_ops.shape(elems)[0]
    elems_ta = tensor_array_ops.TensorArray(dtype=elems.dtype, size=n,
                                            dynamic_size=False)
    elems_ta = elems_ta.unpack(elems)

    if initializer is None:
      i = n - 1
      a = elems_ta.read(i)
    else:
      i = n
      a = ops.convert_to_tensor(initializer)
    def compute(i, a):
      i -= 1
      a = fn(a, elems_ta.read(i))
      return [i, a]
    _, r_a = While(lambda i, a: i > 0, compute, [i, a])
    return r_a


def map(fn, elems, dtype=None, name=None):
  """The map operator on on the unpacked tensors of a tensor.

  This map operator applies the function `fn` to a sequence of elements
  from right to left. The elements are made of the tensors unpacked from
  `elems`.

  Args:
    fn: The function to be performed.
    elems: A tensor to be unpacked to apply `fn`.
    dtype: (optional) The output type of `fn`.
    name: (optional) Name prefix for the returned tensors.

  Returns:
    A tensor that packs the results of applying `fn` on each element
    of `elems`.

  Raises:
    TypeError: if `fn` is not callable.

  Example:
    ```python
    elems = [1, 2, 3, 4, 5, 6]
    squares = map(lambda x: x * x, elems)
    ```
  """
  with ops.op_scope([elems], name, "map") as name:
    if not callable(fn):
      raise TypeError("fn must be callable.")
    dtype = dtype if dtype else elems.dtype

    # Convert elems to tensor array.
    elems_ta = tensor_array_ops.TensorArray(dtype=elems.dtype, size=0,
                                            dynamic_size=True)
    elems_ta = elems_ta.unpack(elems)

    n = elems_ta.size()
    i = constant_op.constant(0)
    acc_ta = tensor_array_ops.TensorArray(dtype=dtype, size=n)
    def compute(i, a):
      a = a.write(i, fn(elems_ta.read(i)))
      i = math_ops.add(i, 1)
      return [i, a]
    _, r_a = While(lambda i, a: math_ops.less(i, n), compute, [i, acc_ta])
    return r_a.pack()


def case(pred_fn_pairs, default, exclusive=False, name="case"):
  """Create a case operation.

  The `pred_fn_pairs` parameter is a dict or list of pairs of size N.
  Each pair contains a boolean scalar tensor and a python callable that
  creates the tensors to be returned if the boolean evaluates to True. `default`
  is a callable generating a list of tensors. All the callables in
  `pred_fn_pairs` as well as `default` should return the same number and types
  of tensors.

  If `exclusive==True`, all predicates are evaluated, and a logging operation
  with an error is returned if more than one of the predicates evaluates to
  True. If `exclusive==False`, execution stops are the first predicate which
  evaluates to True, and the tensors generated by the corresponding function
  are returned immediately. If none of the predicates evaluate to True, this
  operation returns the tensors generated by `default`.

  Example 1:
    Pseudocode:
    ```
      if (x < y) return 17;
      else return 23;
    ```

    Expressions:
    ```
      f1 = lambda: tf.constant(17)
      f2 = lambda: tf.constant(23)
      r = case([(tf.less(x, y), f1)], default=f2)
    ```

  Example 2:
    Pseudocode:
    ```
      if (x < y && x > z) raise OpError("Only one predicate may evaluate true");
      if (x < y) return 17;
      else if (x > z) return 23;
      else return -1;
    ```

    Expressions:
    ```
      def f1(): return tf.constant(17)
      def f2(): return tf.constant(23)
      def f3(): return tf.constant(-1)
      r = case({tf.less(x, y): f1, tf.greater(x, z): f2},
               default=f3, exclusive=True)
    ```

  Args:
    pred_fn_pairs: Dict or list of pairs of a boolean scalar tensor and a
                   callable which returns a list of tensors.
    default: A callable that returns a list of tensors.
    exclusive: True iff more than one predicate is allowed to evaluate to True.
    name: A name for this operation (optional).

  Returns:
    The tensors returned by the first pair whose predicate evaluated to True, or
    those returned by `default` if none does.

  Raises:
    TypeError: If `pred_fn_pairs` is not a list/dictionary.
    TypeError: If `pred_fn_pairs` is a list but does not contain 2-tuples.
    TypeError: If `fns[i]` is not callable for any i, or `default` is not
               callable.
  """
  pfp = pred_fn_pairs  # For readability
  if not (isinstance(pfp, list) or isinstance(pfp, _basetuple)
          or isinstance(pfp, dict)):
    raise TypeError("fns must be a list, tuple, or dict")
  if isinstance(pfp, dict):
    pfp = pfp.items()
    if not exclusive:
      logging.warn("%s: Provided dictionary of predicate/fn pairs, but "
                   "exclusive=False.  Order of conditional tests is "
                   "not guaranteed.", name)
  for tup in pfp:
    if not isinstance(tup, _basetuple) or len(tup) != 2:
      raise TypeError("Each entry in pred_fn_pairs must be a 2-tuple")
    pred, fn = tup
    if pred.dtype != dtypes.bool:
      raise TypeError("pred must be of type bool: %s", pred.name)
    if not callable(fn):
      raise TypeError("fn for pred %s must be callable." % pred.name)
  if not callable(default):
    raise TypeError("default must be callable.")

  preds, fns = map(list, zip(*pfp))
  with ops.op_scope([preds], name, "case"):
    if not preds:
      return default()
    not_preds = []
    for i, p in enumerate(preds):
      with ops.name_scope("not_%d" % i):
        not_preds.append(math_ops.logical_not(p))
    and_not_preds = [constant_op.constant(True, name="and_not_true")]
    for i, notp in enumerate(not_preds[:-1]):
      with ops.name_scope("and_not_%d" % i):
        and_not_preds.append(math_ops.logical_and(and_not_preds[-1], notp))

    # preds = [p1, p2, p3]
    # fns = [f1, f2, f3]
    # not_preds = [~p1, ~p2, ~p3]
    # case_preds = [p1 & True,
    #               p2 & ~p1,
    #               p3 & ~p1 & ~ p2]
    case_preds = []
    for i, (p, and_not_p_prev) in enumerate(zip(preds, and_not_preds)):
      with ops.name_scope("case_%d" % i):
        case_preds.append(math_ops.logical_and(p, and_not_p_prev))

    # case_sequence = [cond(p3 & ..., f3, default),
    #                  cond(p2 & ..., f2, lambda: case_sequence[0]),
    #                  ...
    #                  cond(p1 & True, f1, lambda: case_sequence[i-1])]
    # and prev_case_seq will loop from case_sequence[0] to case_sequence[-1]
    if exclusive:
      # TODO(ebrevdo): Add Where() for DT_BOOL, replace with Size(Where(preds))
      preds_c = array_ops.concat(0, preds, name="preds_c")
      num_true_conditions = math_ops.reduce_sum(
          math_ops.cast(preds_c, dtypes.int32), name="num_true_conds")
      at_most_one_true_condition = math_ops.less(
          num_true_conditions, constant_op.constant(2, name="two_true_conds"))

      error_msg = [
          ("More than one condition evaluated as True but "
           "exclusive=True.  Conditions: (%s), Values:"
           % ", ".join([p.name for p in preds])),
          preds_c]
      with ops.control_dependencies([
          logging_ops.Assert(condition=at_most_one_true_condition,
                             data=error_msg, summarize=len(preds))]):
        prev_case_seq = None
        for i, (cp, fn) in enumerate(zip(case_preds, fns)[::-1]):
          prev_case_seq = cond(
              cp, fn,
              default if i == 0 else lambda: prev_case_seq,
              name="If_%d" % i)
    else:
      prev_case_seq = None
      for i, (cp, fn) in enumerate(zip(case_preds, fns)[::-1]):
        prev_case_seq = cond(
            cp, fn,
            default if i == 0 else lambda: prev_case_seq,
            name="If_%d" % i)

    return prev_case_seq


ops.RegisterShape("Enter")(common_shapes.unchanged_shape)
ops.RegisterShape("Exit")(common_shapes.unchanged_shape)
ops.RegisterShape("NextIteration")(common_shapes.unchanged_shape)
ops.RegisterShape("RefEnter")(common_shapes.unchanged_shape)
ops.RegisterShape("RefExit")(common_shapes.unchanged_shape)
ops.RegisterShape("RefNextIteration")(common_shapes.unchanged_shape)
ops.RegisterShape("ControlTrigger")(common_shapes.no_outputs)
ops.RegisterShape("NoOp")(common_shapes.no_outputs)


@ops.RegisterShape("LoopCond")
def _LoopCondShape(op):
  """Shape function for the LoopCond op."""
  return [op.inputs[0].get_shape().merge_with(tensor_shape.scalar())]


@ops.RegisterShape("Merge")
def _MergeShape(op):
  """Shape function for the Merge op.

  The Merge op takes many inputs of arbitrary shapes, and produces a
  first output that is one of those inputs, and a second scalar
  output.

  If all input shapes are known and have the same rank, the output
  shape must have that rank, otherwise the output shape is unknown.
  Each output dimension is specified only if that dimension in all
  inputs are the same.

  Args:
    op: A Merge Operation.

  Returns:
    A single-element list containing the Shape of the Merge op.

  """
  output_shape = op.inputs[0].get_shape()
  if output_shape.dims is None:
    return [tensor_shape.unknown_shape(), tensor_shape.scalar()]
  else:
    for input_ in op.inputs[1:]:
      input_shape = input_.get_shape()
      if input_shape.dims is None or input_shape.ndims != output_shape.ndims:
        return [tensor_shape.unknown_shape(), tensor_shape.scalar()]
      else:
        output_shape = tensor_shape.TensorShape(
            [input_dim.value if input_dim.value == output_dim.value else None
             for input_dim, output_dim in zip(input_shape.dims,
                                              output_shape.dims)])
    return [output_shape, tensor_shape.scalar()]

ops.RegisterShape("RefMerge")(_MergeShape)


@ops.RegisterShape("RefSelect")
def _RefSelectShape(op):
  """Shape function for the RefSelect op.

  The RefSelect takes one scalar input and N inputs of arbitrary
  shapes, and produces one output, which is one of those N inputs.

  This function conservatively assumes that if any of the N inputs is
  not fully defined, the output shape is unknown. If all of the N
  inputs have the exact same known shape, the output must have that
  shape.

  Args:
    op: A RefSelect Operation.

  Returns:
    A single-element list containing the Shape of the RefSelect op.
  """
  unused_shape = op.inputs[0].get_shape().merge_with(tensor_shape.scalar())
  first_input_shape = op.inputs[1].get_shape()
  if first_input_shape.is_fully_defined():
    for input_ in op.inputs[2:]:
      input_shape = input_.get_shape()
      if (not input_shape.is_fully_defined()
          or not input_shape.is_compatible_with(first_input_shape)):
        return [tensor_shape.unknown_shape()]
    return [first_input_shape]
  else:
    return [tensor_shape.unknown_shape()]


@ops.RegisterShape("RefSwitch")
@ops.RegisterShape("Switch")
def _SwitchShape(op):
  input_shape = op.inputs[0].get_shape()
  unused_pred_shape = op.inputs[1].get_shape().merge_with(tensor_shape.scalar())
  return [input_shape] * 2
