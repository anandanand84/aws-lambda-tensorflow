"""Python wrappers around Brain.

This file is MACHINE GENERATED! Do not edit.
"""

import collections

from google.protobuf import text_format

from tensorflow.core.framework import op_def_pb2
from tensorflow.python.framework import op_def_registry
from tensorflow.python.framework import ops
from tensorflow.python.ops import op_def_library


_abort_outputs = [""]


def abort(error_msg=None, name=None):
  r"""Raise a exception to abort the process when called.

  Returns nothing but an exception.

  Args:
    error_msg: An optional `string`. Defaults to `""`.
      A string which is the message associated with the exception.
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("Abort", error_msg=error_msg, name=name)
  return result


_control_trigger_outputs = [""]


def control_trigger(name=None):
  r"""Does nothing. Serves as a control trigger for scheduling. Only useful as a

  placeholder for control edges.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("ControlTrigger", name=name)
  return result


_enter_outputs = ["output"]


def enter(data, frame_name, is_constant=None, parallel_iterations=None,
          name=None):
  r"""Creates or finds a child frame, and makes `data` available to the child frame.

  This op is used together with `Exit` to create loops in the graph.
  The unique `frame_name` is used by the `Executor` to identify frames. If
  `is_constant` is true, `output` is a constant in the child frame; otherwise
  it may be changed in the child frame. At most `parallel_iterations` iterations
  are run in parallel in the child frame.

  Args:
    data: A `Tensor`. The tensor to be made available to the child frame.
    frame_name: A `string`. The name of the child frame.
    is_constant: An optional `bool`. Defaults to `False`.
      If true, the output is constant within the child frame.
    parallel_iterations: An optional `int`. Defaults to `10`.
      The number of iterations allowed to run in parallel.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`. The same tensor as `data`.
  """
  result = _op_def_lib.apply_op("Enter", data=data, frame_name=frame_name,
                                is_constant=is_constant,
                                parallel_iterations=parallel_iterations,
                                name=name)
  return result


__exit_outputs = ["output"]


def _exit(data, name=None):
  r"""Exits the current frame to its parent frame.

  Exit makes its input `data` available to the parent frame.

  Args:
    data: A `Tensor`. The tensor to be made available to the parent frame.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`. The same tensor as `data`.
  """
  result = _op_def_lib.apply_op("Exit", data=data, name=name)
  return result


_loop_cond_outputs = ["output"]


def loop_cond(input, name=None):
  r"""Forwards the input to the output.

  This operator represents the loop termination condition used by the
  "pivot" switches of a loop.

  Args:
    input: A `Tensor` of type `bool`.
      A boolean scalar, representing the branch predicate of the Switch op.
    name: A name for the operation (optional).

  Returns:
    A `Tensor` of type `bool`. The same tensor as `input`.
  """
  result = _op_def_lib.apply_op("LoopCond", input=input, name=name)
  return result


__merge_outputs = ["output", "value_index"]


_MergeOutput = collections.namedtuple("Merge", __merge_outputs)


def _merge(inputs, name=None):
  r"""Forwards the value of an available tensor from `inputs` to `output`.

  `Merge` waits for at least one of the tensors in `inputs` to become available.
  It is usually combined with `Switch` to implement branching.

  `Merge` forwards the first tensor for become available to `output`, and sets
  `value_index` to its index in `inputs`.

  It is an error if more than one tensor in `inputs` is available.

  Args:
    inputs: A list of at least 1 `Tensor` objects of the same type.
      The input tensors, exactly one of which will become available.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, value_index).
    output: A `Tensor`. Has the same type as `inputs`. Will be set to the available input tensor.
    value_index: A `Tensor` of type `int32`. The index of the chosen input tensor in `inputs`.
  """
  result = _op_def_lib.apply_op("Merge", inputs=inputs, name=name)
  return _MergeOutput._make(result)


_next_iteration_outputs = ["output"]


def next_iteration(data, name=None):
  r"""Makes its input available to the next iteration.

  Args:
    data: A `Tensor`. The tensor to be made available to the next iteration.
    name: A name for the operation (optional).

  Returns:
    A `Tensor`. Has the same type as `data`. The same tensor as `data`.
  """
  result = _op_def_lib.apply_op("NextIteration", data=data, name=name)
  return result


_no_op_outputs = [""]


def no_op(name=None):
  r"""Does nothing. Only useful as a placeholder for control edges.

  Args:
    name: A name for the operation (optional).

  Returns:
    The created Operation.
  """
  result = _op_def_lib.apply_op("NoOp", name=name)
  return result


_ref_enter_outputs = ["output"]


def ref_enter(data, frame_name, is_constant=None, parallel_iterations=None,
              name=None):
  r"""Creates or finds a child frame, and makes `data` available to the child frame.

  The unique `frame_name` is used by the `Executor` to identify frames. If
  `is_constant` is true, `output` is a constant in the child frame; otherwise
  it may be changed in the child frame. At most `parallel_iterations` iterations
  are run in parallel in the child frame.

  Args:
    data: A mutable `Tensor`.
      The tensor to be made available to the child frame.
    frame_name: A `string`. The name of the child frame.
    is_constant: An optional `bool`. Defaults to `False`.
      If true, the output is constant within the child frame.
    parallel_iterations: An optional `int`. Defaults to `10`.
      The number of iterations allowed to run in parallel.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `data`.
    The same tensor as `data`.
  """
  result = _op_def_lib.apply_op("RefEnter", data=data, frame_name=frame_name,
                                is_constant=is_constant,
                                parallel_iterations=parallel_iterations,
                                name=name)
  return result


__ref_exit_outputs = ["output"]


def _ref_exit(data, name=None):
  r"""Exits the current frame to its parent frame.

  Exit makes its input `data` available to the parent frame.

  Args:
    data: A mutable `Tensor`.
      The tensor to be made available to the parent frame.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `data`.
    The same tensor as `data`.
  """
  result = _op_def_lib.apply_op("RefExit", data=data, name=name)
  return result


__ref_merge_outputs = ["output", "value_index"]


_RefMergeOutput = collections.namedtuple("RefMerge", __ref_merge_outputs)


def _ref_merge(inputs, name=None):
  r"""Forwards the value of an available tensor from `inputs` to `output`.

  `Merge` waits for at least one of the tensors in `inputs` to become available.
  It is usually combined with `Switch` to implement branching.

  `Merge` forwards the first tensor for become available to `output`, and sets
  `value_index` to its index in `inputs`.

  It is an error if more than one tensor in `inputs` is available.

  Args:
    inputs: A list of at least 1 mutable `Tensor` objects of the same type.
      The input tensors, exactly one of which will become available.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output, value_index).
    output: A mutable `Tensor`. Has the same type as `inputs`. Will be set to the available input tensor.
    value_index: A `Tensor` of type `int32`. The index of the chosen input tensor in `inputs`.
  """
  result = _op_def_lib.apply_op("RefMerge", inputs=inputs, name=name)
  return _RefMergeOutput._make(result)


_ref_next_iteration_outputs = ["output"]


def ref_next_iteration(data, name=None):
  r"""Makes its input available to the next iteration.

  Args:
    data: A mutable `Tensor`.
      The tensor to be made available to the next iteration.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `data`.
    The same tensor as `data`.
  """
  result = _op_def_lib.apply_op("RefNextIteration", data=data, name=name)
  return result


_ref_select_outputs = ["output"]


def ref_select(index, inputs, name=None):
  r"""Forwards the `index`th element of `inputs` to `output`.

  Args:
    index: A `Tensor` of type `int32`.
      A scalar that determines the input that gets selected.
    inputs: A list of at least 1 mutable `Tensor` objects of the same type.
      A list of ref tensors, one of which will be forwarded to `output`.
    name: A name for the operation (optional).

  Returns:
    A mutable `Tensor`. Has the same type as `inputs`. The forwarded tensor.
  """
  result = _op_def_lib.apply_op("RefSelect", index=index, inputs=inputs,
                                name=name)
  return result


_ref_switch_outputs = ["output_false", "output_true"]


_RefSwitchOutput = collections.namedtuple("RefSwitch", _ref_switch_outputs)


def ref_switch(data, pred, name=None):
  r"""Forwards the ref tensor `data` to the output port determined by `pred`.

  If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
  the data goes to `output_false`.

  See also `Switch` and `Merge`.

  Args:
    data: A mutable `Tensor`.
      The ref tensor to be forwarded to the appropriate output.
    pred: A `Tensor` of type `bool`.
      A scalar that specifies which output port will receive data.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_false, output_true).
    output_false: A mutable `Tensor`. Has the same type as `data`. If `pred` is false, data will be forwarded to this output.
    output_true: A mutable `Tensor`. Has the same type as `data`. If `pred` is true, data will be forwarded to this output.
  """
  result = _op_def_lib.apply_op("RefSwitch", data=data, pred=pred, name=name)
  return _RefSwitchOutput._make(result)


__switch_outputs = ["output_false", "output_true"]


_SwitchOutput = collections.namedtuple("Switch", __switch_outputs)


def _switch(data, pred, name=None):
  r"""Forwards `data` to the output port determined by `pred`.

  If `pred` is true, the `data` input is forwarded to `output_true`. Otherwise,
  the data goes to `output_false`.

  See also `RefSwitch` and `Merge`.

  Args:
    data: A `Tensor`. The tensor to be forwarded to the appropriate output.
    pred: A `Tensor` of type `bool`.
      A scalar that specifies which output port will receive data.
    name: A name for the operation (optional).

  Returns:
    A tuple of `Tensor` objects (output_false, output_true).
    output_false: A `Tensor`. Has the same type as `data`. If `pred` is false, data will be forwarded to this output.
    output_true: A `Tensor`. Has the same type as `data`. If `pred` is true, data will be forwarded to this output.
  """
  result = _op_def_lib.apply_op("Switch", data=data, pred=pred, name=name)
  return _SwitchOutput._make(result)


def _InitOpDefLibrary():
  op_list = op_def_pb2.OpList()
  text_format.Merge(_InitOpDefLibrary.op_list_ascii, op_list)
  op_def_registry.register_op_list(op_list)
  op_def_lib = op_def_library.OpDefLibrary()
  op_def_lib.add_op_list(op_list)
  return op_def_lib


_InitOpDefLibrary.op_list_ascii = """op {
  name: "Abort"
  attr {
    name: "error_msg"
    type: "string"
    default_value {
      s: ""
    }
  }
}
op {
  name: "ControlTrigger"
}
op {
  name: "Enter"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "frame_name"
    type: "string"
  }
  attr {
    name: "is_constant"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "parallel_iterations"
    type: "int"
    default_value {
      i: 10
    }
  }
}
op {
  name: "Exit"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "LoopCond"
  input_arg {
    name: "input"
    type: DT_BOOL
  }
  output_arg {
    name: "output"
    type: DT_BOOL
  }
}
op {
  name: "Merge"
  input_arg {
    name: "inputs"
    type_attr: "T"
    number_attr: "N"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  output_arg {
    name: "value_index"
    type: DT_INT32
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 1
  }
}
op {
  name: "NextIteration"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  output_arg {
    name: "output"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "NoOp"
}
op {
  name: "RefEnter"
  input_arg {
    name: "data"
    type_attr: "T"
    is_ref: true
  }
  output_arg {
    name: "output"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "frame_name"
    type: "string"
  }
  attr {
    name: "is_constant"
    type: "bool"
    default_value {
      b: false
    }
  }
  attr {
    name: "parallel_iterations"
    type: "int"
    default_value {
      i: 10
    }
  }
}
op {
  name: "RefExit"
  input_arg {
    name: "data"
    type_attr: "T"
    is_ref: true
  }
  output_arg {
    name: "output"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "RefMerge"
  input_arg {
    name: "inputs"
    type_attr: "T"
    number_attr: "N"
    is_ref: true
  }
  output_arg {
    name: "output"
    type_attr: "T"
    is_ref: true
  }
  output_arg {
    name: "value_index"
    type: DT_INT32
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 1
  }
}
op {
  name: "RefNextIteration"
  input_arg {
    name: "data"
    type_attr: "T"
    is_ref: true
  }
  output_arg {
    name: "output"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "RefSelect"
  input_arg {
    name: "index"
    type: DT_INT32
  }
  input_arg {
    name: "inputs"
    type_attr: "T"
    number_attr: "N"
    is_ref: true
  }
  output_arg {
    name: "output"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
  }
  attr {
    name: "N"
    type: "int"
    has_minimum: true
    minimum: 1
  }
}
op {
  name: "RefSwitch"
  input_arg {
    name: "data"
    type_attr: "T"
    is_ref: true
  }
  input_arg {
    name: "pred"
    type: DT_BOOL
  }
  output_arg {
    name: "output_false"
    type_attr: "T"
    is_ref: true
  }
  output_arg {
    name: "output_true"
    type_attr: "T"
    is_ref: true
  }
  attr {
    name: "T"
    type: "type"
  }
}
op {
  name: "Switch"
  input_arg {
    name: "data"
    type_attr: "T"
  }
  input_arg {
    name: "pred"
    type: DT_BOOL
  }
  output_arg {
    name: "output_false"
    type_attr: "T"
  }
  output_arg {
    name: "output_true"
    type_attr: "T"
  }
  attr {
    name: "T"
    type: "type"
  }
}
"""


_op_def_lib = _InitOpDefLibrary()
