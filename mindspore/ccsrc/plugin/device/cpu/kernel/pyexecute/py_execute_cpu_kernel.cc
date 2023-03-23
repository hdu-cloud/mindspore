/**
 * Copyright 2022 Huawei Technologies Co., Ltd
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "plugin/device/cpu/kernel/pyexecute/py_execute_cpu_kernel.h"

#include <memory>
#include <vector>
#include <utility>

#include "Eigen/Core"
#include "abstract/utils.h"
#include "plugin/device/cpu/hal/device/cpu_common.h"
#include "include/common/utils/python_adapter.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/ccsrc/pipeline/jit/parse/resolve.h"

namespace mindspore {
namespace kernel {
namespace {
py::object CallPythonGetGlobalParams() {
  constexpr auto python_mod_parse = "mindspore._extends.parse";  // The same as PYTHON_MOD_PARSE_MODULE[]
  py::module mod = python_adapter::GetPyModule(python_mod_parse);
  constexpr auto python_get_dict = "get_global_params";
  return python_adapter::CallPyModFn(mod, python_get_dict);
}

// Call the python script string. The same codes as parse/data_converter.h, we must copy it here.
py::object CallPythonScript(const py::object &script, const py::tuple &args_kwargs) {
  constexpr auto python_mod_parse = "mindspore._extends.parse";  // The same as PYTHON_MOD_PARSE_MODULE[]
  py::module mod = python_adapter::GetPyModule(python_mod_parse);
  constexpr auto python_mode_eval = "eval_script";
  // The `args_kwargs` is a tuple(dict(global), dict(local)).
  return args_kwargs.empty() ? python_adapter::CallPyModFn(mod, python_mode_eval, script)
                             : python_adapter::CallPyModFn(mod, python_mode_eval, script, args_kwargs);
}
}  // namespace

void PyExecuteCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_LOG(DEBUG) << "kernel_node: " << kernel_node << ", " << kernel_node->DebugString();
  inputs_info_.clear();
  kernel_node_ = kernel_node;
  for (size_t i = 1; i < kernel_node->size(); ++i) {
    const auto &input = kernel_node->inputs()[i];

    // Check if PyExecuteOutputData exists.
    py::object obj = py::none();
    if (input->has_user_data<PyExecuteOutputData>()) {
      py::gil_scoped_acquire gil_acquire;
      const auto &output_data = input->user_data<PyExecuteOutputData>();
      obj = output_data->obj;
      MS_LOG(DEBUG) << "Has \'PyExecuteOutputData\', obj: " << obj;
    }

    // Record the inputs' information by their abstract types.
    const auto &input_abstract = input->abstract();
    MS_EXCEPTION_IF_NULL(input_abstract);
    if (input_abstract->isa<abstract::AbstractRefTensor>()) {
      const auto &param = dyn_cast<Parameter>(input);
      MS_EXCEPTION_IF_NULL(param);
      MS_LOG(DEBUG) << "AbstractRefTensor, input[" << i << "]: " << param->default_param()->ToString();
      (void)inputs_info_.emplace_back(PyExecuteInputInfo({obj, input_abstract, kTypeUnknown, {}}));
    } else if (input_abstract->isa<abstract::AbstractTensor>()) {
      const auto &tensor_abstract = dyn_cast<abstract::AbstractTensor>(input_abstract);
      MS_EXCEPTION_IF_NULL(tensor_abstract);
      MS_LOG(DEBUG) << "AbstractTensor, input[" << i << "]: " << tensor_abstract->BuildType()->ToString() << ", "
                    << tensor_abstract->BuildShape()->ToString();
      const auto &in_type = AnfAlgo::GetInputDeviceDataType(kernel_node, i - 1);
      const auto &in_shape = AnfAlgo::GetInputDeviceShape(kernel_node, i - 1);
      (void)inputs_info_.emplace_back(PyExecuteInputInfo({obj, input_abstract, in_type, in_shape}));
    } else {
      MS_LOG(DEBUG) << "Other, input[" << i << "]: " << input->DebugString() << ", " << input_abstract->ToString();
      (void)inputs_info_.emplace_back(PyExecuteInputInfo({obj, input_abstract, kTypeUnknown, {}}));
    }
    MS_LOG(DEBUG) << "Kernel node's input[" << i << "]: " << input->DebugString() << ", " << input_abstract->ToString();
  }
}

void ArrayToRawMemory(const py::array &array, const AddressPtr &address) {
  MS_EXCEPTION_IF_NULL(address);
  if (static_cast<unsigned int>(array.flags()) &
      static_cast<unsigned int>(pybind11::detail::npy_api::NPY_ARRAY_C_CONTIGUOUS_)) {
    const py::buffer_info &buf_info = array.request();
    const auto &res =
      memcpy_s(address->addr, address->size, buf_info.ptr, LongToSize(buf_info.size * buf_info.itemsize));
    if (res != EOK) {
      MS_LOG(EXCEPTION) << "memcpy failed. res: " << res << ", address->size: " << address->size
                        << ", size: " << LongToSize(buf_info.size * buf_info.itemsize);
    }
  } else {
    // Transform numpy array to contiguous data.
    Py_buffer pybuf;
    if (PyObject_GetBuffer(array.ptr(), &pybuf, PyBUF_ANY_CONTIGUOUS) != 0) {
      MS_LOG(EXCEPTION) << "Failed to get buffer from the input!";
    }
    auto buffer = std::make_unique<char[]>(LongToSize(pybuf.len));
    if (PyBuffer_ToContiguous(buffer.get(), &pybuf, pybuf.len, 'C')) {
      PyBuffer_Release(&pybuf);
      MS_LOG(EXCEPTION) << "Can't copy numpy.ndarray to a contiguous buffer.";
    }
    PyBuffer_Release(&pybuf);
    const auto &res = memcpy_s(address->addr, address->size, buffer.get(), LongToSize(pybuf.len));
    if (res != EOK) {
      MS_LOG(EXCEPTION) << "memcpy failed. res: " << res;
    }
  }
}

void PyExecuteCpuKernelMod::AttachPyOutputData(const py::object &py_res) {
  const auto &py_output = std::make_shared<PyExecuteOutputData>();
  py_output->obj = py_res;
  // Set Python data for kernel node.
  kernel_node_->set_user_data<PyExecuteOutputData>(py_output);

  // Set Python data for front node.
  const auto &kernel_graph = std::dynamic_pointer_cast<session::KernelGraph>(kernel_node_->func_graph());
  MS_EXCEPTION_IF_NULL(kernel_graph);
  const auto &graph_output_map = kernel_graph->graph_output_map();
  session::AnfWithOutIndex anf_index = std::make_pair(kernel_node_, 0);
  const auto &iter = graph_output_map.find(anf_index);
  if (iter != graph_output_map.cend()) {
    const auto &front_node = iter->second.first;
    MS_LOG(INFO) << "Found front output for " << kernel_node_ << ", " << kernel_node_->DebugString();
    front_node->set_user_data<PyExecuteOutputData>(py_output);
  } else {
    MS_LOG(DEBUG) << "Not found, kernel node is not output, " << kernel_node_ << ", " << kernel_node_->DebugString();
    if (!IS_OUTPUT_ON(mindspore::kDebug)) {
      return;
    }
    for (const auto &output_pair : graph_output_map) {
      MS_EXCEPTION_IF_NULL(output_pair.first.first);
      MS_EXCEPTION_IF_NULL(output_pair.second.first);
      MS_LOG(DEBUG) << "backend node: " << output_pair.first.first << ", " << output_pair.first.first->DebugString()
                    << ", front node: " << output_pair.second.first << ", " << output_pair.second.first->DebugString();
    }
  }
}

// Notice: Remove here after BE kernel supports tuple input.
py::object PyExecuteCpuKernelMod::BuildLocalTupleParameters(const std::vector<AddressPtr> &inputs) {
  constexpr auto internal_tuple_keys_str = "__internal_tuple_keys__";
  constexpr auto internal_tuple_values_str = "__internal_tuple_values__";
  constexpr auto number_two = 2;
  std::string tuple_key_str;
  py::tuple local_tuple_inputs(inputs_info_.size() - number_two);  // Exclude the script and key.
  MS_LOG(DEBUG) << "Local parameter tuple size: " << (inputs_info_.size() - number_two);
  bool tuple_input_start = false;
  size_t tuple_index = 0;
  py::dict local_tuple_dict;
  for (size_t i = 1; i < inputs.size() && i < inputs_info_.size(); ++i) {
    const auto &input = inputs[i];
    MS_EXCEPTION_IF_NULL(input);
    const auto &input_info = inputs_info_[i];
    const auto &input_abstract = input_info.abstract;
    MS_EXCEPTION_IF_NULL(input_abstract);
    const auto &input_type = input_abstract->BuildType();
    MS_EXCEPTION_IF_NULL(input_type);
    if (!tuple_input_start && input_abstract->isa<abstract::AbstractScalar>() && input_type->isa<String>()) {
      const auto &value = input_abstract->BuildValue();
      MS_EXCEPTION_IF_NULL(value);
      const auto &str_value = dyn_cast<StringImm>(value);
      MS_EXCEPTION_IF_NULL(str_value);
      const auto &str = str_value->value();
      if (str != internal_tuple_keys_str && str != internal_tuple_values_str) {
        return py::none();
      }
      tuple_key_str = str;
      tuple_input_start = true;
      MS_LOG(DEBUG) << "String, key input[" << i << "]: " << input_abstract->ToString();
      continue;
    }

    // Rebuild the tuple with all left inputs.
    if (input_abstract->isa<abstract::AbstractScalar>() && input_type->isa<String>()) {
      const auto &value = input_abstract->BuildValue();
      MS_EXCEPTION_IF_NULL(value);
      const auto &str_value = dyn_cast<StringImm>(value);
      MS_EXCEPTION_IF_NULL(str_value);
      const auto &str = str_value->value();
      local_tuple_inputs[tuple_index++] = py::str(str);
      MS_LOG(DEBUG) << "String, value input[" << i << "]: " << input_abstract->ToString();
    } else if (input_abstract->isa<abstract::AbstractTensor>()) {
      const auto &py_array_value = input_info.py_obj_output;
      bool is_py_middle_data = !py::isinstance<py::none>(py_array_value);
      MS_LOG(DEBUG) << "Tensor, value input[" << i << "]: " << input_abstract->ToString()
                    << ", type: " << input_info.type << ", shape: " << input_info.shape << ", addr: " << inputs[i]->addr
                    << ", size: " << inputs[i]->size << ", py_array_value: " << py_array_value
                    << ", is_py_middle_data: " << is_py_middle_data;
      if (!is_py_middle_data) {
        const auto tensor =
          std::make_shared<tensor::Tensor>(input_info.type, input_info.shape, inputs[i]->addr, inputs[i]->size);
        MS_EXCEPTION_IF_NULL(tensor);
        local_tuple_inputs[tuple_index++] = tensor;
      } else {
        local_tuple_inputs[tuple_index++] = py_array_value;
      }
    } else {
      MS_LOG(ERROR) << "Unsupported value type.";
    }
  }
  local_tuple_dict[py::str(tuple_key_str)] = local_tuple_inputs;
  return local_tuple_dict;
}

py::object PyExecuteCpuKernelMod::BuildLocalParameters(const std::vector<AddressPtr> &inputs) {
  const auto &local_tuple_params = BuildLocalTupleParameters(inputs);
  if (local_tuple_params != py::none()) {
    return local_tuple_params;
  }

  MS_LOG(DEBUG) << "Build normal local parameters.";
  // Build local parameters dict.
  std::vector<std::string> keys;
  std::vector<tensor::TensorPtr> tensor_values;
  std::vector<py::object> py_object_values;
  std::vector<bool> py_array_flags;
  constexpr auto number_two = 2;
  size_t pair_size = (inputs_info_.size() - 1) / number_two;

  // Handle the keys.
  size_t i = 1;
  for (; i < inputs.size() && i < pair_size + 1; ++i) {
    const auto &input = inputs[i];
    MS_EXCEPTION_IF_NULL(input);
    const auto &input_info = inputs_info_[i];
    const auto &input_abstract = input_info.abstract;
    MS_EXCEPTION_IF_NULL(input_abstract);
    const auto &input_type = input_abstract->BuildType();
    MS_EXCEPTION_IF_NULL(input_type);
    if (input_abstract->isa<abstract::AbstractScalar>() && input_type->isa<String>()) {
      const auto &value = input_abstract->BuildValue();
      MS_EXCEPTION_IF_NULL(value);
      const auto &str_value = dyn_cast<StringImm>(value);
      MS_EXCEPTION_IF_NULL(str_value);
      const auto &str = str_value->value();
      (void)keys.emplace_back(str);
      MS_LOG(DEBUG) << "String, input[" << i << "]: " << input_abstract->ToString();
    } else {
      MS_LOG(EXCEPTION) << "Other, input[" << i << "]: " << input_abstract->ToString();
    }
  }
  // Handle the values.
  for (; i < inputs.size() && i < inputs_info_.size(); ++i) {
    const auto &input = inputs[i];
    MS_EXCEPTION_IF_NULL(input);
    const auto &input_info = inputs_info_[i];
    const auto &input_abstract = input_info.abstract;
    MS_EXCEPTION_IF_NULL(input_abstract);
    const auto &input_type = input_abstract->BuildType();
    MS_EXCEPTION_IF_NULL(input_type);
    if (input_abstract->isa<abstract::AbstractScalar>() && input_type->isa<String>()) {
      const auto &value = input_abstract->BuildValue();
      MS_EXCEPTION_IF_NULL(value);
      const auto &str_value = dyn_cast<StringImm>(value);
      MS_EXCEPTION_IF_NULL(str_value);
      const auto &str = str_value->value();
      (void)py_object_values.emplace_back(py::str(str));
      (void)tensor_values.emplace_back(nullptr);
      (void)py_array_flags.emplace_back(true);
      MS_LOG(DEBUG) << "String, input[" << i << "]: " << input_abstract->ToString();
    } else if (input_abstract->isa<abstract::AbstractTensor>()) {
      const auto &py_array_value = input_info.py_obj_output;
      bool is_py_middle_data = !py::isinstance<py::none>(py_array_value);
      MS_LOG(DEBUG) << "Tensor, input[" << i << "]: " << input_abstract->ToString() << ", type: " << input_info.type
                    << ", shape: " << input_info.shape << ", addr: " << inputs[i]->addr << ", size: " << inputs[i]->size
                    << ", py_array_value: " << py_array_value << ", is_py_middle_data: " << is_py_middle_data;
      tensor::TensorPtr tensor = nullptr;
      if (!is_py_middle_data) {
        tensor = std::make_shared<tensor::Tensor>(input_info.type, input_info.shape, inputs[i]->addr, inputs[i]->size);
        MS_EXCEPTION_IF_NULL(tensor);
      }
      (void)py_object_values.emplace_back(py_array_value);
      (void)tensor_values.emplace_back(tensor);
      (void)py_array_flags.emplace_back(is_py_middle_data);
    } else if (input_abstract->isa<abstract::AbstractRefTensor>()) {
      MS_LOG(DEBUG) << "Parameter, input[" << i << "]: " << input_abstract->ToString();
    } else {
      MS_LOG(DEBUG) << "Other, input[" << i << "]: " << input_abstract->ToString();
    }
  }

  if (keys.size() != tensor_values.size() || keys.size() != pair_size) {
    MS_LOG(EXCEPTION) << "The local dict input is invalid, " << keys.size() << ", " << tensor_values.size() << ", "
                      << inputs_info_.size();
  }

  // To call the script with global and local parameters.
  py::dict local_dict;
  for (i = 0; i < keys.size(); ++i) {
    if (py_array_flags[i]) {
      local_dict[py::str(keys[i])] = py_object_values[i];
    } else {
      local_dict[py::str(keys[i])] = tensor_values[i];
    }
  }
  MS_LOG(DEBUG) << "local_dict: " << local_dict;
  return local_dict;
}

bool PyExecuteCpuKernelMod::Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
                                   const std::vector<AddressPtr> &outputs) {
  MS_LOG(DEBUG) << "Launch PyExecute(), inputs.size: " << inputs.size() << ", outputs: " << outputs.size();
  if (Py_IsInitialized() != true) {
    MS_LOG(ERROR) << "Py_IsInitialized failed.";
    return false;
  }
  if (outputs.size() != 1) {
    MS_LOG(EXCEPTION) << "The output num is 1, but got " << outputs.size();
  }

  // Build the script.
  py::gil_scoped_acquire gil_acquire;
  const auto &input0_info = inputs_info_[0];
  const auto &input0_abstract = input0_info.abstract;
  const auto &input0_abstract_scalar = dyn_cast<abstract::AbstractScalar>(input0_abstract);
  MS_EXCEPTION_IF_NULL(input0_abstract_scalar);
  if (!input0_abstract_scalar->BuildType()->isa<String>()) {
    MS_LOG(EXCEPTION) << "Should be a string, but got " << input0_abstract_scalar->ToString();
  }
  const auto &input0_value = input0_abstract_scalar->BuildValue();
  MS_EXCEPTION_IF_NULL(input0_value);
  const auto &input0_str = dyn_cast<StringImm>(input0_value);
  MS_LOG(DEBUG) << "Script: " << input0_str->ToString();
  const std::string &script = input0_str->value();

  // Build local parameters dict.
  const auto &local_dict = BuildLocalParameters(inputs);
  // To call the script with global and local parameters.
  const auto &global_dict = CallPythonGetGlobalParams();
  const auto &py_script = py::str(script);
  auto params = py::tuple(2);
  params[0] = global_dict;
  params[1] = local_dict;
  MS_LOG(DEBUG) << "py_script: " << py_script << ", params: " << params;
  const auto &py_res = CallPythonScript(py_script, params);
  // Check Python result.
  if (py::isinstance<py::none>(py_res)) {
    MS_LOG(EXCEPTION) << "Real output is None.";
  } else if (py::isinstance<py::array>(py_res)) {
    MS_LOG(DEBUG) << "Real output is py::array, py_res: " << py_res;
    ArrayToRawMemory(py_res.cast<py::array>(), outputs[0]);
  } else if (py::isinstance<py::float_>(py_res)) {
    MS_LOG(DEBUG) << "Real output is py::float_, py_res: " << py_res;
  } else if (py::isinstance<py::int_>(py_res)) {
    MS_LOG(DEBUG) << "Real output is py::int_, py_res: " << py_res;
  } else if (py::isinstance<py::bool_>(py_res)) {
    MS_LOG(DEBUG) << "Real output is py::bool_, py_res: " << py_res;
  } else if (py::isinstance<py::str>(py_res)) {
    MS_LOG(DEBUG) << "Real output is py::str, py_res: " << py_res;
  } else if (py::isinstance<py::tuple>(py_res)) {
    MS_LOG(DEBUG) << "Real output is py::tuple, py_res: " << py_res;
  } else if (py::isinstance<py::list>(py_res)) {
    MS_LOG(DEBUG) << "Real output is py::list, py_res: " << py_res;
  } else if (py::isinstance<py::dict>(py_res)) {
    MS_LOG(DEBUG) << "Real output is py::dict, py_res: " << py_res;
  } else if (py::isinstance<py::set>(py_res)) {
    MS_LOG(DEBUG) << "Real output is py::set, py_res: " << py_res;
  } else {
    MS_LOG(EXCEPTION) << "The output is invalid, py_res: " << py_res;
  }
  AttachPyOutputData(py_res);
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, PyExecute, PyExecuteCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
