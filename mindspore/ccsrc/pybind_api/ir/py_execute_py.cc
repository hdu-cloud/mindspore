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

#include "pybind11/pybind11.h"
#include "pybind_api/pybind_patch.h"

#include "mindspore/core/ops/py_execute.h"
#include "mindspore/ccsrc/include/common/utils/python_adapter.h"
#include "mindspore/ccsrc/pipeline/jit/parse/data_converter.h"
#include "mindspore/ccsrc/pybind_api/ir/tensor_py.h"

namespace py = pybind11;
namespace mindspore {
namespace {
py::object CallPythonGetGlobalParams() {
  constexpr auto python_mod_parse = "mindspore._extends.parse";  // The same as PYTHON_MOD_PARSE_MODULE[]
  py::module mod = python_adapter::GetPyModule(python_mod_parse);
  constexpr auto python_get_dict = "get_global_params";
  return python_adapter::CallPyModFn(mod, python_get_dict);
}
}  // namespace

class PyExecuteInitializer {
 public:
  PyExecuteInitializer() { mindspore::ops::PyExecuteInfer::set_infer_handler(InferPy); }

  ~PyExecuteInitializer() = default;

 private:
  // TODO(zh_qh): Will check the abstract shape and type later.
  static void InferPy(const std::vector<AbstractBasePtr> &input_args) {
    const auto &script_abs = input_args[0];
    const auto &script = script_abs->BuildValue();
    const auto &script_str = dyn_cast<StringImm>(script);

    const auto &keys_tuple_abs = input_args[1];
    const auto &keys_tuple = keys_tuple_abs->BuildValue();
    const auto &keys = dyn_cast<ValueSequence>(keys_tuple);
    const auto &values_tuple_abs = input_args[2];
    const auto &values_tuple = values_tuple_abs->BuildValue();
    if (values_tuple == kAnyValue) {
      MS_LOG(EXCEPTION) << "Value tuple should not be anyvalue.";
    }
    const auto &values = dyn_cast<ValueSequence>(values_tuple);
    MS_LOG(DEBUG) << "script: " << script->ToString() << ", keys_tuple: " << keys_tuple->ToString()
                  << ", values_tuple: " << values_tuple->ToString();

    py::gil_scoped_acquire gil_acquire;
    py::dict local_dict;
    for (size_t i = 0; i < keys->size(); ++i) {
      const auto &key = (*keys)[i];
      const auto &key_str = dyn_cast<StringImm>(key);
      MS_EXCEPTION_IF_NULL(key_str);
      const auto &value = (*values)[i];
      const auto &tuple_abs = values_tuple_abs->cast<abstract::AbstractSequencePtr>();
      const auto &value_abs = (*tuple_abs)[i];
      if (value->isa<tensor::Tensor>()) {
        const auto &tensor = value->cast<tensor::TensorPtr>();
        const auto &py_array_value = python_adapter::PyAdapterCallback::TensorToNumpy(*tensor);
        local_dict[py::str(key_str->value())] = py_array_value;
        continue;
      }
      local_dict[py::str(key_str->value())] = value;
    }
    const auto &global_dict = CallPythonGetGlobalParams();
    const auto &py_script = py::str(script_str->value());
    auto params = py::tuple(2);
    params[0] = global_dict;
    params[1] = local_dict;
    MS_LOG(DEBUG) << "py_script: " << py_script << ", params: " << params;
    const auto &py_res = parse::data_converter::CallPythonScript(py_script, params);
    MS_LOG(DEBUG) << "py_res: " << py_res;
    if (py::isinstance<py::none>(py_res)) {
      MS_LOG(EXCEPTION) << "py_res is none";
    } else if (py::isinstance<py::array>(py_res)) {
      MS_LOG(DEBUG) << "is py::array, py_res: " << py_res;
      const auto &res_tensor = tensor::TensorPy::MakeTensorOfNumpy(py_res);
      MS_LOG(DEBUG) << "res_tensor: " << res_tensor->ToString();
    } else if (py::isinstance<py::float_>(py_res)) {
      MS_LOG(DEBUG) << "is py::float_, py_res: " << py_res;
    } else if (py::isinstance<py::int_>(py_res)) {
      MS_LOG(DEBUG) << "is py::int_, py_res: " << py_res;
    } else if (py::isinstance<py::bool_>(py_res)) {
      MS_LOG(DEBUG) << "is py::bool_, py_res: " << py_res;
    } else if (py::isinstance<py::str>(py_res)) {
      MS_LOG(DEBUG) << "is py::str, py_res: " << py_res;
    } else if (py::isinstance<py::tuple>(py_res)) {
      MS_LOG(DEBUG) << "is py::tuple, py_res: " << py_res;
    } else if (py::isinstance<py::list>(py_res)) {
      MS_LOG(DEBUG) << "is py::list, py_res: " << py_res;
    } else if (py::isinstance<py::dict>(py_res)) {
      MS_LOG(DEBUG) << "is py::dict, py_res: " << py_res;
    } else if (py::isinstance<py::set>(py_res)) {
      MS_LOG(DEBUG) << "is py::set, py_res: " << py_res;
    } else {
      MS_LOG(EXCEPTION) << "py_res is invalid, py_res: " << py_res;
    }
  }
};

static PyExecuteInitializer py_execute_initializer;
}  // namespace mindspore
