/**
 * This is the C++ adaptation and derivative work of Myia (https://github.com/mila-iqia/myia/).
 *
 * Copyright 2019-2022 Huawei Technologies Co., Ltd
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

#include "pipeline/jit/pipeline.h"

#include <memory>
#include <sstream>
#include <map>
#include <cstdlib>
#include <algorithm>
#include <iomanip>
#include <unordered_map>

#include "pybind_api/pybind_patch.h"
#include "pybind11/pybind11.h"
#include "ir/param_info.h"
#include "pipeline/jit/action.h"
#include "pipeline/jit/pass.h"
#include "pipeline/jit/parse/data_converter.h"
#include "pipeline/jit/static_analysis/async_eval_result.h"
#include "pipeline/pynative/pynative_execute.h"
#include "frontend/optimizer/ad/dfunctor.h"
#include "frontend/optimizer/ad/prim_bprop_optimizer.h"
#include "include/common/utils/parallel_context.h"
#include "frontend/parallel/step_parallel_utils.h"
#include "frontend/parallel/parameter_manager.h"
#include "frontend/parallel/graph_util/get_parallel_info.h"
#include "frontend/parallel/auto_parallel/graph_costmodel.h"
#include "include/common/utils/config_manager.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/convert_utils_py.h"
#include "utils/ms_context.h"
#include "utils/shape_utils.h"
#include "utils/info.h"
#include "utils/crypto.h"
#include "utils/phase.h"
#include "include/common/utils/comm_manager.h"
#include "utils/interpret_node_recorder.h"
#include "include/common/debug/anf_ir_dump.h"
#include "include/common/debug/dump_proto.h"
#include "pipeline/jit/debug/anf_ir_utils.h"
#include "pipeline/jit/debug/trace.h"
#include "pipeline/jit/event_message_print.h"
#include "include/common/debug/draw.h"
#include "include/common/debug/common.h"
#include "load_mindir/load_model.h"
#include "backend/graph_compiler/segment_runner.h"
#include "backend/common/session/executor_manager.h"
#include "backend/common/session/session_factory.h"
#include "runtime/hardware/device_context_manager.h"
#include "runtime/device/kernel_runtime_manager.h"
#include "runtime/pynative/op_executor.h"
#include "runtime/device/stream_synchronizer.h"
#include "distributed/collective/collective_manager.h"
#include "mindspore/ccsrc/utils/dynamic_obfuscation/dynamic_obfuscation.h"
#include "mindspore/ccsrc/utils/dynamic_obfuscation/registry_opaque_predicate.h"
#include "mindspore/ccsrc/plugin/device/cpu/kernel/pyexecute/py_execute_cpu_kernel.h"
#include "distributed/init.h"
#include "profiler/device/profiling.h"
#include "kernel/akg/akg_kernel_build_manager.h"
#include "kernel/graph_kernel_info.h"
#include "include/backend/data_queue/data_queue_mgr.h"
#include "pybind_api/ir/log_adapter_py.h"
#ifndef ENABLE_SECURITY
#include "debug/data_dump/dump_json_parser.h"
#endif
#if defined(__linux__) && defined(WITH_BACKEND)
#include "ps/constants.h"
#include "ps/util.h"
#include "ps/ps_cache/ps_data/ps_data_prefetch.h"
#include "distributed/cluster/cluster_context.h"
#include "runtime/graph_scheduler/embedding_cache_scheduler.h"
#include "ps/scheduler.h"
#endif
#ifdef ENABLE_DUMP_IR
#include "debug/rdr/graph_recorder.h"
#include "include/common/debug/rdr/recorder_manager.h"
#include "ir/cell.h"
#endif

namespace mindspore {
// namespace to support intermediate representation definition
namespace pipeline {
using Tensor = mindspore::tensor::Tensor;
using MetaTensor = mindspore::tensor::MetaTensor;
using MetaSparseTensor = mindspore::tensor::MetaSparseTensor;
using CSRTensor = mindspore::tensor::CSRTensor;
using COOTensor = mindspore::tensor::COOTensor;
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTensorPtr;
using mindspore::abstract::AbstractTuple;
using mindspore::abstract::AbstractTuplePtr;

const char IR_TYPE_ANF[] = "anf_ir";
const char IR_TYPE_ONNX[] = "onnx_ir";
const char IR_TYPE_MINDIR[] = "mind_ir";

GraphExecutorPyPtr GraphExecutorPy::executor_ = nullptr;
std::mutex GraphExecutorPy::instance_lock_;

std::unordered_map<abstract::AbstractBasePtrList, uint64_t, abstract::AbstractBasePtrListHasher,
                   abstract::AbstractBasePtrListEqual>
  g_args_cache;

namespace {
#ifdef ENABLE_DUMP_IR
std::string GetBaseNameForIR(int64_t stage_idx, const std::string &action_name) {
  std::ostringstream oss;
  int spaces = 2;
  oss << std::setfill('0') << std::setw(spaces) << stage_idx << "_" << action_name;
  return oss.str();
}
#endif

bool CheckAllTensor(const ValueTuplePtr &value_tuple) {
  auto elements = value_tuple->value();
  for (auto element : elements) {
    MS_EXCEPTION_IF_NULL(element);
    if (!(element->isa<ValueTuple>() && CheckAllTensor(element->cast<ValueTuplePtr>())) &&
        !(element->isa<MetaTensor>())) {
      return false;
    }
  }
  return true;
}

bool Mutable(const py::object &obj, const ValuePtr &value) {
  // If a tensor has been set const arg, it should not be mutable.
  if (value->isa<MetaTensor>()) {
    constexpr char const_arg_attr[] = "const_arg";
    if (py::hasattr(obj, const_arg_attr) && py::cast<bool>(py::getattr(obj, const_arg_attr))) {
      return false;
    }
  }
  constexpr char mutable_attr[] = "__ms_mutable__";
  return py::hasattr(obj, mutable_attr) && py::cast<bool>(py::getattr(obj, mutable_attr));
}

void CheckAndConvertToVariableLenSequence(const py::object &obj, AbstractBasePtr abs) {
  constexpr char variable_len_attr[] = "__ms_dynamic_len__";
  bool dynamic_len = (py::hasattr(obj, variable_len_attr) && py::cast<bool>(py::getattr(obj, variable_len_attr)));
  if (!dynamic_len) {
    return;
  }
  if (!abs->isa<abstract::AbstractSequence>()) {
    MS_EXCEPTION(TypeError) << "For mutable, when the variable_len the True, the first input should be"
                            << " list or tuple, but got: " << abs->ToString();
  }
  auto abs_seq = abs->cast<abstract::AbstractSequencePtr>();
  abs_seq->CheckAndConvertToDynamicLenSequence();
}

bool TensorArgMutable(const py::object &obj, const ValuePtr &value) {
  if (!value->isa<MetaTensor>()) {
    return false;
  }
  constexpr char const_arg_attr[] = "const_arg";
  return !py::hasattr(obj, const_arg_attr) || !py::cast<bool>(py::getattr(obj, const_arg_attr));
}

bool EnableTupleBroaden(const ValuePtr &value, bool enable_tuple_broaden) {
  return enable_tuple_broaden && value->isa<ValueTuple>() && CheckAllTensor(value->cast<ValueTuplePtr>());
}

bool GradForScalar(const ValuePtr &value) {
  return MsContext::GetInstance()->get_param<bool>(MS_CTX_GRAD_FOR_SCALAR) && value->isa<Scalar>();
}

AbstractBasePtr ArgsToAbstract(const py::object &arg, const ValuePtr &value, bool enable_tuple_broaden = false) {
  bool broaden = TensorArgMutable(arg, value) || Mutable(arg, value) || value->isa<MetaSparseTensor>() ||
                 EnableTupleBroaden(value, enable_tuple_broaden) || GradForScalar(value);
  auto ret = abstract::FromValue(value, broaden);
  CheckAndConvertToVariableLenSequence(arg, ret);
  return ret;
}

bool CheckArgValid(const py::handle &arg) {
  if (py::isinstance<py::list>(arg) || py::isinstance<py::tuple>(arg)) {
    auto vector_arg = py::cast<py::list>(arg);
    return std::all_of(vector_arg.begin(), vector_arg.end(), CheckArgValid);
  }

  if (py::isinstance<py::dict>(arg)) {
    auto dict_arg = py::cast<py::dict>(arg);
    return std::all_of(dict_arg.begin(), dict_arg.end(), [](const auto &pair) { return CheckArgValid(pair.second); });
  }

  if (py::isinstance<Tensor>(arg)) {
    auto tensor = py::cast<TensorPtr>(arg);
    if (tensor->data_type() == kNumberTypeBool) {
      MS_LOG(INFO) << "It is not recommended to use a tensor of bool data type as network input, which may cause "
                   << "operator compilation failure. For more details, please refer to the FAQ at "
                   << "https://mindspore.cn/search?[AddN]%20input(kNumberTypeBool.";
    }
  }

  return py::isinstance<py::int_>(arg) || py::isinstance<py::float_>(arg) || py::isinstance<py::none>(arg) ||
         py::isinstance<Number>(arg) || py::isinstance<Tensor>(arg) || py::isinstance<CSRTensor>(arg) ||
         py::isinstance<COOTensor>(arg);
}

std::string GetCompileExceptionInfo() {
  std::ostringstream oss;
  trace::GetTraceStackInfo(oss);
  return oss.str();
}

void SetLoopCount(const ResourcePtr &resource) {
  MS_EXCEPTION_IF_NULL(resource);
  auto func_graph = resource->func_graph();
  if (func_graph != nullptr && func_graph->manager() != nullptr) {
    auto manager = func_graph->manager();
    size_t graph_nums = manager->func_graphs().size();
    int64_t loop_size = ConfigManager::GetInstance().iter_num();
    const auto context_ptr = MsContext::GetInstance();
    bool enable_mind_rt = context_ptr->get_param<bool>(MS_CTX_ENABLE_MINDRT);
    if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
      resource->set_vm_loop(!(context_ptr->get_param<bool>(MS_CTX_IS_MULTI_GRAPH_SINK) || enable_mind_rt), loop_size);
    } else if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice) {
      bool run_with_mind_rt = graph_nums == 1 || enable_mind_rt;
      resource->set_vm_loop(!run_with_mind_rt, loop_size);
    }
    MS_LOG(INFO) << "Change vm_loop_flag to " << resource->vm_loop_flag() << ", set loop_size to " << loop_size;
  }
}

std::map<string, string> GenerateJitConfigMap(const py::dict &jit_config) {
  std::map<string, string> ret{};
  for (auto jit_param = jit_config.begin(); jit_param != jit_config.end(); ++jit_param) {
    auto param_name = py::cast<std::string>(jit_param->first);
    auto param_value = py::cast<std::string>(jit_param->second);
    ret[param_name] = param_value;
  }
  return ret;
}

void RecordInitStatus() {
  static bool printed = false;
  if (!printed) {
    MS_LOG(INFO) << "Status record: system init.";
    printed = true;
  }
}

void RecordExitStatus() { MS_LOG(INFO) << "Status record: system exit."; }

std::string ToOrdinal(const size_t &i) {
  auto suffix = "th";
  if (i == kIndex1) {
    suffix = "st";
  } else if (i == kIndex2) {
    suffix = "nd";
  } else if (i == kIndex3) {
    suffix = "rd";
  }
  return std::to_string(i) + suffix;
}

py::object GetPyExecuteOutput(const AnfNodePtr &output) {
  static const auto support_fallback_runtime = (common::GetEnv("MS_DEV_ENABLE_FALLBACK_RUNTIME") == "1");
  if (support_fallback_runtime) {
    std::function<AnfNodePtr(const AnfNodePtr &)> get_real_output = [&get_real_output](const AnfNodePtr &node) {
      if (IsPrimitiveCNode(node, prim::kPrimDepend)) {
        const auto cnode = dyn_cast<CNode>(node);
        MS_EXCEPTION_IF_NULL(cnode);
        return get_real_output(cnode->input(1));
      }
      return node;
    };
    const auto &real_output = get_real_output(output);
    MS_LOG(INFO) << "Real output: " << real_output << ", " << real_output->DebugString()
                 << ", has \'PyExecuteOutputData\': " << real_output->has_user_data<kernel::PyExecuteOutputData>();
    if (real_output->has_user_data<kernel::PyExecuteOutputData>()) {
      py::gil_scoped_acquire gil_acquire;
      const auto &output_data = real_output->user_data<kernel::PyExecuteOutputData>();
      py::object res_obj = output_data->obj;
      MS_LOG(INFO) << "Has \'PyExecuteOutputData\', just return it. res_obj: " << res_obj;
      if (!py::isinstance<py::none>(res_obj)) {
        return res_obj;
      }
    }
  }
  return py::none();
}
}  // namespace

std::string GetObjDesc(const py::object &source_obj) {
  std::string obj_desc;
  if (py::hasattr(source_obj, parse::PYTHON_PARSE_METHOD)) {
    auto cell_class_name = source_obj.attr("__class__").attr("__name__");
    auto ms_function_name = source_obj.attr(parse::PYTHON_PARSE_METHOD);
    obj_desc = "'" + py::cast<std::string>(cell_class_name) + "." + py::cast<std::string>(ms_function_name) + "'";
  } else {
    if (py::hasattr(source_obj, "__name__")) {
      auto ms_function_name = source_obj.attr("__name__");
      obj_desc = "'" + py::cast<std::string>(ms_function_name) + "'";
    } else if (py::isinstance<Cell>(source_obj)) {
      auto cell_class_name = source_obj.attr("__class__").attr("__name__");
      obj_desc = "'" + py::cast<std::string>(cell_class_name) + ".construct'";
    } else {
      MS_EXCEPTION(TypeError) << "The source object is invalid: " << py::str(source_obj);
    }
  }
  return obj_desc;
}

void CheckArgsValid(const py::object &source_obj, const py::tuple &args) {
  std::string obj_desc = GetObjDesc(source_obj);
  for (size_t i = 0; i < args.size(); i++) {
    if (!CheckArgValid(args[i])) {
      MS_EXCEPTION(TypeError)
        << "The inputs types of the outermost network " << obj_desc
        << " support bool, int, float, None, Tensor, Parameter, "
           "mstype.Number(mstype.bool, mstype.int, mstype.float, mstype.uint), "
           "and tuple or list containing only these types, and dict whose values are these types, but the "
        << ToOrdinal(i + 1) << " arg type is " << args[i].get_type() << ", value is '" << py::str(args[i]) << "'.\n"
        << "For more details, please search 'outermost network' at https://www.mindspore.cn.";
    }
  }
}

py::object GraphExecutorPy::GenerateArgumentsKey(const py::tuple &args, bool enable_tuple_broaden) {
  MS_LOG(DEBUG) << "GenerateArgumentsKey args size: " << args.size()
                << ", enable_tuple_broaden: " << enable_tuple_broaden;

  abstract::AbstractBasePtrList args_abs;
  cur_convert_input_.clear();
  std::size_t size = args.size();
  for (std::size_t i = 0; i < size; i++) {
    ValuePtr converted = nullptr;
    if (!parse::ConvertData(args[i], &converted)) {
      MS_EXCEPTION(TypeError) << "parse::ConvertData for " << i << "th argument failed, the argument type is "
                              << args[i].get_type() << ", value is '" << py::str(args[i]) << "'.";
    }
    AbstractBasePtr abs = ArgsToAbstract(args[i], converted, enable_tuple_broaden);
    (void)args_abs.emplace_back(abs);
    // The 'converted' maybe a Parameter, we need connect it to the Parameter of func graph,
    // so we keep all inputs for subsequent procedure.
    (void)cur_convert_input_.emplace(args[i].ptr(), std::make_pair(converted, abs));
  }

  // If cache matched no need CheckArgsValid
  auto iter = g_args_cache.find(args_abs);
  if (iter != g_args_cache.end()) {
    return py::int_(iter->second);
  }

  static uint64_t key_counter = 0;
  g_args_cache[args_abs] = key_counter;
  MS_LOG(INFO) << "Generate a new compile key for new args, key: " << key_counter;
  return py::int_(key_counter++);
}

py::bool_ VerifyInputSignature(const py::list &input_signature, const py::tuple &inputs) {
  MS_LOG(DEBUG) << "Verify args size:" << inputs.size();
  if (inputs.size() != input_signature.size()) {
    MS_LOG(ERROR) << "Signature size not equal to args size";
    return false;
  }

  size_t count = 0;
  for (auto arg_obj : inputs) {
    if (py::isinstance<Tensor>(arg_obj)) {
      MS_LOG(DEBUG) << "Verify Tensor";
      auto m_tensor = arg_obj.cast<std::shared_ptr<Tensor>>();
      if (m_tensor == nullptr) {
        MS_LOG(ERROR) << "Verify Tensor error, get ptr is null";
        return false;
      }
      auto sig = input_signature[count].cast<std::shared_ptr<MetaTensor>>();
      ShapeVector sig_shape = sig->shape();
      TypePtr sig_type = sig->Dtype();

      ShapeVector tensor_shape = m_tensor->shape_c();
      if (tensor_shape != sig_shape) {
        MS_LOG(ERROR) << "Python input shape is incompatible with input_signature";
        return false;
      }

      if (*m_tensor->Dtype() != *sig_type) {
        MS_LOG(ERROR) << "Python input type(" << m_tensor->Dtype()->ToString() << ") incompatible with input_signature("
                      << sig_type->ToString() << ")";
        return false;
      }
    }
    count++;
  }

  return true;
}

ResourcePtr GraphExecutorPy::GetResource(const std::string &phase) {
  MS_LOG(DEBUG) << "Phase size:" << info_.size();
  if (info_.count(phase) == 0) {
    return nullptr;
  }
  return info_[phase]->resource;
}

FuncGraphPtr GraphExecutorPy::GetFuncGraph(const std::string &phase) {
  if (info_.count(phase) == 0) {
    MS_LOG(INFO) << "No executor info. found for phase: " << phase;
    return nullptr;
  }
  return info_[phase]->func_graph;
}

FuncGraphPtr GraphExecutorPy::GetGradGraph(const std::string &phase) {
  if (phase.empty()) {
    MS_LOG(EXCEPTION) << "The input phase is empty.";
  }
  if (info_.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "No phase in executor:" << phase;
  }

  auto execute_info = info_[phase];
  MS_EXCEPTION_IF_NULL(execute_info);
  auto grad_graph = execute_info->grad_graph;
  MS_EXCEPTION_IF_NULL(grad_graph);
  return grad_graph;
}

void GraphExecutorPy::SetGradGraph(const FuncGraphPtr &grad_graph, const std::string &phase) {
  if (phase.empty()) {
    MS_LOG(EXCEPTION) << "The input phase is empty.";
  }
  if (info_.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "No phase in executor: " << phase;
  }

  auto execute_info = info_[phase];
  MS_EXCEPTION_IF_NULL(execute_info);
  if (execute_info->grad_graph != nullptr) {
    MS_LOG(DEBUG) << "The grad graph has existed, phase is: " << phase;
  }
  MS_EXCEPTION_IF_NULL(grad_graph);
  execute_info->grad_graph = grad_graph;
}

compile::VmEvalFuncPtr GraphExecutorPy::GetVmEvalFunc(const std::string &phase) {
  ResourcePtr res = GetResource(phase);
  MS_EXCEPTION_IF_NULL(res);
  if (res->HasResult(kOutput) && res->GetResult(kOutput).is<compile::VmEvalFuncPtr>()) {
    return res->GetResult(kOutput).cast<compile::VmEvalFuncPtr>();
  }
  MS_LOG(ERROR) << "GetVmEvalFunc vm model can't find kOutput:" << kOutput;
  return nullptr;
}

bool GraphExecutorPy::HasCompiled(const std::string &phase) const {
  if (info_.count(phase) == 0) {
    return false;
  }
  return true;
}

py::bytes GraphExecutorPy::GetFuncGraphProto(const std::string &phase, const std::string &ir_type) {
  FuncGraphPtr fg_ptr = GetFuncGraph(phase);
  if (fg_ptr == nullptr) {
    for (const auto &item : info_) {
      MS_LOG(DEBUG) << "Phase key is: " << item.first;
    }
    MS_LOG(EXCEPTION) << "Can not find func graph " << phase;
  }

  if (ir_type == IR_TYPE_ANF) {
    std::string proto_str = GetFuncGraphProtoString(fg_ptr);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Export ANF format model failed.";
    }
    return proto_str;
  }

  if (ir_type == IR_TYPE_ONNX) {
    std::string proto_str = GetOnnxProtoString(fg_ptr);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Export ONNX format model failed.";
    }
    return proto_str;
  }

  if (ir_type == IR_TYPE_MINDIR) {
    // obfuscate model
    std::string proto_str = GetBinaryProtoString(fg_ptr);
    if (proto_str.empty()) {
      MS_LOG(EXCEPTION) << "Export MINDIR format model failed.";
    }
    return proto_str;
  }

  MS_LOG(EXCEPTION) << "Unknown ir type: " << ir_type;
}

py::bytes GraphExecutorPy::GetObfuscateFuncGraphProto(const std::string &phase, const float obf_ratio,
                                                      const int obf_password, const int append_password) {
  FuncGraphPtr fg_ptr = GetFuncGraph(phase);
  // obfuscate model
  if (obf_password == 0) {
    (void)mindspore::kernel::CustomizedOpaquePredicate::GetInstance().set_func_names();
    MS_LOG(DEBUG) << "[GetObfuscateFuncGraphProto] set customized function names finished";
  }
  mindspore::DynamicObfuscator dynamic_obfuscator(obf_ratio, obf_password, append_password);
  mindspore::FuncGraphPtr obfuscated_graph = dynamic_obfuscator.ObfuscateMindIR(fg_ptr);

  std::string proto_str = GetBinaryProtoString(obfuscated_graph);
  if (proto_str.empty()) {
    MS_LOG(EXCEPTION) << "GetBinaryProtoString failed.";
  }
  return proto_str;
}

py::bytes GraphExecutorPy::GetOptimizeGraphProto(const std::string &phase) {
  if (info_.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "No phase in executor: " << phase;
  }
  FuncGraphPtr fg_ptr = info_[phase]->resource->optimize_graph();
  if (fg_ptr == nullptr) {
    MS_LOG(WARNING) << "Can not find optimize graph.";
    return "";
  }
  std::string proto_str = GetFuncGraphProtoString(fg_ptr);
  if (proto_str.empty()) {
    MS_LOG(EXCEPTION) << "Export optimize graph proto string failed.";
  }
  return proto_str;
}

void GraphExecutorPy::SetJitConfig(const py::dict &jit_config) { jit_config_ = GenerateJitConfigMap(jit_config); }

py::dict GraphExecutorPy::GetParallelGraphInfo(const std::string &phase) {
  MS_LOG(DEBUG) << "GetParallelGraphInfo!";
  std::string parallel_phase = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(parallel_phase);
  if (graph == nullptr) {
    MS_LOG(EXCEPTION) << "Can not access FuncGraph according to phase: " << parallel_phase;
  }

  return mindspore::parallel::GetParallelCNodeInfoFromGraph(graph);
}

py::dict GraphExecutorPy::GetParameterLayout(const std::string &phase) {
  MS_LOG(DEBUG) << "GetParameterLayout!";
  std::string layout_graph = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(layout_graph);
  if (graph == nullptr) {
    auto resource = info_[phase]->resource;
    return mindspore::parallel::GetParameterLayoutFromResource(resource);
  }
  return mindspore::parallel::GetParameterLayoutFromGraph(graph);
}

py::dict GraphExecutorPy::GetCNodeStrategy(const std::string &phase) {
  MS_LOG(DEBUG) << "GetCNodeStrategy!";
  return stra_dict_[phase];
}

py::list GraphExecutorPy::GetParallelParameterNameList(const std::string &phase) {
  std::string param_graph = phase + kStepParallelGraph;
  auto graph = GetFuncGraph(param_graph);
  if (graph == nullptr) {
    auto resource = info_[phase]->resource;
    return mindspore::parallel::GetParallelParameterNameListFromResource(resource);
  }
  return mindspore::parallel::GetParallelParameterNameListFromGraph(graph);
}

void GraphExecutorPy::SetCNodeStrategy(const std::string &name, const parallel::Strategies &strategy) {
  MS_LOG(DEBUG) << "SetCNodeStrategy!";
  stra_dict_[phase_][py::str(name)] = strategy;
}

size_t GraphExecutorPy::GetNumOpsInfo(const std::string &phase) {
  MS_LOG(DEBUG) << "GetNumOpsInfo!";
  return phase_to_num_op_info_[phase];
}

void GraphExecutorPy::SetNumOpsInfo(size_t num_ops) {
  MS_LOG(DEBUG) << "SetNumOpsInfo!";
  phase_to_num_op_info_[phase_] = num_ops;
}

py::dict GraphExecutorPy::GetAllreduceFusion(const std::string &phase) {
  MS_LOG(INFO) << "GetAllreduceFusion!";
  auto graph = GetFuncGraph(phase);
  return mindspore::parallel::GetAllreduceFusion(graph);
}

// Not support multi thread, not support nested call too.
// Here using nested_called flg to avoid nested call.
void GraphExecutorPy::DelNetRes(const py::set &id) {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  auto device_target = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  std::string backend = ms_context->backend_policy();
  if (device_target == kAscendDevice && backend == "ge") {
    FinalizeBackend();
  }
  for (auto item : id) {
    DelOneNetRes(item);
  }
#ifdef WITH_BACKEND
  if (backend == "ge" && !id.empty() && info_.size() == 0) {
    DeviceContext *device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({"GE", 0});
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
    // because Ge only support one Session exist at the same time ,so we delete the old one
    device_context->GetDeprecatedInterface()->EraseGeResource();
  }
#endif
}

void GraphExecutorPy::DelOneNetRes(const py::handle &py_phase) {
  MS_LOG(INFO) << "Delete one net resource start";
  if (!pybind11::isinstance<py::str>(py_phase)) {
    MS_LOG(ERROR) << "Expect string phase, but got " << py::str(py_phase);
    return;
  }
  auto phase = pybind11::cast<std::string>(py_phase);
  auto iter = info_.find(phase);
  auto clear = false;
  if (iter != info_.end()) {
    clear = true;
    auto res = iter->second->resource;
    if (res->HasResult(kStepParallelGraph)) {
      std::string layout_graph = phase + kStepParallelGraph;
      (void)info_.erase(layout_graph);
    }
    (void)info_.erase(phase);
    MS_LOG(DEBUG) << "Delete phase: " << phase << ", info size: " << info_.size();
  }
  if (clear) {
    // Do clear here to avoid any pointer for resource.
    FuncGraphLoopBreaker::Inst().ClearCellGraphs(phase);
  }
  MS_LOG(INFO) << "Delete one net resource end.";
}

void GraphExecutorPy::ClearRes() {
  MS_LOG(INFO) << "Clean executor resource!";
  executor_ = nullptr;
}

GraphExecutorPy::~GraphExecutorPy() {
  MS_LOG(INFO) << "Release Executor!";
  ConfigManager::GetInstance().ResetConfig();
}

void GraphExecutorPy::GetWeightInfo(
  const CNodePtr &root_node, const AnfNodePtr &weight_node,
  std::map<std::string, std::pair<PrimitivePyAdapterPtr, std::string>> *fake_quant_table) const {
  MS_EXCEPTION_IF_NULL(root_node);
  MS_EXCEPTION_IF_NULL(fake_quant_table);
  std::string weight_name;
  auto x = root_node->input(1);
  MS_EXCEPTION_IF_NULL(x);
  if (IsPrimitiveCNode(weight_node, prim::kPrimLoad)) {
    weight_name = weight_node->cast_ptr<CNode>()->input(1)->cast_ptr<Parameter>()->name();
  } else {
    auto para = weight_node->cast_ptr<Parameter>();
    MS_EXCEPTION_IF_NULL(para);
    weight_name = para->name();
  }
  // find the fakequant from input
  int64_t count = 0;
  const int64_t max_depth = 5;
  auto is_quant_cnode = [](const AnfNodePtr &node) {
    return IsPrimitiveCNode(node, prim::kPrimFakeQuantPerLayer) ||
           IsPrimitiveCNode(node, prim::kPrimFakeQuantPerChannel) ||
           IsPrimitiveCNode(node, prim::kPrimFakeLearnedScaleQuantPerLayer) ||
           IsPrimitiveCNode(node, prim::kPrimFakeLearnedScaleQuantPerChannel);
  };
  while (!is_quant_cnode(x)) {
    if (count >= max_depth) {
      break;
    }
    auto cnode = x->cast_ptr<CNode>();
    if (cnode == nullptr || cnode->size() <= 1) {
      break;
    }
    x = cnode->input(1);
    count += 1;
  }
  if (x->isa<Parameter>() || IsPrimitiveCNode(x, prim::kPrimLoad)) {
    (*fake_quant_table)[weight_name] = std::make_pair(nullptr, "input");
  }
  // get the fakequant parameter minq's name
  if (!is_quant_cnode(x)) {
    return;
  }
  auto cnode = x->cast_ptr<CNode>();
  constexpr size_t expect_input_size = 4;
  if (cnode == nullptr || cnode->IsApply(prim::kPrimLoad) || cnode->size() != expect_input_size) {
    return;
  }
  const size_t fakequant_index = 2;
  auto fakequant_min_node = cnode->input(fakequant_index);
  if (!fakequant_min_node->isa<Parameter>() && !IsPrimitiveCNode(fakequant_min_node, prim::kPrimLoad)) {
    return;
  }
  std::string fakequant_min_node_name;
  if (IsPrimitiveCNode(fakequant_min_node, prim::kPrimLoad)) {
    fakequant_min_node_name = fakequant_min_node->cast_ptr<CNode>()->input(1)->cast_ptr<Parameter>()->name();
  } else {
    auto param = fakequant_min_node->cast_ptr<Parameter>();
    MS_EXCEPTION_IF_NULL(param);
    fakequant_min_node_name = param->name();
  }
  auto quant_op = GetValuePtr<PrimitivePy>(cnode->input(0));
  if (quant_op == nullptr) {
    return;
  }
  (*fake_quant_table)[weight_name] = std::make_pair(quant_op->adapter(), fakequant_min_node_name);
}

std::map<std::string, std::pair<PrimitivePyAdapterPtr, std::string>> GraphExecutorPy::FetchInfoForQuantExport(
  const std::string &phase) {
  FuncGraphPtr func_graph = info_[phase]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "FetchInfoForQuantExport func graph(" << func_graph->ToString() << ") phase(" << phase << ")!";
  std::map<std::string, std::pair<PrimitivePyAdapterPtr, std::string>> fake_quant_table;
  auto filter = [](const AnfNodePtr &node) {
    return !(IsPrimitiveCNode(node, prim::kPrimConv2D) || IsPrimitiveCNode(node, prim::kPrimMatMul) ||
             IsPrimitiveCNode(node, prim::kPrimDepthwiseConv2dNative));
  };
  std::vector<AnfNodePtr> nodes = DeepScopedGraphSearchWithFilter(func_graph->get_return(), AlwaysInclude, filter);
  auto is_quant_cnode = [](const AnfNodePtr &node) {
    return IsPrimitiveCNode(node, prim::kPrimFakeQuantPerLayer) ||
           IsPrimitiveCNode(node, prim::kPrimFakeQuantPerChannel) ||
           IsPrimitiveCNode(node, prim::kPrimFakeLearnedScaleQuantPerLayer) ||
           IsPrimitiveCNode(node, prim::kPrimFakeLearnedScaleQuantPerChannel);
  };
  const size_t root_node_size = 3;
  const size_t weight_index = 2;
  for (const auto &node : nodes) {
    auto root_node = node->cast<CNodePtr>();
    if (root_node == nullptr || root_node->size() != root_node_size) {
      continue;
    }
    auto weight = root_node->input(weight_index);
    if (!is_quant_cnode(weight)) {
      auto tuple_node = weight->cast_ptr<CNode>();
      if (tuple_node != nullptr) {
        auto fake_node = tuple_node->input(1);
        if (!is_quant_cnode(fake_node)) {
          continue;
        } else {
          weight = fake_node;
        }
      }
    }
    // get parameter weight's name
    auto cnode = weight->cast_ptr<CNode>();
    MS_EXCEPTION_IF_NULL(cnode);
    auto weight_node = cnode->input(weight_index);
    MS_EXCEPTION_IF_NULL(weight_node);
    if (!weight_node->isa<Parameter>() && !IsPrimitiveCNode(weight_node, prim::kPrimLoad)) {
      continue;
    }
    GetWeightInfo(root_node, weight_node, &fake_quant_table);
  }
  return fake_quant_table;
}

void GraphExecutorPy::SaveCompiledGraph(const std::string &phase) {
  // save the graph to GraphExecutorPy
  FuncGraphPtr func_graph = info_[phase]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(INFO) << "Save compiled func graph(" << func_graph->ToString() << ") phase(" << phase << ")!";
  info_[phase]->func_graph = func_graph;

  if ((func_graph != nullptr) && parallel::IsAutoParallelCareGraph(func_graph)) {
    MS_LOG(DEBUG) << "Save model parallel parameter layout graph!";
    auto res = info_[phase]->resource;
    // When using frontend compile cache, model parallel parameter layout graph is not saved.
    if (res->HasResult(kStepParallelGraph)) {
      func_graph = res->GetResult(kStepParallelGraph).cast<FuncGraphPtr>();
      ExecutorInfoPtr executor_info = std::make_shared<ExecutorInfo>();
      std::string layout_graph = phase + kStepParallelGraph;
      executor_info->func_graph = func_graph;
      info_[layout_graph] = executor_info;
    }
  } else {
    MS_LOG(DEBUG) << "Save model parallel parameter layout graph null!";
  }
  MS_LOG(INFO) << "End save compiled func graph!";
}

void GraphExecutorPy::GetGeBackendPolicy() const {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->backend_policy();
  if (backend != "ge") {
    MS_LOG(EXCEPTION) << backend << " backend policy is not supported under ge backend!";
  }
}

bool IsPhaseExportAir(const std::string &phase) {
  auto phase_to_export = "export.air";
  return phase.rfind(phase_to_export) != std::string::npos;
}

bool IsPhaseTrain(const std::string &phase) {
  const std::string phase_to_train = "train";
  return phase.rfind(phase_to_train) != std::string::npos;
}

bool IsPhaseLoadFromMindIR(const std::string &phase) {
  const std::string mindir_graph = "graph_load_from_mindir";
  return phase.rfind(mindir_graph) != std::string::npos;
}

std::vector<ActionItem> GetPipeline(const ResourcePtr &resource, const std::string &phase, bool use_vm) {
  MS_EXCEPTION_IF_NULL(resource);
  bool is_air = IsPhaseExportAir(phase);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  std::string backend = ms_context->backend_policy();
#if defined(__linux__) && defined(WITH_BACKEND)
  if (distributed::cluster::ClusterContext::instance()->initialized()) {
    auto node = distributed::cluster::ClusterContext::instance()->node();
    MS_EXCEPTION_IF_NULL(node);
    const auto &cluster_ctx = distributed::cluster::ClusterContext::instance();
    MS_EXCEPTION_IF_NULL(cluster_ctx);
    MS_LOG(INFO) << "Cluster is initialized. This node role is " << cluster_ctx->node_role();
    // If this process is not scheduler, it should be a computing graph node so common pipeline is returned.
    if (cluster_ctx->node_role() == distributed::kEnvRoleOfScheduler) {
      return PSchedulerPipeline(resource);
    }
  } else {
    if (ps::PSContext::instance()->is_scheduler()) {
      return PSchedulerPipeline(resource);
    }
  }
#endif

  compile::SetMindRTEnable();
  if (use_vm && backend != "ge" && !is_air) {
    if (IsPhaseLoadFromMindIR(phase)) {
      return MindIRPipeline();
    }
    return VmPipeline(resource);
  }
  return GePipeline();
}

void GraphExecutorPy::InitCompileCacheInfo(const ResourcePtr &resource, const std::string &phase) {
  // The compilation cache only support for training cell or functions decorated with 'jit' currently.
  // If enable compilation cache, it will get a non-empty dependent files list from python.
  if (compile_cache_dep_files_.empty()) {
    return;
  }
#ifdef ENABLE_PROFILE
  double t1 = GetTime();
#endif
  static size_t idx = 0;
  MS_EXCEPTION_IF_NULL(resource);
  resource->GetCompileCacheResource(compile_cache_dep_files_, weights_, queue_name_, idx++, &compile_cache_consistent_);
#ifdef ENABLE_PROFILE
  double t2 = GetTime();
  MsProfile::StatTime("LoadCachedFuncGraph", t2 - t1);
#endif
}

void GraphExecutorPy::ParallelPostProcess(const std::string &phase) {
  // Slice Python parameter obj
  auto layout_graph = phase + kStepParallelGraph;
  // only Parallel graph has tensor_layout
  auto root = GetFuncGraph(layout_graph);
  bool after_shard = false;
  if (phase.find("after_shard") != std::string::npos) {
    after_shard = true;
  }
  if (root == nullptr && !after_shard) {
    auto graph = info_[phase]->resource->func_graph();
    MS_EXCEPTION_IF_NULL(graph);
    parallel::InitOptimizerState(graph);
    return;
  }
  MS_EXCEPTION_IF_NULL(root);
  parallel::AutoParallelPostProcess(root);
}

// Clean all resource not used in the future and cache generated during compiling.
void CleanCompileRes(const ResourcePtr &resource) {
  MS_LOG(INFO) << "Clean compile resource start";
  abstract::AnalysisContext::ClearContext();
  ad::PrimBpropOptimizer::GetPrimBpropOptimizerInst().Clear();
  ad::g_k_prims.clear();
  ad::DFunctor::Clear();
  ReclaimOptimizer();
  resource->Clean();
  FuncGraphLoopBreaker::Inst().CleanMetaFuncGraphCache();
  MS_LOG(INFO) << "Clean compile resource end";
}

bool GraphExecutorPy::CompileInner(const py::object &source_obj, const py::tuple &args, const py::object &phase_obj,
                                   bool use_vm) {
  // Check if the phase is valid.
  if ((!py::isinstance<py::str>(phase_obj))) {
    MS_LOG(ERROR) << "The `phase` must be string.";
    return false;
  }
  // Check if the function or net is valid.
  if (py::isinstance<py::none>(source_obj)) {
    MS_LOG(ERROR) << "The source object to compile should not be None.";
    return false;
  }

  // Check if the args of function or net is valid.
  CheckArgsValid(source_obj, args);

  auto phase = py::cast<std::string>(phase_obj);
  PhaseManager::GetInstance().set_phase(phase);
  phase_ = phase;
  auto obj_desc = GetObjDesc(source_obj);
  MS_LOG(INFO) << "Start compiling, phase: " << phase;
  MS_LOG(DEBUG) << "source: {" << py::str(source_obj) << "}\nargs: " << py::str(const_cast<py::tuple &>(args));
  EventMessage::PrintCompileStartMsg(phase, obj_desc);

  ExecutorInfoPtr executor_info = std::make_shared<ExecutorInfo>();
  ResourcePtr resource = std::make_shared<Resource>(source_obj);
  InitCompileCacheInfo(resource, phase);
  ConfigManager::GetInstance().ResetQueue(queue_name_);

  auto actions = GetPipeline(resource, phase, use_vm);
  std::shared_ptr<Pipeline> pip = std::make_shared<Pipeline>(resource, FilterActions(actions, phase));

  if (pip->NeedCreateBackend()) {
    // Create backend asynchronously.
    resource->SetBackendAsync([]() {
      auto backend = compile::CreateBackend();
#ifdef ENABLE_DEBUGGER
      // Connect session to debugger.
      backend->SetDebugger();
#endif
      return backend;
    });
  }

  // Get the parameters items and add the value to args_abs.
  abstract::AbstractBasePtrList args_abs;
  std::vector<ValuePtr> arguments;
  std::size_t size = args.size();
  MS_EXCEPTION_IF_NULL(parallel::ParallelContext::GetInstance());
  bool is_auto_parallel = (parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kSemiAutoParallel ||
                           parallel::ParallelContext::GetInstance()->parallel_mode() == parallel::kAutoParallel) &&
                          !py::hasattr(source_obj, parallel::kSkipAutoParallelCompile) &&
                          !py::hasattr(source_obj, parallel::kKeepInputUnchanged);
  for (std::size_t i = 0; i < size; i++) {
    ValuePtr converted = nullptr;
    // In some parallel mode need full_tensor which cause the args of GenerateArgumentsKey not same to compile,
    // So can't use cur_convert_input_ directly.
    auto iter = cur_convert_input_.find(args[i].ptr());
    if (iter != cur_convert_input_.end()) {
      (void)arguments.emplace_back(iter->second.first);
      if (is_auto_parallel) {
        auto abs_item = iter->second.second->Clone();
        (void)parallel::ExtendInputArgsAbstractShape(abs_item, i);
        (void)args_abs.emplace_back(abs_item);
        continue;
      }
      (void)args_abs.emplace_back(iter->second.second);
      continue;
    }
    bool succ = parse::ConvertData(args[i], &converted);
    if (!succ) {
      MS_LOG(EXCEPTION) << "Fail to convert the " << i << "th argument, args[" << i << "]: " << py::str(args[i]);
    }
    (void)arguments.emplace_back(converted);
    auto args_abstract_item = ArgsToAbstract(args[i], converted, enable_tuple_broaden_);
    if (is_auto_parallel) {
      (void)parallel::ExtendInputArgsAbstractShape(args_abstract_item, i);
    }
    (void)args_abs.emplace_back(args_abstract_item);
  }
  resource->set_arguments(arguments);
  resource->set_args_abs(args_abs);
  executor_info->arg_list_size = size;
  executor_info->resource = resource;
  info_[phase] = executor_info;
  pip->Run();

  // Save the compiled graph to MsPipeLine.
  SaveCompiledGraph(phase);
  if (is_auto_parallel) {
    ParallelPostProcess(phase);
  }
#ifdef ENABLE_DUMP_IR
  mindspore::RDR::Snapshot();
#endif
  CleanCompileRes(resource);
  EventMessage::PrintCompileEndMsg(phase, obj_desc);
  PhaseManager::GetInstance().ClearPhase();
  MS_LOG(INFO) << "Finish compiling.";
  return true;
}

std::vector<ActionItem> GraphExecutorPy::FilterActions(const std::vector<ActionItem> &actions,
                                                       const std::string &phase) {
  // filter action after validate when 'export'.
  if (GetPhasePrefix(phase).rfind("export", 0) == std::string::npos) {
    return actions;
  }
  MS_LOG(INFO) << "Phase is '" << phase << "', filter out actions after stage 'validate'";
  std::vector<ActionItem> filtered_actions;
  for (const auto &item : actions) {
    (void)filtered_actions.emplace_back(item);
    if (item.first == "validate") {
      break;
    }
  }
  return filtered_actions;
}

void GraphExecutorPy::ReleaseResource(const py::object &phase) {
  bool clear = false;
  // Be sure the pointer res destroyed before do DelOneNetRes.
  {
    ResourcePtr res = GetResource(py::cast<std::string>(phase));
    if (res != nullptr) {
      clear = true;
      CleanCompileRes(res);
    }
  }
  if (clear) {
    DelOneNetRes(phase);
  }
}

bool GraphExecutorPy::Compile(const py::object &source_obj, const py::tuple &args, const py::object &phase,
                              bool use_vm) {
  bool ret_value = false;
  try {
    ret_value = CompileInner(source_obj, args, phase, use_vm);
  } catch (const py::error_already_set &ex) {
    if (!StaticAnalysisException::Instance().HasException()) {
      // print function call stack info before release
      std::string compile_exception_info = GetCompileExceptionInfo();
      if (!compile_exception_info.empty()) {
        MS_LOG(ERROR) << compile_exception_info;
      }
    }
    ReleaseResource(phase);

    // re-throw this exception to Python interpreter to handle it
    throw(py::error_already_set(ex));
  } catch (const py::type_error &ex) {
    ReleaseResource(phase);
    throw py::type_error(ex);
  } catch (const py::value_error &ex) {
    ReleaseResource(phase);
    throw py::value_error(ex);
  } catch (const py::index_error &ex) {
    ReleaseResource(phase);
    throw py::index_error(ex);
  } catch (const py::key_error &ex) {
    ReleaseResource(phase);
    throw py::key_error(ex);
  } catch (const py::attribute_error &ex) {
    ReleaseResource(phase);
    throw py::attribute_error(ex);
  } catch (const py::name_error &ex) {
    ReleaseResource(phase);
    throw py::name_error(ex);
  } catch (const py::assertion_error &ex) {
    ReleaseResource(phase);
    throw py::assertion_error(ex);
  } catch (const py::base_exception &ex) {
    ReleaseResource(phase);
    throw py::base_exception(ex);
  } catch (const py::keyboard_interrupt &ex) {
    ReleaseResource(phase);
    throw py::keyboard_interrupt(ex);
  } catch (const py::stop_iteration &ex) {
    ReleaseResource(phase);
    throw py::stop_iteration(ex);
  } catch (const py::overflow_error &ex) {
    ReleaseResource(phase);
    throw py::overflow_error(ex);
  } catch (const py::zero_division_error &ex) {
    ReleaseResource(phase);
    throw py::zero_division_error(ex);
  } catch (const py::environment_error &ex) {
    ReleaseResource(phase);
    throw py::environment_error(ex);
  } catch (const py::io_error &ex) {
    ReleaseResource(phase);
    throw py::io_error(ex);
  } catch (const py::os_error &ex) {
    ReleaseResource(phase);
    throw py::os_error(ex);
  } catch (const py::memory_error &ex) {
    ReleaseResource(phase);
    throw py::memory_error(ex);
  } catch (const py::unbound_local_error &ex) {
    ReleaseResource(phase);
    throw py::unbound_local_error(ex);
  } catch (const py::not_implemented_error &ex) {
    ReleaseResource(phase);
    throw py::not_implemented_error(ex);
  } catch (const py::indentation_error &ex) {
    ReleaseResource(phase);
    throw py::indentation_error(ex);
  } catch (const py::runtime_warning &ex) {
    ReleaseResource(phase);
    throw py::runtime_warning(ex);
  } catch (const std::exception &ex) {
    ReleaseResource(phase);
    // re-throw this exception to Python interpreter to handle it
    throw(std::runtime_error(ex.what()));
  } catch (...) {
    ReleaseResource(phase);
#ifndef _MSC_VER
    std::string exName(abi::__cxa_current_exception_type()->name());
    MS_LOG(EXCEPTION) << "Error occurred when compile graph. Exception name: " << exName;
#else
    MS_LOG(EXCEPTION) << "Error occurred when compile graph. Exception name: ";
#endif
  }
  return ret_value;
}

void CacheValidateFuncGraph(const ResourcePtr &resource) {
  if (!resource->EnableCompileCache()) {
    return;
  }
#ifdef ENABLE_PROFILE
  double t1 = GetTime();
#endif
  resource->CacheFuncGraph();
#ifdef ENABLE_PROFILE
  double t2 = GetTime();
  MsProfile::StatTime("SaveCacheFuncGraph", t2 - t1);
#endif
}

void CheckInterpretNodeLineInfos() {
  auto &line_infos = InterpretNodeRecorder::GetInstance().LineInfos();
  if (line_infos.empty()) {
    return;
  }
  std::stringstream ss;
  ss << "Found unsupported syntax in graph mode, those codes would be fallen back to Python interpreter:\n";
  ss << "-----\n";
  size_t num = 1;
  for (auto &line : line_infos) {
    ss << "# No. " << num << ":\n" << line << "\n";
    ++num;
  }
  ss << "-----\n";
  // Print the codes run in JIT Fallback.
  MS_LOG(INFO) << ss.str();
  InterpretNodeRecorder::GetInstance().Clear();
}

#ifdef ENABLE_DUMP_IR
void RDRRecordGraph(const size_t action_index, const size_t action_size, const std::string &filename,
                    const FuncGraphPtr &graph) {
  if (mindspore::RecorderManager::Instance().RdrEnable()) {
    MS_LOG(INFO) << "Recording FuncGraph in pipeline using RDR.";
    if (graph != nullptr) {
      auto graph_clone = BasicClone(graph);
      if (graph_clone != nullptr) {
        DumpGraphParams dump_params = {false, static_cast<int>(kTopStack)};
        if (action_index == action_size) {
          dump_params.dump_mode = static_cast<int>(kWholeStack);
        }
        (void)mindspore::RDR::RecordAnfGraph(SUBMODULE_ID, filename, graph_clone, dump_params, ".ir");
      } else {
        MS_LOG(WARNING) << "Clone FuncGraph failed in pipeline, no FuncGraph recording in RDR.";
      }
    } else {
      MS_LOG(WARNING) << "Pipeline Resource has no FuncGraph, no FuncGraph recording in RDR";
    }
    MS_LOG(INFO) << "Recording FuncGraph in pipeline end.";
  }
}
#endif

#ifdef ENABLE_DUMP_IR
void RecordIR(const size_t action_index, const size_t action_size, const std::string &action_name,
              const FuncGraphPtr &graph, FuncGraphPtr *user_graph) {
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG) && graph != nullptr) {
    *user_graph = graph;
    std::string base_name = GetBaseNameForIR(SizeToLong(action_index), action_name);

    // Generate IR file in human-readable format
    static const auto switch_order = (common::GetEnv("MS_DEV_SAVE_GRAPHS_SORT_MODE") == "1");
    if (switch_order) {
      ExportIR(base_name + ".ir", graph);
    } else {
      if (action_index == action_size - 1) {
        DumpIR(base_name + ".ir", graph, false, kWholeStack);
      } else {
        DumpIR(base_name + ".ir", graph, false, kTopStack);
      }
    }
    if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPH_DOT)) {
      draw::Draw(base_name + ".dot", graph);
    }
  }
}
#endif

#ifndef ENABLE_SECURITY
void SaveGraphForReadability(const std::string &action_name, const FuncGraphPtr &graph, const ResourcePtr &resource) {
  if (graph != nullptr && action_name.find("optimize") != string::npos) {
#ifdef ENABLE_DUMP_IR
    if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG)) {
      DumpIRProto(graph, action_name);
    }
#endif
    resource->set_optimize_graph(graph);
  }
}
#endif

void Pipeline::Run() {
  MS_LOG(INFO) << "Pipeline run";
  MS_EXCEPTION_IF_NULL(resource_);
  FuncGraphPtr user_graph = nullptr;
  WITH(MsProfile::GetProfile())[&user_graph, this]() {
    size_t i = 0;
    for (auto &action : actions_) {
#ifdef ENABLE_TIMELINE
      DumpTime &dump_time = DumpTime::GetInstance();
      dump_time.Record(action.first, GetTime(), true);
#endif
      bool result = true;
      WITH(MsProfile::GetProfile()->Step(action.first))[&result, &action, this]() {
        MS_LOG(INFO) << "Status record: start " << action.first << " action.";
        result = action.second(resource_);
        MS_LOG(INFO) << "Status record: end " << action.first << " action.";
      };
      if (action.first == "task_emit") {
        SetLoopCount(resource_);
      } else if (action.first == "validate") {
        CheckInterpretNodeLineInfos();
        CacheValidateFuncGraph(resource_);
#ifndef ENABLE_SECURITY
#ifdef WITH_BACKEND
        MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
        if (MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
          const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
            {kAscendDevice, MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
          MS_EXCEPTION_IF_NULL(device_context);
          MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
          device_context->GetDeprecatedInterface()->DumpProfileParallelStrategy(resource_->func_graph());
        }
#endif
#endif
      }
      if (!result) {
        MS_LOG(EXCEPTION) << "Pipeline running to end, failed in step:" << action.first;
      }

      FuncGraphPtr graph = resource_->func_graph();
#ifdef ENABLE_DUMP_IR
      std::string filename = GetBaseNameForIR(SizeToLong(i), action.first);
      RDRRecordGraph(i, actions_.size(), filename, graph);
      RecordIR(i, actions_.size(), action.first, graph, &user_graph);
#endif
#ifndef ENABLE_SECURITY
      SaveGraphForReadability(action.first, graph, resource_);
#endif
      i++;
#ifdef ENABLE_TIMELINE
      dump_time.Record(action.first, GetTime(), false);
#endif
    }
  };
#ifdef ENABLE_PROFILE
  MsProfile::Print();
  MsProfile::Reset();
#endif

#ifdef ENABLE_DUMP_IR
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG) && (user_graph != nullptr)) {
    if (MsContext::GetInstance()->get_param<bool>(MS_CTX_SAVE_GRAPH_DOT)) {
      draw::DrawUserFuncGraph("ModelDigraph.dot", user_graph);
    }
  }
#endif
  MS_LOG(INFO) << "End";
}

bool Pipeline::NeedCreateBackend() {
  return std::any_of(actions_.begin(), actions_.end(),
                     [](const ActionItem &action) { return action.first == "task_emit" || action.first == "execute"; });
}

void ProcessVmArgInner(const py::tuple &args, const ResourcePtr &res, VectorRef *const arg_list) {
  MS_EXCEPTION_IF_NULL(arg_list);
  std::size_t size = args.size();
  bool arg_list_inited = !arg_list->empty();
  for (std::size_t i = 0; i < size; i++) {
    py::object arg = args[i];
    ValuePtr converted = nullptr;
    bool succ = parse::ConvertData(arg, &converted);
    if (!succ) {
      MS_LOG(EXCEPTION) << "The " << i << "th arg convert failed.";
    }
    if (!arg_list_inited) {
      arg_list->push_back(converted);
      continue;
    }
    if (i >= arg_list->size()) {
      MS_LOG(EXCEPTION) << "i:" << i << " output of range:" << arg_list->size();
    }
    (*arg_list)[i] = converted;
  }

  MS_EXCEPTION_IF_NULL(res);
  auto graph = res->func_graph();
  MS_EXCEPTION_IF_NULL(graph);
  std::vector<AnfNodePtr> graph_params = graph->parameters();
  std::size_t graph_params_size = graph_params.size();
  if ((*arg_list).size() != graph_params_size) {
    // Maybe some default parameter
    for (std::size_t i = (*arg_list).size(); i < graph_params_size; i++) {
      MS_EXCEPTION_IF_NULL(graph_params[i]);
      auto param_ptr = (graph_params[i])->cast_ptr<Parameter>();
      MS_EXCEPTION_IF_NULL(param_ptr);
      if (!param_ptr->has_default()) {
        MS_LOG(EXCEPTION) << "Parameter[" << i << "] has no default param";
      }
      if (!param_ptr->default_param()->isa<Tensor>()) {
        MS_LOG(EXCEPTION) << "Parameter[" << param_ptr->ToString()
                          << "] is not initialized, need to call `.init_data()`";
      }
      arg_list->push_back(param_ptr->default_param());
    }
  }
}

void GraphExecutorPy::ProcessVmArg(const py::tuple &args, const std::string &phase, VectorRef *const arg_list) {
  ProcessVmArgInner(args, GetResource(phase), arg_list);
}

#ifdef ENABLE_DEBUGGER
void GraphExecutorPy::TerminateDebugger() {
  if (Common::GetDebugTerminate()) {
    MS_LOG(INFO) << "Terminate debugger and clear resources!";
    ClearResAtexit();
    exit(static_cast<int>(!Common::GetDebugExitSuccess()));
  }
}
#endif

py::object GraphExecutorPy::Run(const py::tuple &args, const py::object &phase_obj) {
  // init for dynamic-obfuscated model infer
  (void)mindspore::kernel::CustomizedOpaquePredicate::GetInstance().init_calling_count();
  // Mindspore debugger notify main thread to exit after one step, and will not run next step
#ifdef ENABLE_DEBUGGER
  TerminateDebugger();
#endif
  if (!py::isinstance<py::str>(phase_obj)) {
    MS_LOG(EXCEPTION) << "Run failed, phase input is not a str";
  }
  auto phase = py::cast<std::string>(phase_obj);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
#ifdef WITH_BACKEND
  if (ms_context->backend_policy() == "ge") {
    std::string phase_prefix = GetPhasePrefix(phase);
    if (phase_prefix == "save") {
      auto pos = phase.find('.');
      std::string origin_phase = phase.substr(pos + 1);
      FuncGraphPtr func_graph = info_["train." + origin_phase]->func_graph;
      MS_EXCEPTION_IF_NULL(func_graph);
      MS_EXCEPTION_IF_NULL(MsContext::GetInstance());
      auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {MsContext::GetInstance()->get_param<std::string>(MS_CTX_DEVICE_TARGET),
         MsContext::GetInstance()->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
      MS_EXCEPTION_IF_NULL(device_context);
      MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
      device_context->GetDeprecatedInterface()->DoExecNonInputGraph("save." + func_graph->ToString());
      ConfigManager::GetInstance().ResetConfig();
      return py::none();
    }
  }
#endif
  auto ret_val = std::make_shared<py::object>();
  if (info_.count(phase) != 0 && info_[phase]->func_graph != nullptr) {
    if (IsGraphOutputValueNodeOrParameter(info_[phase]->func_graph->output(), args, ret_val)) {
      return *ret_val;
    }
  }
#ifndef WITH_BACKEND
  if (ms_context->backend_policy() == "ge") {
    // Virtual output constructed for test cases.
    if (!args.empty()) {
      return args[0];
    }
    return args;
  }
#endif
  auto iter = info_.find(phase);
  if (iter == info_.end()) {
    MS_LOG(EXCEPTION) << "No executor info. found for phase: " << phase;
  }
  auto &execute_info = iter->second;
  MS_EXCEPTION_IF_NULL(execute_info);
  if (args.size() > execute_info->arg_list_size) {
    MS_LOG(WARNING) << "The args size: " << args.size() << ", full_arg_size: " << execute_info->arg_list_size;
  }
  ProcessVmArg(args, phase, &execute_info->arg_list);
  // Start to run phase.
  compile::VmEvalFuncPtr run = GetVmEvalFunc(phase);
  if (run == nullptr) {
    MS_LOG(EXCEPTION) << "Can't find run graph func for " << phase;
  }
  // Set loopsink size for each phase.
  bool vm_loop_flag = info_[phase]->resource->vm_loop_flag();
  int64_t loop_size = info_[phase]->resource->loop_size();
  int64_t vm_loop = 1;
  if (vm_loop_flag) {
    vm_loop = loop_size;
  } else {
    // Set the loop size in config if graphs nums is 1(is_loop_sin=True), then there will be a loop embrace
    // 'Execute(graph)' in GPUSession.
    ConfigManager::GetInstance().set_gpu_loopsink_size(loop_size);
  }
  MS_LOG(INFO) << "VM loop size " << vm_loop << ", loopsink size " << vm_loop;
  py::object res;
  MS_LOG(DEBUG) << "Eval run" << ms_context->backend_policy();
  const auto &output = execute_info->func_graph->output();
  MS_EXCEPTION_IF_NULL(output);
  const auto &output_abs = output->abstract();
  MS_EXCEPTION_IF_NULL(output_abs);
  for (int64_t i = 0; i < vm_loop; i++) {
    BaseRef value = (*run)(execute_info->arg_list);
    res = BaseRefToPyData(value, output_abs);
  }

  // Replace the output if it's not Tensor, but Python data.
  const auto &py_res = GetPyExecuteOutput(output);
  if (py_res != py::none()) {
    return py_res;
  }

  MS_LOG(DEBUG) << "Run end";
  return res;
}  // namespace pipeline

FuncGraphPtr GraphExecutorPy::BuildGraph(const py::dict &init_params, const std::string &phase) const {
  MS_LOG(INFO) << "Start build df graph, phase = " << phase;
  if (info_.count(phase) == 0) {
    MS_LOG(EXCEPTION) << "No phase in executor: " << GetPhasePrefix(phase);
  }
  DeviceContext *device_context = nullptr;
  try {
    device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({"GE", 0});
  } catch (const std::exception &) {
    return nullptr;
  }
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
  return device_context->GetDeprecatedInterface()->BuildDFGraph(info_.at(phase)->func_graph, init_params);
}

void GraphExecutorPy::UpdataParamNodeDefaultInput(
  const std::string &phase, const std::unordered_map<std::string, tensor::TensorPtr> &params_value) {
  FuncGraphPtr func_graph = info_[phase]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  MS_LOG(DEBUG) << "UpdataParamNodeDefaultInput for func graph(" << func_graph->ToString() << ") phase(" << phase
                << ")!";
  auto &params = func_graph->parameters();
  for (const auto &param : params) {
    MS_EXCEPTION_IF_NULL(param);
    auto param_cast = param->cast_ptr<Parameter>();
    MS_EXCEPTION_IF_NULL(param_cast);
    auto iter = params_value.find(param_cast->name());
    if (iter != params_value.end()) {
      param_cast->set_default_param(iter->second);
    }
  }
}

py::dict GraphExecutorPy::GetParams(const std::string &phase) {
  FuncGraphPtr func_graph = info_[phase]->resource->func_graph();
  MS_EXCEPTION_IF_NULL(func_graph);
  py::dict parameter_dict;
  std::vector<AnfNodePtr> graph_params = func_graph->parameters();
  for (auto &param : graph_params) {
    MS_EXCEPTION_IF_NULL(param);
    auto param_ptr = std::static_pointer_cast<Parameter>(param);
    std::string name = param_ptr->name();
    auto tensor = std::dynamic_pointer_cast<tensor::Tensor>(param_ptr->default_param());
    if (tensor != nullptr) {
      parameter_dict[py::str(name)] = *tensor;
    }
  }
  return parameter_dict;
}

void GraphExecutorPy::PyExePath(const py::object &py_exe_path) const {
  if (!py::isinstance<py::str>(py_exe_path)) {
    MS_LOG(EXCEPTION) << "Failed, py_exe_path input is not a str";
  }
  auto py_exe_path_s = py::cast<std::string>(py_exe_path);
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_PYTHON_EXE_PATH, py_exe_path_s);
}

void GraphExecutorPy::KernelBuildServerDir(const py::object &kernel_build_server_dir) const {
  if (!py::isinstance<py::str>(kernel_build_server_dir)) {
    MS_LOG(EXCEPTION) << "Failed, kernel_build_server_dir input is not a str";
  }
  auto kernel_build_server_dir_s = py::cast<std::string>(kernel_build_server_dir);
  auto ms_context = MsContext::GetInstance();
  ms_context->set_param<std::string>(MS_CTX_KERNEL_BUILD_SERVER_DIR, kernel_build_server_dir_s);
}

bool InitExecDataset(const std::string &queue_name, int64_t iter_num, int64_t batch_size,
                     const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                     const std::vector<int64_t> &input_indexes, const std::string &phase, bool need_run) {
  std::string name = MsContext::GetInstance()->backend_policy();
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
#ifdef WITH_BACKEND
  if (ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {kAscendDevice, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
    if (!device_context->GetDeprecatedInterface()->IsTsdOpened(ms_context)) {
      InitPipeline();
    }
  }

#endif
  if (iter_num == -1) {
    iter_num = INT32_MAX;
  }
  if (name == kMsConvert || name == kMsVm) {
    return InitExecDatasetVm(queue_name, iter_num, batch_size, types, shapes, input_indexes, need_run);
  }
  std::string backend = ms_context->backend_policy();
#ifdef WITH_BACKEND
  if (backend == "ge") {
    auto device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET), ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());

    return device_context->GetDeprecatedInterface()->InitExecDataset(queue_name, iter_num, batch_size, types, shapes,
                                                                     input_indexes, phase);
  }
#endif
  return backend == "ge" ? true : false;
}

bool InitExecDatasetVm(const std::string &queue_name, int64_t size, int64_t batch_size,
                       const std::vector<TypePtr> &types, const std::vector<std::vector<int64_t>> &shapes,
                       const std::vector<int64_t> &input_indexes, bool need_run) {
#if defined(__linux__) && defined(WITH_BACKEND)
  if ((ps::PSContext::instance()->is_ps_mode()) && (!ps::PSContext::instance()->is_worker())) {
    return true;
  }
  const auto &cluster_ctx = distributed::cluster::ClusterContext::instance();
  MS_EXCEPTION_IF_NULL(cluster_ctx);
  if (cluster_ctx->initialized() && cluster_ctx->node_role() == distributed::kEnvRoleOfScheduler) {
    return true;
  }
#endif
  MS_LOG(INFO) << "Start InitDataSet Entry";
  mindspore::python_adapter::set_python_env_flag(true);
  ShapeVector int_input_indexes;
  (void)std::transform(input_indexes.begin(), input_indexes.end(), std::back_inserter(int_input_indexes),
                       [](int64_t item) { return static_cast<int64_t>(item); });
  std::vector<ShapeVector> int_shapes;
  (void)std::transform(shapes.begin(), shapes.end(), std::back_inserter(int_shapes),
                       [](const std::vector<int64_t> &item) {
                         ShapeVector vector_item;
                         (void)std::transform(item.begin(), item.end(), std::back_inserter(vector_item),
                                              [](int64_t inner_item) { return static_cast<int64_t>(inner_item); });
                         return vector_item;
                       });
  auto p_init = std::make_shared<Primitive>("InitDataSetQueue");
  p_init->set_attr("queue_name", MakeValue(queue_name));
  p_init->set_attr("size", MakeValue(static_cast<int64_t>(size)));
  p_init->set_attr("batch_size", MakeValue(static_cast<int64_t>(batch_size)));
  p_init->set_attr("types", MakeValue(types));
  p_init->set_attr("shapes", MakeValue(int_shapes));
  p_init->set_attr("input_indexes", MakeValue(int_input_indexes));

  const std::vector<std::string> empty_str_list;
  p_init->set_attr("input_names", MakeValue(empty_str_list));
  p_init->set_attr("output_names", MakeValue(empty_str_list));

  FuncGraphPtr func_graph = std::make_shared<FuncGraph>();
  auto app_init = std::make_shared<CNode>(AnfNodePtrList{NewValueNode(p_init)}, func_graph);
  func_graph->set_output(app_init);
  auto manager = MakeManager();
  manager->AddFuncGraph(func_graph);

  // AbstractNone indicates there is no output for this apply node.
  auto abstract_none = std::make_shared<abstract::AbstractNone>();
  app_init->set_abstract(abstract_none);
  // Before the graph compiling, need reset the iter num.
  ConfigManager::GetInstance().ResetIterNum();
#ifdef ENABLE_DUMP_IR
  mindspore::RDR::ResetRecorder();
#endif

  compile::SetMindRTEnable();
  auto backend = compile::CreateBackend();
  MS_EXCEPTION_IF_NULL(backend);
  // The data set graph compiling and running of mindRT.
  if (MsContext::GetInstance()->get_param<bool>(MS_CTX_ENABLE_MINDRT)) {
#if defined(__linux__) && defined(WITH_BACKEND)
    if (ps::PSContext::instance()->is_worker() && ps::PSContext::instance()->cache_enable()) {
      ps::PsDataPrefetch::GetInstance().CreateDataChannel(queue_name, LongToSize(size));
    }
#endif

    const auto &mindrt_backend = std::dynamic_pointer_cast<compile::MindRTBackend>(backend);
    MS_EXCEPTION_IF_NULL(mindrt_backend);
    SetRunMode(func_graph, mindrt_backend.get());
    auto &actor_info = mindrt_backend->CompileGraphs(func_graph);
    VectorRef args;
    if (need_run) {
      VectorRef outputs;
      mindrt_backend->RunGraph(actor_info, args, &outputs);
    }
    ConfigManager::GetInstance().set_iter_num(queue_name, size);
    return true;
  }

  auto convert_fn = backend->convert_fn();
  MS_EXCEPTION_IF_NULL(convert_fn);
  // Convert CNodeList to LinConvertResult.
  auto segment = std::make_shared<GraphSegment>(std::vector<AnfNodePtr>{app_init}, false);
  auto runner = convert_fn(segment, "");
  ConfigManager::GetInstance().set_iter_num(queue_name, size);
  // PS cache does not support loop sink.
#if defined(__linux__) && defined(WITH_BACKEND)
  if (ps::PSContext::instance()->is_worker() && ps::PsDataPrefetch::GetInstance().cache_enable()) {
    ps::PsDataPrefetch::GetInstance().CreateDataChannel(queue_name, LongToSize(size));
    ConfigManager::GetInstance().set_iter_num(queue_name, 1);
  }
#endif

  if (!(*runner.run)) {
    // empty function
    MS_LOG(EXCEPTION) << "Backend " << backend->name() << " unsupported tdt dataset.";
  }

  // launch init dataset runner without inputs and outputs
  VectorRef args;
  auto fn = runner.run;
  if (need_run) {
    (void)(*fn)(args);
  }
  MS_LOG(DEBUG) << "InitDataSetVm End.";
  return true;
}  // namespace pipeline

std::string GetJitLevel() {
  auto jit_config = GraphExecutorPy::GetInstance()->jit_config();
  auto iter = jit_config.find("jit_level");
  if (iter != jit_config.end()) {
    return iter->second;
  }
  return "";
}

void ResetOpId() { mindspore::id_generator::reset_id(); }
void ResetOpIdWithOffset() { mindspore::id_generator::reset_id_with_offset(); }

void InitHccl() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
#ifdef WITH_BACKEND
  auto backend = ms_context->backend_policy();
  if (backend == "ge") {
    InitPipeline();
    return;
  }
#endif
  mindspore::python_adapter::set_python_env_flag(true);
  ms_context->set_param<bool>(MS_CTX_ENABLE_HCCL, true);
  std::string device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (ms_context->backend_policy() == "ms" && device_name == kAscendDevice) {
    if (!mindspore::distributed::Initialize()) {
      MS_LOG(EXCEPTION) << "InitHccl failed.";
    }
  }
}

void FinalizeHccl() {
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
#ifdef WITH_BACKEND
  auto backend = ms_context->backend_policy();
  if (backend == "ge") {
    FinalizeBackend();
    return;
  }
#endif
  session::ExecutorManager::Instance().Clear();
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();
  device::DeviceContextManager::GetInstance().ClearDeviceContexts();
  device::DeviceContextManager::GetInstance().UnloadPlugin();
}

uint32_t GetHcclRankId() {
  uint32_t rank_id = 0;
  bool ret = CommManager::GetInstance().GetRankID("", &rank_id);
  if (!ret) {
    MS_LOG(ERROR) << "Get rank id failed, return rank id " << rank_id << " as default.";
  }
  return rank_id;
}

uint32_t GetHcclRankSize() {
  uint32_t rank_size = 0;
  bool ret = CommManager::GetInstance().GetRankSize("", &rank_size);
  if (!ret) {
    MS_LOG(ERROR) << "Get rank size failed, return rank size " << rank_size << " as default.";
  }
  return rank_size;
}

void GraphExecutorPy::ExportGraph(const std::string &file_name, const std::string &phase, const py::object encrypt,
                                  char *key) {
  DeviceContext *device_context = nullptr;
  try {
    device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({"GE", 0});
  } catch (const std::exception &) {
    MS_EXCEPTION(ValueError) << "Only support export file in 'AIR' format with Ascend backend.";
  }
  MS_EXCEPTION_IF_NULL(device_context);
  MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
  FuncGraphPtr func_graph = info_[phase]->func_graph;
  MS_EXCEPTION_IF_NULL(func_graph);
  device_context->GetDeprecatedInterface()->ExportDFGraph(file_name, func_graph->ToString(), encrypt, key);
}

FuncGraphPtr LoadMindIR(const std::string &file_name, const char *dec_key, const size_t key_len,
                        const std::string &dec_mode, const py::object decrypt, const bool obfuscated) {
  if (obfuscated) {
    MS_LOG(DEBUG) << "[LoadMindIR] Set customized function.";
    (void)mindspore::kernel::CustomizedOpaquePredicate::GetInstance().set_func_names();
    (void)mindspore::kernel::CustomizedOpaquePredicate::GetInstance().init_calling_count();
  }
  FuncGraphPtr func_graph = nullptr;
  if (dec_mode == "Customized") {
    py::bytes key_bytes(dec_key);
    py::bytes model_stream = decrypt(file_name, key_bytes);
    std::string model_string(model_stream);

    MindIRLoader mindir_loader;
    func_graph = mindir_loader.LoadMindIR(model_string.c_str(), model_string.size());
  } else {
    MindIRLoader mindir_loader(false, reinterpret_cast<const unsigned char *>(dec_key), key_len, dec_mode, false);
    func_graph = mindir_loader.LoadMindIR(file_name);
  }
#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    DumpIR("load.ir", func_graph);
  }
#endif
  return func_graph;
}

FuncGraphPtr DynamicObfuscateMindIR(const std::string &file_name, float obf_ratio, int obf_password,
                                    int append_password, char *dec_key, const size_t key_len,
                                    const std::string &dec_mode) {
  if (obf_password == 0) {
    (void)mindspore::kernel::CustomizedOpaquePredicate::GetInstance().set_func_names();
    MS_LOG(DEBUG) << "[DynamicObfuscateMindIR] set function names finished.";
  }
  mindspore::DynamicObfuscator dynamic_obfuscator(obf_ratio, obf_password, append_password);
  MindIRLoader mindir_loader(false, reinterpret_cast<unsigned char *>(dec_key), key_len, dec_mode, false);
  FuncGraphPtr func_graph = mindir_loader.LoadMindIR(file_name);
  if (func_graph == nullptr) {
    MS_LOG(ERROR) << "[DynamicObfuscateMindIR] load mindir failed, please check the mindir file.";
    return nullptr;
  }
  mindspore::FuncGraphPtr obfuscated_graph = dynamic_obfuscator.ObfuscateMindIR(func_graph);
  if (obfuscated_graph == nullptr) {
    MS_LOG(ERROR) << "[DynamicObfuscateMindIR] obfuscate model failed.";
    return nullptr;
  }
  return obfuscated_graph;
}

void CloseTsd(bool force) {
#ifdef WITH_BACKEND
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  if (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice) {
    const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {kAscendDevice, context_ptr->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
    (void)device_context->GetDeprecatedInterface()->CloseTsd(context_ptr, force);
  }
#endif
}

void InitPipeline() {
  // set python env flag
  RecordInitStatus();
  mindspore::python_adapter::set_python_env_flag(true);
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
#ifdef WITH_BACKEND
  auto backend = ms_context->backend_policy();
  auto device_name = ms_context->get_param<std::string>(MS_CTX_DEVICE_TARGET);
  if (backend == "ge") {
    const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
      {device_name, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
    MS_EXCEPTION_IF_NULL(device_context);
    device_context->Initialize();
  }
  if (!common::UseDynamicCluster()) {
    if (device_name == kAscendDevice) {
      const auto &device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext(
        {device_name, ms_context->get_param<uint32_t>(MS_CTX_DEVICE_ID)});
      MS_EXCEPTION_IF_NULL(device_context);
      MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
      if (!device_context->GetDeprecatedInterface()->OpenTsd(ms_context)) {
        MS_LOG(EXCEPTION) << "Open tsd failed";
      }
    }
  }
#endif
}

void FinalizeBackend() { CloseTsd(); }

void MemoryRecycle() {
#ifdef ENABLE_DUMP_IR
  mindspore::RDR::ResetRecorder();
#endif
  ReclaimOptimizer();
  session::ExecutorManager::Instance().ClearDoneTasks();
  ad::g_k_prims.clear();
  ad::ClearPyNativeAutoGradStaticRes();
  ad::PrimBpropOptimizer::GetPrimBpropOptimizerInst().Clear();
  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  abstract::AnalysisContext::ClearContext();
  g_args_cache.clear();
  // clean static variable to prevent from crash. As static variable is released after
  // Python threads is released.
  parse::data_converter::ClearObjectCache();
  parse::Parser::CleanParserResource();
  trace::ClearTraceStack();
  pynative::PyNativeExecutor::GetInstance()->ClearRes();
  pynative::PyNativeExecutor::GetInstance()->WorkerJoin();
  ConfigManager::GetInstance().ResetConfig();
  ScopeManager::GetInstance().ClearScope();
  FuncGraphLoopBreaker::Inst().CleanMetaFuncGraphCache();
  FuncGraphLoopBreaker::Inst().BreakLoop();
}

void ClearResPart1() {
  runtime::OpExecutor::GetInstance().WorkerJoin();
  // When the python process exits, the kernels on the device may not have finished executing.
  device::KernelRuntimeManager::Instance().WaitTaskFinishOnDevice();
  device::DeviceContextManager::GetInstance().WaitTaskFinishOnDevice();

  RecordExitStatus();
#ifdef ENABLE_DUMP_IR
  mindspore::RDR::Snapshot();
  mindspore::RDR::ResetRecorder();
#endif
  session::ExecutorManager::Instance().Clear();
  runtime::GraphScheduler::GetInstance().Clear();

  MS_LOG(INFO) << "Start clear kernel runtime...";
  device::KernelRuntimeManager::Instance().ClearRuntimeResource();
  MS_LOG(INFO) << "End clear kernel runtime.";

  MS_LOG(INFO) << "Start Finalize StreamSynchronizer...";
  device::StreamSynchronizer::GetInstance()->Finalize();
  MS_LOG(INFO) << "End Finalize StreamSynchronizer...";

  (void)distributed::collective::CollectiveManager::instance()->Finalize();
  PrimitivePy::ClearHookRes();
  ad::g_k_prims.clear();
  ad::ClearPyNativeAutoGradStaticRes();
  ad::PrimBpropOptimizer::GetPrimBpropOptimizerInst().Clear();

  abstract::ClearPrimEvaluatorMap();
  pipeline::GetMethodMap().clear();
  pipeline::GetAttrMap().clear();
  pipeline::GraphExecutorPy::ClearRes();
  pipeline::ReclaimOptimizer();
}

void ClearResPart2() {
  MS_LOG(INFO) << "Start clear PyNativeExecutor...";
  pynative::PyNativeExecutor::GetInstance()->ClearRes();
  MS_LOG(INFO) << "End clear PyNativeExecutor.";

#ifdef WITH_BACKEND
  auto ms_context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(ms_context);
  if (ms_context->backend_policy() == "ge") {
    DeviceContext *device_context = device::DeviceContextManager::GetInstance().GetOrCreateDeviceContext({"GE", 0});
    MS_EXCEPTION_IF_NULL(device_context);
    MS_EXCEPTION_IF_NULL(device_context->GetDeprecatedInterface());
    device_context->GetDeprecatedInterface()->ClearGraphWrapper();
    device_context->GetDeprecatedInterface()->ClearOpAdapterMap();
  } else {
    MS_LOG(INFO) << "Start clear ConfigManager...";
    ConfigManager::GetInstance().ResetIterNum();
    MS_LOG(INFO) << "End clear ConfigManager.";
  }
#else
  MS_LOG(INFO) << "Start clear ConfigManager...";
  ConfigManager::GetInstance().ResetIterNum();
  MS_LOG(INFO) << "End clear ConfigManager.";
#endif
  MS_LOG(INFO) << "Start clear device context...";
  device::DeviceContextManager::GetInstance().ClearDeviceContexts();
  MS_LOG(INFO) << "End clear device context.";

  MS_LOG(INFO) << "Start clear AnalysisResultCacheMgr...";
  abstract::AnalysisResultCacheMgr::GetInstance().Clear();
  MS_LOG(INFO) << "End clear AnalysisResultCacheMgr.";

  MS_LOG(INFO) << "Start clear AnalysisContext...";
  abstract::AnalysisContext::ClearContext();
  MS_LOG(INFO) << "End clear AnalysisContext...";

  MS_LOG(INFO) << "Start clear AnalysisSchedule...";
  abstract::AnalysisSchedule::GetInstance().Stop();
  MS_LOG(INFO) << "End clear AnalysisSchedule...";

  // Python object needs to be freed after AnalysisResultCacheMgr and AnalysisContext.
  MS_LOG(INFO) << "Start clear python_adapter...";
  python_adapter::ResetPythonScope();
  MS_LOG(INFO) << "End clear python_adapter.";
#ifdef ENABLE_DEBUGGER
  Debugger::GetInstance()->Reset();
#endif
  g_args_cache.clear();
}

void ClearResPart3() {
  // clean static variable to prevent from crash. As static variable is released after
  // Python threads is released.
  MS_LOG(INFO) << "Start clear ClearObjectCache...";
  parse::data_converter::ClearObjectCache();
  MS_LOG(INFO) << "End clear ClearObjectCache...";

  MS_LOG(INFO) << "Start clear Parser...";
  parse::Parser::CleanParserResource();
  MS_LOG(INFO) << "End clear Parser...";

  MS_LOG(INFO) << "Start ClearTraceStack...";
  trace::ClearTraceStack();
  MS_LOG(INFO) << "End ClearTraceStack...";

  MS_LOG(INFO) << "Start clear InterpretNodeRecorder...";
  InterpretNodeRecorder::GetInstance().Clear();
  MS_LOG(INFO) << "End clear InterpretNodeRecorder...";

  MS_LOG(INFO) << "Start clear parallel::entire_costgraph...";
  parallel::entire_costgraph.reset();
  MS_LOG(INFO) << "End clear parallel::entire_costgraph...";

  MS_LOG(INFO) << "Start clear ProtobufLibrary...";
  google::protobuf::ShutdownProtobufLibrary();
  MS_LOG(INFO) << "End clear ProtobufLibrary...";
}

void ClearSingleton() {
  MS_LOG(INFO) << "Start clear singleton...";
  profiler::Profiler::Clear();
#ifdef ENABLE_AKG
  kernel::AkgKernelBuildManager::Instance().Clear();
#endif
  somas::SomasManager::Instance().Clear();
  GraphKernelInfoManager::Instance().Clear();
  device::DataQueueMgr::GetInstance().Clear();
  session::SessionFactory::Get().Clear();
  device::KernelRuntimeManager::Instance().Clear();
#ifndef ENABLE_SECURITY
  DumpJsonParser::Finalize();
#endif
  CommManager::Clear();
  MS_LOG(INFO) << "End clear singleton.";
}

void ClearResAtexit() {
  MS_LOG(INFO) << "Pipeline clear all resource";
  ClearResPart1();
  ClearResPart2();
  ClearResPart3();
  ClearSingleton();
  MS_LOG(INFO) << "Start unload dynamic lib...";
  device::DeviceContextManager::GetInstance().UnloadPlugin();
  MS_LOG(INFO) << "End unload dynamic lib...";
}

py::bytes PyEncrypt(char *plain_data, size_t plain_len, char *key, size_t key_len, const std::string &enc_mode) {
  size_t encrypt_len;
  auto encrypt_data = mindspore::Encrypt(&encrypt_len, reinterpret_cast<Byte *>(plain_data), plain_len,
                                         reinterpret_cast<Byte *>(key), key_len, enc_mode);
  if (encrypt_data == nullptr) {
    MS_EXCEPTION(ValueError) << "Encrypt failed";
  }
  auto py_encrypt_data = py::bytes(reinterpret_cast<char *>(encrypt_data.get()), encrypt_len);
  return py_encrypt_data;
}

py::bytes PyDecrypt(const std::string &encrypt_data_path, char *key, size_t key_len, const std::string &dec_mode) {
  size_t decrypt_len;
  auto decrypt_data =
    mindspore::Decrypt(&decrypt_len, encrypt_data_path, reinterpret_cast<Byte *>(key), key_len, dec_mode);
  if (decrypt_data == nullptr) {
    MS_LOG(ERROR) << "Decrypt failed";
    return py::none();
  }
  auto py_decrypt_data = py::bytes(reinterpret_cast<char *>(decrypt_data.get()), decrypt_len);
  return py_decrypt_data;
}

bool PyIsCipherFile(const std::string &file_path) { return mindspore::IsCipherFile(file_path); }

void FinalizeCluster() {
#if defined(__linux__) && defined(WITH_BACKEND)
  if (distributed::cluster::ClusterContext::instance()->initialized()) {
    MS_LOG(INFO) << "Start finalize the EmbeddingCacheScheduler.";
    runtime::EmbeddingCacheScheduler::GetInstance().Finalize();
    MS_LOG(INFO) << "End finalize the EmbeddingCacheScheduler.";

    if (!distributed::cluster_exit_with_exception()) {
      MS_LOG(INFO) << "Start finalize the cluster instance.";
      // Finalize MindSpore cluster only when this process exits without any exception.
      (void)distributed::cluster::ClusterContext::instance()->Finalize(UINT32_MAX);
      MS_LOG(INFO) << "End finalize the cluster instance.";
    }
  }
#endif
}
}  // namespace pipeline
}  // namespace mindspore
