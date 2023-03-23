/**
 * Copyright 2019-2021  uawei Technologies Co., Ltd
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

#include <algorithm>
#include <functional>
#include <string>
#include <vector>

#include "src/extendrt/session/single_op_session.h"
#include "src/extendrt/infer_device_address.h"

#include "plugin/factory/ms_factory.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/cpu_kernel_mod.h"
#include "src/extendrt/utils/kernel_build_utils.h"
#include "src/extendrt/kernel/ascend/plugin/ascend_kernel_plugin.h"
#include "extendrt/session/factory.h"
#include "extendrt/utils/runtime_utils.h"
#include "extendrt/utils/tensor_default_impl.h"

namespace mindspore {
const size_t tensor_max_size = 0x1000000;
constexpr auto kNameCustomAscend = "CustomAscend";

SingleOpInferSession::~SingleOpInferSession() {
  kernel::Factory<kernel::KernelMod>::Instance().UnRegister(kNameCustomAscend);
  kernel::AscendKernelPlugin::GetInstance().UpdateRegisterStatus(false);
}

Status SingleOpInferSession::AscendInit(const std::shared_ptr<Context> &context) {
  auto device_list = context->MutableDeviceInfo();
  for (const auto &device_info : device_list) {
    MS_EXCEPTION_IF_NULL(device_info);
    if (device_info->GetDeviceType() == DeviceType::kAscend) {
      kernel::AscendKernelPlugin::GetInstance().Register();
      auto ascend_device_info = device_info->Cast<mindspore::AscendDeviceInfo>();
      MS_EXCEPTION_IF_NULL(ascend_device_info);
      device_id_ = ascend_device_info->GetDeviceID();

      // AIPP config path is specified, DVPP mode
      if (ascend_device_info->GetInsertOpConfigPath() != "") {
        is_dvpp_ = true;
      }
      return kSuccess;
    }
  }
  MS_LOG(DEBUG) << "There is no ascend device info, no need to register ascend plugin.";
  return kSuccess;
}

Status SingleOpInferSession::Init(const std::shared_ptr<Context> &context) {
  MS_LOG(INFO) << "SingleOpInferSession::Init";
  MS_EXCEPTION_IF_NULL(context);
  kernel_graph_utils_ = std::make_shared<mindspore::KernelGraphUtils>();
  if (AscendInit(context) != kSuccess) {
    MS_LOG(ERROR) << "Init ascend failed.";
    return kLiteError;
  }
  return kSuccess;
}

void InitInputSizeList(const std::shared_ptr<CNode> &kernel_node, std::vector<size_t> *input_size_list) {
  MS_EXCEPTION_IF_NULL(input_size_list);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    TypeId type_id = AnfAlgo::GetInputDeviceDataType(kernel_node, input_index);
    size_t type_size = GetTypeByte(TypeIdToType(type_id));
    auto shape = AnfAlgo::GetInputDeviceShape(kernel_node, input_index);
    size_t tensor_size;
    if (std::any_of(shape.begin(), shape.end(), [](int64_t tmp) { return tmp < 0; })) {
      tensor_size = type_size;
    } else {
      tensor_size =
        shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
    }
    tensor_size = std::max(tensor_size, type_size);
    (void)input_size_list->emplace_back(tensor_size);
  }
}

Status SingleOpInferSession::UpdateKernelGraphInputs(const std::vector<std::vector<int64_t>> &dims,
                                                     const std::vector<TypeId> &type_ids, bool use_type_from_graph) {
  auto graph_inputs = RuntimeUtils::GetGraphDataInputs(kernel_graph_);
  if (graph_inputs.size() != dims.size()) {
    MS_LOG(ERROR) << "Number of graph inputs [" << graph_inputs.size() << "] is not equal to the given dims num ["
                  << dims.size() << "]";
    return kLiteError;
  }
  if (!use_type_from_graph && (graph_inputs.size() != type_ids.size())) {
    MS_LOG(ERROR) << "Number of graph inputs [" << graph_inputs.size() << "] is not equal to the given type ids num ["
                  << type_ids.size() << "]";
    return kLiteError;
  }
  for (size_t i = 0; i < graph_inputs.size(); ++i) {
    auto &graph_input = graph_inputs[i];
    if (utils::isa<mindspore::abstract::AbstractTuplePtr>(graph_input->abstract())) {
      MS_LOG(ERROR) << "The abstract of input does not support abstract tuple.";
      return kLiteError;
    }
    auto graph_input_addr = AnfAlgo::GetMutableOutputAddr(graph_input, 0);
    if (graph_input_addr == nullptr) {
      MS_LOG(ERROR) << "Graph input addr is nullptr.";
      return kLiteError;
    }
    TypeId type_id = graph_input_addr->type_id();
    if (!use_type_from_graph) {
      type_id = type_ids[i];
    }
    size_t type_size = GetTypeByte(TypeIdToType(type_id));
    const std::vector<int64_t> &dim = dims[i];
    size_t tensor_size =
      dim.empty() ? type_size : std::accumulate(dim.begin(), dim.end(), type_size, std::multiplies<size_t>());
    // update input size
    if (graph_input_addr->ptr_ != nullptr) {
      free(graph_input_addr->ptr_);
      auto new_addr = malloc(tensor_size);
      if (new_addr == nullptr) {
        MS_LOG(ERROR) << " malloc memory of input " << i << " failed, memory size " << tensor_size;
        return kLiteError;
      }
      graph_input_addr->set_ptr(new_addr);
      graph_input_addr->SetSize(tensor_size);
    }
    // update input shape
    auto abstract = std::make_shared<abstract::AbstractTensor>(TypeIdToType(type_id), dim);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Abstract is nullptr.";
      return kLiteError;
    }
    graph_input->set_abstract(abstract);
  }
  return kSuccess;
}

Status SingleOpInferSession::UpdateGraphInputsForDVPP(const std::vector<kernel::KernelTensorPtr> &inputs) {
  std::vector<std::vector<int64_t>> dims = {};
  std::vector<TypeId> type_ids = {};
  for (auto &input : inputs) {
    dims.push_back(input->GetShapeVector());
    type_ids.push_back(input->GetDtype());
  }
  auto ret = UpdateKernelGraphInputs(dims, type_ids, false);
  if (ret != kSuccess) {
    return ret;
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    // update session inputs_
    auto data_type = static_cast<enum DataType>(type_ids[i]);
    auto impl = std::make_shared<TensorDefaultImpl>(input_names_[i], data_type, dims[i]);
    impl->SetFormat(inputs[i]->GetFormat());
    inputs_.push_back(impl);
  }
  return kSuccess;
}

Status SingleOpInferSession::CompileGraph(FuncGraphPtr graph, const void *data, size_t size) {
  MS_LOG(INFO) << "SingleOpInferSession::CompileGraph";
  std::vector<KernelGraphPtr> all_out_graph;
  kernel_graph_ = kernel_graph_utils_->ConstructKernelGraph(graph, &all_out_graph, mindspore::device::DeviceType::kCPU);
  MS_EXCEPTION_IF_NULL(kernel_graph_);

  auto &nodes = kernel_graph_->nodes();
  for (const auto &node : nodes) {
    std::string node_name = common::AnfAlgo::GetCNodeName(node);
    MS_LOG(INFO) << "SingleOpInferSession::Nodes " << node_name;
  }

  auto &kernel_nodes = kernel_graph_->execution_order();
  bool update_flag = false;
  std::vector<kernel::KernelTensorPtr> update_inputs;
  std::vector<kernel::KernelTensorPtr> update_outputs;
  for (const auto &kernel_node : kernel_nodes) {
    mindspore::infer::SetKernelInfo(kernel_node);
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    std::shared_ptr<kernel::KernelMod> kernel_mod = kernel::Factory<kernel::KernelMod>::Instance().Create(kernel_name);
    if (kernel_mod == nullptr) {
      MS_LOG(EXCEPTION) << "Kernel mod is nullptr, kernel name: " << kernel_name;
    }
    MS_LOG(INFO) << "SingleOpInferSession::Kernels " << kernel_name;
    if (kernel_name == kNameCustomAscend) {
      kernel_mod->SetDevicedId(device_id_);
    }
    auto args = kernel::AbstractArgsFromCNode(kernel_node);
    mindspore::infer::CopyInputWeights(kernel_node, args.inputs);
    auto ret = kernel_mod->Init(args.op, args.inputs, args.outputs);
    MS_LOG(INFO) << "SingleOpInferSession::Kernels ret " << ret;
    if (!ret) {
      MS_LOG(EXCEPTION) << "kernel init failed " << kernel_name;
    }

    std::vector<size_t> input_size_list;
    std::vector<size_t> output_size_list;
    InitInputSizeList(kernel_node, &input_size_list);
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    for (size_t output_index = 0; output_index < output_num; ++output_index) {
      TypeId type_id = AnfAlgo::GetOutputDeviceDataType(kernel_node, output_index);
      size_t type_size = GetTypeByte(TypeIdToType(type_id));
      auto shape = AnfAlgo::GetOutputDeviceShape(kernel_node, output_index);
      size_t tensor_size =
        shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
      tensor_size = std::max(tensor_size, type_size);
      (void)output_size_list.emplace_back(tensor_size);
    }
    kernel_mod->SetInputSizeList(input_size_list);
    kernel_mod->SetOutputSizeList(output_size_list);

    AnfAlgo::SetKernelMod(kernel_mod, kernel_node.get());

    if (kernel_name == kNameCustomAscend) {
      if (is_dvpp_) {
        update_inputs = kernel_mod->GetInputKernelTensor();
      }
      update_flag = true;
      update_outputs = kernel_mod->RetrieveOutputShape();
    }
  }

  RuntimeUtils::AssignKernelGraphAddress(kernel_graph_);

  std::vector<tensor::TensorPtr> graph_inputs, graph_outputs;
  kernel_graph_utils_->GetModelInputsInfo(kernel_graph_->graph_id(), &graph_inputs, &input_names_);
  kernel_graph_utils_->GetModelOutputsInfo(kernel_graph_->graph_id(), &graph_outputs, &output_names_);
  if (graph_inputs.size() != input_names_.size()) {
    MS_LOG(ERROR) << "Graph input size " << graph_inputs.size() << " != input names size " << input_names_.size();
    return kCoreFailed;
  }
  if (graph_outputs.size() != output_names_.size()) {
    MS_LOG(ERROR) << "Graph output size " << graph_outputs.size() << " != output names size " << output_names_.size();
    return kCoreFailed;
  }
  if (is_dvpp_) {
    MS_LOG(INFO) << "Update input kernel tensor shape, data type, and format for CustomAscend DVPP";
    auto ret = UpdateGraphInputsForDVPP(update_inputs);
    if (ret != kSuccess) {
      return ret;
    }
  } else {
    for (size_t i = 0; i < input_names_.size(); i++) {
      auto &input = graph_inputs[i];
      auto data_type = static_cast<enum DataType>(input->data_type());
      auto impl = std::make_shared<TensorDefaultImpl>(input_names_[i], data_type, input->shape_c());
      inputs_.push_back(impl);
    }
  }
  for (size_t i = 0; i < output_names_.size(); i++) {
    auto &output = graph_outputs[i];
    auto data_type = static_cast<enum DataType>(output->data_type());
    auto impl = std::make_shared<TensorDefaultImpl>(output_names_[i], data_type, output->shape_c());
    outputs_.push_back(impl);
  }

  if (update_flag) {
    for (size_t i = 0; i < update_outputs.size(); ++i) {
      outputs_.at(i)->SetShape(update_outputs.at(i)->GetShapeVector());
    }
  }
  return kSuccess;
}

Status SingleOpInferSession::RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs,
                                      const MSKernelCallBack &before, const MSKernelCallBack &after) {
  return RunGraph(inputs, outputs);
}

Status SingleOpInferSession::RunGraph(const std::vector<tensor::Tensor> &inputs, std::vector<tensor::Tensor> *outputs) {
  MS_LOG(INFO) << "SingleOpInferSession::RunGraph with input and outputs";
  MS_EXCEPTION_IF_NULL(kernel_graph_);

  RuntimeUtils::CopyInputTensorsToKernelGraph(inputs, kernel_graph_);

  auto &kernel_nodes = kernel_graph_->execution_order();
  for (const auto &kernel_node : kernel_nodes) {
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    MS_LOG(INFO) << "SingleOpInferSession::RunGraph " << kernel_name;
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel_node);
    MS_EXCEPTION_IF_NULL(kernel_mod);
    std::vector<kernel::AddressPtr> kernel_inputs;
    size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
    for (size_t i = 0; i < input_num; ++i) {
      auto device_address = AnfAlgo::GetPrevNodeMutableOutputAddr(kernel_node, i);
      auto input = RuntimeUtils::GetAddressFromDevice(device_address);
      kernel_inputs.push_back(input);
    }
    std::vector<kernel::AddressPtr> kernel_outputs;
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    for (size_t i = 0; i < output_num; ++i) {
      auto device_address = AnfAlgo::GetMutableOutputAddr(kernel_node, i);
      auto output = RuntimeUtils::GetAddressFromDevice(device_address);
      kernel_outputs.push_back(output);
    }
    std::vector<kernel::AddressPtr> kernel_workspaces;
    bool ret = true;
    try {
      ret = kernel_mod->Launch(kernel_inputs, kernel_workspaces, kernel_outputs, 0);
      RuntimeUtils::UpdateKernelNodeOutputInfo(kernel_node, kernel_outputs);
    } catch (std::exception &e) {
      MS_LOG(EXCEPTION) << e.what();
    }
    if (!ret) {
      MS_LOG(EXCEPTION) << "Launch kernel failed.";
    }
  }

  RuntimeUtils::CopyOutputTensorsFromKernelGraph(outputs, kernel_graph_);
  return kSuccess;
}

Status SingleOpInferSession::ResizeGraphInputs(const std::vector<tensor::Tensor> &inputs,
                                               const std::vector<std::vector<int64_t>> &dims) {
  if (inputs_.size() != inputs.size()) {
    MS_LOG(ERROR) << "Graph inputs tensor size[" << inputs_.size() << " is not equal with user input tensor size["
                  << inputs.size() << "]";
    return kLiteError;
  }
  auto ret = UpdateKernelGraphInputs(dims, {}, true);
  if (ret != kSuccess) {
    return ret;
  }
  for (size_t i = 0; i < inputs.size(); i++) {
    // update session inputs_
    inputs_[i]->SetShape(dims[i]);
  }
  return kSuccess;
}

Status SingleOpInferSession::Resize(const std::vector<tensor::Tensor> &inputs,
                                    const std::vector<std::vector<int64_t>> &dims) {
  if (ResizeGraphInputs(inputs, dims) != kSuccess) {
    MS_LOG(EXCEPTION) << "Resize graph input error. ";
  }
  auto &kernel_nodes = kernel_graph_->execution_order();
  for (const auto &kernel_node : kernel_nodes) {
    std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
    MS_LOG(INFO) << "SingleOpInferSession::Resize " << kernel_name;
    auto kernel_mod = AnfAlgo::GetKernelMod(kernel_node);
    if (kernel_mod == nullptr) {
      MS_LOG(EXCEPTION) << "Kernel mod is nullptr, kernel name: " << kernel_name;
    }
    auto args = kernel::AbstractArgsFromCNode(kernel_node);
    if (kernel_mod->Resize(args.op, args.inputs, args.outputs) != kSuccess) {
      MS_LOG(EXCEPTION) << "Kernel mod resize failed, kernel name: " << kernel_name;
    }
  }
  return kSuccess;
}

std::vector<MutableTensorImplPtr> SingleOpInferSession::GetOutputs() { return outputs_; }
std::vector<MutableTensorImplPtr> SingleOpInferSession::GetInputs() { return inputs_; }
std::vector<std::string> SingleOpInferSession::GetOutputNames() { return output_names_; }
std::vector<std::string> SingleOpInferSession::GetInputNames() { return input_names_; }

MutableTensorImplPtr SingleOpInferSession::GetOutputByTensorName(const std::string &tensor_name) {
  for (size_t idx = 0; idx < output_names_.size(); ++idx) {
    if (output_names_[idx] == tensor_name) {
      if (idx < outputs_.size()) {
        return outputs_[idx];
      }
    }
  }
  MS_LOG(ERROR) << "Can't found tensor name " << tensor_name;
  return nullptr;
}

MutableTensorImplPtr SingleOpInferSession::GetInputByTensorName(const std::string &tensor_name) {
  for (size_t idx = 0; idx < input_names_.size(); ++idx) {
    if (input_names_[idx] == tensor_name) {
      if (idx < inputs_.size()) {
        return inputs_[idx];
      }
    }
  }
  MS_LOG(ERROR) << "Can't found tensor name " << tensor_name;
  return nullptr;
}

static std::shared_ptr<InferSession> SingleOpSessionCreator(const std::shared_ptr<Context> &ctx,
                                                            const ConfigInfos &config_infos) {
  auto session = std::make_shared<SingleOpInferSession>();
  session->Init(ctx);
  return session;
}
REG_SESSION(kSingleOpSession, SingleOpSessionCreator);
}  // namespace mindspore
