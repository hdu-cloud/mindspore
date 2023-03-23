/**
 * Copyright 2020-2022 Huawei Technologies Co., Ltd
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

#include "plugin/device/cpu/hal/device/kernel_select_cpu.h"
#include <string>
#include <memory>
#include <algorithm>
#include "kernel/common_utils.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "kernel/oplib/oplib.h"
#include "plugin/device/cpu/kernel/pyfunc/py_func_cpu_kernel.h"
#include "plugin/device/cpu/kernel/custom/custom_aot_cpu_kernel.h"
#include "plugin/device/cpu/kernel/custom/custom_julia_cpu_kernel.h"
#include "utils/trace_base.h"
#include "include/common/utils/convert_utils.h"
#include "include/common/utils/utils.h"
#include "mindspore/core/ops/core_ops.h"
#include "mindspore/core/ops/op_name.h"

namespace mindspore {
namespace device {
namespace cpu {
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;
using mindspore::kernel::KernelBuildInfo;
namespace {
constexpr auto kParamDynamic = "dynamic";

static const std::set<std::string> kVmapCPUWhiteList = {kUnsortedSegmentMinOpName,
                                                        kUnsortedSegmentMaxOpName,
                                                        kUnsortedSegmentSumOpName,
                                                        kUnsortedSegmentProdOpName,
                                                        kUniqueWithPadOpName,
                                                        kMaskedFillOpName,
                                                        kDataFormatDimMapOpName,
                                                        kSTFTOpName,
                                                        kRandomChoiceWithMaskOpName,
                                                        kApplyAdamOpName,
                                                        kUniformCandidateSamplerOpName,
                                                        kSplitOpName,
                                                        kLinSpaceOpName,
                                                        kSquareSumAllOpName,
                                                        kApplyAdaMaxOpName,
                                                        kApplyAdadeltaOpName,
                                                        kApplyProximalAdagradOpName,
                                                        kApplyGradientDescentOpName,
                                                        kApplyProximalGradientDescentOpName,
                                                        kApplyPowerSignOpName,
                                                        kApplyAdagradV2OpName,
                                                        kApplyAdagradDAOpName,
                                                        kApplyRMSPropOpName,
                                                        kApplyCenteredRMSPropOpName,
                                                        kSparseApplyAdagradOpName,
                                                        kSparseApplyAdagradV2OpName,
                                                        kSparseApplyFtrlOpName,
                                                        kRandomShuffleOpName,
                                                        kApplyAdamWithAmsgradOpName,
                                                        kApplyFtrlOpName,
                                                        prim::kMatrixBandPart,
                                                        prim::kGer,
                                                        prim::kCdist,
                                                        prim::kCdistGrad,
                                                        prim::kSparseSegmentMean};

void GetOutputDtypes(const CNodePtr &kernel_node, std::vector<TypeId> *output_types) {
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    TypeId dtype = common::AnfAlgo::GetOutputInferDataType(kernel_node, output_index);
    output_types->emplace_back(dtype);
  }
}

void GetOutputFormat(const CNodePtr &kernel_node, std::vector<std::string> *output_formats) {
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    output_formats->emplace_back(kOpFormat_DEFAULT);
  }
}

void GetInputDtypes(const CNodePtr &kernel_node, std::vector<TypeId> *input_types) {
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    TypeId dtype = kTypeUnknown;
    dtype = common::AnfAlgo::GetPrevNodeOutputInferDataType(kernel_node, input_index);
    input_types->emplace_back(dtype);
  }
}

void GetInputFormat(const CNodePtr &kernel_node, std::vector<std::string> *input_formats) {
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  for (size_t input_index = 0; input_index < input_num; ++input_index) {
    input_formats->emplace_back(kOpFormat_DEFAULT);
  }
}

void GetOutputFormatsAndDtypes(const CNodePtr &kernel_node, const kernel::KernelAttr &kernel_attr,
                               std::vector<std::string> *output_formats, std::vector<TypeId> *output_types) {
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    output_formats->emplace_back(kernel_attr.GetOutputAttr(output_index).second);
    auto dtype = kernel_attr.GetOutputAttr(output_index).first;
    output_types->emplace_back(dtype);
  }
}

bool InputDtypeMatch(TypeId InputAttr, TypeId input_type, bool strict) {
  if (InputAttr == input_type) {
    return true;
  }
  if (!strict && InputAttr == kNumberTypeInt32 && (input_type == kNumberTypeInt16 || input_type == kNumberTypeInt64)) {
    return true;
  }
  if (!strict && InputAttr == kNumberTypeFloat32 &&
      (input_type == kNumberTypeFloat16 || input_type == kNumberTypeFloat64)) {
    return true;
  }
  return false;
}

int GetOutputDtypeMatchedNum(const kernel::KernelAttr &kernel_attr, const std::vector<TypeId> &output_types) {
  if (kernel_attr.GetOutputSize() != output_types.size()) {
    MS_LOG(DEBUG) << "required output num:" << kernel_attr.GetInputSize()
                  << ", actual output num:" << output_types.size();
    return 0;
  }
  int data_type_matched_num = 0;
  auto output_num = output_types.size();
  for (size_t i = 0; i < output_num; ++i) {
    if (kernel_attr.GetOutputAttr(i).first != output_types[i]) {
      MS_LOG(DEBUG) << "required dtype:" << kernel_attr.GetOutputAttr(i).first
                    << ", actual output dtype:" << output_types[i];
    } else {
      data_type_matched_num++;
    }
  }
  return data_type_matched_num;
}

int GetInputDtypeFormatMatchedNum(const kernel::KernelAttr &kernel_attr, const std::vector<TypeId> &input_types,
                                  bool strict) {
  if (kernel_attr.GetInputSize() != input_types.size()) {
    MS_LOG(DEBUG) << "required input num:" << kernel_attr.GetInputSize() << ", actual input num:" << input_types.size();
    return 0;
  }
  int data_type_matched_num = 0;
  auto input_num = input_types.size();
  for (size_t i = 0; i < input_num; ++i) {
    if (!InputDtypeMatch(kernel_attr.GetInputAttr(i).first, input_types[i], strict)) {
      MS_LOG(DEBUG) << "required dtype:" << kernel_attr.GetInputAttr(i).first
                    << ", actual input dtype:" << input_types[i];
    } else {
      data_type_matched_num++;
    }
  }
  return data_type_matched_num;
}

void ExpandKernelAttr(const CNodePtr &kernel_node, kernel::KernelAttr *kernel_attr) {
  MS_EXCEPTION_IF_NULL(kernel_attr);
  size_t attr_num = kernel_attr->GetInputSize();
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (attr_num == 0) {
    MS_LOG(EXCEPTION) << "Input size is empty";
    return;  // To pass the CI Check_Cppcheck
  }
  // Only support one dynamic input like Concat or
  // many dynamic input but each input has same number like DynamicStitch
  std::string format = kOpFormat_DEFAULT;
  std::vector<DataType> attr_list;
  size_t each_attr_input_num = input_num / attr_num;
  for (size_t i = 0; i < attr_num; ++i) {
    TypeId input_dtype = kernel_attr->GetInputAttr(i).first;
    for (size_t j = 0; j < each_attr_input_num; ++j) {
      (void)attr_list.emplace_back(input_dtype, format);
    }
  }
  kernel_attr->SetInputAttrList(attr_list);

  TypeId output_dtype = kernel_attr->GetOutputAttr(0).first;
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  for (size_t i = 1; i < output_num; ++i) {
    (void)kernel_attr->AddOutputAttr(output_dtype);
  }
}

void ExpandMultiDynamicAttr(const CNodePtr &kernel_node, const std::vector<int64_t> &dyn_input_sizes,
                            kernel::KernelAttr *kernel_attr) {
  MS_EXCEPTION_IF_NULL(kernel_attr);
  // Judge whether the inputs are consistent
  std::unordered_set<int64_t> dyn_input_sizes_no_repetition(dyn_input_sizes.begin(), dyn_input_sizes.end());
  if (dyn_input_sizes_no_repetition.size() == 1 && kernel_attr->GetInputSize() == 1) {
    MS_LOG(EXCEPTION)
      << "For single dynamic input, the cpu kernel should register the 'AddSkipCheckAttr' or 'AddAllSameAttr'.";
  }
  MS_LOG(DEBUG) << "Process multi dynamic inputs.";

  size_t inpyt_attr_num = kernel_attr->GetInputSize();
  size_t dyn_input_size = dyn_input_sizes.size();
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (inpyt_attr_num == 0 || input_num == 0) {
    MS_LOG(EXCEPTION) << "Input size is empty";
  }
  if (inpyt_attr_num != dyn_input_size) {
    MS_LOG(EXCEPTION) << "Input size: " << inpyt_attr_num << ", is not equal to dynamic input size: " << dyn_input_size;
  }
  // Expand input kernel attr, support multi dynamic inputs
  std::string format = kOpFormat_DEFAULT;
  std::vector<DataType> input_attr_list;
  for (size_t input_index = 0; input_index < inpyt_attr_num; ++input_index) {
    TypeId input_dtype = kernel_attr->GetInputAttr(input_index).first;
    int64_t dyn_input_num = dyn_input_sizes[input_index];
    if (dyn_input_num < 0) {
      dyn_input_num = 1;
    }
    for (size_t j = 0; j < LongToSize(dyn_input_num); ++j) {
      (void)input_attr_list.emplace_back(input_dtype, format);
    }
  }
  kernel_attr->SetInputAttrList(input_attr_list);

  size_t output_attr_num = kernel_attr->GetOutputSize();
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_attr_num == output_num) {
    MS_LOG(DEBUG) << "Output is not dynamic.";
    return;
  }
  if (output_attr_num == 0) {
    MS_LOG(EXCEPTION) << "Output size is empty";
  }
  // Expand output kernel attr, only support one dynamic output. TODO: support multi dynamic outputs
  std::vector<DataType> output_attr_list;
  for (size_t output_index = 0; output_index < output_num; ++output_index) {
    TypeId output_dtype = kernel_attr->GetOutputAttr(0).first;
    (void)output_attr_list.emplace_back(output_dtype, format);
  }
  kernel_attr->SetOutputAttrList(output_attr_list);
}

void SetKernelBuildInfo(const std::vector<std::string> &input_formats, const std::vector<TypeId> &input_types,
                        const std::vector<std::string> &output_formats, const std::vector<TypeId> &output_types,
                        AnfNode *kernel_node) {
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  MS_EXCEPTION_IF_NULL(builder);
  builder->SetInputsFormat(input_formats);
  builder->SetInputsDeviceType(input_types);
  builder->SetOutputsFormat(output_formats);
  builder->SetOutputsDeviceType(output_types);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), kernel_node);
}

std::pair<std::string, ExceptionType> KernelNotSupportWarning(const AnfNodePtr &kernel_node,
                                                              const std::vector<TypeId> &input_types,
                                                              const std::vector<TypeId> &infer_output_types,
                                                              bool is_kernel_exist) {
  std::string kernel_name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (!is_kernel_exist) {
    std::stringstream ss;
    ss << "Unsupported op [" << kernel_name << "] on CPU, Please confirm whether the device target setting is correct, "
       << "or refer to 'mindspore.ops' at https://www.mindspore.cn to query the operator support list."
       << trace::DumpSourceLines(kernel_node);
    return {ss.str(), NotSupportError};
  }

  std::stringstream operator_info;
  operator_info << "Operator[" << kernel_name << "] ";
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  if (input_num > 0) {
    operator_info << " input(";
    for (size_t i = 0; i < input_num; ++i) {
      operator_info << TypeIdLabel(input_types[i]);
      if (i != input_num - 1) {
        operator_info << ",";
      }
    }
    operator_info << ") ";
  }
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  if (output_num > 0) {
    operator_info << "output(";
    for (size_t i = 0; i < output_num; ++i) {
      operator_info << TypeIdLabel(infer_output_types[i]);
      if (i != output_num - 1) {
        operator_info << ",";
      }
    }
    operator_info << ") ";
  }
  operator_info
    << "is not supported. This error means the current input type is not supported, please refer to the MindSpore "
       "doc for supported types.";
  operator_info << trace::DumpSourceLines(kernel_node);
  return {operator_info.str(), TypeError};
}

void UpdateDynamicKernelBuildInfo(const CNodePtr &kernel_node) {
  const std::string &op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  MS_LOG(INFO) << "Operator name: " << op_name;
  // Set kernel build info
  std::vector<TypeId> input_types;
  GetInputDtypes(kernel_node, &input_types);
  std::vector<TypeId> output_types;
  GetOutputDtypes(kernel_node, &output_types);
  std::vector<std::string> input_formats;
  GetInputFormat(kernel_node, &input_formats);
  std::vector<std::string> output_formats;
  GetOutputFormat(kernel_node, &output_formats);
  SetKernelBuildInfo(input_formats, input_types, output_formats, output_types, kernel_node.get());
}

bool CheckKernelInfo(const std::shared_ptr<KernelBuildInfo> &alternative_kernel_info,
                     const std::shared_ptr<KernelBuildInfo> &selected_kernel_info) {
  MS_EXCEPTION_IF_NULL(selected_kernel_info);
  MS_EXCEPTION_IF_NULL(alternative_kernel_info);
  size_t selected_input_num = selected_kernel_info->GetInputNum();
  size_t alternative_input_num = alternative_kernel_info->GetInputNum();
  if (selected_input_num != alternative_input_num) {
    return false;
  }
  for (size_t i = 0; i < selected_input_num; i++) {
    auto format = alternative_kernel_info->GetInputFormat(i);
    if (selected_kernel_info->GetInputFormat(i) != format && (!format.empty())) {
      return false;
    }
    auto type = alternative_kernel_info->GetInputDeviceType(i);
    if (selected_kernel_info->GetInputDeviceType(i) != type && (type != TypeId::kMetaTypeNone)) {
      return false;
    }
  }

  size_t selected_output_num = selected_kernel_info->GetOutputNum();
  size_t alternative_output_num = alternative_kernel_info->GetOutputNum();
  if (selected_output_num != alternative_output_num) {
    return false;
  }
  for (size_t i = 0; i < selected_output_num; i++) {
    auto format = alternative_kernel_info->GetOutputFormat(i);
    if (selected_kernel_info->GetOutputFormat(i) != format && (!format.empty())) {
      return false;
    }
    auto type = alternative_kernel_info->GetOutputDeviceType(i);
    if (selected_kernel_info->GetOutputDeviceType(i) != type && (type != TypeId::kMetaTypeNone)) {
      return false;
    }
  }
  return true;
}

void UpdateCustomKernelBuildInfo(const CNodePtr &kernel_node, bool is_akg_op) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto builder = std::make_shared<kernel::KernelBuildInfo::KernelBuildInfoBuilder>();
  const std::string &op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  std::shared_ptr<mindspore::kernel::OpInfo> kernel_attr = nullptr;
  if (is_akg_op) {
#ifndef USE_LLVM
    MS_LOG(EXCEPTION) << "When calling AKG-CPU operator, found LLVM 12.0.1 not installed, please check: "
                         "https://www.mindspore.cn/install for installing LLVM on MindSpore.";
#else
    builder->SetKernelType(KernelType::AKG_KERNEL);
#endif
    kernel_attr = mindspore::kernel::OpLib::FindOp(op_name, kernel::OpImplyType::kImplyAKG);
    if (kernel_attr == nullptr) {
      MS_LOG(WARNING) << "Not find operator information for Custom operator[" << op_name << "]. "
                      << "Infer operator information from inputs. For more details, "
                      << "please refer to 'mindspore.ops.Custom' at https://www.mindspore.cn.";
    }
  } else {
    builder->SetKernelType(KernelType::CPU_KERNEL);
  }
  builder->SetProcessor(kernel::Processor::CPU);
  // Set inputs info
  std::vector<TypeId> input_types;
  GetInputDtypes(kernel_node, &input_types);
  std::vector<std::string> input_formats;
  GetInputFormat(kernel_node, &input_formats);
  builder->SetInputsDeviceType(input_types);
  builder->SetInputsFormat(input_formats);
  // Set inputs info
  std::vector<TypeId> output_types;
  GetOutputDtypes(kernel_node, &output_types);
  std::vector<std::string> output_formats;
  GetOutputFormat(kernel_node, &output_formats);
  builder->SetOutputsDeviceType(output_types);
  builder->SetOutputsFormat(output_formats);
  AnfAlgo::SetSelectKernelBuildInfo(builder->Build(), kernel_node.get());

  // check reg info if kernel_attr is not null
  if (kernel_attr != nullptr) {
    std::vector<std::shared_ptr<KernelBuildInfo>> kernel_info_list;
    if (!ParseMetadata(kernel_node, kernel_attr, kernel::Processor::CPU, &kernel_info_list)) {
      MS_LOG(EXCEPTION) << "Parsed metadata of op[" << op_name << "] failed.";
    }
    if (kernel_info_list.empty()) {
      MS_LOG(EXCEPTION) << "Not find valid metadata of op[" << op_name << "].";
    }
    bool match = std::any_of(kernel_info_list.begin(), kernel_info_list.end(),
                             [&](const std::shared_ptr<KernelBuildInfo> &alternative_kernel_info) {
                               return CheckKernelInfo(alternative_kernel_info, builder->Build());
                             });
    if (!match) {
      auto [msg, etype] = KernelNotSupportWarning(kernel_node, input_types, output_types, true);
      MS_EXCEPTION(etype) << msg;
    }
  }
}

kernel::KernelAttr FillNoneInKernelAttr(const CNodePtr &kernel_node, const std::vector<TypeId> &input_types,
                                        const std::vector<TypeId> &output_types,
                                        const kernel::KernelAttr &kernel_attr) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  // Only process Custom op
  if (!IsPrimitiveCNode(kernel_node, prim::kPrimCustom)) {
    return kernel_attr;
  }
  auto input_num = input_types.size();
  auto output_num = output_types.size();
  if (kernel_attr.GetInputSize() != input_types.size() || kernel_attr.GetOutputSize() != output_types.size()) {
    MS_LOG(DEBUG) << "required input num:" << kernel_attr.GetInputSize() << ", actual input num:" << input_num;
    MS_LOG(DEBUG) << "required input num:" << kernel_attr.GetOutputSize() << ", actual input num:" << output_num;
    return kernel_attr;
  }
  kernel::KernelAttr result;
  // Fill inputs info.
  for (size_t i = 0; i < input_num; ++i) {
    auto type_format = kernel_attr.GetInputAttr(i);
    if (type_format.first == TypeId::kMetaTypeNone) {
      type_format.first = input_types[i];
    }
    if (type_format.second.empty()) {
      type_format.second = kOpFormat_DEFAULT;
    }
    (void)result.AddInputAttr(type_format.first, type_format.second);
  }
  // Fill outputs info.
  for (size_t i = 0; i < output_num; ++i) {
    auto type_format = kernel_attr.GetOutputAttr(i);
    if (type_format.first == TypeId::kMetaTypeNone) {
      type_format.first = output_types[i];
    }
    if (type_format.second.empty()) {
      type_format.second = kOpFormat_DEFAULT;
    }
    (void)result.AddOutputAttr(type_format.first, type_format.second);
  }
  return result;
}
}  // namespace

bool IsDynamicParamKernel(const std::string &op_name) {
  const auto &op_info = kernel::OpLib::FindOp(op_name, kernel::OpImplyType::kImplyCPU);
  if (op_info == nullptr) {
    return false;
  }

  const auto &input_io_info = op_info->inputs_ptr();
  if (input_io_info.size() != 1 || input_io_info[0]->param_type() != kParamDynamic) {
    return false;
  }

  const auto &output_io_info = op_info->outputs_ptr();
  if (output_io_info.size() != 1 || output_io_info[0]->param_type() != kParamDynamic) {
    return false;
  }

  return true;
}

bool SelectKernel(const CNodePtr &kernel_node, kernel::KernelAttr *selected_kernel_attr,
                  const std::vector<kernel::KernelAttr> &kernel_attrs, const std::vector<TypeId> &input_types,
                  const std::vector<TypeId> &output_types, std::pair<bool, bool> *matched, bool strict) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  MS_EXCEPTION_IF_NULL(selected_kernel_attr);
  MS_EXCEPTION_IF_NULL(matched);
  MS_LOG(DEBUG) << "Select kernel for op: " << common::AnfAlgo::GetCNodeName(kernel_node);
  std::vector<int64_t> dyn_input_sizes = {};
  if (common::AnfAlgo::HasNodeAttr(kAttrDynInputSizes, kernel_node)) {
    dyn_input_sizes = common::AnfAlgo::GetNodeAttr<std::vector<int64_t>>(kernel_node, kAttrDynInputSizes);
  }
  for (auto kernel_attr : kernel_attrs) {
    if (kernel_attr.GetAllSame()) {
      ExpandKernelAttr(kernel_node, &kernel_attr);
    } else if (!kernel_attrs[0].GetSkipCheck() && !dyn_input_sizes.empty()) {
      // if there are multi dynamic inputs, expand the kernel attr.
      ExpandMultiDynamicAttr(kernel_node, dyn_input_sizes, &kernel_attr);
    }
    size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
    if (kernel_attr.GetOutputSize() != output_num) {
      MS_LOG(DEBUG) << "Output num is not equal!";
      continue;
    }

    auto new_kernel_attr = FillNoneInKernelAttr(kernel_node, input_types, output_types, kernel_attr);
    int input_dtype_matched_num = GetInputDtypeFormatMatchedNum(new_kernel_attr, input_types, strict);
    int output_dtype_matched_num = GetOutputDtypeMatchedNum(new_kernel_attr, output_types);
    // All formats and data types matched
    if (input_dtype_matched_num == SizeToInt(input_types.size())) {
      *selected_kernel_attr = new_kernel_attr;
      matched->first = true;
      if (output_dtype_matched_num == SizeToInt(output_types.size())) {
        matched->second = true;
        return true;
      }
    }
  }
  return false;
}

kernel::KernelAttr BuildKernelFromInput(const std::vector<TypeId> &inputs, const std::vector<TypeId> &outputs,
                                        const kernel::KernelAttr &origin_attr) {
  kernel::KernelAttr attr = origin_attr;
  for (auto in_dtype : inputs) {
    (void)attr.AddInputAttr(in_dtype);
  }
  for (auto out_dtype : outputs) {
    (void)attr.AddOutputAttr(out_dtype);
  }
  (void)attr.AddSkipCheckAttr(true);
  return attr;
}

void SetCustomOpKernelInfo(const std::string &custom_op_type, const std::string &op_name) {
  if (custom_op_type == kCustomTypePyfunc) {
    kernel::Factory<kernel::NativeCpuKernelMod>::Instance().Register(
      op_name, []() { return std::make_shared<kernel::PyFuncCpuKernelMod>(); });
  } else if (custom_op_type == kCustomTypeAOT) {
    kernel::Factory<kernel::NativeCpuKernelMod>::Instance().Register(
      op_name, []() { return std::make_shared<kernel::CustomAOTCpuKernelMod>(); });
  } else if (custom_op_type == kCustomTypeJULIA) {
    kernel::Factory<kernel::NativeCpuKernelMod>::Instance().Register(
      op_name, []() { return std::make_shared<kernel::CustomJULIACpuKernelMod>(); });
  } else {
    MS_LOG(EXCEPTION) << "Unsupported func type for Custom operator on CPU, it should be "
                      << "'hybrid', 'akg', 'pyfunc' or 'aot' or 'julia', "
                      << "but got [" << custom_op_type << "] for Custom operator [" << op_name << "]";
  }
}

std::pair<std::string, ExceptionType> SetKernelInfoWithMsg(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  const std::string &op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  if (common::AnfAlgo::HasNodeAttr(ops::kBatchRank, kernel_node) && !kVmapCPUWhiteList.count(op_name)) {
    std::stringstream ss;
    ss << op_name << " does not support 'batch_rank' on CPU, which means that 'vmap' cannot support " << op_name
       << " on CPU currently.";
    return {ss.str(), NotSupportError};
  }
  if (IsPrimitiveCNode(kernel_node, prim::kPrimCustom)) {
    auto tp = common::AnfAlgo::GetNodeAttr<std::string>(kernel_node, kAttrFuncType);
    if (IsOneOfCustomAkgType(tp)) {
      UpdateCustomKernelBuildInfo(kernel_node, true);
      return {};
    }
    if (!kernel::Factory<kernel::NativeCpuKernelMod>::Instance().IsRegistered(op_name)) {
      SetCustomOpKernelInfo(tp, op_name);
    }
    // If Custom op has not set reg info,
    // or the no info about inputs in reg info(the case of undetermined input size),
    // then infer info from inputs
    auto op_reg_info = mindspore::kernel::OpLib::FindOp(op_name, kernel::OpImplyType::kImplyCPU);
    if (op_reg_info == nullptr || op_reg_info->inputs_ptr().size() == 0) {
      MS_LOG(WARNING) << "Not find operator information for Custom operator[" << op_name << "]. "
                      << "Infer operator information from inputs. For more details, "
                      << "please refer to 'mindspore.ops.Custom' at https://www.mindspore.cn.";
      UpdateCustomKernelBuildInfo(kernel_node, false);
      return {};
    }
  } else if (IsDynamicParamKernel(op_name)) {
    // Select for dynamic kernel(both the number and data type are undetermined).
    UpdateDynamicKernelBuildInfo(kernel_node);
    return {};
  } else if (IsAKGSparseOP(kernel_node)) {
    UpdateCustomKernelBuildInfo(kernel_node, true);
    return {};
  }

  std::vector<std::string> input_formats;
  std::vector<TypeId> input_types;
  std::vector<std::string> selected_output_formats;
  std::vector<TypeId> output_types;
  std::vector<TypeId> selected_output_types;
  MS_LOG(INFO) << "SetKernelInfo, CNode Name: " << op_name;
  GetInputDtypes(kernel_node, &input_types);
  GetOutputDtypes(kernel_node, &output_types);
  kernel::KernelAttr selected_kernel_attr;
  std::pair<bool, bool> matched = std::make_pair(false, false);
  auto kernel_attrs = kernel::NativeCpuKernelMod::GetCpuSupportedList(op_name);
  // If GetSkipCheck is true, that means we do not check the build info between input and registered.
  // Take the input attrs to build the kernel.
  if (!kernel_attrs.empty() && kernel_attrs[0].GetSkipCheck()) {
    kernel_attrs[0] = BuildKernelFromInput(input_types, output_types, kernel_attrs[0]);
    MS_LOG(DEBUG) << "Build kernel form input for " << op_name;
  }
  if (!SelectKernel(kernel_node, &selected_kernel_attr, kernel_attrs, input_types, output_types, &matched, true)) {
    if (op_name == "Cast") {
      return KernelNotSupportWarning(kernel_node, input_types, output_types, !kernel_attrs.empty());
    }
    matched = std::make_pair(false, false);
    (void)SelectKernel(kernel_node, &selected_kernel_attr, kernel_attrs, input_types, output_types, &matched, false);
    if (!matched.first) {
      return KernelNotSupportWarning(kernel_node, input_types, output_types, !kernel_attrs.empty());
    }
  }

  if (matched.first) {
    MS_LOG(INFO) << "Input format and dtype is matched";
    GetOutputFormatsAndDtypes(kernel_node, selected_kernel_attr, &selected_output_formats, &selected_output_types);
    for (size_t index = 0; index < selected_kernel_attr.GetInputSize(); index++) {
      input_types[index] = selected_kernel_attr.GetInputAttr(index).first;
      (void)input_formats.emplace_back(selected_kernel_attr.GetInputAttr(index).second);
    }
  }
  SetKernelBuildInfo(input_formats, input_types, selected_output_formats, selected_output_types, kernel_node.get());
  return {};
}

void CPUGraphKernelInfo::SetKernelInfo(const CNodePtr &kernel_node, KernelType) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  auto [msg, etype] = SetKernelInfoWithMsg(kernel_node);
  if (msg.empty()) {
    return;
  }
  MS_EXCEPTION(etype) << msg;
}
}  // namespace cpu
}  // namespace device
}  // namespace mindspore
