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

#include "plugin/device/cpu/kernel/nllloss_cpu_kernel.h"
#include <map>
#include <string>
#include "mindspore/core/ops/nllloss.h"
#include "nnacl/errorcode.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kNLLLossInputsNum = 3;
constexpr size_t kNLLLossOutputsNum = 2;
const std::map<Reduction, ReductionType> kReductionMap = {
  {Reduction::MEAN, Reduction_Mean}, {Reduction::REDUCTION_SUM, Reduction_Sum}, {Reduction::NONE, Reduction_None}};
}  // namespace

bool NLLLossCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::NLLLoss>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR) << "cast NLLLoss ops failed!";
    return false;
  }

  kernel_name_ = kernel_ptr->GetPrim()->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);

  bool is_match = MatchKernelAttr(kernel_attr, GetOpSupport()).first;
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', it does not support this kernel data type: " << kernel_attr;
    return false;
  }

  auto reduction = kernel_ptr->get_reduction();
  auto pair = kReductionMap.find(reduction);
  if (pair == kReductionMap.end()) {
    MS_LOG(EXCEPTION) << "For " << kernel_name_
                      << ", the attr 'reduction' only support 'mean', 'sum' and 'none', but got " << reduction;
  }
  nllloss_param_.reduction_type_ = pair->second;
  return true;
}

int NLLLossCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  int ret = 0;
  if ((ret = KernelMod::Resize(base_operator, inputs, outputs, inputsOnHost)) != 0) {
    return ret;
  }

  auto logits_shape = inputs[kIndex0]->GetShapeVector();
  nllloss_param_.batch_ = LongToInt(logits_shape[kIndex0]);
  nllloss_param_.class_num_ = LongToInt(logits_shape[kIndex1]);

  return KRET_OK;
}

bool NLLLossCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                 const std::vector<kernel::AddressPtr> &workspace,
                                 const std::vector<kernel::AddressPtr> &outputs) {
  CHECK_KERNEL_INPUTS_NUM(kNLLLossInputsNum, inputs.size(), kernel_name_);
  CHECK_KERNEL_OUTPUTS_NUM(kNLLLossOutputsNum, outputs.size(), kernel_name_);

  const auto *logits = reinterpret_cast<float *>(inputs[kIndex0]->addr);
  const auto *labels = reinterpret_cast<int *>(inputs[kIndex1]->addr);
  const auto *weight = reinterpret_cast<float *>(inputs[kIndex2]->addr);
  auto *loss = reinterpret_cast<float *>(outputs[kIndex0]->addr);
  auto *total_weight = reinterpret_cast<float *>(outputs[kIndex1]->addr);

  int ret = NLLLoss(logits, labels, weight, loss, total_weight, &nllloss_param_);
  if (ret != static_cast<int>(NNACL_OK)) {
    MS_LOG(EXCEPTION) << "Launch " << kernel_name_ << " failed, the nnacl error code " << ret;
  }
  return true;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, NLLLoss, NLLLossCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
