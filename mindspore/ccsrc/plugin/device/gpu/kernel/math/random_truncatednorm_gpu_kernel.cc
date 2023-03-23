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

#include "plugin/device/gpu/kernel/math/random_truncatednorm_gpu_kernel.h"

namespace mindspore {
namespace kernel {
bool TruncatedNormalGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                       const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->GetPrim()->name();
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' does not support this kernel type: " << kernel_attr;
    return false;
  }
  kernel_func_ = func_list_[index].second;
  auto kernel_ptr = std::make_shared<ops::TruncatedNormal>(base_operator->GetPrim());
  seed_ = static_cast<int>(kernel_ptr->get_seed());
  seed2_ = static_cast<int>(kernel_ptr->get_seed2());
  unit_input_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  unit_output_size_ = abstract::TypeIdSize(kernel_attr.GetOutputAttr(kIndex0).first);
  return true;
}

int TruncatedNormalGpuKernelMod::Resize(const BaseOperatorPtr &base_operator,
                                        const std::vector<KernelTensorPtr> &inputs,
                                        const std::vector<KernelTensorPtr> &outputs,
                                        const std::map<uint32_t, tensor::TensorPtr> &) {
  if (!IsValidShape(inputs[0]->GetShapeVector())) {
    return KRET_UNKNOWN_SHAPE;
  }
  if (inputs[0] == 0 && inputs.size() == 0) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "'input shape shouldn't be null";
    return false;
  }
  ResetResource();
  std::vector<int64_t> input_shape_ = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                           inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  std::vector<int64_t> output_shape_ = std::vector<int64_t>(outputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                            outputs.at(kIndex0)->GetDeviceShapeAdaptively().end());

  for (size_t i = 0; i < input_shape_.size(); i++) {
    input_num_ *= input_shape_[i];
  }
  for (size_t i = 0; i < output_shape_.size(); i++) {
    output_num_ *= output_shape_[i];
  }
  input_size_list_.emplace_back(input_num_ * unit_input_size_);
  output_size_list_.emplace_back(output_num_ * unit_output_size_);
  workspace_size_list_.emplace_back(output_num_ * sizeof(curandState));
  return KRET_OK;
}

void TruncatedNormalGpuKernelMod::ResetResource() noexcept {
  input_num_ = 1;
  output_num_ = 1;
  workspace_size_list_.clear();
  input_size_list_.clear();
  output_size_list_.clear();
}

template <typename S>
bool TruncatedNormalGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs,
                                               const std::vector<AddressPtr> &workspace,
                                               const std::vector<AddressPtr> &outputs) {
  S *output = GetDeviceAddress<S>(outputs, 0);
  curandState *devStates = nullptr;
  void *workspace_addr = GetDeviceAddress<void *>(workspace, 0);
  devStates = reinterpret_cast<curandState *>(workspace_addr);
  TruncatedNormal(seed_, seed2_, devStates, output, output_num_, reinterpret_cast<cudaStream_t>(cuda_stream_));
  return true;
}

std::vector<std::pair<KernelAttr, TruncatedNormalGpuKernelMod::TruncatedNormalFunc>>
  TruncatedNormalGpuKernelMod::func_list_ = {
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat16),
     &TruncatedNormalGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat32),
     &TruncatedNormalGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeFloat64),
     &TruncatedNormalGpuKernelMod::LaunchKernel<double>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat16),
     &TruncatedNormalGpuKernelMod::LaunchKernel<half>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat32),
     &TruncatedNormalGpuKernelMod::LaunchKernel<float>},
    {KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeFloat64),
     &TruncatedNormalGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> TruncatedNormalGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, TruncatedNormalFunc> &pair) { return pair.first; });
  return support_list;
}
MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, TruncatedNormal, TruncatedNormalGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
