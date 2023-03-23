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

#include "plugin/device/gpu/kernel/math/betainc_gpu_kernel.h"
#include <utility>
#include <algorithm>
#include "kernel/common_utils.h"
#include "abstract/utils.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/betainc_impl.cuh"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBetaincInputsNum = 3;
constexpr size_t kAIndex = 0;
constexpr size_t kBIndex = 1;
constexpr size_t kXIndex = 2;
}  // namespace

bool BetaincGpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                               const std::vector<KernelTensorPtr> &outputs) {
  kernel_name_ = base_operator->name();
  auto kernel_attr = GetKernelAttrFromTensors(inputs, outputs);
  auto [is_match, index] = MatchKernelAttr(kernel_attr, GetOpSupport());
  if (inputs.empty() || outputs.empty()) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "' got empty inputs or outputs, which is invalid.";
    return false;
  }
  if (inputs.size() != kBetaincInputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of inputs should be 3, but got " << inputs.size();
    return false;
  }
  constexpr int OUTPUT_NUM = 1;
  if (outputs.size() != OUTPUT_NUM) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "', the number of outputs should be 1, but got " << outputs.size();
    return false;
  }
  if (!is_match) {
    MS_LOG(ERROR) << "For '" << kernel_name_
                  << "', the kernel type should be in [float32, float64], but got: " << kernel_attr << ".";
    return false;
  }
  kernel_func_ = func_list_[index].second;
  input_size_ = abstract::TypeIdSize(kernel_attr.GetInputAttr(kIndex0).first);
  return true;
}
int BetaincGpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                const std::vector<KernelTensorPtr> &outputs,
                                const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) {
  for (const auto &input : inputs) {
    // If any input shape contains -1, means input shape is dynamic, so just return do nothing.
    auto input_shape = input->GetShapeVector();
    if (!IsValidShape(input_shape)) {
      return KRET_UNKNOWN_SHAPE;
    }
  }
  ResetResource();
  std::vector<int64_t> input_shape_ = std::vector<int64_t>(inputs.at(kIndex0)->GetDeviceShapeAdaptively().begin(),
                                                           inputs.at(kIndex0)->GetDeviceShapeAdaptively().end());
  for (int64_t i = 0; i < static_cast<int64_t>(input_shape_.size()); i++) {
    input_size_ *= input_shape_[i];
  }
  InitSizeLists();
  return KRET_OK;
}

bool BetaincGpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                 const std::vector<kernel::AddressPtr> &workspace,
                                 const std::vector<kernel::AddressPtr> &outputs, void *cuda_stream) {
  if (is_null_input_) {
    return true;
  }
  return kernel_func_(this, inputs, outputs, workspace, cuda_stream);
}

template <typename T>
bool BetaincGpuKernelMod::LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs,
                                       const std::vector<AddressPtr> &workspace, void *cuda_stream) {
  T *input_a = GetDeviceAddress<T>(inputs, kAIndex);
  T *input_b = GetDeviceAddress<T>(inputs, kBIndex);
  T *input_x = GetDeviceAddress<T>(inputs, kXIndex);
  T *output = GetDeviceAddress<T>(outputs, kAIndex);

  CalBetainc(input_size_ / sizeof(T), input_a, input_b, input_x, output, device_id_,
             reinterpret_cast<cudaStream_t>(cuda_stream));

  return true;
}

std::vector<std::pair<KernelAttr, BetaincGpuKernelMod::KernelFunc>> BetaincGpuKernelMod::func_list_ = {
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddInputAttr(kNumberTypeFloat32)
     .AddOutputAttr(kNumberTypeFloat32),
   &BetaincGpuKernelMod::LaunchKernel<float>},
  {KernelAttr()
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddInputAttr(kNumberTypeFloat64)
     .AddOutputAttr(kNumberTypeFloat64),
   &BetaincGpuKernelMod::LaunchKernel<double>}};

std::vector<KernelAttr> BetaincGpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list;
  (void)std::transform(func_list_.begin(), func_list_.end(), std::back_inserter(support_list),
                       [](const std::pair<KernelAttr, KernelFunc> &pair) { return pair.first; });
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeGpuKernelMod, Betainc, BetaincGpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
