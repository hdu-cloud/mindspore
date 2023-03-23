/**
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
#include <functional>
#include <map>
#include "unsupported/Eigen/CXX11/Tensor"
#include "plugin/device/cpu/kernel/eigen/bessel_k1e_cpu_kernel.h"
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "mindspore/core/ops/bessel_k1e.h"
#include "abstract/utils.h"

namespace mindspore {
namespace kernel {
namespace {
constexpr size_t kBesselK1eInputsNum = 1;
constexpr size_t kBesselK1eOutputsNum = 1;
}  // namespace

bool BesselK1eCpuKernelMod::Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                 const std::vector<KernelTensorPtr> &outputs) {
  auto kernel_ptr = std::dynamic_pointer_cast<ops::BesselK1e>(base_operator);
  if (!kernel_ptr) {
    MS_LOG(ERROR)
      << "For 'BesselK1eCpuKernelMod', BaseOperatorPtr can not dynamic cast to BesselK1e before initialize!";
    return false;
  }
  kernel_name_ = kernel_ptr->name();
  if (inputs.size() != kBesselK1eInputsNum || outputs.size() != kBesselK1eOutputsNum) {
    MS_LOG(ERROR) << "For '" << kernel_name_ << "': input and output size should be " << kBesselK1eInputsNum << " and "
                  << kBesselK1eOutputsNum << ", but get " << inputs.size() << " and " << outputs.size();
    return false;
  }

  input_shape_ = inputs[0]->GetShapeVector();
  output_shape_ = outputs[0]->GetShapeVector();
  input_dtype_ = inputs[0]->GetDtype();
  input_size_ = std::accumulate(input_shape_.begin(), input_shape_.end(), size_t(1), std::multiplies<size_t>());

  switch (input_dtype_) {
    case kNumberTypeFloat64:
      kernel_func_ = &BesselK1eCpuKernelMod::LaunchKernel<double>;
      break;
    case kNumberTypeFloat32:
      kernel_func_ = &BesselK1eCpuKernelMod::LaunchKernel<float>;
      break;
    case kNumberTypeFloat16:
      kernel_func_ = &BesselK1eCpuKernelMod::LaunchKernel<Eigen::half>;
      break;
    default:
      MS_LOG(ERROR) << "BesselK1e kernel does not support " << TypeIdToString(input_dtype_);
      return false;
  }
  return true;
}

int BesselK1eCpuKernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                                  const std::vector<KernelTensorPtr> &outputs,
                                  const std::map<uint32_t, tensor::TensorPtr> &others) {
  int ret = 0;
  if ((ret = NativeCpuKernelMod::Resize(base_operator, inputs, outputs, others)) != 0) {
    MS_LOG(WARNING) << kernel_name_ << " reinit failed.";
    return ret;
  }
  return 0;
}

template <typename T>
bool BesselK1eCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                         const std::vector<kernel::AddressPtr> &outputs) {
  int block_size = 1000;
  size_t tensor_size = inputs[0]->size / sizeof(T);
  Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> input(static_cast<T *>(inputs[0]->addr), tensor_size);
  Eigen::TensorMap<Eigen::Tensor<T, 1>, Eigen::Aligned> output(static_cast<T *>(outputs[0]->addr), tensor_size);

  auto task = [this, &input, &output](size_t start, size_t end) {
    Eigen::array<Eigen::Index, 1> offsets = {static_cast<int64_t>(start)};
    Eigen::array<Eigen::Index, 1> extends = {static_cast<int64_t>(end - start)};
    output.slice(offsets, extends) = input.slice(offsets, extends).bessel_k1e();
  };
  ParallelLaunch(task, input_size_, block_size);

  return true;
}

std::vector<KernelAttr> BesselK1eCpuKernelMod::GetOpSupport() {
  std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat64).AddOutputAttr(kNumberTypeFloat64),
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeFloat16).AddOutputAttr(kNumberTypeFloat16)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, BesselK1e, BesselK1eCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
