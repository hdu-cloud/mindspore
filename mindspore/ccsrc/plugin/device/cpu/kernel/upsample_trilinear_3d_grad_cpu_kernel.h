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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_TRILINEAR_3D_GRAD_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_TRILINEAR_3D_GRAD_CPU_KERNEL_H_

#include <algorithm>
#include <map>
#include <string>
#include <memory>
#include <vector>
#include "kernel/common_utils.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "mindspore/core/ops/grad/upsample_trilinear_3d_grad.h"

namespace mindspore {
namespace kernel {
class UpsampleTrilinear3DGradCpuKernelMod : public NativeCpuKernelMod {
 public:
  UpsampleTrilinear3DGradCpuKernelMod() = default;
  ~UpsampleTrilinear3DGradCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs) override;

 protected:
  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T, typename S>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);

  TypeId in_type_{kTypeUnknown};
  std::vector<int64_t> input_shape_;
  std::vector<int64_t> output_shape_;
  std::vector<float> attr_scales_;
  bool attr_align_corners_{false};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_UPSAMLE_TRILINEAR_3D_GRAD_CPU_KERNEL_H_
