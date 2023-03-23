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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UNARYOP_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UNARYOP_GPU_KERNEL_H_

#include <functional>
#include <vector>
#include <string>
#include <map>
#include <utility>
#include <algorithm>
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/factory/ms_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/unary_op_impl.cuh"

namespace mindspore {
namespace kernel {
class UnaryOpGpuKernelMod : public NativeGpuKernelMod {
 public:
  explicit UnaryOpGpuKernelMod(const std::string &kernel_name) { kernel_name_ = kernel_name; }
  ~UnaryOpGpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(
    const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
    const std::vector<KernelTensorPtr> &outputs,
    const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost = std::map<uint32_t, tensor::TensorPtr>()) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *cuda_stream) override {
    if (is_null_input_) {
      return true;
    }
    cuda_stream_ = cuda_stream;
    return kernel_func_(this, inputs, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using UnaryOpFunc = std::function<bool(UnaryOpGpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                         const std::vector<kernel::AddressPtr> &)>;
  static std::map<std::string, std::vector<std::pair<KernelAttr, UnaryOpGpuKernelMod::UnaryOpFunc>>> kernel_attr_map_;
  UnaryOpFunc kernel_func_;
  bool is_null_input_{true};
  void *cuda_stream_{nullptr};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MATH_UNARYOP_GPU_KERNEL_H_
