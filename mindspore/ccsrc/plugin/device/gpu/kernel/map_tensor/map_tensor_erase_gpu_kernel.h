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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MAP_TENSOR_ERASE_GPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MAP_TENSOR_ERASE_GPU_KERNEL_H_

#include <vector>
#include <string>
#include <map>
#include <utility>
#include "mindspore/core/ops/map_tensor_erase.h"
#include "plugin/device/gpu/kernel/map_tensor/map_tensor_gpu_kernel.h"

namespace mindspore {
namespace kernel {
using device::gpu::GPUHashTable;
constexpr size_t kMapTensorEraseInputNum = 2;
constexpr size_t kMapTensorEraseOutputNum = 1;

class MapTensorEraseGpuKernelMod : public MapTensorGpuKernelMod {
 public:
  MapTensorEraseGpuKernelMod() = default;
  ~MapTensorEraseGpuKernelMod() override = default;

  std::vector<KernelAttr> GetOpSupport() override;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    return kernel_launch_func_(this, inputs, workspace, outputs, stream_ptr);
  }

 private:
  template <typename KeyType>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs, void *stream_ptr);

  void InitSizeLists(const ShapeVector &keys_shape);

  size_t input_key_type_size_{0};

  using MapTensorEraseLaunchFunc =
    std::function<bool(MapTensorEraseGpuKernelMod *, const std::vector<AddressPtr> &, const std::vector<AddressPtr> &,
                       const std::vector<AddressPtr> &, void *)>;
  static std::vector<std::pair<KernelAttr, MapTensorEraseLaunchFunc>> map_tensor_erase_func_list_;
  MapTensorEraseLaunchFunc kernel_launch_func_;
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_GPU_MAP_TENSOR_ERASE_GPU_KERNEL_H_
