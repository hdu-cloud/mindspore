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

#ifndef MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_ROI_ALIGN_GPU_KERNEL_H
#define MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_ROI_ALIGN_GPU_KERNEL_H

#include <vector>
#include <map>
#include <utility>
#include "mindspore/core/ops/roi_align.h"
#include "plugin/device/gpu/kernel/gpu_kernel.h"
#include "plugin/device/gpu/kernel/gpu_kernel_factory.h"
#include "plugin/device/gpu/kernel/cuda_impl/cuda_ops/roi_align_impl.cuh"

namespace mindspore {
namespace kernel {
class ROIAlignGpuKernelMod : public NativeGpuKernelMod, public MatchKernelHelper<ROIAlignGpuKernelMod> {
 public:
  ROIAlignGpuKernelMod() { ResetResource(); }
  ~ROIAlignGpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs,
             const std::map<uint32_t, tensor::TensorPtr> &inputsOnHost) override;

  const std::vector<std::pair<KernelAttr, KernelRunFunc>> &GetFuncList() const override;

  std::vector<KernelAttr> GetOpSupport() override { return OpSupport(); }

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
              const std::vector<AddressPtr> &outputs, void *stream_ptr) override {
    if (is_null_input_) {
      return true;
    }
    stream_ptr_ = stream_ptr;
    return kernel_func_(this, inputs, workspace, outputs);
  }

 protected:
  using FuncList = std::vector<std::pair<KernelAttr, ROIAlignGpuKernelMod::KernelRunFunc>>;

  template <typename T>
  bool LaunchKernel(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &workspace,
                    const std::vector<AddressPtr> &outputs);

  void ResetResource() noexcept {
    is_null_input_ = false;
    stream_ptr_ = nullptr;
    input_size_list_.clear();
    output_size_list_.clear();
    workspace_size_list_.clear();
  }

  void InitSizeLists() {
    input_size_list_.push_back(x_size_);
    input_size_list_.push_back(rois_size_);
    output_size_list_.push_back(output_size_);
  }

 private:
  void *stream_ptr_{nullptr};
  bool is_null_input_{false};

  int64_t pooled_height_{0};
  int64_t pooled_width_{0};
  float spatial_scale_{0.0};
  int64_t sample_num_{0};
  int64_t roi_end_mode_{0};

  int64_t roi_rows_{0};
  int64_t roi_cols_{0};
  int64_t batch_{0};
  int64_t channel_{0};
  int64_t height_{0};
  int64_t width_{0};

  size_t x_size_{0};
  size_t rois_size_{0};
  size_t output_size_{0};
};
}  // namespace kernel
}  // namespace mindspore

#endif  // MINDSPORE_CCSRC_PLUGIN_DEVICE_GPU_KERNEL_NN_ROI_ALIGN_GPU_KERNEL_H
