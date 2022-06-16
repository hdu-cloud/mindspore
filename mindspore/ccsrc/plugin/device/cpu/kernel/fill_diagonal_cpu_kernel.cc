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

#include "plugin/device/cpu/kernel/fill_diagonal_cpu_kernel.h"
#include <algorithm>
#include <vector>
#include <cmath>
#include <type_traits>
#include <memory>
#include <functional>
#include "plugin/device/cpu/hal/device/cpu_device_address.h"
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/device/cpu/kernel/arithmetic_cpu_kernel.h"

namespace mindspore {
namespace kernel {
namespace {
const size_t kFillDiagonalInputNum = 1;
const size_t kFillDiagonalOutputNum = 1;
const size_t kInputDimIndex0 = 0;
const size_t kInputDimIndex1 = 1;
const size_t kInputMinDim = 2;
constexpr int64_t kParallelDataNums = 512 * 1024;
}  // namespace

void FillDiagonalCpuKernelMod::InitKernel(const CNodePtr &kernel_node) {
  MS_EXCEPTION_IF_NULL(kernel_node);
  kernel_name_ = common::AnfAlgo::GetCNodeName(kernel_node);
  size_t input_num = common::AnfAlgo::GetInputTensorNum(kernel_node);
  CHECK_KERNEL_INPUTS_NUM(input_num, kFillDiagonalInputNum, kernel_name_);
  size_t output_num = common::AnfAlgo::GetOutputTensorNum(kernel_node);
  CHECK_KERNEL_OUTPUTS_NUM(output_num, kFillDiagonalOutputNum, kernel_name_);

  input_type_ = AnfAlgo::GetInputDeviceDataType(kernel_node, 0);
  input_shape_ = AnfAlgo::GetInputDeviceShape(kernel_node, 0);
  if (input_shape_.size() < kInputMinDim) {
    MS_EXCEPTION(ValueError) << "For '" << kernel_name_ << "', the input dims must larger than 1.";
  }
  if (input_shape_.size() > kInputMinDim) {
    for (size_t i = 1; i < input_shape_.size(); i++) {
      if (input_shape_[i] != input_shape_[i - 1]) {
        MS_EXCEPTION(ValueError) << "For '" << kernel_name_
                                 << "', each dim of input must be of equal length while dims > 2.";
      }
    }
  }

  fill_value_ = common::AnfAlgo::GetNodeAttr<float>(kernel_node, "fill_value");
  wrap_ = common::AnfAlgo::GetNodeAttr<bool>(kernel_node, "wrap");
}

bool FillDiagonalCpuKernelMod::Launch(const std::vector<kernel::AddressPtr> &inputs,
                                      const std::vector<kernel::AddressPtr> &workspace,
                                      const std::vector<kernel::AddressPtr> &outputs) {
  if (input_type_ == kNumberTypeFloat32) {
    return LaunchKernel<float>(inputs, outputs);
  } else if (input_type_ == kNumberTypeInt32) {
    return LaunchKernel<int32_t>(inputs, outputs);
  } else if (input_type_ == kNumberTypeInt64) {
    return LaunchKernel<int64_t>(inputs, outputs);
  } else {
    MS_LOG(EXCEPTION) << "For '" << kernel_name_
                      << "the datatype of the input not support, support datatype: float, int32, int64.";
  }

  return true;
}

template <typename T>
bool FillDiagonalCpuKernelMod::LaunchKernel(const std::vector<kernel::AddressPtr> &inputs,
                                            const std::vector<kernel::AddressPtr> &outputs) {
  T *input_ptr = reinterpret_cast<T *>(inputs[0]->addr);
  MS_EXCEPTION_IF_NULL(input_ptr);
  T *output_ptr = reinterpret_cast<T *>(outputs[0]->addr);
  MS_EXCEPTION_IF_NULL(output_ptr);

  int64_t data_nums = static_cast<int64_t>(outputs[0]->size / sizeof(T));
  if (data_nums <= kParallelDataNums) {
    std::memcpy(output_ptr, input_ptr, data_nums * sizeof(T));
  } else {
    auto task = [this, input_ptr, output_ptr](size_t start, size_t end) {
      std::memcpy(output_ptr + start, input_ptr + start, (end - start) * sizeof(T));
    };
    CPUKernelUtils::ParallelFor(task, data_nums);
  }

  int64_t height = input_shape_[kInputDimIndex0];
  int64_t width = input_shape_[kInputDimIndex1];
  int64_t size = std::min(height, width);

  int64_t stride = 0;
  for (int64_t i = (input_shape_.size() - 1); i >= 0; i--) {
    stride += pow(width, i);
  }
  for (int64_t i = 0; i < size; ++i) {
    output_ptr[stride * i] = static_cast<T>(fill_value_);
  }

  if (wrap_ && input_shape_.size() == kInputMinDim && height > width + 1) {
    int64_t location = size * (size + 1);
    while (location < data_nums) {
      output_ptr[location] = static_cast<T>(fill_value_);
      location += stride;
    }
  }

  return true;
}

std::vector<KernelAttr> FillDiagonalCpuKernelMod::GetOpSupport() {
  static std::vector<KernelAttr> support_list = {
    KernelAttr().AddInputAttr(kNumberTypeFloat32).AddOutputAttr(kNumberTypeFloat32),
    KernelAttr().AddInputAttr(kNumberTypeInt32).AddOutputAttr(kNumberTypeInt32),
    KernelAttr().AddInputAttr(kNumberTypeInt64).AddOutputAttr(kNumberTypeInt64)};
  return support_list;
}

MS_KERNEL_FACTORY_REG(NativeCpuKernelMod, FillDiagonal, FillDiagonalCpuKernelMod);
}  // namespace kernel
}  // namespace mindspore
