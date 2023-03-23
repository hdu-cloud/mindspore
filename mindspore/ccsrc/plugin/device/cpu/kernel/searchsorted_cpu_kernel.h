/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEARCHSORTED_CPU_KERNEL_H_
#define MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEARCHSORTED_CPU_KERNEL_H_

#include <map>
#include <vector>
#include <utility>
#include "plugin/device/cpu/kernel/cpu_kernel.h"
#include "plugin/factory/ms_factory.h"

namespace mindspore {
namespace kernel {
class SearchSortedCpuKernelMod : public NativeCpuKernelMod {
 public:
  SearchSortedCpuKernelMod() = default;
  ~SearchSortedCpuKernelMod() override = default;

  bool Init(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
            const std::vector<KernelTensorPtr> &outputs) override;

  int Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
             const std::vector<KernelTensorPtr> &outputs, const std::map<uint32_t, tensor::TensorPtr> &) override;

  bool Launch(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &,
              const std::vector<AddressPtr> &outputs) override {
    return kernel_func_(this, inputs, outputs);
  }

  std::vector<KernelAttr> GetOpSupport() override;

 private:
  template <typename S>
  const S *CustomizedLowerBound(const S *seq_start, const S *seq_end, const S key) const;
  template <typename S, typename T>
  void CheckParam(const std::vector<AddressPtr> &inputs, const std::vector<AddressPtr> &outputs);
  template <typename S, typename T>
  bool LaunchKernel(const std::vector<kernel::AddressPtr> &inputs, const std::vector<kernel::AddressPtr> &outputs);
  using SearchSortedFunc = std::function<bool(SearchSortedCpuKernelMod *, const std::vector<kernel::AddressPtr> &,
                                              const std::vector<kernel::AddressPtr> &)>;
  static std::vector<std::pair<KernelAttr, SearchSortedFunc>> func_list_;
  SearchSortedFunc kernel_func_;

  bool right_{false};
  size_t search_len_{0};
  std::vector<int64_t> sequence_shape_;
  std::vector<int64_t> values_shape_;
  std::vector<int64_t> output_shape_;
};
}  // namespace kernel
}  // namespace mindspore
#endif  // MINDSPORE_CCSRC_BACKEND_KERNEL_COMPILER_CPU_SEARCHSORTED_CPU_KERNEL_H_
