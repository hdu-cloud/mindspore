/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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
#ifndef AICPU_KERNELS_NORMALIZED_CROP_AND_RESIZE_H_
#define AICPU_KERNELS_NORMALIZED_CROP_AND_RESIZE_H_

#include <string>

#include "Eigen/Core"
#include "inc/cpu_ops_kernel.h"
#include "inc/cpu_types.h"

namespace aicpu {

class CropAndResizeCpuKernel : public CpuKernel {
 public:
  ~CropAndResizeCpuKernel() = default;
  uint32_t Compute(CpuKernelContext &ctx) override;

 private:
  template <typename T1, typename T2>
  uint32_t DoCompute(const CpuKernelContext &ctx);

  uint32_t GetInputAndCheck(CpuKernelContext &ctx);

  std::vector<int64_t> in_shape_{};
  std::vector<int64_t> out_shape_{};
  DataType dtype_ = DT_FLOAT;
};
}  // namespace aicpu
#endif
