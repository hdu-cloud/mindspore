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
#include "minddata/dataset/kernels/image/normalize_pad_op.h"

#include <random>

#include "minddata/dataset/kernels/image/image_utils.h"
#include "minddata/dataset/util/status.h"

namespace mindspore {
namespace dataset {
NormalizePadOp::NormalizePadOp(const std::vector<float> &mean, const std::vector<float> &std, const std::string dtype,
                               bool is_hwc)
    : mean_(mean), std_(std), dtype_(dtype), is_hwc_(is_hwc) {}

Status NormalizePadOp::Compute(const std::shared_ptr<Tensor> &input, std::shared_ptr<Tensor> *output) {
  IO_CHECK(input, output);
  // Doing the Normalization + pad
  return NormalizePad(input, output, mean_, std_, dtype_, is_hwc_);
}

void NormalizePadOp::Print(std::ostream &out) const {
  out << "NormalizePadOp, mean: ";
  for (const auto &m : mean_) {
    out << m << ", ";
  }
  out << "}" << std::endl << "std: ";
  for (const auto &s : std_) {
    out << s << ", ";
  }
  out << "}" << std::endl << "is_hwc: " << is_hwc_;
  out << "}" << std::endl;
}
}  // namespace dataset
}  // namespace mindspore
