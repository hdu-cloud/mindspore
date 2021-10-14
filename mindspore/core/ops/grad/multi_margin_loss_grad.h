/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#ifndef MINDSPORE_CORE_OPS_MULTI_MARGIN_LOSS_GRAD_H_
#define MINDSPORE_CORE_OPS_MULTI_MARGIN_LOSS_GRAD_H_

#include <algorithm>
#include <map>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "ops/primitive_c.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
constexpr auto kNameMultiMarginLossGrad = "MultiMarginLossGrad";
class MS_CORE_API MultiMarginLossGrad : public PrimitiveC {
 public:
  MultiMarginLossGrad() : PrimitiveC(kNameMultiMarginLossGrad) {
    InitIOName({"y_grad", "x", "target", "weight"}, {"x_grad"});
  }
  ~MultiMarginLossGrad() = default;
  MS_DECLARE_PARENT(MultiMarginLossGrad, PrimitiveC);
};

AbstractBasePtr MultiMarginLossGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args);
using PrimMultiMarginLossGradPtr = std::shared_ptr<MultiMarginLossGrad>;
}  // namespace ops
}  // namespace mindspore

#endif  // MINDSPORE_CORE_OPS_MULTI_MARGIN_LOSS_GRAD_H_
