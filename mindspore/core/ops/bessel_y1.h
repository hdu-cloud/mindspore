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
#ifndef MINDSPORE_CORE_OPS_BESSEL_Y1_H_
#define MINDSPORE_CORE_OPS_BESSEL_Y1_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "mindapi/base/types.h"
#include "ops/base_operator.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBesselY1 = "BesselY1";

/// \brief BesselY1 is used to compute bessel y1 value for input tensor.
/// \note Param x type must be float16, float32 or float64.
class MIND_API BesselY1 : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(BesselY1);
  /// \brief Constructor.
  BesselY1() : BaseOperator(kNameBesselY1) { InitIOName({"x"}, {"output"}); }
};

MIND_API abstract::AbstractBasePtr BesselY1Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                                 const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_Bessel_Y1_H_
