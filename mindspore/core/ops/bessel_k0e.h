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
#ifndef MINDSPORE_CORE_OPS_BESSEL_K0e_H_
#define MINDSPORE_CORE_OPS_BESSEL_K0e_H_

#include <map>
#include <memory>
#include <string>
#include <vector>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
namespace ops {
constexpr auto kNameBesselK0e = "BesselK0e";
class MIND_API BesselK0e : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(BesselK0e);
  BesselK0e() : BaseOperator(kNameBesselK0e) { InitIOName({"x"}, {"y"}); }
};

abstract::AbstractBasePtr BesselK0eInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                         const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_Bessel_K0e_H_
