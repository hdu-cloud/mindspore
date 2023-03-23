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

#ifndef MINDSPORE_CORE_OPS_CHOLESKYGRAD_H
#define MINDSPORE_CORE_OPS_CHOLESKYGRAD_H

#include <vector>
#include <memory>

#include "ops/base_operator.h"
#include "mindapi/base/types.h"

namespace mindspore {
/// \brief Computes the reverse mode backpropgated gradient of the Cholesky algorithm.
/// Refer to Python API @ref mindspore.ops.CholeskyGrad for more details.
namespace ops {
constexpr auto kNameCholeskyGrad = "CholeskyGrad";

class MIND_API CholeskyGrad : public BaseOperator {
 public:
  MIND_API_BASE_MEMBER(CholeskyGrad);
  /// \brief Constructor.
  CholeskyGrad() : BaseOperator(kNameCholeskyGrad) { InitIOName({"x", "grad"}, {"y"}); }
};

abstract::AbstractBasePtr CholeskyGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                            const std::vector<abstract::AbstractBasePtr> &input_args);
}  // namespace ops
}  // namespace mindspore
#endif  // MINDSPORE_CORE_OPS_CHOLESKYGRAD_H
