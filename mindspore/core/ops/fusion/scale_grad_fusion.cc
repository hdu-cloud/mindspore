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

#include "ops/fusion/scale_grad_fusion.h"
#include <string>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ScaleGrad, BaseOperator);
AbstractBasePtr ScaleGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterThan, kInputNum, primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
    (void)item->Broaden();
  }

  auto infer_shape = input_args[0]->BuildShape();
  auto infer_type = input_args[0]->BuildType();
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(ScaleGrad, prim::kPrimScaleGrad, ScaleGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
