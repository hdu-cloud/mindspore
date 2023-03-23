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

#include "ops/grad/hswish_grad.h"

#include <string>
#include <algorithm>
#include <map>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore::ops {
namespace {
abstract::ShapePtr HSwishGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                           primitive->name());
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape());
  auto shape = input_shape[kShape];
  return std::make_shared<abstract::Shape>(shape);
}

TypePtr HSwishGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                           prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  (void)types.emplace("y_grad", input_args[0]->BuildType());
  (void)types.emplace("x", input_args[1]->BuildType());
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace
MIND_API_OPERATOR_IMPL(HSwishGrad, BaseOperator);
AbstractBasePtr HSwishGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = HSwishGradInferType(primitive, input_args);
  auto infer_shape = HSwishGradInferShape(primitive, input_args);
  MS_EXCEPTION_IF_NULL(infer_shape);
  return std::make_shared<abstract::AbstractTensor>(infer_type, infer_shape->shape());
}
REGISTER_PRIMITIVE_EVAL_IMPL(HSwishGrad, prim::kPrimHSwishGrad, HSwishGradInfer, nullptr, true);
}  // namespace mindspore::ops
