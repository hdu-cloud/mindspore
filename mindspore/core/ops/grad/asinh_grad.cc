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

#include "ops/grad/asinh_grad.h"
#include "utils/check_convert_utils.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "ops/grad/elewise_grad_infer_shape.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr AsinhGradInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  return mindspore::ElewiseGradInferShape(primitive, input_args);
}

TypePtr AsinhGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kComplex64, kComplex128};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("y", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("dy", input_args[kInputIndex1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim_name);
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(AsinhGrad, BaseOperator);
AbstractBasePtr AsinhGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t InputNum = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, InputNum, prim_name);
  auto types = AsinhGradInferType(primitive, input_args);
  auto shapes = AsinhGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

REGISTER_PRIMITIVE_EVAL_IMPL(AsinhGrad, prim::kPrimAsinhGrad, AsinhGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
