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

#include "ops/grad/adaptive_avg_pool_2d_grad_v1.h"

#include <set>

#include "ops/op_utils.h"
#include "abstract/param_validator.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr AdaptiveAvgPool2DGradV1InferShape(const PrimitivePtr &primitive,
                                                     const std::vector<AbstractBasePtr> &input_args) {
  auto input_grad_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  const int64_t input_grad_dims = SizeToLong(input_grad_shape.size());
  (void)CheckAndConvertUtils::CheckInRange("dim of input_grad", input_grad_dims, kIncludeBoth, {3, 4},
                                           kNameAdaptiveAvgPool2DGradV1);
  auto orig_input_shape = GetValue<std::vector<int64_t>>(primitive->GetAttr("orig_input_shape"));
  const int64_t orig_input_shape_shape = SizeToLong(orig_input_shape.size());
  (void)CheckAndConvertUtils::CheckInRange("length of orig_input_shape", orig_input_shape_shape, kIncludeBoth, {3, 4},
                                           kNameAdaptiveAvgPool2DGradV1);
  std::vector<int64_t> orig_input_shapeList(input_grad_dims);
  for (int64_t i = 1; i <= input_grad_dims; i++) {
    orig_input_shapeList.end()[-i] = orig_input_shape.end()[-i];
  }
  return std::make_shared<abstract::Shape>(orig_input_shapeList);
}

TypePtr AdaptiveAvgPool2DGradV1InferType(const PrimitivePtr &, const std::vector<AbstractBasePtr> &input_args) {
  auto input_grad_dtype = input_args[0]->BuildType();
  const std::set<TypePtr> input_grad_valid = {kFloat16, kFloat32, kFloat64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("input_grad", input_grad_dtype, input_grad_valid,
                                                   kNameAdaptiveAvgPool2DGradV1);
  return input_grad_dtype;
}
}  // namespace

MIND_API_OPERATOR_IMPL(AdaptiveAvgPool2DGradV1, BaseOperator);
AbstractBasePtr AdaptiveAvgPool2DGradV1Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr int64_t input_num = 1;
  (void)CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto types = AdaptiveAvgPool2DGradV1InferType(primitive, input_args);
  auto shapes = AdaptiveAvgPool2DGradV1InferShape(primitive, input_args);
  return abstract::MakeAbstract(shapes, types);
}

REGISTER_PRIMITIVE_EVAL_IMPL(AdaptiveAvgPool2DGradV1, prim::kPrimAdaptiveAvgPool2DGradV1, AdaptiveAvgPool2DGradV1Infer,
                             nullptr, true);
}  // namespace ops
}  // namespace mindspore
