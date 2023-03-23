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

#include <set>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include "ops/gpu_convert_to_dynamic_shape.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr GpuConvertToDynamicShapeInferShape(const PrimitivePtr &primitive,
                                                      const std::vector<AbstractBasePtr> &input_args) {
  auto input_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamicRank(input_shape)) {
    return std::make_shared<abstract::Shape>(ShapeVector{abstract::Shape::kShapeRankAny});
  }
  ShapeVector output_shape_dyn;
  for (size_t i = 0; i < input_shape.size(); ++i) {
    output_shape_dyn.push_back(abstract::Shape::kShapeDimAny);
  }
  return std::make_shared<abstract::Shape>(output_shape_dyn);
}

TypePtr GpuConvertToDynamicShapeInferType(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  auto x_type = input_args[0]->BuildType();
  return x_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(GpuConvertToDynamicShape, BaseOperator);
AbstractBasePtr GpuConvertToDynamicShapeInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto infer_type = GpuConvertToDynamicShapeInferType(primitive, input_args);
  auto infer_shape = GpuConvertToDynamicShapeInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(GpuConvertToDynamicShape, prim::kPrimGpuConvertToDynamicShape,
                             GpuConvertToDynamicShapeInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
