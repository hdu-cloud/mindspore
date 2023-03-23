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
#include "ops/clip_by_norm_no_div_sum.h"

#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ClipByNormNoDivSumInferShape(const PrimitivePtr &primitive,
                                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, kIndex4, prim_name);
  auto shape_element = CheckAndConvertUtils::GetTensorInputShape(prim_name, input_args, kIndex0);
  return shape_element;
}

TypePtr ClipByNormNoDivSumInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input_x", input_args[kIndex0]->BuildType());
  (void)types.emplace("input_1", input_args[kIndex1]->BuildType());
  (void)types.emplace("input_2", input_args[kIndex2]->BuildType());
  (void)types.emplace("input_3", input_args[kIndex2]->BuildType());
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace

MIND_API_OPERATOR_IMPL(ClipByNormNoDivSum, BaseOperator);
AbstractBasePtr ClipByNormNoDivSumInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                        const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  const int64_t kInputNum = 4;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kGreaterEqual, kInputNum,
                                           op_name);
  auto infer_type = ClipByNormNoDivSumInferType(primitive, input_args);
  auto infer_shape = ClipByNormNoDivSumInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(ClipByNormNoDivSum, prim::kPrimClipByNormNoDivSum, ClipByNormNoDivSumInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
