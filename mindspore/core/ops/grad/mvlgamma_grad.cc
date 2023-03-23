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
#include <algorithm>
#include <set>
#include <string>
#include "ops/grad/mvlgamma_grad.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MvlgammaGradInferShape(const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[1]->BuildShape())[kShape];
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr MvlgammaGradInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("y_grad", input_args[0]->BuildType());
  (void)types.emplace("x", input_args[1]->BuildType());
  const std::set<TypePtr> valid_types = {kFloat32, kFloat64};
  return CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
}
}  // namespace

void MvlgammaGrad::Init(const int64_t p) { set_p(p); }

void MvlgammaGrad::set_p(const int64_t p) { (void)this->AddAttr(kP, api::MakeValue(p)); }

int64_t MvlgammaGrad::get_p() const { return GetValue<int64_t>(GetAttr(kP)); }

AbstractBasePtr MvlgammaGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = MvlgammaGradInferType(primitive, input_args);
  auto infer_shape = MvlgammaGradInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
MIND_API_OPERATOR_IMPL(MvlgammaGrad, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(MvlgammaGrad, prim::kPrimMvlgammaGrad, MvlgammaGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
