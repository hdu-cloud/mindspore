/**
 * Copyright 2021-2022 Huawei Technologies Co., Ltd
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

#include "ops/ctc_loss_v2.h"
#include <vector>
#include <string>
#include <memory>
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "mindapi/src/helper.h"
#include "include/common/utils/utils.h"
namespace mindspore {
namespace ops {
int64_t CTCLossV2::get_blank() const { return GetValue<int64_t>(GetAttr(kAttrBlank)); }
std::string CTCLossV2::get_reduction() const { return GetValue<std::string>(GetAttr(kAttrReduction)); }
bool CTCLossV2::get_zero_infinity() const { return GetValue<bool>(GetAttr(kAttrZeroInfinity)); }
namespace {
abstract::TupleShapePtr CTCLossV2InferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  constexpr size_t kLenLogProbs = 3;
  constexpr size_t kLenTarget = 2;
  constexpr int64_t kMulti = 2;
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  auto log_probs_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex0]->BuildShape())[kShape];
  auto targets_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex1]->BuildShape())[kShape];
  auto input_lengths_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex2]->BuildShape())[kShape];
  auto target_lengths_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kIndex3]->BuildShape())[kShape];

  if (IsDynamicRank(log_probs_shape) || IsDynamicRank(targets_shape) || IsDynamicRank(input_lengths_shape) ||
      IsDynamicRank(target_lengths_shape)) {
    std::vector<int64_t> dyn_shape = {abstract::Shape::kShapeRankAny};
    abstract::ShapePtr neg_log_shape = std::make_shared<abstract::Shape>(dyn_shape);
    abstract::ShapePtr log_alpha_shape = std::make_shared<abstract::Shape>(dyn_shape);
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{neg_log_shape, log_alpha_shape});
  }

  (void)CheckAndConvertUtils::CheckValue("dim of log_probs", log_probs_shape.size(), kEqual, kLenLogProbs, prim_name);
  (void)CheckAndConvertUtils::CheckValue("dim of targets", targets_shape.size(), kEqual, kLenTarget, prim_name);

  int64_t T = log_probs_shape[kIndex0];
  int64_t N = log_probs_shape[kIndex1];
  int64_t C = log_probs_shape[kIndex2];
  int64_t S = targets_shape[kIndex1];

  int64_t padded_S = (S == abstract::Shape::kShapeDimAny) ? abstract::Shape::kShapeDimAny : (kMulti * S + 1);
  abstract::ShapePtr neg_log_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{N});
  abstract::ShapePtr log_alpha_shape = std::make_shared<abstract::Shape>(std::vector<int64_t>{N, T, padded_S});

  if (IsDynamicShape(log_probs_shape) || IsDynamicShape(targets_shape) || IsDynamicShape(input_lengths_shape) ||
      IsDynamicShape(target_lengths_shape)) {
    return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{neg_log_shape, log_alpha_shape});
  }

  (void)CheckAndConvertUtils::CheckValue<size_t>("dim of input_lengths", input_lengths_shape.size(), kEqual, kDim1,
                                                 prim_name);
  (void)CheckAndConvertUtils::CheckValue<size_t>("dim of target_lengths", target_lengths_shape.size(), kEqual, kDim1,
                                                 prim_name);
  (void)CheckAndConvertUtils::CheckValue<int64_t>("input_lengths[0]", input_lengths_shape[0], kEqual, N, prim_name);
  (void)CheckAndConvertUtils::CheckValue<int64_t>("target_lengths[0]", target_lengths_shape[0], kEqual, N, prim_name);

  // check blank
  auto blank = GetValue<int64_t>(primitive->GetAttr(kAttrBlank));
  CheckAndConvertUtils::CheckInRange(kAttrBlank, blank, kIncludeLeft, {0, C}, prim_name);

  return std::make_shared<abstract::TupleShape>(std::vector<abstract::BaseShapePtr>{neg_log_shape, log_alpha_shape});
}

TuplePtr CTCLossV2InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto name = primitive->name();
  auto type = CheckAndConvertUtils::CheckTypeValid("log_probs", input_args[kInputIndex0]->BuildType(),
                                                   {kFloat32, kFloat64}, name);
  (void)CheckAndConvertUtils::CheckTypeValid("targets", input_args[kInputIndex1]->BuildType(), {kInt32, kInt64}, name);
  (void)CheckAndConvertUtils::CheckTypeValid("input_lengths", input_args[kInputIndex2]->BuildType(), {kInt32, kInt64},
                                             name);
  (void)CheckAndConvertUtils::CheckTypeValid("target_lengths", input_args[kInputIndex3]->BuildType(), {kInt32, kInt64},
                                             name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, type});
}
}  // namespace

MIND_API_OPERATOR_IMPL(CTCLossV2, BaseOperator);
AbstractBasePtr CTCLossV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                               const std::vector<AbstractBasePtr> &input_args) {
  constexpr int64_t kInputNum = 4;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, primitive->name());
  auto type = CTCLossV2InferType(primitive, input_args);
  auto shape = CTCLossV2InferShape(primitive, input_args);
  return abstract::MakeAbstract(shape, type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(CTCLossV2, prim::kPrimCTCLossV2, CTCLossV2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
