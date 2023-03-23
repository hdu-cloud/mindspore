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

#include "ops/nllloss.h"
#include <memory>
#include <string>
#include <vector>
#include <map>
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"
#include "include/common/utils/utils.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(NLLLoss, BaseOperator);
class NLLLossInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    const auto prim_name = primitive->name();
    auto logits_shape_ptr = input_args[kInputIndex0]->BuildShape();
    auto logits_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(logits_shape_ptr)[kShape];
    auto target_shape_ptr = input_args[kInputIndex1]->BuildShape();
    auto target_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(target_shape_ptr)[kShape];
    auto weight_shape_ptr = input_args[kInputIndex2]->BuildShape();
    auto weight_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(weight_shape_ptr)[kShape];

    (void)CheckAndConvertUtils::CheckInteger("rank of target", SizeToLong(target_shape.size()), kEqual, 1, prim_name);
    (void)CheckAndConvertUtils::CheckInteger("rank of weight", SizeToLong(weight_shape.size()), kEqual, 1, prim_name);
    CheckAndConvertUtils::CheckInRange("rank of logits", SizeToLong(logits_shape.size()), kIncludeBoth, {1, 2},
                                       prim_name);

    if (!logits_shape_ptr->IsDynamic()) {
      if (!target_shape_ptr->IsDynamic() && logits_shape[kInputIndex0] != target_shape[kInputIndex0]) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', the 'logits_dim0' and the shape of 'target' should be equal, but got "
                                 << logits_shape[kInputIndex0] << " and " << target_shape[kInputIndex0] << ".";
      }

      size_t weight_dim = logits_shape.size() - 1;
      if (!weight_shape_ptr->IsDynamic() && logits_shape[weight_dim] != weight_shape[kInputIndex0]) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name
                                 << "', the last dim of 'logits' and the shape of 'weight' should be equal, but got "
                                 << logits_shape[weight_dim] << " and " << weight_shape[kInputIndex0] << ".";
      }
    }
    ShapeVector loss_shape;
    ShapeVector total_weight_shape;
    auto reduction_ptr = primitive->GetAttr(kReduction);
    bool reduction_is_none;
    if (reduction_ptr->isa<StringImm>()) {
      auto reduction = GetValue<std::string>(reduction_ptr);
      reduction_is_none = reduction == kNone;
    } else {
      auto reduction = Reduction(GetValue<int64_t>(reduction_ptr));
      reduction_is_none = reduction == Reduction::NONE;
    }
    if (reduction_is_none) {
      loss_shape.push_back(logits_shape[kInputIndex0]);
    }
    abstract::ShapePtr loss_shape_ptr = std::make_shared<abstract::Shape>(loss_shape);
    abstract::ShapePtr total_weight_shape_ptr = std::make_shared<abstract::Shape>(total_weight_shape);

    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{loss_shape_ptr, total_weight_shape_ptr});
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    const std::set valid_types = {kFloat16, kFloat32};
    auto logits_data_type = input_args[kIndex0]->BuildType();
    auto target_type = input_args[kIndex1]->BuildType();
    auto weight_data_type = input_args[kIndex2]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("target", target_type, {kInt32}, prim->name());
    (void)CheckAndConvertUtils::CheckTensorTypeValid("logits", logits_data_type, valid_types, prim->name());
    (void)CheckAndConvertUtils::CheckTensorTypeValid("weight", weight_data_type, valid_types, prim->name());
    return std::make_shared<Tuple>(std::vector<TypePtr>{logits_data_type, weight_data_type});
  }
};

void NLLLoss::Init(const Reduction &reduction) { this->set_reduction(reduction); }

void NLLLoss::set_reduction(const Reduction &reduction) {
  std::string reduce;
  if (reduction == Reduction::REDUCTION_SUM) {
    reduce = "sum";
  } else if (reduction == Reduction::MEAN) {
    reduce = "mean";
  } else {
    reduce = "none";
  }
  (void)this->AddAttr(kReduction, api::MakeValue(reduce));
}

Reduction NLLLoss::get_reduction() const {
  auto value_ptr = MakeValue(GetValue<std::string>(GetAttr(kReduction)));
  int64_t reduction = 0;
  CheckAndConvertUtils::GetReductionEnumValue(value_ptr, &reduction);
  return Reduction(reduction);
}
REGISTER_PRIMITIVE_OP_INFER_IMPL(NLLLoss, prim::kPrimNLLLoss, NLLLossInfer, false);
}  // namespace ops
}  // namespace mindspore
