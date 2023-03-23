/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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

#include <string>
#include <algorithm>
#include <memory>
#include <vector>
#include "ops/expand_dims.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "utils/log_adapter.h"
#include "mindapi/src/helper.h"
#include "include/common/utils/utils.h"

#include "ops/primitive_c.h"
#include "ops/op_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr ExpandDimsInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  auto shape_ptr = input_args[kInputIndex0]->BuildShape();

  auto shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(shape_ptr);
  auto x_shape = shape_map[kShape];

  // ExpandDims could handle -1, but could not handle -2
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }

  const int64_t rank = SizeToLong(x_shape.size());

  constexpr auto kExpandDimsInputsNum = 2;
  int64_t axis = 0;
  if (input_args.size() == kExpandDimsInputsNum) {
    auto input_value = input_args[kInputIndex1]->BuildValue();
    if (input_args[kInputIndex1]->isa<abstract::AbstractScalar>()) {
      axis = GetValue<int64_t>(input_value);
    } else if (input_args[kInputIndex1]->isa<abstract::AbstractTensor>()) {
      if (input_value->isa<tensor::Tensor>()) {
        auto axis_vec = CheckAndConvertUtils::CheckTensorIntValue("axis", input_value, prim_name);
        if (axis_vec.size() != 1) {
          MS_LOG(EXCEPTION) << " The input number of ExpandDims axis must be int, but got " << axis_vec;
        }
        axis = axis_vec[0];
      } else {
        ShapeVector out_shape;
        (void)out_shape.insert(out_shape.end(), x_shape.size() + 1, -1);
        return std::make_shared<abstract::Shape>(out_shape);
      }
    }
  } else if (input_args.size() == 1) {
    auto value_ptr = primitive->GetAttr(kAxis);
    if (value_ptr->isa<tensor::Tensor>()) {
      auto axis_vec = CheckAndConvertUtils::CheckTensorIntValue("axis", value_ptr, prim_name);
      if (axis_vec.size() != 1) {
        MS_LOG(EXCEPTION) << " The input number of ExpandDims axis must be int, but got " << axis_vec;
      }
      axis = axis_vec[0];
    } else {
      axis = GetValue<int64_t>(primitive->GetAttr(kAxis));
    }
  } else {
    MS_LOG(EXCEPTION) << " The input number of ExpandDims must be 1 or 2, but got " << input_args.size();
  }

  CheckAndConvertUtils::CheckInRange<int64_t>("axis", axis, kIncludeBoth, {-rank - 1, rank}, prim_name);
  if (axis < 0) {
    axis += rank + 1;
  }

  (void)x_shape.insert(x_shape.begin() + axis, 1);
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr ExpandDimsInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, kInputIndex0);
  constexpr auto kExpandDimsInputsNum = 2;
  if (input_args.size() == kExpandDimsInputsNum) {
    auto dim_type = input_args[kInputIndex1]->BuildType();
    if (dim_type->isa<TensorType>()) {
      std::set<TypePtr> check_list = {kInt32, kInt64};
      (void)CheckAndConvertUtils::CheckTensorTypeValid("dim dtype", dim_type, check_list, prim_name);
    } else if (dim_type->type_id() != kNumberTypeInt64 && dim_type->type_id() != kNumberTypeInt32) {
      MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the 'axis' must be a Int, but got "
                              << dim_type->ToString();
    }
  } else if (input_args.size() == 1) {
    auto num_value = prim->GetAttr("axis");
    MS_EXCEPTION_IF_NULL(num_value);
    if (!num_value->isa<Int64Imm>() && !num_value->isa<Int32Imm>()) {
      MS_EXCEPTION(TypeError) << "For primitive[" << prim_name << "], the 'axis' must be a Int, but got "
                              << num_value->ToString();
    }
  } else {
    MS_LOG(EXCEPTION) << " The num of ExpandDims must be 1 or 2, but got " << input_args.size();
  }

  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(ExpandDims, BaseOperator);
AbstractBasePtr ExpandDimsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  constexpr size_t input_num = 2;
  if (input_args.size() > input_num) {
    MS_EXCEPTION(ValueError) << "For primitive[" << primitive->name() << "], the input numbe must be 1 or 2, but got "
                             << input_args.size();
  }
  // Only for checking nullptr
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, SizeToLong(input_args.size()), primitive->name());
  auto infer_type = ExpandDimsInferType(primitive, input_args);
  auto infer_shape = ExpandDimsInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_HOST_DEPENDS(kNameExpandDims, {1});
REGISTER_PRIMITIVE_EVAL_IMPL(ExpandDims, prim::kPrimExpandDims, ExpandDimsInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
