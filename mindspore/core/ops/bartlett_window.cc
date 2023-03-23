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

#include "ops/bartlett_window.h"
#include <map>
#include <set>
#include <string>
#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr BartlettWindowInferShape(const PrimitivePtr &primitive,
                                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_length = GetValue<int64_t>(max_length_ptr);
  if (input_args[0]->isa<abstract::AbstractTensor>() && !input_args[0]->BuildValue()->isa<AnyValue>() &&
      !input_args[0]->BuildValue()->isa<None>()) {
    auto window_length = input_args[0]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(window_length);
    auto window_length_value_ptr = window_length->BuildValue();
    MS_EXCEPTION_IF_NULL(window_length_value_ptr);
    auto window_length_tensor = window_length_value_ptr->cast<tensor::TensorPtr>();
    MS_EXCEPTION_IF_NULL(window_length_tensor);
    auto input_type = input_args[0]->BuildType();
    MS_EXCEPTION_IF_NULL(input_type);
    auto input_type_id = input_type->cast<TensorTypePtr>();
    MS_EXCEPTION_IF_NULL(input_type_id);
    auto input_type_element = input_type_id->element();
    MS_EXCEPTION_IF_NULL(input_type_element);
    auto window_length_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape());
    auto window_length_shape = window_length_shape_map[kShape];
    if (IsDynamicRank(window_length_shape)) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
    }

    if (window_length_shape.size() != 0) {
      if (window_length_shape[0] == 0) {
        MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the input window_length can not be empty.";
      } else {
        MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                                 << "', the dim of window_length must be 0, but got: " << window_length_shape.size()
                                 << ".";
      }
    }

    std::vector<int64_t> out_shape;
    int64_t window_length_value = 0;
    if (input_type_element->type_id() == kNumberTypeInt32) {
      auto window_length_ptr = reinterpret_cast<int *>(window_length_tensor->data_c());
      window_length_value = static_cast<int64_t>(*window_length_ptr);
    } else if (input_type_element->type_id() == kNumberTypeInt64) {
      auto window_length_ptr = reinterpret_cast<int64_t *>(window_length_tensor->data_c());
      window_length_value = static_cast<int64_t>(*window_length_ptr);
    }

    if (window_length_value >= 0 && window_length_value <= max_length) {
      out_shape.push_back(window_length_value);
      return std::make_shared<abstract::Shape>(out_shape);
    } else {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the value range of window_length must be [0, "
                               << max_length << "], but got: " << window_length_value << ".";
    }
  } else {
    auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
    if (IsDynamicRank(x_shape)) {
      return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
    }
    auto x_size = x_shape.size();
    if (x_size != 0) {
      if (x_shape[0] == 0) {
        MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', the input window_length can not be empty.";
      } else {
        MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                                 << "', the dim of window_length must be 0, but got: " << x_size << ".";
      }
    }
    std::vector<int64_t> out_shape = {abstract::Shape::kShapeDimAny};
    std::vector<int64_t> infer_shape_min = {0};
    std::vector<int64_t> infer_shape_max = {max_length};
    return std::make_shared<abstract::Shape>(out_shape, infer_shape_min, infer_shape_max);
  }
}

TypePtr BartlettWindowInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto input_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  const std::set<TypePtr> valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("window_length", input_type, valid_types, prim->name());
  auto dtype_attr = prim->GetAttr("dtype");
  MS_EXCEPTION_IF_NULL(dtype_attr);
  auto infer_type = dtype_attr->cast<TypePtr>();
  MS_EXCEPTION_IF_NULL(infer_type);
  return infer_type;
}
}  // namespace

void BartlettWindow::Init(const bool periodic) { set_periodic(periodic); }

void BartlettWindow::set_periodic(const bool periodic) { (void)this->AddAttr(kPeriodic, api::MakeValue(periodic)); }

bool BartlettWindow::get_periodic() const { return GetValue<bool>(GetAttr(kPeriodic)); }

AbstractBasePtr BartlettWindowInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                    const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = BartlettWindowInferType(primitive, input_args);
  auto infer_shape = BartlettWindowInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(BartlettWindow, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(BartlettWindow, prim::kPrimBartlettWindow, BartlettWindowInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
