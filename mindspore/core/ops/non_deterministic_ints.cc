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
#include "ops/non_deterministic_ints.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr NonDeterministicIntsInferShape(const PrimitivePtr &primitive,
                                                  const std::vector<AbstractBasePtr> &input_args) {
  if (!input_args[0]->isa<abstract::AbstractTensor>()) {
    MS_EXCEPTION(TypeError) << "For '" << primitive->name()
                            << "', input must be a tensor, but got: " << input_args[0]->BuildShape()->ToString() << ".";
  }
  MS_EXCEPTION_IF_NULL(primitive);
  const uint32_t kInpuDims = 1;
  const uint32_t kInpuSizes = 2;
  auto max_length_ptr = primitive->GetAttr("max_length");
  MS_EXCEPTION_IF_NULL(max_length_ptr);
  int64_t max_length = GetValue<int64_t>(max_length_ptr);
  auto input_shape = input_args[0]->cast<abstract::AbstractTensorPtr>();
  MS_EXCEPTION_IF_NULL(input_shape);
  auto input_shape_value_ptr = input_shape->BuildValue();
  MS_EXCEPTION_IF_NULL(input_shape_value_ptr);
  auto input_shape_tensor = input_shape_value_ptr->cast<tensor::TensorPtr>();
  auto input_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(input_type);
  auto input_type_id = input_type->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(input_type_id);
  auto input_type_element = input_type_id->element();
  MS_EXCEPTION_IF_NULL(input_type_element);
  auto shape_ptr = std::make_shared<abstract::Shape>(
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape]);
  auto shape_v = shape_ptr->shape();
  if (IsDynamicRank(shape_v)) {
    return std::make_shared<abstract::Shape>(ShapeVector({abstract::Shape::kShapeRankAny}));
  }

  if (shape_v.size() != kInpuDims) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                             << "', input tensor must be a 1-D tensor, but got shape size: " << shape_v.size() << ".";
  }
  if (shape_v[0] < kInpuSizes) {
    MS_EXCEPTION(ValueError) << "For '" << primitive->name() << "', input tensor must have a least 2 elements, but got "
                             << shape_v[0] << ".";
  }
  if (!input_args[0]->BuildValue()->isa<AnyValue>() && !input_args[0]->BuildValue()->isa<None>()) {
    std::vector<int64_t> out_shape;
    int64_t shape_m = 1;
    if (input_type_element->type_id() == kNumberTypeInt32) {
      auto input_shape_ptr = reinterpret_cast<int32_t *>(input_shape_tensor->data_c());
      for (auto i = 0; i < shape_v[0]; ++i) {
        if (input_shape_ptr[i] > 0) {
          out_shape.push_back(input_shape_ptr[i]);
          shape_m *= static_cast<int64_t>(input_shape_ptr[i]);
        } else {
          MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                                   << "', each dimension of input must be greater than 0, but got input_shape[" << i
                                   << "]: " << input_shape_ptr[i] << ".";
        }
      }
    } else if (input_type_element->type_id() == kNumberTypeInt64) {
      auto input_shape_ptr = reinterpret_cast<int64_t *>(input_shape_tensor->data_c());
      for (auto i = 0; i < shape_v[0]; ++i) {
        if (input_shape_ptr[i] > 0) {
          out_shape.push_back(input_shape_ptr[i]);
          shape_m *= static_cast<int64_t>(input_shape_ptr[i]);
        } else {
          MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                                   << "', each dimension of input must be greater than 0, but got input_shape[" << i
                                   << "]: " << input_shape_ptr[i] << ".";
        }
      }
    }
    if (shape_m > max_length) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', the number of elements of output must be less than max length: " << max_length
                               << ", but got " << shape_m
                               << ". The shape of output must be reduced or max_length must be increased";
    }
    return std::make_shared<abstract::Shape>(out_shape);
  } else {
    std::vector<int64_t> output_shape;
    for (int i = 0; i < shape_v[0]; i++) {
      output_shape.push_back(abstract::Shape::kShapeDimAny);
    }
    return std::make_shared<abstract::Shape>(output_shape);
  }
}

TypePtr NonDeterministicIntsInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = prim->name();
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  const std::set<TypePtr> valid_input_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTypeValid("shape", input_args[0]->BuildType(), valid_input_types, prim_name);
  auto dtype_value = prim->GetAttr("dtype");
  if (!dtype_value->isa<Type>()) {
    MS_EXCEPTION(TypeError) << "The dtype of NonDeterministicInts is invalid!";
  }
  auto output_type = dtype_value->cast<TypePtr>();
  const std::set<TypePtr> valid_output_types = {kInt32, kInt64};
  return CheckAndConvertUtils::CheckSubClass("dtype", output_type, valid_output_types, prim_name);
}
}  // namespace

MIND_API_OPERATOR_IMPL(NonDeterministicInts, BaseOperator);
AbstractBasePtr NonDeterministicIntsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t kInputNum = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kGreaterEqual, kInputNum, primitive->name());
  auto infer_type = NonDeterministicIntsInferType(primitive, input_args);
  auto infer_shape = NonDeterministicIntsInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(NonDeterministicInts, prim::kPrimNonDeterministicInts, NonDeterministicIntsInfer, nullptr,
                             true);
}  // namespace ops
}  // namespace mindspore
