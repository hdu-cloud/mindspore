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
#include "ops/sinc.h"

#include <set>
#include <map>
#include <string>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SincInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr SincInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto x_dtype = input_args[0]->BuildType();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_dtype, common_valid_types_with_complex_and_bool,
                                                   prim->name());
  auto tensor_type = x_dtype->cast<TensorTypePtr>();
  MS_EXCEPTION_IF_NULL(tensor_type);
  TypePtr type = tensor_type->element();
  MS_EXCEPTION_IF_NULL(type);
  auto type_id = type->type_id();
  const std::set<TypeId> valid_types = {kNumberTypeUInt8,  kNumberTypeInt8,   kNumberTypeUInt16,
                                        kNumberTypeInt16,  kNumberTypeUInt32, kNumberTypeInt32,
                                        kNumberTypeUInt64, kNumberTypeInt64,  kNumberTypeBool};
  if (valid_types.count(type_id) > 0) {
    return std::make_shared<TensorType>(kFloat32);
  } else {
    return x_dtype;
  }
}
}  // namespace

MIND_API_OPERATOR_IMPL(Sinc, BaseOperator);
AbstractBasePtr SincInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = SincInferType(primitive, input_args);
  auto infer_shape = SincInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(Sinc, prim::kPrimSinc, SincInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
