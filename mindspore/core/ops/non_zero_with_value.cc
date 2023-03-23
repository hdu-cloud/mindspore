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

#include "ops/non_zero_with_value.h"

#include <set>
#include <memory>
#include <algorithm>
#include <functional>

#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"

namespace mindspore {
namespace ops {
// NonZeroWithValue
MIND_API_OPERATOR_IMPL(NonZeroWithValue, BaseOperator);
AbstractBasePtr NonZeroWithValueInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                      const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const std::string &op_name = primitive->name();
  constexpr size_t input_num = 1;
  abstract::CheckArgsSize(op_name, input_args, input_num);
  abstract::AbstractTensorPtr x = CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(op_name, input_args, 0);

  MS_EXCEPTION_IF_NULL(x);
  auto x_shape = x->shape();
  MS_EXCEPTION_IF_NULL(x_shape);
  ShapeVector y_shape;

  int64_t rank_base = SizeToLong(x_shape->shape().size());
  int64_t max_size = std::accumulate(x_shape->shape().begin(), x_shape->shape().end(), 1, std::multiplies<int64_t>());

  (void)y_shape.emplace_back(rank_base);
  // Indices of elements that are non-zero
  (void)y_shape.emplace_back(abstract::Shape::kShapeDimAny);
  ShapeVector min_shape = {rank_base, 1};
  ShapeVector max_shape = {rank_base, max_size};

  auto value = std::make_shared<abstract::AbstractTensor>(
    x->element(), std::make_shared<abstract::Shape>(y_shape, min_shape, max_shape));
  auto index = std::make_shared<abstract::AbstractTensor>(
    kInt32, std::make_shared<abstract::Shape>(y_shape, min_shape, max_shape));
  auto count = std::make_shared<abstract::AbstractTensor>(
    kInt32, std::make_shared<abstract::Shape>(y_shape, min_shape, max_shape));
  AbstractBasePtrList result = {value, index, count};
  return std::make_shared<abstract::AbstractTuple>(result);
}
REGISTER_PRIMITIVE_EVAL_IMPL(NonZeroWithValue, prim::kPrimNonZeroWithValue, NonZeroWithValueInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
