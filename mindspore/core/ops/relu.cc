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
#include "ops/relu.h"
#include <string>
#include <map>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
MIND_API_OPERATOR_IMPL(ReLU, BaseOperator);
class ReLUInfer : public abstract::OpInferBase {
 public:
  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kGreaterEqual, 1,
                                             prim_name);
    (void)CheckAndConvertUtils::CheckArgs<abstract::AbstractTensor>(prim_name, input_args, 0);
    return input_args[0]->BuildShape();
  }

  TypePtr InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(prim);
    auto prim_name = prim->name();
    (void)CheckAndConvertUtils::CheckInteger("input numbers", SizeToLong(input_args.size()), kEqual, 1, prim_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto x_type = input_args[0]->BuildType();
    (void)CheckAndConvertUtils::CheckTensorTypeValid("input_x", x_type, common_valid_types, prim_name);
    return x_type;
  }
};
REGISTER_PRIMITIVE_OP_INFER_IMPL(ReLU, prim::kPrimReLU, ReLUInfer, false);
}  // namespace ops
}  // namespace mindspore
