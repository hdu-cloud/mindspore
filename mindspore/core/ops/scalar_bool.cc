/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#include <memory>
#include <set>
#include <string>
#include <vector>

#include "abstract/abstract_value.h"
#include "abstract/dshape.h"
#include "abstract/ops/op_infer.h"
#include "abstract/ops/primitive_infer_map.h"
#include "base/base.h"
#include "ir/anf.h"
#include "ir/dtype/number.h"
#include "ir/dtype/type.h"
#include "ir/primitive.h"
#include "ir/value.h"
#include "mindapi/base/type_id.h"
#include "mindapi/src/helper.h"
#include "mindspore/core/ops/comparison_ops.h"
#include "ops/primitive_c.h"
#include "ops/scalar_bool.h"
#include "utils/check_convert_utils.h"
#include "utils/convert_utils_base.h"
#include "utils/log_adapter.h"

namespace mindspore {
namespace ops {
class ScalarBoolInfer : public abstract::OpInferBase {
 public:
  TypePtr InferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto prim_name = primitive->name();
    auto x_type = input_args[0]->BuildType();
    std::set<TypePtr> check_types = {kInt32, kInt64, kFloat32, kFloat64, kBool};
    (void)CheckAndConvertUtils::CheckSubClass("x_dtype", x_type, check_types, prim_name);
    return kBool;
  }

  BaseShapePtr InferShape(const PrimitivePtr &primitive,
                          const std::vector<AbstractBasePtr> &input_args) const override {
    MS_EXCEPTION_IF_NULL(primitive);
    auto op_name = primitive->name();
    const int64_t input_len = 1;
    (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_len, op_name);
    auto elem = input_args[0];
    if (!elem->isa<abstract::AbstractScalar>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "', the input should be scalar but got : " << elem->ToString();
    }
    return abstract::kNoShape;
  }

  ValuePtr InferValue(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) const {
    MS_EXCEPTION_IF_NULL(primitive);
    const int64_t input_num = 1;
    auto op_name = primitive->name();
    CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
    MS_EXCEPTION_IF_NULL(input_args[0]);
    auto elem = input_args[0];
    if (!elem->isa<abstract::AbstractScalar>()) {
      MS_EXCEPTION(TypeError) << "For '" << op_name << "', the input should be scalar but got : " << elem->ToString();
    }

    auto x_valueptr = elem->BuildValue();
    if (x_valueptr == kValueAny) {
      return nullptr;
    }
    auto x_type = input_args[0]->BuildType();
    bool res;
    switch (x_type->type_id()) {
      case kNumberTypeInt32: {
        auto elem_value = GetValue<int32_t>(x_valueptr);
        res = static_cast<bool>(elem_value);
        break;
      }
      case kNumberTypeInt64: {
        auto elem_value = GetValue<int64_t>(x_valueptr);
        res = static_cast<bool>(elem_value);
        break;
      }
      case kNumberTypeFloat32: {
        auto elem_value = GetValue<float>(x_valueptr);
        res = static_cast<bool>(elem_value);
        break;
      }
      case kNumberTypeFloat64: {
        auto elem_value = GetValue<double>(x_valueptr);
        res = static_cast<bool>(elem_value);
        break;
      }
      case kNumberTypeBool: {
        auto elem_value = GetValue<bool>(x_valueptr);
        res = static_cast<bool>(elem_value);
        break;
      }
      default: {
        MS_EXCEPTION(TypeError)
          << "For '" << op_name
          << "', the supported type is in the list: [int32, int64, float32, float64, bool], but got "
          << x_type->ToString() << ".";
      }
    }
    return MakeValue(res);
  }
};
MIND_API_OPERATOR_IMPL(ScalarBool, BaseOperator);
REGISTER_PRIMITIVE_OP_INFER_IMPL(ScalarBool, prim::kPrimScalarBool, ScalarBoolInfer, true);
}  // namespace ops
}  // namespace mindspore
