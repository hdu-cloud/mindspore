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
#include "ops/multinomial.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include <iostream>
#include <map>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/ms_context.h"
#include "abstract/ops/primitive_infer_map.h"
#include "abstract/param_validator.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr MultinomialInferShape(const PrimitivePtr &primitive,
                                         const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->BuildShape())[kShape];
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  const int64_t x_rank_max = 2;
  const int64_t x_rank_min = 1;
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  auto is_ascend = (context_ptr->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  if (is_ascend) {
    if (x_shape.size() != x_rank_max) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', input[x] dimension must be 2 in Ascend, but got rank " << x_shape.size() << ".";
    }
  } else {
    if (x_shape.size() > x_rank_max || x_shape.size() < x_rank_min) {
      MS_EXCEPTION(ValueError) << "For '" << primitive->name()
                               << "', input[x] dimension must be 1 or 2 in CPU and GPU, but got rank " << x_shape.size()
                               << ".";
    }
  }

  int64_t num_samples_val = 0;
  if (input_args[1]->isa<abstract::AbstractScalar>()) {
    auto num_samples = input_args[1]->cast<abstract::AbstractScalarPtr>();
    auto num_samples_value_ptr = num_samples->BuildValue();
    if (!num_samples_value_ptr->isa<Int64Imm>()) {
      MS_EXCEPTION(TypeError) << "For '" << prim_name << "', the num_samples"
                              << " must be a int, but got " << num_samples_value_ptr->ToString() << ".";
    }
    num_samples_val = GetValue<int64_t>(num_samples->BuildValue());
    if (num_samples_val < 0) {
      MS_EXCEPTION(ValueError) << "For '" << prim_name << "', num_samples"
                               << " should be a nonnegative number, but got " << num_samples_val << ".";
    }
  } else if (input_args[1]->cast<abstract::AbstractTensorPtr>()) {
    auto num_samples = input_args[1]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(num_samples);
    auto num_samples_value_ptr = num_samples->BuildValue();
    MS_EXCEPTION_IF_NULL(num_samples_value_ptr);
    if (num_samples_value_ptr->isa<tensor::Tensor>()) {
      auto num_samples_tensor = num_samples_value_ptr->cast<tensor::TensorPtr>();
      MS_EXCEPTION_IF_ZERO("num_samples_tensor->ElementsNum()", num_samples_tensor->ElementsNum());
      if (num_samples_tensor->data_type() == kNumberTypeInt64) {
        num_samples_val = static_cast<int64_t *>(num_samples_tensor->data_c())[0];
      } else if (num_samples_tensor->data_type() == kNumberTypeInt32) {
        num_samples_val = static_cast<int32_t *>(num_samples_tensor->data_c())[0];
      } else {
        MS_EXCEPTION(TypeError) << "For '" << prim_name << "' the num_samples"
                                << " must be a int, but got " << TypeIdToString(num_samples_tensor->data_type()) << ".";
      }
      if (num_samples_val < 0) {
        MS_EXCEPTION(ValueError) << "For '" << prim_name << "', num_samples"
                                 << " should be a nonnegative number, but got " << num_samples_val << ".";
      }
    } else {
      num_samples_val = -1;
    }
  }

  std::vector<int64_t> output_shape;
  if (x_shape.size() == x_rank_max) {
    output_shape.push_back(x_shape[0]);
  }
  output_shape.push_back(num_samples_val);
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr MultinomialInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto x_type = input_args[0]->BuildType();
  auto num_samples_type = input_args[1]->BuildType();
  const std::set valid_types_1 = {kFloat16, kFloat32, kFloat64};
  const std::set valid_types_2 = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types_1, prim_name);
  (void)CheckAndConvertUtils::CheckTypeValid("num_samples", num_samples_type, valid_types_2, prim_name);
  auto dtype = GetValue<TypePtr>(prim->GetAttr("dtype"));
  const std::set valid_types_3 = {kInt32, kInt64};
  auto out_type = CheckAndConvertUtils::CheckTypeValid("dtype", dtype->cast<TypePtr>(), valid_types_3, prim->name());
  return out_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(Multinomial, BaseOperator);

void Multinomial::Init(const int64_t seed, const int64_t seed2) {
  this->set_seed(seed);
  this->set_seed2(seed2);
}

int64_t Multinomial::get_seed() const {
  auto value_ptr = this->GetAttr(kSeed);
  return GetValue<int64_t>(value_ptr);
}
void Multinomial::set_seed(const int64_t seed) { (void)this->AddAttr(kSeed, api::MakeValue(seed)); }

int64_t Multinomial::get_seed2() const {
  auto value_ptr = this->GetAttr(kSeed2);
  return GetValue<int64_t>(value_ptr);
}
void Multinomial::set_seed2(const int64_t seed2) { (void)this->AddAttr(kSeed2, api::MakeValue(seed2)); }

AbstractBasePtr MultinomialInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  const int64_t input_num = 2;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infertype = MultinomialInferType(primitive, input_args);
  auto infershape = MultinomialInferShape(primitive, input_args);
  return abstract::MakeAbstract(infershape, infertype);
}

REGISTER_PRIMITIVE_EVAL_IMPL(Multinomial, prim::kPrimMultinomial, MultinomialInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
