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

#include "ops/population_count.h"
#include <string>
#include <algorithm>
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "utils/ms_context.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr PopulationCountInferShape(const PrimitivePtr &primitive,
                                             const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto shape_element = x->cast<abstract::ShapePtr>();
  MS_EXCEPTION_IF_NULL(shape_element);
  return shape_element;
}

TypePtr PopulationCountInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto context = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context);
  bool is_cpu = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kCPUDevice);
  bool is_ascend = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kAscendDevice);
  bool is_gpu = (context->get_param<std::string>(MS_CTX_DEVICE_TARGET) == kGPUDevice);
  auto input_type = input_args[0]->BuildType();
  if (is_cpu || is_gpu) {
    std::set<TypePtr> check_list = {kInt8, kInt16, kInt32, kInt64, kUInt8, kUInt16, kUInt32, kUInt64};
    CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_type, check_list, prim->name());
  }
  if (is_ascend) {
    std::set<TypePtr> check_list = {kInt16, kUInt16};
    CheckAndConvertUtils::CheckTensorTypeValid("input_x", input_type, check_list, prim->name());
  }
  auto output_type = kUInt8;
  return output_type;
}
}  // namespace

MIND_API_OPERATOR_IMPL(PopulationCount, BaseOperator);
AbstractBasePtr PopulationCountInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                     const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, primitive->name());
  auto infer_type = PopulationCountInferType(primitive, input_args);
  auto infer_shape = PopulationCountInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

REGISTER_PRIMITIVE_EVAL_IMPL(PopulationCount, prim::kPrimPopulationCount, PopulationCountInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
