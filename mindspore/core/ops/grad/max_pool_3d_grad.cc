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

#include "ops/grad/max_pool_3d_grad.h"
#include <memory>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void MaxPool3DGrad::Init(const std::vector<int64_t> &kernel_size, const std::vector<int64_t> &strides,
                         const PadMode &pad_mode, const std::vector<int64_t> &pad_list, const Format &format) {
  set_kernel_size(kernel_size);
  set_strides(strides);
  set_pad_mode(pad_mode);
  set_pad_list(pad_list);
  set_format(format);
}

void MaxPool3DGrad::set_pad_list(const std::vector<int64_t> &pad_list) {
  const int64_t pad_size = 4;
  (void)CheckAndConvertUtils::CheckInteger(kPadList, SizeToLong(pad_list.size()), kEqual, pad_size, name());
  (void)AddAttr(kPadList, api::MakeValue(pad_list));
}

std::vector<int64_t> MaxPool3DGrad::get_pad_list() const {
  auto value_ptr = GetAttr(kPadList);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

abstract::ShapePtr MaxPool3DGradInferShape(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[0]->GetShapeTrack())[kShape];
  constexpr int64_t k5DInputDims = 5;
  (void)CheckAndConvertUtils::CheckInteger("input_rank", SizeToLong(x_shape.size()), kEqual, k5DInputDims, op_name);
  return std::make_shared<abstract::Shape>(x_shape);
}

TypePtr MaxPool3DGradInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  const int64_t input_num = 3;
  (void)CheckAndConvertUtils::CheckInteger("input size", SizeToLong(input_args.size()), kEqual, input_num, op_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  auto x_dtype = input_args[0]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  return CheckAndConvertUtils::CheckTensorTypeValid("input", x_dtype, valid_types, op_name);
}

MIND_API_OPERATOR_IMPL(MaxPool3DGrad, PoolGrad);
AbstractBasePtr MaxPool3DGradInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                   const std::vector<AbstractBasePtr> &input_args) {
  auto res = std::make_shared<abstract::AbstractTensor>(MaxPool3DGradInferType(primitive, input_args),
                                                        MaxPool3DGradInferShape(primitive, input_args)->shape());
  return res;
}

REGISTER_PRIMITIVE_EVAL_IMPL(MaxPool3DGrad, prim::kPrimMaxPool3DGrad, MaxPool3DGradInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
