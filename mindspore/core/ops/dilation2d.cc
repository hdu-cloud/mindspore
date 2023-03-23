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

#include "ops/dilation2d.h"

#include <algorithm>
#include <set>

#include "mindapi/base/types.h"
#include "ops/op_utils.h"
#include "mindapi/src/helper.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"

namespace mindspore {
namespace ops {
namespace {
void CheckDilation2DShapeAnyAndPositive(const std::string &op, const ShapeVector &shape) {
  for (size_t i = 0; i < shape.size(); ++i) {
    if ((shape[i] < 0) && (shape[i] != abstract::Shape::kShapeDimAny)) {
      MS_EXCEPTION(ValueError) << op << " shape element [" << i
                               << "] must be positive integer or kShapeDimAny, but got " << shape[i];
    }
  }
}

abstract::ShapePtr Dilation2DInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                           primitive->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto filter_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  auto filter_shape = filter_shape_map[kShape];

  const int64_t x_shape_size = 4;
  const int64_t filter_shape_size = 3;
  (void)CheckAndConvertUtils::CheckInteger("x shape size", SizeToLong(x_shape.size()), kEqual, x_shape_size,
                                           primitive->name());
  (void)CheckAndConvertUtils::CheckInteger("filter shape size", SizeToLong(filter_shape.size()), kEqual,
                                           filter_shape_size, primitive->name());
  const uint64_t n_axis = 0;
  const uint64_t shapeIndex1 = 1;
  const uint64_t shapeIndex2 = 2;
  const uint64_t shapeIndex3 = 3;
  uint64_t h_axis = shapeIndex1;
  uint64_t w_axis = shapeIndex2;
  uint64_t c_axis = shapeIndex3;
  Format data_format = Format(CheckAndConvertUtils::GetAndCheckFormat(primitive->GetAttr("format")));
  if (data_format == Format::NCHW) {
    c_axis = shapeIndex1;
    h_axis = shapeIndex2;
    w_axis = shapeIndex3;
  }
  std::string pad_mode = GetValue<std::string>(primitive->GetAttr("pad_mode"));
  std::vector<int64_t> kernel_size{filter_shape[h_axis - 1], filter_shape[w_axis - 1]};
  int64_t depth = filter_shape[c_axis - 1];
  std::vector<int64_t> stride = GetValue<std::vector<int64_t>>(primitive->GetAttr("stride"));
  std::vector<int64_t> dilation = GetValue<std::vector<int64_t>>(primitive->GetAttr("dilation"));
  int window_h = static_cast<int>((kernel_size[0] - 1) * dilation[h_axis] + 1);
  int window_w = static_cast<int>((kernel_size[1] - 1) * dilation[w_axis] + 1);
  const int64_t wLengthMaxLimit = 255;
  const int64_t wSizeMaxLimit = 512;
  if (window_h < 1 || window_h > wLengthMaxLimit || window_w < 1 || window_w > wLengthMaxLimit ||
      window_h * window_w > wSizeMaxLimit) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", size of window which is equal to (filter-1)*dilation+1 is not supported, "
                             << "window should be in the range of [1, 255], "
                             << "window_h * window_w should not be greater than 512, "
                             << "but got window_h: " << window_h << ", window_w: " << window_w << ".";
  }
  if (stride[h_axis] < 1 || stride[h_axis] > wSizeMaxLimit || stride[w_axis] < 1 || stride[w_axis] > wSizeMaxLimit) {
    MS_EXCEPTION(ValueError) << "For " << prim_name
                             << ", size of stride is not supported, the range of "
                                "stride should be [1, 255], but stride_h is "
                             << stride[h_axis] << " and stride_w is" << stride[w_axis];
  }
  if (window_h > x_shape[h_axis] || window_w > x_shape[w_axis]) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", size of window which is equal to (filter-1)*dilation+1, "
                             << "window_h should not be greater than x_h, window_w should not be greater than x_w, "
                             << "but got window_h: " << window_h << ", window_w: " << window_w
                             << ", x_h: " << x_shape[h_axis] << ", x_w: " << x_shape[w_axis] << ".";
  }
  std::vector<int64_t> output_hw;
  if (pad_mode == "VALID") {
    output_hw.push_back(static_cast<int64_t>(
      std::ceil(((x_shape[h_axis] * 1.0) - dilation[h_axis] * (kernel_size[0] - 1)) / stride[h_axis])));
    output_hw.push_back(static_cast<int64_t>(
      std::ceil(((x_shape[w_axis] * 1.0) - dilation[w_axis] * (kernel_size[1] - 1)) / stride[w_axis])));
  } else if (pad_mode == "SAME") {
    output_hw.push_back(static_cast<int64_t>(std::ceil((x_shape[h_axis] * 1.0) / stride[h_axis])));
    output_hw.push_back(static_cast<int64_t>(std::ceil((x_shape[w_axis] * 1.0) / stride[w_axis])));
  }
  ShapeVector output_shape;
  if (data_format == Format::NHWC) {
    output_shape = {x_shape[n_axis], output_hw[0], output_hw[1], depth};
  } else {
    output_shape = {x_shape[n_axis], depth, output_hw[0], output_hw[1]};
  }
  CheckDilation2DShapeAnyAndPositive(prim_name + " output_shape", output_shape);
  return std::make_shared<abstract::Shape>(output_shape);
}

TypePtr Dilation2DInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  const int64_t input_num = 2;
  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, input_num,
                                           prim->name());
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64, kUInt8, kUInt16, kUInt32,
                                         kUInt64,  kInt8,    kInt16,   kInt32, kInt64};
  std::map<std::string, TypePtr> types;
  (void)types.emplace("x", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("filter", input_args[kInputIndex1]->BuildType());
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, prim->name());
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(Dilation2D, BaseOperator);
AbstractBasePtr Dilation2DInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto infer_type = Dilation2DInferType(primitive, input_args);
  auto infer_shape = Dilation2DInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

std::vector<int64_t> Dilation2D::get_stride() const {
  auto value_ptr = GetAttr("stride");
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::vector<int64_t> Dilation2D::get_dilation() const {
  auto value_ptr = GetAttr("dilation");
  return GetValue<std::vector<int64_t>>(value_ptr);
}
std::string Dilation2D::get_pad_mode() const {
  auto value_ptr = GetAttr("pad_mode");
  return GetValue<string>(value_ptr);
}
std::string Dilation2D::get_format() const {
  auto value_ptr = GetAttr("format");
  return GetValue<std::string>(value_ptr);
}

REGISTER_PRIMITIVE_EVAL_IMPL(Dilation2D, prim::kPrimDilation2D, Dilation2DInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
