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

#include "ops/im2col.h"
#include <algorithm>
#include <memory>
#include <set>
#include <string>
#include <vector>
#include "abstract/abstract_value.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
void Im2Col::set_ksizes(const std::vector<int64_t> &ksizes) { (void)this->AddAttr(kKsizes, api::MakeValue(ksizes)); }

std::vector<int64_t> Im2Col::get_ksizes() const {
  auto value_ptr = GetAttr(kKsizes);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Im2Col::set_strides(const std::vector<int64_t> &strides) {
  (void)this->AddAttr(kStrides, api::MakeValue(strides));
}

std::vector<int64_t> Im2Col::get_strides() const {
  auto value_ptr = GetAttr(kStrides);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Im2Col::set_dilations(const std::vector<int64_t> &dilations) {
  (void)this->AddAttr(kDilations, api::MakeValue(dilations));
}

std::vector<int64_t> Im2Col::get_dilations() const {
  auto value_ptr = GetAttr(kDilations);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

void Im2Col::set_pad_mode(const std::string &pad_mode) { (void)this->AddAttr(kPaddingMode, api::MakeValue(pad_mode)); }

std::string Im2Col::get_pad_mode() const {
  auto value_ptr = GetAttr(kPaddingMode);
  MS_EXCEPTION_IF_NULL(value_ptr);
  auto mode_str = GetValue<std::string>(value_ptr);
  std::transform(mode_str.begin(), mode_str.end(), mode_str.begin(), ::toupper);
  return mode_str;
}

void Im2Col::set_pads(const std::vector<int64_t> &pads) { (void)this->AddAttr(kPads, api::MakeValue(pads)); }

std::vector<int64_t> Im2Col::get_pads() const {
  auto value_ptr = GetAttr(kPads);
  MS_EXCEPTION_IF_NULL(value_ptr);
  return GetValue<std::vector<int64_t>>(value_ptr);
}

namespace {
abstract::ShapePtr Im2ColInferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  auto op_name = primitive->name();
  constexpr size_t size_2 = 2;
  constexpr size_t size_4 = 4;

  auto in_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("dimension of input x", SizeToLong(in_shape.size()), kEqual,
                                           SizeToLong(size_4), op_name);
  if (input_args[kInputIndex0]->BuildShape()->IsDynamic()) {
    return std::make_shared<abstract::Shape>(in_shape);
  }
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero("spatial size of input", in_shape, op_name);
  auto ksizes_ptr = primitive->GetAttr(kKsizes);
  MS_EXCEPTION_IF_NULL(ksizes_ptr);
  auto ksizes = GetValue<std::vector<int64_t>>(ksizes_ptr);
  auto strides_ptr = primitive->GetAttr(kStrides);
  MS_EXCEPTION_IF_NULL(strides_ptr);
  auto strides = GetValue<std::vector<int64_t>>(strides_ptr);
  auto dilations_ptr = primitive->GetAttr(kDilations);
  MS_EXCEPTION_IF_NULL(dilations_ptr);
  auto dilations = GetValue<std::vector<int64_t>>(dilations_ptr);
  auto padding_mode_ptr = primitive->GetAttr(kPaddingMode);
  MS_EXCEPTION_IF_NULL(padding_mode_ptr);
  auto padding_mode = GetValue<string>(padding_mode_ptr);
  auto pads_ptr = primitive->GetAttr(kPads);
  MS_EXCEPTION_IF_NULL(pads_ptr);
  auto pads = GetValue<std::vector<int64_t>>(pads_ptr);

  if (ksizes.empty() || ksizes.size() > size_2) {
    MS_EXCEPTION(ValueError)
      << "For Im2Col, the element number of ksizes must be 1 or 2 when x_format only support NCHW, but get "
      << ksizes.size() << " elements in ksizes.";
  }
  if (strides.empty() || strides.size() > size_2) {
    MS_EXCEPTION(ValueError)
      << "For Im2Col, the element number of strides must be 1 or 2 when x_format only support NCHW, but get "
      << strides.size() << " elements in strides.";
  }
  if (dilations.empty() || dilations.size() > size_2) {
    MS_EXCEPTION(ValueError)
      << "For Im2Col, the element number of dilations must be 1 or 2 when x_format only support NCHW, but get "
      << dilations.size() << " elements in dilations.";
  }
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero(kKsizes, ksizes, op_name);
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero(kDilations, dilations, op_name);
  (void)CheckAndConvertUtils::CheckPositiveVectorExcludeZero(kStrides, strides, op_name);

  const int64_t in_n = in_shape[kInputIndex0];
  const int64_t in_c = in_shape[kInputIndex1];
  const int64_t in_h = in_shape[kInputIndex2];
  const int64_t in_w = in_shape[kInputIndex3];

  int64_t filter_h = ksizes.front();
  int64_t filter_w = ksizes.back();
  int64_t dilation_h = dilations.front();
  int64_t dilation_w = dilations.back();
  int64_t stride_h = strides.front();
  MS_EXCEPTION_IF_ZERO("stride_h", stride_h);
  int64_t stride_w = strides.back();
  MS_EXCEPTION_IF_ZERO("stride_w", stride_w);

  int64_t effective_filter_h = (filter_h - 1) * dilation_h + 1;
  int64_t effective_filter_w = (filter_w - 1) * dilation_w + 1;
  int64_t out_h{0}, out_w{0}, out_c{0};
  int64_t pad_h_top{0}, pad_h_bottom{0}, pad_w_before{0}, pad_w_after{0};
  if (padding_mode == "VALID") {
    out_h = (in_h - effective_filter_h + stride_h) / stride_h;
    out_w = (in_w - effective_filter_w + stride_w) / stride_w;
  } else if (padding_mode == "SAME") {
    out_h = (in_h + stride_h - 1) / stride_h;
    out_w = (in_w + stride_w - 1) / stride_w;
  } else if (padding_mode == "CALCULATED") {
    (void)CheckAndConvertUtils::CheckPositiveVector(kPads, pads, op_name);
    if (!pads.empty() && pads.size() <= size_2) {
      pad_h_top = pad_h_bottom = pads.front();
      pad_w_before = pad_w_after = pads.back();
    } else if (!pads.empty() && pads.size() == size_4) {
      pad_h_top = pads[kInputIndex0];
      pad_h_bottom = pads[kInputIndex1];
      pad_w_before = pads[kInputIndex2];
      pad_w_after = pads[kInputIndex3];
    } else {
      MS_EXCEPTION(ValueError) << "For Im2Col, the size of pads must be 1, 2 or 4, but get " << pads.size()
                               << "elements in pads.";
    }
    out_h = (in_h + pad_h_top + pad_h_bottom - (dilation_h * (filter_h - 1) + 1)) / stride_h + 1;
    out_w = (in_w + pad_w_before + pad_w_after - (dilation_w * (filter_w - 1) + 1)) / stride_w + 1;
  } else {
    MS_EXCEPTION(ValueError) << "For Im2Col, the padding_mode only support VALID, SAME and CALCULATED, but get "
                             << padding_mode << ".";
  }
  out_c = in_c * filter_h * filter_w;
  if (out_h < 1 || out_w < 1) {
    MS_EXCEPTION(ValueError) << "For Im2Col, given input with spatial size (" << in_n << ", " << in_c << ", " << in_h
                             << ", " << in_w << "), ksizes=(" << filter_h << ", " << filter_w << "), dilation=("
                             << dilation_h << ", " << dilation_w << "), padding_mode=" << padding_mode << ", pads=("
                             << pad_h_top << ", " << pad_h_bottom << ", " << pad_w_before << ", " << pad_w_after
                             << "), calculated shape of output as (" << in_h << ", " << out_c << ", " << out_h << ", "
                             << out_w << "), which is too small (non-positive).";
  }
  // current only support NCHW
  std::vector<int64_t> out_dim = {in_n, out_c, out_h, out_w};
  ShapeVector out_shape = out_dim;
  return std::make_shared<abstract::Shape>(out_shape);
}

TypePtr Im2ColInferType(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  return CheckAndConvertUtils::CheckTensorTypeValid("x", input_args[kInputIndex0]->BuildType(), common_valid_types,
                                                    primitive->name());
}
}  // namespace

AbstractBasePtr Im2ColInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                            const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  constexpr int64_t input_num = 1;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, op_name);
  auto infer_type = Im2ColInferType(primitive, input_args);
  auto infer_shape = Im2ColInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(Im2Col, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(Im2Col, prim::kPrimIm2Col, Im2ColInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
