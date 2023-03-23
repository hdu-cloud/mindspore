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

#include <set>
#include <map>
#include <string>
#include <vector>
#include <memory>
#include <algorithm>

#include "ops/sparse_segment_sqrt_n_with_num_segments.h"
#include "abstract/dshape.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
abstract::ShapePtr SparseSegmentSqrtNWithNumSegmentsInferShape(const PrimitivePtr &prim,
                                                               const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  auto x_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape())[kShape];
  auto indices_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape())[kShape];
  auto segment_ids_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape())[kShape];
  auto num_segments_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape())[kShape];
  (void)CheckAndConvertUtils::CheckInteger("indices_shape", indices_shape.size(), kEqual, kInputIndex1, prim->name());
  (void)CheckAndConvertUtils::CheckInteger("segment_ids_shape", segment_ids_shape.size(), kEqual, kInputIndex1,
                                           prim->name());
  if (x_shape.size() < kInputIndex1) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', "
                             << "x's rank must be greater than 1, but got [" << x_shape.size() << "].";
  }
  if (!(IsDynamic(indices_shape) || IsDynamic(segment_ids_shape)) &&
      indices_shape[kInputIndex0] != segment_ids_shape[kInputIndex0]) {
    MS_EXCEPTION(ValueError) << "For '" << prim_name << "', the rank of indices and segment_ids must be the same, "
                             << "but got indices [" << indices_shape[kInputIndex0] << "] "
                             << "and segment_ids [" << segment_ids_shape[kInputIndex0] << "].";
  }
  if (num_segments_shape.size() > kInputIndex1) {
    MS_EXCEPTION(ValueError) << "For " << prim_name << ", num_segments should be at most 1-D, but got ["
                             << num_segments_shape.size() << "].";
  }
  if (IsDynamicRank(x_shape)) {
    return std::make_shared<abstract::Shape>(std::vector<int64_t>{-2});
  }
  if (!input_args[kInputIndex3]->BuildValue()->isa<AnyValue>() &&
      !input_args[kInputIndex3]->BuildValue()->isa<None>()) {
    if (!IsDynamic(num_segments_shape) && num_segments_shape.size() == kInputIndex1) {
      if (num_segments_shape[kInputIndex0] != kInputIndex1) {
        MS_EXCEPTION(ValueError) << "For " << prim_name << ", the num element of num_segments should be 1, but got ["
                                 << num_segments_shape[kInputIndex0] << "].";
      }
    }
    auto num_segments_value = input_args[kInputIndex3]->cast<abstract::AbstractTensorPtr>();
    MS_EXCEPTION_IF_NULL(num_segments_value);
    auto num_segments_value_ptr = num_segments_value->BuildValue();
    MS_EXCEPTION_IF_NULL(num_segments_value_ptr);
    auto num_segments_value_ptr_tensor =
      CheckAndConvertUtils::CheckTensorIntValue("num_segments", num_segments_value_ptr, prim->name());
    size_t dim_zero = static_cast<size_t>(num_segments_value_ptr_tensor.back());
    if (dim_zero < kInputIndex1) {
      MS_EXCEPTION(ValueError) << "For " << prim_name
                               << ", num_segments must bigger than the last number of segment_ids, "
                               << "but got " << dim_zero << ".";
    } else {
      ShapeVector y_shape = x_shape;
      y_shape[kInputIndex0] = static_cast<int64_t>(dim_zero);
      return std::make_shared<abstract::Shape>(y_shape);
    }
  } else {
    ShapeVector output_shape = x_shape;
    output_shape[kInputIndex0] = -1;
    return std::make_shared<abstract::Shape>(output_shape);
  }
}

TypePtr SparseSegmentSqrtNWithNumSegmentsInferType(const PrimitivePtr &prim,
                                                   const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto x_type = input_args[kInputIndex0]->BuildType();
  auto indices_type = input_args[kInputIndex1]->BuildType();
  auto segment_ids_type = input_args[kInputIndex2]->BuildType();
  auto num_segments_type = input_args[kInputIndex3]->BuildType();
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32, kFloat64};
  const std::set<TypePtr> common_valid_types = {kInt32, kInt64};
  (void)CheckAndConvertUtils::CheckTensorTypeValid("x", x_type, valid_types, prim->name());
  std::map<std::string, TypePtr> types;
  (void)types.emplace("indices", indices_type);
  (void)types.emplace("segment_ids", segment_ids_type);
  (void)types.emplace("num_segments", num_segments_type);
  (void)CheckAndConvertUtils::CheckTensorTypeSame(types, common_valid_types, prim->name());
  return input_args[kInputIndex0]->BuildType();
}
}  // namespace

MIND_API_OPERATOR_IMPL(SparseSegmentSqrtNWithNumSegments, BaseOperator);
AbstractBasePtr SparseSegmentSqrtNWithNumSegmentsInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &prim,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(prim);
  auto prim_name = prim->name();
  const int64_t input_num = static_cast<int64_t>(kInputIndex4);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, input_num, prim_name);
  auto types = SparseSegmentSqrtNWithNumSegmentsInferType(prim, input_args);
  auto shapes = SparseSegmentSqrtNWithNumSegmentsInferShape(prim, input_args);
  return abstract::MakeAbstract(shapes, types);
}
REGISTER_HOST_DEPENDS(kNameSparseSegmentSqrtNWithNumSegments, {3});
REGISTER_PRIMITIVE_EVAL_IMPL(SparseSegmentSqrtNWithNumSegments, prim::kPrimSparseSegmentSqrtNWithNumSegments,
                             SparseSegmentSqrtNWithNumSegmentsInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
