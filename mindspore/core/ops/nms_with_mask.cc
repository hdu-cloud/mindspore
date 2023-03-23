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

#include "ops/nms_with_mask.h"
#include <algorithm>
#include <memory>
#include <vector>
#include <set>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
constexpr size_t kBboxesShapeSize = 2;
constexpr int64_t kBboxesShapeIn2ndDimAscendAfterPad = 8;
constexpr int64_t kBboxesShapeIn2ndDimNormal = 5;
constexpr int64_t kDynamicShapeDim = -1;

void NMSWithMask::set_iou_threshold(const std::vector<float> &iou_thredshold) {
  (void)this->AddAttr(kNmsIouThreshold, api::MakeValue(iou_thredshold));
}

std::vector<float> NMSWithMask::get_iou_threshold() const {
  auto value_ptr = GetAttr(kNmsIouThreshold);
  return GetValue<std::vector<float>>(value_ptr);
}

void NMSWithMask::Init(const float iou_threshold) {
  auto op_name = this->name();
  std::vector<float> iou_threshold_vec = {iou_threshold};
  (void)CheckAndConvertUtils::CheckInteger("iou_threshold_len", SizeToLong(iou_threshold_vec.size()), kEqual, 1,
                                           op_name);
  for (auto &item : iou_threshold_vec) {
    CheckAndConvertUtils::CheckInRange<float>("iou_threshold", item, kIncludeBoth, {0, 1}, op_name);
  }
  this->set_iou_threshold(iou_threshold_vec);
}

namespace {
abstract::TupleShapePtr NMSWithMaskInferShape(const PrimitivePtr &primitive,
                                              const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  auto x = input_args[0]->BuildShape();
  MS_EXCEPTION_IF_NULL(x);
  auto bboxes_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(x);
  auto bboxes_shape = bboxes_shape_map[kShape];
  auto bboxes_shape_min = bboxes_shape_map[kMinShape];
  auto bboxes_shape_max = bboxes_shape_map[kMaxShape];

  (void)CheckAndConvertUtils::CheckValue<size_t>("shape of bboxes", bboxes_shape.size(), kEqual, kBboxesShapeSize,
                                                 op_name);
  if (bboxes_shape[1] != kDynamicShapeDim && bboxes_shape[1] != kBboxesShapeIn2ndDimNormal &&
      bboxes_shape[1] != kBboxesShapeIn2ndDimAscendAfterPad) {
    MS_EXCEPTION(ValueError) << " For " << op_name
                             << ", the 2nd dim in shape of bboxes should equal to 5 or 8, but got " << bboxes_shape[1];
  }

  // for ascend
  if (bboxes_shape[1] == kBboxesShapeIn2ndDimAscendAfterPad) {
    bboxes_shape[1] = kBboxesShapeIn2ndDimNormal;
  }

  // output_idx, selected_mask output shape
  abstract::ShapePtr output_idx_shape;
  ShapeVector output_idx_shape_real = {bboxes_shape[0]};
  if (!bboxes_shape_min.empty() && !bboxes_shape_max.empty()) {
    if (bboxes_shape_min[1] == kBboxesShapeIn2ndDimAscendAfterPad) {
      bboxes_shape_min[1] = kBboxesShapeIn2ndDimNormal;
    }
    if (bboxes_shape_max[1] == kBboxesShapeIn2ndDimAscendAfterPad) {
      bboxes_shape_max[1] = kBboxesShapeIn2ndDimNormal;
    }
    ShapeVector output_idx_shape_min = {bboxes_shape_min[0]};
    ShapeVector output_idx_shape_max = {bboxes_shape_max[0]};
    output_idx_shape =
      std::make_shared<abstract::Shape>(output_idx_shape_real, output_idx_shape_min, output_idx_shape_max);
  } else {
    output_idx_shape = std::make_shared<abstract::Shape>(output_idx_shape_real);
  }

  abstract::ShapePtr output_boxes_shape =
    std::make_shared<abstract::Shape>(bboxes_shape, bboxes_shape_min, bboxes_shape_max);

  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{output_boxes_shape, output_idx_shape, output_idx_shape});
}

TypePtr NMSWithMaskInferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  auto infer_type = input_args[0]->BuildType();
  MS_EXCEPTION_IF_NULL(infer_type);
  const std::set<TypePtr> valid_types = {kFloat16, kFloat32};
  auto type = CheckAndConvertUtils::CheckTensorTypeValid("bboxes", infer_type, valid_types, prim->name());
  std::vector<TypePtr> type_tuple;
  type_tuple.push_back(type);
  type_tuple.push_back(kInt32);
  type_tuple.push_back(kBool);
  return std::make_shared<Tuple>(type_tuple);
}
}  // namespace

MIND_API_OPERATOR_IMPL(NMSWithMask, BaseOperator);
AbstractBasePtr NMSWithMaskInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                 const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();

  (void)CheckAndConvertUtils::CheckInteger("input number", SizeToLong(input_args.size()), kEqual, 1, prim_name);
  for (const auto &item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }

  auto infer_type = NMSWithMaskInferType(primitive, input_args);
  auto infer_shape = NMSWithMaskInferShape(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}
REGISTER_PRIMITIVE_EVAL_IMPL(NMSWithMask, prim::kPrimNMSWithMask, NMSWithMaskInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
