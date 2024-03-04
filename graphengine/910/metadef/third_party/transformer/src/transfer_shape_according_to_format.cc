/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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

#include "transfer_shape_according_to_format.h"
#include <algorithm>
#include "axis_constants.h"
#include "graph/utils/attr_utils.h"

namespace transformer {
namespace {
  const std::string kAttrHiddenSize = "hidden_size";
  const std::string kAttrInputSize = "input_size";
  const std::string kAttrStateSize = "state_size";
  const int64_t kM0DefaultVal = 16;

  void GeShapeToRtShape(const ge::GeShape &ge_shape, gert::Shape &rt_shape) {
    rt_shape.SetDimNum(0);
    for (size_t i = 0; i < ge_shape.GetDimNum(); ++i) {
      rt_shape.AppendDim(ge_shape.GetDim(i));
    }
  }

  void RtShapeToGeShape(const gert::Shape &rt_shape, ge::GeShape &ge_shape) {
    ge_shape.SetDimNum(0);
    for (size_t i = 0; i < rt_shape.GetDimNum(); ++i) {
      ge_shape.AppendDim(rt_shape.GetDim(i));
    }
  }
}

ShapeTransferAccordingToFormat::ShapeTransferAccordingToFormat() {}

bool ShapeTransferAccordingToFormat::GetShapeAccordingToFormat(const ge::OpDescPtr &op_desc,
                                                               ShapeAndFormat &shapeAndFormatInfo) {
  if (shapeAndFormatInfo.oldShape.IsUnknownDimNum()) {
    return true;
  }
  gert::Shape shape;
  GeShapeToRtShape(shapeAndFormatInfo.oldShape, shape);
  ExtAxisValue ext_axis;
  InitExtAxisValue(op_desc, ext_axis);
  bool ret = TransferShapeUtils::TransferShape(shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.newFormat,
                                               shapeAndFormatInfo.currentDataType, ext_axis, shape);
  RtShapeToGeShape(shape, shapeAndFormatInfo.oldShape);
  return ret;
}

bool ShapeTransferAccordingToFormat::GetShapeAccordingToFormat(ShapeAndFormat &shapeAndFormatInfo) {
  if (shapeAndFormatInfo.oldShape.IsUnknownDimNum()) {
    return true;
  }
  gert::Shape shape;
  GeShapeToRtShape(shapeAndFormatInfo.oldShape, shape);
  ExtAxisValue ext_axis = {shapeAndFormatInfo.extra_attr.input_size, shapeAndFormatInfo.extra_attr.hidden_size,
                           shapeAndFormatInfo.extra_attr.state_size, kM0DefaultVal};
  bool ret = TransferShapeUtils::TransferShape(shapeAndFormatInfo.oldFormat, shapeAndFormatInfo.newFormat,
                                               shapeAndFormatInfo.currentDataType, ext_axis, shape);
  RtShapeToGeShape(shape, shapeAndFormatInfo.oldShape);
  return ret;
}

bool ShapeTransferAccordingToFormat::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                                   const ge::DataType &data_type, const ExtAxisValue &ext_axis,
                                                   ge::GeShape &shape) {
  gert::Shape rt_shape;
  GeShapeToRtShape(shape, rt_shape);
  bool ret = TransferShapeUtils::TransferShape(origin_format, format, data_type, ext_axis, rt_shape);
  RtShapeToGeShape(rt_shape, shape);
  return ret;
}

bool ShapeTransferAccordingToFormat::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                                   const ge::DataType &data_type, const ExtAxisValue &ext_axis,
                                                   const ge::GeShape &origin_shape, ge::GeShape &shape) {
  gert::Shape rt_origin_shape;
  GeShapeToRtShape(origin_shape, rt_origin_shape);
  gert::Shape rt_shape;
  GeShapeToRtShape(shape, rt_shape);
  bool ret = TransferShapeUtils::TransferShape(origin_format, format, data_type, ext_axis, rt_origin_shape, rt_shape);
  RtShapeToGeShape(rt_shape, shape);
  return ret;
}

bool ShapeTransferAccordingToFormat::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                                   const ge::DataType &data_type, gert::Shape &shape,
                                                   const ge::OpDescPtr op_desc) {
  ExtAxisValue ext_axis;
  InitExtAxisValue(op_desc, ext_axis);
  return TransferShapeUtils::TransferShape(origin_format, format, data_type, ext_axis, shape);
}

bool ShapeTransferAccordingToFormat::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                                   const ge::DataType &data_type, const gert::Shape &origin_shape,
                                                   gert::Shape &shape, const ge::OpDescPtr op_desc) {
  ExtAxisValue ext_axis;
  InitExtAxisValue(op_desc, ext_axis);
  return TransferShapeUtils::TransferShape(origin_format, format, data_type, ext_axis, origin_shape, shape);
}

void ShapeTransferAccordingToFormat::InitExtAxisValue(const ge::OpDescPtr &op_desc, ExtAxisValue &ext_axis) {
  int64_t input_size = 1;
  int64_t hidden_size = 1;
  int64_t state_size = -1;
  if (op_desc != nullptr) {
    (void)ge::AttrUtils::GetInt(op_desc, kAttrInputSize, input_size);
    (void)ge::AttrUtils::GetInt(op_desc, kAttrHiddenSize, hidden_size);
    (void)ge::AttrUtils::GetInt(op_desc, kAttrStateSize, state_size);
  }

  ext_axis[EXT_INDEX_INPUT_SIZE] = input_size;
  ext_axis[EXT_INDEX_HIDEEN_SIZE] = hidden_size;
  ext_axis[EXT_INDEX_STATE_SIZE] = state_size;
  ext_axis[EXT_INDEX_M0_VAL] = kM0DefaultVal;
}
} // namespace transformer
