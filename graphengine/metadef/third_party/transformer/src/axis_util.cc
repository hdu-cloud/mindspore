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

#include "axis_util.h"
#include "axis_constants.h"
#include "framework/common/debug/ge_log.h"

namespace transformer {
const size_t DIM_SIZE_TWO = 2;
const size_t DIM_SIZE_FOUR = 4;
const size_t DIM_SIZE_FIVE = 5;
const size_t DIM_SIZE_SIX = 6;

const size_t EXT_INDEX_INPUT_SIZE = 0;
const size_t EXT_INDEX_HIDEEN_SIZE = 1;
const size_t EXT_INDEX_STATE_SIZE = 2;

const int32_t AXIS_NCHW_DIM_N = 0;
const int32_t AXIS_NCHW_DIM_C = 1;
const int32_t AXIS_NCHW_DIM_H = 2;
const int32_t AXIS_NCHW_DIM_W = 3;

const int32_t AXIS_NHWC_DIM_N = 0;
const int32_t AXIS_NHWC_DIM_H = 1;
const int32_t AXIS_NHWC_DIM_W = 2;
const int32_t AXIS_NHWC_DIM_C = 3;

const int32_t AXIS_HWCN_DIM_H = 0;
const int32_t AXIS_HWCN_DIM_W = 1;
const int32_t AXIS_HWCN_DIM_C = 2;
const int32_t AXIS_HWCN_DIM_N = 3;

const int32_t AXIS_CHWN_DIM_C = 0;
const int32_t AXIS_CHWN_DIM_H = 1;
const int32_t AXIS_CHWN_DIM_W = 2;
const int32_t AXIS_CHWN_DIM_N = 3;

const int32_t NDHWC_DIM_N = 0;
const int32_t NDHWC_DIM_D = 1;
const int32_t NDHWC_DIM_H = 2;
const int32_t NDHWC_DIM_W = 3;
const int32_t NDHWC_DIM_C = 4;

const int32_t NCDHW_DIM_N = 0;
const int32_t NCDHW_DIM_C = 1;
const int32_t NCDHW_DIM_D = 2;
const int32_t NCDHW_DIM_H = 3;
const int32_t NCDHW_DIM_W = 4;

const int32_t DHWCN_DIM_D = 0;
const int32_t DHWCN_DIM_H = 1;
const int32_t DHWCN_DIM_W = 2;
const int32_t DHWCN_DIM_C = 3;
const int32_t DHWCN_DIM_N = 4;

const int32_t DHWNC_DIM_D = 0;
const int32_t DHWNC_DIM_H = 1;
const int32_t DHWNC_DIM_W = 2;
const int32_t DHWNC_DIM_N = 3;
const int32_t DHWNC_DIM_C = 4;

const int32_t AXIS_NC1HWC0_DIM_N = 0;
const int32_t AXIS_NC1HWC0_DIM_C1 = 1;
const int32_t AXIS_NC1HWC0_DIM_H = 2;
const int32_t AXIS_NC1HWC0_DIM_W = 3;
const int32_t AXIS_NC1HWC0_DIM_C0 = 4;

const int32_t AXIS_C1HWNCoC0_DIM_C1 = 0;
const int32_t AXIS_C1HWNCoC0_DIM_H = 1;
const int32_t AXIS_C1HWNCoC0_DIM_W = 2;
const int32_t AXIS_C1HWNCoC0_DIM_N = 3;
const int32_t AXIS_C1HWNCoC0_DIM_Co = 4;

bool AxisUtil::GetAxisValueByOriginFormat(const ge::Format &format, const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.IsScalar(), GELOGI("Original dim vector is empty!"), return true);
  switch (format) {
    case ge::FORMAT_NCHW:
      return GetAxisValueByNCHW(shape, axis_value);
    case ge::FORMAT_NHWC:
      return GetAxisValueByNHWC(shape, axis_value);
    case ge::FORMAT_HWCN:
      return GetAxisValueByHWCN(shape, axis_value);
    case ge::FORMAT_ND:
      return GetAxisValueByND(shape, axis_value);
    case ge::FORMAT_NCDHW:
      return GetAxisValueByNCDHW(shape, axis_value);
    case ge::FORMAT_NDHWC:
      return GetAxisValueByNDHWC(shape, axis_value);
    case ge::FORMAT_DHWCN:
      return GetAxisValueByDHWCN(shape, axis_value);
    case ge::FORMAT_DHWNC:
      return GetAxisValueByDHWNC(shape, axis_value);
    case ge::FORMAT_NC1HWC0:
      return GetAxisValueByNC1HWC0(shape, axis_value);
    case ge::FORMAT_C1HWNCoC0:
      return GetAxisValueByC1HWNCoC0(shape, axis_value);
    default:
      GELOGI("Can not get axis value of old format %d.", format);
      return false;
  }
}

bool AxisUtil::GetAxisValueByND(const gert::Shape &shape, AxisValue &axis_value) {
  /* To differentiate the input datatype of int8 and others */
  if (shape.GetDimNum() == DIM_SIZE_FOUR) {
    axis_value[AXIS_N] = shape.GetDim(AXIS_NCHW_DIM_N);
    axis_value[AXIS_C] = shape.GetDim(AXIS_NCHW_DIM_C);
    axis_value[AXIS_H] = shape.GetDim(AXIS_NCHW_DIM_H);
    axis_value[AXIS_W] = shape.GetDim(AXIS_NCHW_DIM_W);
    axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
    axis_value[AXIS_Co] = axis_value[AXIS_C0];
  }
  return true;
}

bool AxisUtil::GetAxisValueByNCHW(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FOUR, GELOGI("Dim size is less than 4."), return false);
  /* C0 Must be set for case ND or 2D-NCHW to NZ */
  axis_value[AXIS_N] = shape.GetDim(AXIS_NCHW_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(AXIS_NCHW_DIM_C);
  axis_value[AXIS_H] = shape.GetDim(AXIS_NCHW_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(AXIS_NCHW_DIM_W);
  axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  axis_value[AXIS_Co] = axis_value[AXIS_C0];
  return true;
}

bool AxisUtil::GetAxisValueByNHWC(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FOUR, GELOGI("Dim size is less than 4."), return false);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axis_value[AXIS_N] = shape.GetDim(AXIS_NHWC_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(AXIS_NHWC_DIM_C);
  axis_value[AXIS_H] = shape.GetDim(AXIS_NHWC_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(AXIS_NHWC_DIM_W);
  axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  axis_value[AXIS_Co] = axis_value[AXIS_C0];
  return true;
}

bool AxisUtil::GetAxisValueByNC1HWC0(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FOUR, GELOGI("Dim size is less than 4."), return false);
  if (shape.GetDimNum() == DIM_SIZE_FIVE) {
    axis_value[AXIS_C0] = shape.GetDim(AXIS_NC1HWC0_DIM_C0);
    axis_value[AXIS_C1] = shape.GetDim(AXIS_NC1HWC0_DIM_C1);
    axis_value[AXIS_C] = axis_value[AXIS_C1] * axis_value[AXIS_C0];
  } else {
    axis_value[AXIS_C] = shape.GetDim(AXIS_NCHW_DIM_C);
    axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  }

  axis_value[AXIS_N] = shape.GetDim(AXIS_NC1HWC0_DIM_N);
  axis_value[AXIS_H] = shape.GetDim(AXIS_NC1HWC0_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(AXIS_NC1HWC0_DIM_W);
  return true;
}

bool AxisUtil::GetAxisValueByHWCN(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FOUR, GELOGI("Dim size is less than 4."), return false);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axis_value[AXIS_N] = shape.GetDim(AXIS_HWCN_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(AXIS_HWCN_DIM_C);
  axis_value[AXIS_H] = shape.GetDim(AXIS_HWCN_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(AXIS_HWCN_DIM_W);
  axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  axis_value[AXIS_Co] = axis_value[AXIS_C0];
  return true;
}

bool AxisUtil::GetAxisValueByC1HWNCoC0(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_SIX, GELOGI("Dim size is less than 6."), return false);
  /* C0 Must be set for case ND or 2D-NHWC to NZ */
  axis_value[AXIS_N] = shape.GetDim(AXIS_C1HWNCoC0_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(AXIS_C1HWNCoC0_DIM_C1) * axis_value[AXIS_C0];
  axis_value[AXIS_H] = shape.GetDim(AXIS_C1HWNCoC0_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(AXIS_C1HWNCoC0_DIM_W);
  axis_value[AXIS_C1] = shape.GetDim(AXIS_C1HWNCoC0_DIM_C1);
  axis_value[AXIS_Co] = shape.GetDim(AXIS_C1HWNCoC0_DIM_Co);
  return true;
}

bool AxisUtil::GetAxisValueByNDHWC(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FIVE, GELOGI("Dim size is less than 5."), return false);

  axis_value[AXIS_N] = shape.GetDim(NDHWC_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(NDHWC_DIM_C);
  axis_value[AXIS_H] = shape.GetDim(NDHWC_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(NDHWC_DIM_W);
  axis_value[AXIS_D] = shape.GetDim(NDHWC_DIM_D);
  axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  axis_value[AXIS_Co] = axis_value[AXIS_C0];
  return true;
}

bool AxisUtil::GetAxisValueByNCDHW(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FIVE, GELOGI("Dim size is less than 5."), return false);

  axis_value[AXIS_N] = shape.GetDim(NCDHW_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(NCDHW_DIM_C);
  axis_value[AXIS_H] = shape.GetDim(NCDHW_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(NCDHW_DIM_W);
  axis_value[AXIS_D] = shape.GetDim(NCDHW_DIM_D);
  axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  axis_value[AXIS_Co] = axis_value[AXIS_C0];
  return true;
}

bool AxisUtil::GetAxisValueByDHWCN(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FIVE, GELOGI("Dim size is less than 5."), return false);

  axis_value[AXIS_N] = shape.GetDim(DHWCN_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(DHWCN_DIM_C);
  axis_value[AXIS_H] = shape.GetDim(DHWCN_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(DHWCN_DIM_W);
  axis_value[AXIS_D] = shape.GetDim(DHWCN_DIM_D);
  axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  axis_value[AXIS_Co] = axis_value[AXIS_C0];
  return true;
}

bool AxisUtil::GetAxisValueByDHWNC(const gert::Shape &shape, AxisValue &axis_value) {
  CHECK(shape.GetDimNum() < DIM_SIZE_FIVE, GELOGI("Dim size is less than 5."), return false);
  axis_value[AXIS_N] = shape.GetDim(DHWNC_DIM_N);
  axis_value[AXIS_C] = shape.GetDim(DHWNC_DIM_C);
  axis_value[AXIS_H] = shape.GetDim(DHWNC_DIM_H);
  axis_value[AXIS_W] = shape.GetDim(DHWNC_DIM_W);
  axis_value[AXIS_D] = shape.GetDim(DHWNC_DIM_D);
  axis_value[AXIS_C1] = DivisionCeiling(axis_value[AXIS_C], axis_value[AXIS_C0]);
  axis_value[AXIS_Co] = axis_value[AXIS_C0];

  return true;
}
} // namespace transformer
