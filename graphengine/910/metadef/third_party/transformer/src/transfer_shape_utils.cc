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

#include "transfer_shape_utils.h"
#include <mutex>
#include "axis_constants.h"
#include "common/ge_common/string_util.h"
#include "external/graph/ge_error_codes.h"
#include "external/platform/platform_info.h"
#include "graph/utils/type_utils.h"

namespace transformer {
std::array<uint32_t, static_cast<size_t>(ge::DataType::DT_MAX)> TransferShapeUtils::m0_list_{};
std::array<uint32_t, static_cast<size_t>(ge::DataType::DT_MAX)> TransferShapeUtils::k0_list_{};
std::array<uint32_t, static_cast<size_t>(ge::DataType::DT_MAX)> TransferShapeUtils::n0_list_{};
namespace {
  const int64_t SHAPE_NUMBER_16 = 16;
  const int64_t SHAPE_NUMBER_4 = 4;
  const int64_t NI = 16;
  const int64_t LSTM_NI = 4;
  const int64_t GROUPS_DEFAULT_VALUE = 1;
  const int64_t UNKNOWN_SHAPE_VALUE = -1;
  const int64_t RNN_STATE_SIZE_DEFAULT_VALUE = -1;
  const size_t NUMBER_2 = 2;
  const size_t MINUS_VALUE_ONE = 1;
  const size_t MINUS_VALUE_TWO = 2;

  const size_t DIM_INDEX_N = 0;
  const size_t DIM_INDEX_C = 1;
  const size_t DIM_INDEX_H = 2;
  const size_t DIM_INDEX_W = 3;
  const size_t DIM_INDEX_D = 4;
  const size_t DIM_INDEX_ZERO = 0;
  const size_t DIM_INDEX_ONE = 1;
  const size_t DIM_INDEX_TWO = 2;
  const size_t DIM_INDEX_THREE = 3;
  const size_t DIM_INDEX_FOUR = 4;
  const size_t kM0Index = 0;
  const size_t kK0Index = 1;
  const size_t kN0Index = 2;
  const std::string kPltDtypeMKN = "DtypeMKN";
  const std::string kPltDefault = "Default";
  const std::map<ge::Format, FormatIndex> kFormatIndexMap = {
          {ge::FORMAT_NCHW, {DIM_INDEX_ZERO, DIM_INDEX_ONE, DIM_INDEX_TWO, DIM_INDEX_THREE, DIM_INDEX_FOUR}},
          {ge::FORMAT_NHWC, {DIM_INDEX_ZERO, DIM_INDEX_THREE, DIM_INDEX_ONE, DIM_INDEX_TWO, DIM_INDEX_FOUR}},
          {ge::FORMAT_HWCN, {DIM_INDEX_THREE, DIM_INDEX_TWO, DIM_INDEX_ZERO, DIM_INDEX_ONE, DIM_INDEX_FOUR}},
          {ge::FORMAT_CHWN, {DIM_INDEX_THREE, DIM_INDEX_ZERO, DIM_INDEX_ONE, DIM_INDEX_TWO, DIM_INDEX_FOUR}},
          {ge::FORMAT_ND, {DIM_INDEX_ZERO, DIM_INDEX_ONE, DIM_INDEX_TWO, DIM_INDEX_THREE, DIM_INDEX_FOUR}},
          {ge::FORMAT_NCDHW, {DIM_INDEX_ZERO, DIM_INDEX_ONE, DIM_INDEX_THREE, DIM_INDEX_FOUR, DIM_INDEX_TWO}},
          {ge::FORMAT_NDHWC, {DIM_INDEX_ZERO, DIM_INDEX_FOUR, DIM_INDEX_TWO, DIM_INDEX_THREE, DIM_INDEX_ONE}},
          {ge::FORMAT_DHWCN, {DIM_INDEX_FOUR, DIM_INDEX_THREE, DIM_INDEX_ONE, DIM_INDEX_TWO, DIM_INDEX_ZERO}},
          {ge::FORMAT_DHWNC, {DIM_INDEX_THREE, DIM_INDEX_FOUR, DIM_INDEX_ONE, DIM_INDEX_TWO, DIM_INDEX_ZERO}}
  };

  const std::set<ge::Format> kOriginFormatVec = {
          ge::FORMAT_NCHW,  ge::FORMAT_NHWC,  ge::FORMAT_HWCN,
          ge::FORMAT_CHWN,  ge::FORMAT_NDHWC, ge::FORMAT_NCDHW,
          ge::FORMAT_DHWCN, ge::FORMAT_DHWNC, ge::FORMAT_ND
  };

  inline int64_t GetGreatestCommonDivisor(int64_t x, int64_t y) {
    if (y == 0) {
      return x;
    }
    int64_t z = y;
    while (x % y != 0) {
      z = x % y;
      x = y;
      y = z;
    }
    return z;
  }

  inline int64_t GetLeastCommonMultiple(int64_t x, int64_t y) {
    if (x == 0 || y == 0) {
      return 0;
    }
    return (x * y) / GetGreatestCommonDivisor(x, y);
  }

  inline int64_t GetAsisEnlargeValue(int64_t cin, int64_t cout, int64_t c0, int64_t group) {
    if (cin == 0 || cout == 0) {
      return 0;
    }

    return std::min(GetLeastCommonMultiple(c0 / GetGreatestCommonDivisor(cin, c0),
                                           NI / GetGreatestCommonDivisor(cout, NI)), group);
  }
}

bool TransferShapeUtils::InitPlatformInfo() {
  static std::once_flag flag;
  std::call_once(flag, []() {
    m0_list_.fill(SHAPE_NUMBER_16);
    k0_list_.fill(SHAPE_NUMBER_16);
    n0_list_.fill(SHAPE_NUMBER_16);
    fe::PlatformInfoManager::GeInstance().InitializePlatformInfo();
    fe::PlatFormInfos platform_infos;
    fe::OptionalInfos optional_infos;
    if (fe::PlatformInfoManager::GeInstance().GetPlatformInfoWithOutSocVersion(platform_infos, optional_infos) != 0) {
      GELOGW("Get platform info failed, use default MKN value.");
      return false;
    }
    std::string default_mkn;
    if (platform_infos.GetPlatformResWithLock(kPltDtypeMKN, kPltDefault, default_mkn)) {
      GELOGD("Default MKN value from platform is [%s].", default_mkn.c_str());
      std::vector<string> infos = ge::StringUtils::Split(default_mkn, ',');
      if (infos.size() != 3) {
        return false;
      }
      m0_list_.fill(static_cast<uint32_t>(std::atoi(infos[kM0Index].c_str())));
      k0_list_.fill(static_cast<uint32_t>(std::atoi(infos[kK0Index].c_str())));
      n0_list_.fill(static_cast<uint32_t>(std::atoi(infos[kN0Index].c_str())));
    }

    std::map<std::string, std::string> m0_k0_n0_info;
    platform_infos.GetPlatformResWithLock(kPltDtypeMKN, m0_k0_n0_info);
    for (auto &item : m0_k0_n0_info) {
      if (item.first == kPltDefault) {
        continue;
      }
      ge::DataType dtype = ge::TypeUtils::SerialStringToDataType(item.first);
      if (dtype == ge::DT_UNDEFINED) {
        continue;
      }
      std::vector<string> infos = ge::StringUtils::Split(item.second, ',');
      if (infos.size() != 3) {
        continue;
      }
      m0_list_[static_cast<size_t>(dtype)] = static_cast<uint32_t>(std::atoi(infos[kM0Index].c_str()));
      k0_list_[static_cast<size_t>(dtype)] = static_cast<uint32_t>(std::atoi(infos[kK0Index].c_str()));
      n0_list_[static_cast<size_t>(dtype)] = static_cast<uint32_t>(std::atoi(infos[kN0Index].c_str()));
    }
    return true;
  });
  return true;
}

bool TransferShapeUtils::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                       const ge::DataType &data_type, const ExtAxisValue &ext_axis,
                                       gert::Shape &shape) {
  if (!InitPlatformInfo()) {
    GELOGW("Init platform info failed");
  }
  GELOGD("Original format is %u, new format %u", origin_format, format);
  ge::Format primary_format = static_cast<ge::Format>(GetPrimaryFormat(format));
  ge::Format origin_primary_format = static_cast<ge::Format>(GetPrimaryFormat(origin_format));
  if (!IsNeedTransferShape(origin_primary_format, primary_format, shape)) {
    return true;
  }

  if (!CheckInputParam(origin_primary_format, primary_format, data_type)) {
    return false;
  }

  AxisValue axis_value;
  axis_value.fill(1);
  int64_t group = static_cast<int64_t>(ge::GetSubFormat(format));
  if (group > GROUPS_DEFAULT_VALUE) {
    axis_value[AXIS_G] = group;
  }

  axis_value[AXIS_C0] = GetC0Value(data_type, format);
  axis_value[AXIS_M0] = GetM0ByDtype(data_type);
  if (primary_format == ge::FORMAT_FRACTAL_ZN_RNN || primary_format == ge::FORMAT_ND_RNN_BIAS) {
    axis_value[AXIS_INPUT_SIZE] = ext_axis[EXT_INDEX_INPUT_SIZE];
    axis_value[AXIS_HIDEEN_SIZE] = ext_axis[EXT_INDEX_HIDEEN_SIZE];
    axis_value[AXIS_STATE_SIZE] = ext_axis[EXT_INDEX_STATE_SIZE];
  }

  if (!IsNeedAxisValue(primary_format, shape.GetDimNum())) {
    return TransferShapeByFormat(primary_format, axis_value, shape);
  }

  if (!AxisUtil::GetAxisValueByOriginFormat(origin_primary_format, shape, axis_value)) {
    return true;
  }

  return TransferShapeByAxisValue(primary_format, axis_value, shape);
}

bool TransferShapeUtils::TransferShape(const ge::Format &origin_format, const ge::Format &format,
                                       const ge::DataType &data_type, const ExtAxisValue &ext_axis,
                                       const gert::Shape &origin_shape, gert::Shape &shape) {
  if (!InitPlatformInfo()) {
    GELOGW("Init platform info failed");
  }
  GELOGD("Tranfer shape from original format[%d] to format [%d].", origin_format, format);
  ge::Format primary_format = static_cast<ge::Format>(GetPrimaryFormat(format));
  ge::Format origin_primary_format = static_cast<ge::Format>(GetPrimaryFormat(origin_format));
  if (!IsNeedTransferShape(origin_primary_format, primary_format, origin_shape)) {
    return true;
  }

  if (!CheckInputParam(origin_primary_format, primary_format, data_type)) {
    return false;
  }

  int64_t c0 = GetC0Value(data_type, format);
  int64_t m0 = GetM0ByDtype(data_type);
  if (!IsNeedAxisValue(primary_format, origin_shape.GetDimNum())) {
    return TransferShapeByOriginShape(primary_format, c0, m0, ext_axis, origin_shape, shape);
  } else {
    return TransferShapeByFormatIndex(origin_primary_format, format, c0, origin_shape, shape);
  }
}

int64_t TransferShapeUtils::GetC0ByDtype(const ge::DataType &data_type) {
  if (static_cast<size_t>(data_type) < k0_list_.size()) {
    return static_cast<int64_t>(k0_list_[static_cast<size_t>(data_type)]);
  }
  return SHAPE_NUMBER_16;
}

int64_t TransferShapeUtils::GetM0ByDtype(const ge::DataType &data_type) {
  if (static_cast<size_t>(data_type) < m0_list_.size()) {
    return static_cast<int64_t>(m0_list_[static_cast<size_t>(data_type)]);
  }
  return SHAPE_NUMBER_16;
}

bool TransferShapeUtils::IsNeedTransferShape(const ge::Format &origin_format, const ge::Format &format,
                                             const gert::Shape &shape) {
  if (origin_format == ge::FORMAT_ND && kOriginFormatVec.count(format) > 0) {
    GELOGD("Do not need to do shape transformation from ND to original format.");
    return false;
  }

  if (shape.IsScalar()) {
    GELOGD("Do not need to do shape transformation if the shape is scalar.");
    return false;
  }
  return true;
}

bool TransferShapeUtils::CheckInputParam(const ge::Format &origin_format, const ge::Format &primary_format,
                                         const ge::DataType &data_type) {
  bool invalid_format = (origin_format == ge::FORMAT_RESERVED || origin_format >= ge::FORMAT_END) ||
                        (primary_format == ge::FORMAT_RESERVED || primary_format >= ge::FORMAT_END);
  if (invalid_format) {
    GELOGE(ge::GRAPH_FAILED, "Origin format %u or new format %u is invalid.", origin_format, primary_format);
    return false;
  }

  if (data_type == ge::DT_UNDEFINED || data_type >= ge::DT_MAX) {
    GELOGE(ge::GRAPH_FAILED, "DataType %u is invalid.", origin_format);
    return false;
  }

  return true;
}

int64_t TransferShapeUtils::GetC0Value(const ge::DataType &data_type, const ge::Format &format) {
  // The value of C0 should be 4 while format is 5HD-4 or FRAZ-4
  ge::Format primary_format = static_cast<ge::Format>(GetPrimaryFormat(format));
  if (primary_format == ge::FORMAT_NC1HWC0_C04) {
    return SHAPE_NUMBER_4;
  }

  if (ge::HasC0Format(format)) {
    return ge::GetC0Value(format);
  }

  return GetC0ByDtype(data_type);
}

bool TransferShapeUtils::IsNeedAxisValue(const ge::Format &format, const size_t &origin_dim_size) {
  if (format == ge::FORMAT_FRACTAL_NZ || format == ge::FORMAT_FRACTAL_ZN_RNN ||
      format == ge::FORMAT_ND_RNN_BIAS || format == ge::FORMAT_NYUV_A) {
    return false;
  }
  if (format == ge::FORMAT_FRACTAL_Z && origin_dim_size == DIM_SIZE_TWO) {
    return false;
  }
  return true;
}

bool TransferShapeUtils::TransferShapeByFormat(const ge::Format &primary_format, const AxisValue &axis_value,
                                               gert::Shape &shape) {
  switch (primary_format) {
    case ge::FORMAT_FRACTAL_Z:
      return GetFzShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_FRACTAL_NZ:
      return GetNzShapeByAxisValue(axis_value, shape); // need c0
    case ge::FORMAT_FRACTAL_ZN_RNN:
      return GetFznRNNShapeByAxisValue(axis_value, shape); // need c0, input, hidden, state
    case ge::FORMAT_ND_RNN_BIAS:
      return GetNDRNNShapeByAxisValue(axis_value, shape); // need c0, input, hidden, state
    case ge::FORMAT_NYUV_A:
      return GetNYUVShape(shape);
    default:
      GELOGD("Can not get new shape by new format %d.", primary_format);
      return true;
  }
}

bool TransferShapeUtils::TransferShapeByAxisValue(const ge::Format &primary_format, const AxisValue &axis_value,
                                                  gert::Shape &shape) {
  switch (primary_format) {
    case ge::FORMAT_NCHW:
      return GetNCHWShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_NHWC:
      return GetNHWCShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_HWCN:
      return GetHWCNShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_CHWN:
      return GetCHWNShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_NC1HWC0:
    case ge::FORMAT_NC1HWC0_C04:
      return GetNC1HWC0ShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_NDC1HWC0:
      return GetNDC1HWC0ShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_C1HWNCoC0:
      return GetC1HWNCoC0ShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_FRACTAL_Z:
      return GetFzShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_FRACTAL_Z_C04:
      return GetFzC04ShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_FRACTAL_Z_3D:
      return GetFz3DShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE:
      return GetFz3DTransposeShapeByAxisValue(axis_value, shape);
    case ge::FORMAT_FRACTAL_ZN_LSTM:
      return GetFzLstmShapeByAxisValue(axis_value, shape);
    default:
      GELOGD("Can not get new shape by new format %d.", primary_format);
      return true;
  }
}

bool TransferShapeUtils::TransferShapeByOriginShape(const ge::Format &primary_format,
                                                    const int64_t &c0, const int64_t &m0, const ExtAxisValue &ext_axis,
                                                    const gert::Shape &origin_shape, gert::Shape &shape) {
  switch (primary_format) {
    case ge::FORMAT_FRACTAL_Z:
      return GetFractalZShape(c0, origin_shape, shape);
    case ge::FORMAT_FRACTAL_NZ:
      return GetFractalNzShape(c0, m0, origin_shape, shape); // need c0
    case ge::FORMAT_FRACTAL_ZN_RNN:
      return GetFractalZnRnnShape(ext_axis, c0, origin_shape, shape); // need c0, input, hidden, state
    case ge::FORMAT_ND_RNN_BIAS:
      return GetNdRnnBiasShape(ext_axis, c0, origin_shape, shape); // need c0, input, hidden, state
    default:
      GELOGD("Can not get new shape by new format %d.", primary_format);
      return true;
  }
}

bool TransferShapeUtils::TransferShapeByFormatIndex(const ge::Format &origin_format, const ge::Format &format,
                                                    const int64_t &c0, const gert::Shape &origin_shape,
                                                    gert::Shape &shape) {
  std::map<ge::Format, FormatIndex>::const_iterator iter = kFormatIndexMap.find(origin_format);
  if (iter == kFormatIndexMap.end()) {
    return true;
  }
  ge::Format primary_format = static_cast<ge::Format>(GetPrimaryFormat(format));
  int64_t group = static_cast<int64_t>(ge::GetSubFormat(format));
  switch (primary_format) {
    case ge::FORMAT_NCHW:
      return GetNCHWShape(iter->second, origin_shape, shape);
    case ge::FORMAT_NHWC:
      return GetNHWCShape(iter->second, origin_shape, shape);
    case ge::FORMAT_HWCN:
      return GetHWCNShape(iter->second, origin_shape, shape);
    case ge::FORMAT_CHWN:
      return GetCHWNShape(iter->second, origin_shape, shape);
    case ge::FORMAT_NC1HWC0:
    case ge::FORMAT_NC1HWC0_C04:
      return GetNC1HWC0Shape(iter->second, c0, origin_shape, shape);
    case ge::FORMAT_NDC1HWC0:
      return GetNDC1HWC0Shape(iter->second, c0, origin_shape, shape);
    case ge::FORMAT_C1HWNCoC0:
      return GetC1HWNCoC0Shape(iter->second, c0, origin_shape, shape);
    case ge::FORMAT_FRACTAL_Z:
      return GetFractalZShape(iter->second, c0, group, origin_shape, shape);
    case ge::FORMAT_FRACTAL_Z_3D:
      return GetFractalZ3DShape(iter->second, c0, group, origin_shape, shape);
    case ge::FORMAT_FRACTAL_Z_C04:
      return GetFractalZC04Shape(iter->second, c0, origin_shape, shape);
    case ge::FORMAT_FRACTAL_Z_3D_TRANSPOSE:
      return GetFractalZ3DTransposeShape(iter->second, c0, origin_shape, shape);
    case ge::FORMAT_FRACTAL_ZN_LSTM:
      return GetFractalZLstmShape(iter->second, origin_shape, shape);
    default:
      GELOGD("Can not get new shape by new format %d.", primary_format);
      return true;
  }
}

bool TransferShapeUtils::GetNCHWShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_N]);
  shape.AppendDim(axis_value[AXIS_C]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  return true;
}

bool TransferShapeUtils::GetNHWCShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_N]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(axis_value[AXIS_C]);
  return true;
}

bool TransferShapeUtils::GetHWCNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(axis_value[AXIS_C]);
  shape.AppendDim(axis_value[AXIS_N]);
  return true;
}

bool TransferShapeUtils::GetCHWNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_C]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(axis_value[AXIS_N]);
  return true;
}

bool TransferShapeUtils::GetNC1HWC0ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_N]);
  shape.AppendDim(axis_value[AXIS_C1]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetNDC1HWC0ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_N]);
  shape.AppendDim(axis_value[AXIS_D]);
  shape.AppendDim(axis_value[AXIS_C1]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetC1HWNCoC0ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  shape.AppendDim(axis_value[AXIS_C1]);
  shape.AppendDim(axis_value[AXIS_H]);
  shape.AppendDim(axis_value[AXIS_W]);
  shape.AppendDim(DivisionCeiling(axis_value[AXIS_N], NI));
  shape.AppendDim(axis_value[AXIS_Co]);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetNzShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  CHECK(shape.IsScalar(), GELOGD("Origin shape is empty."), return true);
  size_t dim_size = shape.GetDimNum();
  if (dim_size < DIM_SIZE_TWO) {
    GELOGD("nd_value's dim num is less than 2.");
    shape.AppendDim(1);
    dim_size++;
  }
  /* dim_size - 1 mean the last value of original vec
   * dim_size - 2 mean the second last value of original vec */
  int64_t dim_back_two = shape.GetDim(dim_size - MINUS_VALUE_TWO);
  int64_t dim_back_one = shape.GetDim(dim_size - MINUS_VALUE_ONE);
  shape.SetDim((dim_size - MINUS_VALUE_ONE), DivisionCeiling(dim_back_two, axis_value[AXIS_M0]));
  shape.SetDim((dim_size - MINUS_VALUE_TWO), DivisionCeiling(dim_back_one, axis_value[AXIS_C0]));
  shape.AppendDim(axis_value[AXIS_M0]);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetFzShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  size_t size_of_original_vec = shape.GetDimNum();
  if (size_of_original_vec == DIM_SIZE_TWO) {
    /* size_of_original_vec - 1 mean the last value of original vec
     * size_of_original_vec - 2 mean the second last value of original vec */
    shape.SetDim((size_of_original_vec - MINUS_VALUE_ONE),
                 DivisionCeiling(shape.GetDim(size_of_original_vec - MINUS_VALUE_ONE), SHAPE_NUMBER_16));
    shape.SetDim((size_of_original_vec - MINUS_VALUE_TWO),
                 DivisionCeiling(shape.GetDim(size_of_original_vec - MINUS_VALUE_TWO), axis_value[AXIS_C0]));
    shape.AppendDim(SHAPE_NUMBER_16);
    shape.AppendDim(axis_value[AXIS_C0]);
    return true;
  }
  return GetFz3DShapeByAxisValue(axis_value, shape);
}

bool TransferShapeUtils::GetFz3DShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  bool has_unknown_shape = axis_value[AXIS_D] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_H] == UNKNOWN_SHAPE_VALUE ||
                           axis_value[AXIS_W] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_C] == UNKNOWN_SHAPE_VALUE;
  int64_t gdhwc1 = UNKNOWN_SHAPE_VALUE;
  int64_t axis_g_val = GROUPS_DEFAULT_VALUE;
  int64_t axis_n_val = axis_value[AXIS_N];
  int64_t axis_c_val = axis_value[AXIS_C];
  int64_t axis_c1_val = axis_value[AXIS_C1];
  if (!has_unknown_shape) {
    if (axis_value[AXIS_G] > GROUPS_DEFAULT_VALUE && axis_n_val >= axis_value[AXIS_G]) {
      axis_n_val = axis_n_val / axis_value[AXIS_G];
      int64_t enlarge_value = GetAsisEnlargeValue(axis_c_val, axis_n_val, axis_value[AXIS_C0], axis_value[AXIS_G]);
      axis_g_val = DivisionCeiling(axis_value[AXIS_G], enlarge_value);
      MUL_OVERFLOW(axis_c_val, enlarge_value, axis_c_val);
      MUL_OVERFLOW(axis_n_val, enlarge_value, axis_n_val);
      axis_c1_val = DivisionCeiling(axis_c_val, axis_value[AXIS_C0]);
    }
    MUL_OVERFLOW(axis_g_val, axis_c1_val, gdhwc1);
    MUL_OVERFLOW(gdhwc1, axis_value[AXIS_D], gdhwc1);
    MUL_OVERFLOW(gdhwc1, axis_value[AXIS_H], gdhwc1);
    MUL_OVERFLOW(gdhwc1, axis_value[AXIS_W], gdhwc1);
  }
  shape.SetDimNum(0);
  shape.AppendDim(gdhwc1);
  shape.AppendDim(DivisionCeiling(axis_n_val, NI));
  shape.AppendDim(NI);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetFz3DTransposeShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  int64_t dhwn1 = UNKNOWN_SHAPE_VALUE;
  if (axis_value[AXIS_N] != UNKNOWN_SHAPE_VALUE && axis_value[AXIS_H] != UNKNOWN_SHAPE_VALUE &&
      axis_value[AXIS_W] != UNKNOWN_SHAPE_VALUE && axis_value[AXIS_D] != UNKNOWN_SHAPE_VALUE) {
    dhwn1 = DivisionCeiling(axis_value[AXIS_N], NI);
    MUL_OVERFLOW(dhwn1, axis_value[AXIS_D], dhwn1);
    MUL_OVERFLOW(dhwn1, axis_value[AXIS_H], dhwn1);
    MUL_OVERFLOW(dhwn1, axis_value[AXIS_W], dhwn1);
  }

  shape.SetDimNum(0);
  shape.AppendDim(dhwn1);
  if (axis_value[AXIS_C] == UNKNOWN_SHAPE_VALUE) {
    shape.AppendDim(UNKNOWN_SHAPE_VALUE);
  } else {
    shape.AppendDim(axis_value[AXIS_C1]);
  }
  shape.AppendDim(NI);
  shape.AppendDim(axis_value[AXIS_C0]);

  return true;
}

bool TransferShapeUtils::GetFzLstmShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  int64_t h = axis_value[AXIS_N] >> NUMBER_2;
  int64_t i = axis_value[AXIS_C] - h;
  int64_t first_element_of_fz_lstm = DivisionCeiling(i, NI) + DivisionCeiling(h, NI);
  int64_t second_element_of_fz_lstm = DivisionCeiling(h, NI);
  MUL_OVERFLOW(second_element_of_fz_lstm, LSTM_NI, second_element_of_fz_lstm);
  if (axis_value[AXIS_N] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_C] == UNKNOWN_SHAPE_VALUE) {
    first_element_of_fz_lstm = UNKNOWN_SHAPE_VALUE;
    second_element_of_fz_lstm = UNKNOWN_SHAPE_VALUE;
  }
  shape.SetDimNum(0);
  shape.AppendDim(first_element_of_fz_lstm);
  shape.AppendDim(second_element_of_fz_lstm);
  shape.AppendDim(NI);
  shape.AppendDim(NI);
  return true;
}

bool TransferShapeUtils::GetFzC04ShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  shape.SetDimNum(0);
  if (axis_value[AXIS_H] == UNKNOWN_SHAPE_VALUE || axis_value[AXIS_W] == UNKNOWN_SHAPE_VALUE) {
    shape.AppendDim(UNKNOWN_SHAPE_VALUE);
  } else {
    int64_t x = SHAPE_NUMBER_4;
    MUL_OVERFLOW(x, axis_value[AXIS_H], x);
    MUL_OVERFLOW(x, axis_value[AXIS_W], x);
    shape.AppendDim(DivisionCeiling(x, axis_value[AXIS_C0]));
  }
  shape.AppendDim(DivisionCeiling(axis_value[AXIS_N], NI));
  shape.AppendDim(NI);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetFznRNNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  size_t origin_shape_size = shape.GetDimNum();
  CHECK(origin_shape_size < DIM_SIZE_TWO, GELOGW("ndValue's dim num is less than 2!"), return true);
  /* check nd shape value */
  int64_t k_value = shape.GetDim(origin_shape_size - MINUS_VALUE_TWO);
  int64_t hidden_or_state_size = axis_value[AXIS_HIDEEN_SIZE];
  if (axis_value[AXIS_STATE_SIZE] != RNN_STATE_SIZE_DEFAULT_VALUE) {
    hidden_or_state_size = axis_value[AXIS_STATE_SIZE];
  }

  if (k_value == hidden_or_state_size + axis_value[AXIS_INPUT_SIZE]) {
    // use input size and hidden size
    shape.SetDim(origin_shape_size - MINUS_VALUE_TWO,
                 DivisionCeiling(axis_value[AXIS_INPUT_SIZE], SHAPE_NUMBER_16) +
                 DivisionCeiling(hidden_or_state_size, SHAPE_NUMBER_16));
  } else if (k_value == hidden_or_state_size || k_value == axis_value[AXIS_INPUT_SIZE]) {
    // only use hidden size or input size
    shape.SetDim(origin_shape_size - MINUS_VALUE_TWO, DivisionCeiling(k_value, SHAPE_NUMBER_16));
  } else {
    return true;
  }

  int64_t n_value = shape.GetDim(origin_shape_size - MINUS_VALUE_ONE);
  INT64_ZEROCHECK(axis_value[AXIS_HIDEEN_SIZE]);
  int64_t n_num = n_value / axis_value[AXIS_HIDEEN_SIZE];
  MUL_OVERFLOW(n_num, DivisionCeiling(axis_value[AXIS_HIDEEN_SIZE], axis_value[AXIS_C0]), n_num);
  shape.SetDim(origin_shape_size - MINUS_VALUE_ONE, n_num);
  shape.AppendDim(SHAPE_NUMBER_16);
  shape.AppendDim(axis_value[AXIS_C0]);
  return true;
}

bool TransferShapeUtils::GetNDRNNShapeByAxisValue(const AxisValue &axis_value, gert::Shape &shape) {
  CHECK(axis_value[AXIS_HIDEEN_SIZE] == 0, GELOGD("hidden_size is zero"), return true);
  size_t size_of_original_vec = shape.GetDimNum();
  /* check nd shape value */
  int64_t n_num = shape.GetDim(size_of_original_vec - MINUS_VALUE_ONE) / axis_value[AXIS_HIDEEN_SIZE];
  MUL_OVERFLOW(n_num, DivisionCeiling(axis_value[AXIS_HIDEEN_SIZE], axis_value[AXIS_C0]), n_num);
  MUL_OVERFLOW(n_num, axis_value[AXIS_C0], n_num);
  shape.SetDim(size_of_original_vec - MINUS_VALUE_ONE, n_num);
  return true;
}

bool TransferShapeUtils::GetNCHWShape(const FormatIndex& format_index, const gert::Shape &origin_shape,
                                      gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_C]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  return true;
}

bool TransferShapeUtils::GetNHWCShape(const FormatIndex& format_index, const gert::Shape &origin_shape,
                                      gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_C]));
  return true;
}

bool TransferShapeUtils::GetHWCNShape(const FormatIndex& format_index, const gert::Shape &origin_shape,
                                      gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_C]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  return true;
}

bool TransferShapeUtils::GetCHWNShape(const FormatIndex& format_index, const gert::Shape &origin_shape,
                                      gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_C]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  return true;
}

bool TransferShapeUtils::GetNC1HWC0Shape(const FormatIndex& format_index, const int64_t &c0,
                                         const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_C]), c0));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetNDC1HWC0Shape(const FormatIndex& format_index, const int64_t &c0,
                                          const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_N]));
  if (origin_shape.GetDimNum() == DIM_SIZE_FOUR) {
    shape.AppendDim(1);
  } else {
    shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_D]));
  }
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_C]), c0));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetC1HWNCoC0Shape(const FormatIndex& format_index, const int64_t &c0,
                                           const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  shape.SetDimNum(0);
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_C]), c0));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_H]));
  shape.AppendDim(origin_shape.GetDim(format_index[DIM_INDEX_W]));
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_N]), NI));
  shape.AppendDim(c0);
  shape.AppendDim(c0);
  return true;
}
bool TransferShapeUtils::GetFractalNzShape(const int64_t &c0, const int64_t &m0,
                                           const gert::Shape &origin_shape, gert::Shape &shape) {
  size_t dim_size = origin_shape.GetDimNum();
  shape.SetDimNum(0);
  if (dim_size > DIM_SIZE_TWO) {
    for (size_t i = 0; i < dim_size - DIM_SIZE_TWO; i++) {
      shape.AppendDim(origin_shape.GetDim(i));
    }
  }

  /* dim_size - 1 mean the last value of original vec
   * dim_size - 2 mean the second last value of original vec */
  if (dim_size < DIM_SIZE_TWO) {
    shape.AppendDim(1);
    shape.AppendDim(DivisionCeiling(origin_shape.GetDim(dim_size - MINUS_VALUE_ONE), m0));
  } else {
    shape.AppendDim(DivisionCeiling(origin_shape.GetDim(dim_size - MINUS_VALUE_ONE), c0));
    shape.AppendDim(DivisionCeiling(origin_shape.GetDim(dim_size - MINUS_VALUE_TWO), m0));
  }
  shape.AppendDim(m0);
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetFractalZShape(const int64_t &c0, const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() != DIM_SIZE_TWO, GELOGD("Dim size is not 2."), return true);
  /* size_of_original_vec - 1 mean the last value of original vec
   * size_of_original_vec - 2 mean the second last value of original vec */
  shape.SetDimNum(0);
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(DIM_SIZE_TWO - MINUS_VALUE_TWO), c0));
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(DIM_SIZE_TWO - MINUS_VALUE_ONE), SHAPE_NUMBER_16));
  shape.AppendDim(SHAPE_NUMBER_16);
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetFractalZShape(const FormatIndex& format_index, const int64_t &c0, const int64_t &group,
                                          const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  int64_t axis_n_val = origin_shape.GetDim(format_index[DIM_INDEX_N]);
  int64_t axis_c_val = origin_shape.GetDim(format_index[DIM_INDEX_C]);
  int64_t axis_h_val = origin_shape.GetDim(format_index[DIM_INDEX_H]);
  int64_t axis_w_val = origin_shape.GetDim(format_index[DIM_INDEX_W]);
  int64_t ghwc1 = UNKNOWN_SHAPE_VALUE;
  bool is_unknown_shape = axis_c_val == UNKNOWN_SHAPE_VALUE || axis_h_val == UNKNOWN_SHAPE_VALUE ||
                          axis_w_val == UNKNOWN_SHAPE_VALUE;
  if (!is_unknown_shape) {
    int64_t axis_g_val = GROUPS_DEFAULT_VALUE;
    int64_t axis_c1_val = 0;
    if (group > GROUPS_DEFAULT_VALUE && axis_n_val >= group) {
      axis_n_val = axis_n_val / group;
      int64_t enlarge_value = GetAsisEnlargeValue(axis_c_val, axis_n_val, c0, group);
      axis_g_val = DivisionCeiling(group, enlarge_value);
      MUL_OVERFLOW(axis_c_val, enlarge_value, axis_c_val);
      MUL_OVERFLOW(axis_n_val, enlarge_value, axis_n_val);
      axis_c1_val = DivisionCeiling(axis_c_val, c0);
    } else {
      axis_c1_val = DivisionCeiling(axis_c_val, c0);
    }

    MUL_OVERFLOW(axis_g_val, axis_c1_val, ghwc1);
    MUL_OVERFLOW(ghwc1, origin_shape.GetDim(format_index[DIM_INDEX_H]), ghwc1);
    MUL_OVERFLOW(ghwc1, origin_shape.GetDim(format_index[DIM_INDEX_W]), ghwc1);
  }

  shape.SetDimNum(0);
  shape.AppendDim(ghwc1);
  shape.AppendDim(DivisionCeiling(axis_n_val, NI));
  shape.AppendDim(NI);
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetFractalZ3DShape(const FormatIndex& format_index, const int64_t &c0, const int64_t &group,
                                            const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FIVE, GELOGD("Dim size is less than 5."), return true);
  int64_t axis_n_val = origin_shape.GetDim(format_index[DIM_INDEX_N]);
  int64_t axis_c_val = origin_shape.GetDim(format_index[DIM_INDEX_C]);
  int64_t axis_h_val = origin_shape.GetDim(format_index[DIM_INDEX_H]);
  int64_t axis_w_val = origin_shape.GetDim(format_index[DIM_INDEX_W]);
  int64_t axis_d_val = origin_shape.GetDim(format_index[DIM_INDEX_D]);
  int64_t gdhwc1 = UNKNOWN_SHAPE_VALUE;
  bool is_unknown_shape = axis_c_val == UNKNOWN_SHAPE_VALUE || axis_d_val == UNKNOWN_SHAPE_VALUE ||
                          axis_h_val == UNKNOWN_SHAPE_VALUE || axis_w_val == UNKNOWN_SHAPE_VALUE;
  if (!is_unknown_shape) {
    int64_t axis_c1_val = 0;
    int64_t axis_g_val = GROUPS_DEFAULT_VALUE;
    if (group > GROUPS_DEFAULT_VALUE && axis_n_val >= group) {
      axis_n_val = axis_n_val / group;
      int64_t enlarge_value = GetAsisEnlargeValue(axis_c_val, axis_n_val, c0, group);
      axis_g_val = DivisionCeiling(group, enlarge_value);
      MUL_OVERFLOW(axis_c_val, enlarge_value, axis_c_val);
      MUL_OVERFLOW(axis_n_val, enlarge_value, axis_n_val);
      axis_c1_val = DivisionCeiling(axis_c_val, c0);
    } else {
      axis_c1_val = DivisionCeiling(axis_c_val, c0);
    }
    MUL_OVERFLOW(axis_g_val, axis_c1_val, gdhwc1);
    MUL_OVERFLOW(gdhwc1, origin_shape.GetDim(format_index[DIM_INDEX_D]), gdhwc1);
    MUL_OVERFLOW(gdhwc1, origin_shape.GetDim(format_index[DIM_INDEX_H]), gdhwc1);
    MUL_OVERFLOW(gdhwc1, origin_shape.GetDim(format_index[DIM_INDEX_W]), gdhwc1);
  }

  shape.SetDimNum(0);
  shape.AppendDim(gdhwc1);
  shape.AppendDim(DivisionCeiling(axis_n_val, NI));
  shape.AppendDim(NI);
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetFractalZ3DTransposeShape(const FormatIndex& format_index, const int64_t &c0,
                                                     const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FIVE, GELOGD("Dim size is less than 5."), return true);
  int64_t dhwn1 = UNKNOWN_SHAPE_VALUE;
  if (origin_shape.GetDim(format_index[DIM_INDEX_N]) != UNKNOWN_SHAPE_VALUE &&
      origin_shape.GetDim(format_index[DIM_INDEX_H]) != UNKNOWN_SHAPE_VALUE &&
      origin_shape.GetDim(format_index[DIM_INDEX_W]) != UNKNOWN_SHAPE_VALUE &&
      origin_shape.GetDim(format_index[DIM_INDEX_D]) != UNKNOWN_SHAPE_VALUE) {
    dhwn1 = DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_N]), NI);
    MUL_OVERFLOW(dhwn1, origin_shape.GetDim(format_index[DIM_INDEX_D]), dhwn1);
    MUL_OVERFLOW(dhwn1, origin_shape.GetDim(format_index[DIM_INDEX_H]), dhwn1);
    MUL_OVERFLOW(dhwn1, origin_shape.GetDim(format_index[DIM_INDEX_W]), dhwn1);
  }

  shape.SetDimNum(0);
  shape.AppendDim(dhwn1);
  if (origin_shape.GetDim(format_index[DIM_INDEX_C]) == UNKNOWN_SHAPE_VALUE) {
    shape.AppendDim(UNKNOWN_SHAPE_VALUE);
  } else {
    shape.AppendDim(DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_C]), c0));
  }
  shape.AppendDim(NI);
  shape.AppendDim(c0);

  return true;
}

bool TransferShapeUtils::GetFractalZLstmShape(const FormatIndex& format_index, const gert::Shape &origin_shape,
                                              gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  int64_t axis_n_val = origin_shape.GetDim(format_index[DIM_INDEX_N]);
  int64_t axis_c_val = origin_shape.GetDim(format_index[DIM_INDEX_C]);
  int64_t h = axis_n_val / LSTM_NI;
  int64_t i = axis_c_val - h;
  int64_t first_element_of_fz_lstm = DivisionCeiling(i, NI) + DivisionCeiling(h, NI);
  int64_t second_element_of_fz_lstm = DivisionCeiling(h, NI);
  MUL_OVERFLOW(second_element_of_fz_lstm, LSTM_NI, second_element_of_fz_lstm);
  if (axis_n_val == UNKNOWN_SHAPE_VALUE || axis_c_val == UNKNOWN_SHAPE_VALUE) {
    first_element_of_fz_lstm = UNKNOWN_SHAPE_VALUE;
    second_element_of_fz_lstm = UNKNOWN_SHAPE_VALUE;
  }
  shape.SetDimNum(0);
  shape.AppendDim(first_element_of_fz_lstm);
  shape.AppendDim(second_element_of_fz_lstm);
  shape.AppendDim(NI);
  shape.AppendDim(NI);
  return true;
}

bool TransferShapeUtils::GetFractalZC04Shape(const FormatIndex& format_index, const int64_t &c0,
                                             const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(origin_shape.GetDimNum() < DIM_SIZE_FOUR, GELOGD("Dim size is less than 4."), return true);
  int64_t axis_h_val = origin_shape.GetDim(format_index[DIM_INDEX_H]);
  int64_t axis_w_val = origin_shape.GetDim(format_index[DIM_INDEX_W]);
  shape.SetDimNum(0);
  if (axis_h_val == UNKNOWN_SHAPE_VALUE || axis_w_val == UNKNOWN_SHAPE_VALUE) {
    shape.AppendDim(UNKNOWN_SHAPE_VALUE);
  } else {
    int64_t x = SHAPE_NUMBER_4;
    MUL_OVERFLOW(x, origin_shape.GetDim(format_index[DIM_INDEX_H]), x);
    MUL_OVERFLOW(x, origin_shape.GetDim(format_index[DIM_INDEX_W]), x);
    shape.AppendDim(DivisionCeiling(x, c0));
  }
  shape.AppendDim(DivisionCeiling(origin_shape.GetDim(format_index[DIM_INDEX_N]), NI));
  shape.AppendDim(NI);
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetFractalZnRnnShape(const ExtAxisValue &ext_axis, const int64_t &c0,
                                              const gert::Shape &origin_shape, gert::Shape &shape) {
  size_t origin_shape_size = origin_shape.GetDimNum();
  CHECK(origin_shape_size < DIM_SIZE_TWO, GELOGD("Dim size is less than 2."), return true);
  shape.SetDimNum(0);
  for (size_t i = 0; i < origin_shape_size - DIM_SIZE_TWO; i++) {
    shape.AppendDim(origin_shape.GetDim(i));
  }
  /* check nd shape value */
  int64_t k_value = origin_shape.GetDim(origin_shape_size - MINUS_VALUE_TWO);
  int64_t hidden_or_state_size = ext_axis[EXT_INDEX_HIDEEN_SIZE];
  if (ext_axis[EXT_INDEX_STATE_SIZE] != RNN_STATE_SIZE_DEFAULT_VALUE) {
    hidden_or_state_size = ext_axis[EXT_INDEX_STATE_SIZE];
  }
  if (k_value == hidden_or_state_size + ext_axis[EXT_INDEX_INPUT_SIZE]) {
    // use input size and hidden size
    shape.AppendDim(DivisionCeiling(ext_axis[EXT_INDEX_INPUT_SIZE], SHAPE_NUMBER_16) +
                    DivisionCeiling(hidden_or_state_size, SHAPE_NUMBER_16));
  } else if (k_value == hidden_or_state_size || k_value == ext_axis[EXT_INDEX_INPUT_SIZE]) {
    // only use hidden size or input size
    shape.AppendDim(DivisionCeiling(k_value, SHAPE_NUMBER_16));
  } else {
    return true;
  }

  int64_t n_value = origin_shape.GetDim(origin_shape_size - MINUS_VALUE_ONE);
  INT64_ZEROCHECK(ext_axis[EXT_INDEX_HIDEEN_SIZE]);
  int64_t n_num = n_value / ext_axis[EXT_INDEX_HIDEEN_SIZE];
  MUL_OVERFLOW(n_num, DivisionCeiling(ext_axis[EXT_INDEX_HIDEEN_SIZE], c0), n_num);
  shape.AppendDim(n_num);
  shape.AppendDim(SHAPE_NUMBER_16);
  shape.AppendDim(c0);
  return true;
}

bool TransferShapeUtils::GetNdRnnBiasShape(const ExtAxisValue &ext_axis, const int64_t &c0,
                                           const gert::Shape &origin_shape, gert::Shape &shape) {
  CHECK(ext_axis[EXT_INDEX_HIDEEN_SIZE] == 0, GELOGD("hidden_size is zero"), return true);
  size_t size_of_original_vec = origin_shape.GetDimNum();
  shape.SetDimNum(0);
  for (size_t i = 0; i < size_of_original_vec - MINUS_VALUE_ONE; i++) {
    shape.AppendDim(origin_shape.GetDim(i));
  }
  /* check nd shape value */
  int64_t n_num = origin_shape.GetDim(size_of_original_vec - MINUS_VALUE_ONE) / ext_axis[EXT_INDEX_HIDEEN_SIZE];
  MUL_OVERFLOW(n_num, DivisionCeiling(ext_axis[EXT_INDEX_HIDEEN_SIZE], c0), n_num);
  MUL_OVERFLOW(n_num, c0, n_num);
  shape.AppendDim(n_num);
  return true;
}

bool TransferShapeUtils::GetNYUVShape(gert::Shape &shape) {
  const size_t kSize4 = 4U;
  const size_t kSize3 = 3U;
  size_t shape_size = shape.GetDimNum();
  CHECK(((shape_size != kSize3) && (shape_size != kSize4)),
        GELOGD("Dim size is not 3 or 4."), return false);
  const size_t kWdimOffset = 2U;
  const size_t kHdimOffset = 3U;
  const int64_t kAlaign64 = 64;
  const int64_t kAlaign16 = 16;
  auto width  = shape.GetDim(shape_size - kWdimOffset);
  auto height = shape.GetDim(shape_size - kHdimOffset);
  width  = (width + kAlaign64 - 1) / kAlaign64 * kAlaign64;
  height = (height + kAlaign16 - 1) / kAlaign16 * kAlaign16;
  shape.SetDim(shape_size - kWdimOffset, width);
  shape.SetDim(shape_size - kHdimOffset, height);
  return true;
}
}
