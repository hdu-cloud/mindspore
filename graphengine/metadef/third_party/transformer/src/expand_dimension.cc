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

#include "expand_dimension.h"
#include <map>
#include <set>
#include <unordered_map>
#include <unordered_set>
#include <bitset>
#include "axis_constants.h"
#include "exe_graph/runtime/expand_dims_type.h"
#include "framework/common/debug/ge_log.h"

namespace transformer {
namespace {
  const std::string RESHAPE_TYPE_FORBIDDEN = "FORBIDDEN";
  const uint32_t kBitsOfByte = 8;
  const uint32_t kBitSetDisplaySize = 8;
  const uint32_t kMaxReshapeTypeSize = 56;

  const std::set<ge::Format> kSupportedTransFormat = {ge::FORMAT_ND, ge::FORMAT_FRACTAL_NZ,
                                                      ge::FORMAT_ND_RNN_BIAS, ge::FORMAT_FRACTAL_ZN_RNN};

  const std::map<ge::Format, size_t> FULL_SIZE_OF_FORMAT {
          {ge::FORMAT_NCHW,  DIM_SIZE_FOUR},
          {ge::FORMAT_NHWC,  DIM_SIZE_FOUR},
          {ge::FORMAT_HWCN,  DIM_SIZE_FOUR},
          {ge::FORMAT_CHWN,  DIM_SIZE_FOUR},
          {ge::FORMAT_NDHWC, DIM_SIZE_FIVE},
          {ge::FORMAT_NCDHW, DIM_SIZE_FIVE},
          {ge::FORMAT_DHWCN, DIM_SIZE_FIVE},
          {ge::FORMAT_ND,    DIM_SIZE_FOUR}
  };

  inline uint32_t GenerateFormatKey(ge::Format format) {
    return ((static_cast<uint32_t>(format) & 0xff) << kBitsOfByte);
  }

  inline uint32_t GenerateReshapeTypeKey(ge::Format format, size_t size) {
    return ((static_cast<uint32_t>(format) & 0xff) << kBitsOfByte) | (static_cast<uint32_t>(size) & 0xff);
  }

  inline uint32_t GenerateAxisIndexKey(ge::Format format, char ch) {
    return ((static_cast<uint32_t>(format) & 0xff) << kBitsOfByte) | (static_cast<uint32_t>(ch) & 0xff);
  }

  const std::unordered_map<uint32_t, std::string> DEFAULT_RESHAPE_TYPE {
          {GenerateReshapeTypeKey(ge::FORMAT_NCHW, 0), ""},
          {GenerateReshapeTypeKey(ge::FORMAT_NHWC, 0), ""},
          {GenerateReshapeTypeKey(ge::FORMAT_HWCN, 0), ""},
          {GenerateReshapeTypeKey(ge::FORMAT_CHWN, 0), ""},
          {GenerateReshapeTypeKey(ge::FORMAT_NDHWC, 0), ""},
          {GenerateReshapeTypeKey(ge::FORMAT_NCDHW, 0), ""},
          {GenerateReshapeTypeKey(ge::FORMAT_DHWCN, 0), ""},

          {GenerateReshapeTypeKey(ge::FORMAT_NCHW, 1), "C"},
          {GenerateReshapeTypeKey(ge::FORMAT_NHWC, 1), "C"},
          {GenerateReshapeTypeKey(ge::FORMAT_HWCN, 1), "C"},
          {GenerateReshapeTypeKey(ge::FORMAT_CHWN, 1), "C"},
          {GenerateReshapeTypeKey(ge::FORMAT_NDHWC, 1), "C"},
          {GenerateReshapeTypeKey(ge::FORMAT_NCDHW, 1), "C"},
          {GenerateReshapeTypeKey(ge::FORMAT_DHWCN, 1), "C"},

          {GenerateReshapeTypeKey(ge::FORMAT_NCHW, 2), "CH"},
          {GenerateReshapeTypeKey(ge::FORMAT_NHWC, 2), "HW"},
          {GenerateReshapeTypeKey(ge::FORMAT_HWCN, 2), "CN"},
          {GenerateReshapeTypeKey(ge::FORMAT_CHWN, 2), "WN"},
          {GenerateReshapeTypeKey(ge::FORMAT_NDHWC, 2), "WC"},
          {GenerateReshapeTypeKey(ge::FORMAT_NCDHW, 2), "HW"},
          {GenerateReshapeTypeKey(ge::FORMAT_DHWCN, 2), "CN"},

          {GenerateReshapeTypeKey(ge::FORMAT_NCHW, 3), "CHW"},
          {GenerateReshapeTypeKey(ge::FORMAT_NHWC, 3), "HWC"},
          {GenerateReshapeTypeKey(ge::FORMAT_HWCN, 3), "WCN"},
          {GenerateReshapeTypeKey(ge::FORMAT_CHWN, 3), "HWN"},
          {GenerateReshapeTypeKey(ge::FORMAT_NDHWC, 3), "HWC"},
          {GenerateReshapeTypeKey(ge::FORMAT_NCDHW, 3), "DHW"},
          {GenerateReshapeTypeKey(ge::FORMAT_DHWCN, 3), "WCN"},

          {GenerateReshapeTypeKey(ge::FORMAT_NDHWC, 4), "DHWC"},
          {GenerateReshapeTypeKey(ge::FORMAT_NCDHW, 4), "CDHW"},
          {GenerateReshapeTypeKey(ge::FORMAT_DHWCN, 4), "HWCN"}
  };

  const std::unordered_map<uint32_t, int32_t> AXIS_INDEX_OF_FORMAT {
          {GenerateAxisIndexKey(ge::FORMAT_NCHW, 'N'), AXIS_NCHW_DIM_N},
          {GenerateAxisIndexKey(ge::FORMAT_NCHW, 'C'), AXIS_NCHW_DIM_C},
          {GenerateAxisIndexKey(ge::FORMAT_NCHW, 'H'), AXIS_NCHW_DIM_H},
          {GenerateAxisIndexKey(ge::FORMAT_NCHW, 'W'), AXIS_NCHW_DIM_W},

          {GenerateAxisIndexKey(ge::FORMAT_HWCN, 'N'), AXIS_HWCN_DIM_N},
          {GenerateAxisIndexKey(ge::FORMAT_HWCN, 'C'), AXIS_HWCN_DIM_C},
          {GenerateAxisIndexKey(ge::FORMAT_HWCN, 'H'), AXIS_HWCN_DIM_H},
          {GenerateAxisIndexKey(ge::FORMAT_HWCN, 'W'), AXIS_HWCN_DIM_W},

          {GenerateAxisIndexKey(ge::FORMAT_NHWC, 'N'), AXIS_NHWC_DIM_N},
          {GenerateAxisIndexKey(ge::FORMAT_NHWC, 'C'), AXIS_NHWC_DIM_C},
          {GenerateAxisIndexKey(ge::FORMAT_NHWC, 'H'), AXIS_NHWC_DIM_H},
          {GenerateAxisIndexKey(ge::FORMAT_NHWC, 'W'), AXIS_NHWC_DIM_W},

          {GenerateAxisIndexKey(ge::FORMAT_CHWN, 'N'), AXIS_CHWN_DIM_N},
          {GenerateAxisIndexKey(ge::FORMAT_CHWN, 'C'), AXIS_CHWN_DIM_C},
          {GenerateAxisIndexKey(ge::FORMAT_CHWN, 'H'), AXIS_CHWN_DIM_H},
          {GenerateAxisIndexKey(ge::FORMAT_CHWN, 'W'), AXIS_CHWN_DIM_W},

          {GenerateAxisIndexKey(ge::FORMAT_NDHWC, 'N'), NDHWC_DIM_N},
          {GenerateAxisIndexKey(ge::FORMAT_NDHWC, 'C'), NDHWC_DIM_C},
          {GenerateAxisIndexKey(ge::FORMAT_NDHWC, 'H'), NDHWC_DIM_H},
          {GenerateAxisIndexKey(ge::FORMAT_NDHWC, 'W'), NDHWC_DIM_W},
          {GenerateAxisIndexKey(ge::FORMAT_NDHWC, 'D'), NDHWC_DIM_D},

          {GenerateAxisIndexKey(ge::FORMAT_NCDHW, 'N'), NCDHW_DIM_N},
          {GenerateAxisIndexKey(ge::FORMAT_NCDHW, 'C'), NCDHW_DIM_C},
          {GenerateAxisIndexKey(ge::FORMAT_NCDHW, 'H'), NCDHW_DIM_H},
          {GenerateAxisIndexKey(ge::FORMAT_NCDHW, 'W'), NCDHW_DIM_W},
          {GenerateAxisIndexKey(ge::FORMAT_NCDHW, 'D'), NCDHW_DIM_D},

          {GenerateAxisIndexKey(ge::FORMAT_DHWCN, 'N'), DHWCN_DIM_N},
          {GenerateAxisIndexKey(ge::FORMAT_DHWCN, 'C'), DHWCN_DIM_C},
          {GenerateAxisIndexKey(ge::FORMAT_DHWCN, 'H'), DHWCN_DIM_H},
          {GenerateAxisIndexKey(ge::FORMAT_DHWCN, 'W'), DHWCN_DIM_W},
          {GenerateAxisIndexKey(ge::FORMAT_DHWCN, 'D'), DHWCN_DIM_D},

          {GenerateAxisIndexKey(ge::FORMAT_DHWNC, 'N'), DHWNC_DIM_N},
          {GenerateAxisIndexKey(ge::FORMAT_DHWNC, 'C'), DHWNC_DIM_C},
          {GenerateAxisIndexKey(ge::FORMAT_DHWNC, 'H'), DHWNC_DIM_H},
          {GenerateAxisIndexKey(ge::FORMAT_DHWNC, 'W'), DHWNC_DIM_W},
          {GenerateAxisIndexKey(ge::FORMAT_DHWNC, 'D'), DHWNC_DIM_D}
  };

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

bool GetDefaultReshapeType(const ge::Format &original_format, const size_t &old_dims_size, std::string &reshape_type) {
  int32_t default_key = GenerateReshapeTypeKey(original_format, old_dims_size);
  auto iter = DEFAULT_RESHAPE_TYPE.find(default_key);
  if (iter == DEFAULT_RESHAPE_TYPE.end()) {
    GELOGW("dim size %zu is invalid.", old_dims_size);
    return false;
  }

  reshape_type = iter->second;
  return true;
}

bool IsExpandNecessary(const size_t &old_dims_size, const ge::Format &original_format, const ge::Format &final_format,
                       const std::string &reshape_type, size_t &full_size) {
  /* 1. Check whether the old dim size is full. Full size is not necessary for expand. */
  auto iter_full_size = FULL_SIZE_OF_FORMAT.find(original_format);
  if (iter_full_size == FULL_SIZE_OF_FORMAT.end()) {
    GELOGW("Original Format %u is invalid.", original_format);
    return false;
  } else {
    if (old_dims_size >= iter_full_size->second) {
      return false;
    }
  }
  /* 2. Check whether the final format does not need expanding demension. */
  bool no_need_reshape_flag = reshape_type == RESHAPE_TYPE_FORBIDDEN || final_format == ge::FORMAT_FRACTAL_NZ ||
                              (original_format == ge::FORMAT_ND && final_format == ge::FORMAT_FRACTAL_Z);
  if (no_need_reshape_flag) {
    return false;
  }
  full_size = iter_full_size->second;
  return true;
}

bool IsReshapeTypeValid(const ge::Format &original_format, const size_t &old_dims_size,
                        const std::string &reshape_type) {
  if (reshape_type.empty()) {
    return old_dims_size == 0;
  }
  int32_t pos = -1;
  uint32_t format_key = GenerateFormatKey(original_format);
  uint32_t axis_key = 0;
  for (const char &dim : reshape_type) {
    axis_key = format_key | (static_cast<uint32_t>(dim) & 0xff);
    auto iter = AXIS_INDEX_OF_FORMAT.find(axis_key);
    if (iter == AXIS_INDEX_OF_FORMAT.end()) {
      return false;
    }
    if (iter->second > pos) {
      pos = iter->second;
    } else {
      return false;
    }
  }

  return true;
}

void ExpandByReshapeType(ge::GeShape &shape, const ge::Format &original_format,
                         const size_t &old_dims_size, const size_t &full_size, const std::string &reshape_type) {
  GELOGD("Expand tensor by reshape type %s.", reshape_type.c_str());
  /* Build a array with all 1 of full size. Then we will substitute some of the 1 with the original axis value. */
  for (size_t i = old_dims_size; i < full_size; i++) {
    shape.AppendDim(1);
  }
  if (reshape_type.empty() || old_dims_size == 0) {
    return;
  }

  uint32_t format_key = GenerateFormatKey(original_format);
  uint32_t axis_key = 0;
  for (int32_t i = static_cast<int32_t>(old_dims_size) - 1; i >= 0; i--) {
    axis_key = format_key | (static_cast<uint32_t>(reshape_type.at(i)) & 0xff);
    auto iter_axis_index = AXIS_INDEX_OF_FORMAT.find(axis_key);
    if (iter_axis_index == AXIS_INDEX_OF_FORMAT.end()) {
      continue;
    }
    if (iter_axis_index->second == i) {
      continue;
    }
    shape.SetDim(iter_axis_index->second, shape.GetDim(i));
    shape.SetDim(i, 1);
  }
}

bool ExpandDimension(const std::string &op_type, const ge::Format &original_format, const ge::Format &final_format,
                     const uint32_t &tensor_index, const std::string &reshape_type, ge::GeShape &shape) {
  /* 1. Check expanding necessary. */
  size_t full_size = 0;
  size_t old_dims_size = shape.GetDimNum();
  if (!IsExpandNecessary(old_dims_size, original_format, final_format, reshape_type, full_size)) {
    return true;
  }

  /* 2. Check whether the reshape type is consistent with the original format.
   * If not consistent, just return and report a warning. */
  std::string valid_reshape_type = reshape_type;
  if (!IsReshapeTypeValid(original_format, old_dims_size, reshape_type)) {
    if (!GetDefaultReshapeType(original_format, old_dims_size, valid_reshape_type)) {
      return true;
    }
  }

  /* 3. Check whether the dimension of original shape is less than or equal to
   * the length of reshape type. If the dimension of original shape if larger,
   * we cannot find suitable posotion for all axis in original shape and we just return. */
  if (old_dims_size > valid_reshape_type.length()) {
    GELOGW("Dimension %zu of tensor %u of %s is larger than the length of reshape type which is %zu.",
           old_dims_size, tensor_index, op_type.c_str(), valid_reshape_type.length());
    return true;
  }

  /* 4. Expand dimension. */
  ExpandByReshapeType(shape, original_format, old_dims_size, full_size, valid_reshape_type);
  return true;
}

bool ExpandRangeDimension(const std::string &op_type, const ge::Format &original_format,
    const ge::Format &final_format, const uint32_t &tensor_index, const std::string &reshape_type,
    std::vector<std::pair<int64_t, int64_t>> &ranges) {
  std::vector<int64_t> range_upper;
  std::vector<int64_t> range_low;
  for (auto &i : ranges) {
    range_low.emplace_back(i.first);
    range_upper.emplace_back(i.second);
  }

  ge::GeShape shape_low(range_low);
  ge::GeShape shape_upper(range_upper);
  bool res = ExpandDimension(op_type, original_format, final_format, tensor_index, reshape_type, shape_low) &&
      ExpandDimension(op_type, original_format, final_format, tensor_index, reshape_type, shape_upper);
  if (!res || (shape_low.GetDimNum() != shape_upper.GetDimNum())) {
    return false;
  }
  ranges.clear();
  for (size_t idx = 0; idx < shape_low.GetDimNum(); ++idx) {
    ranges.emplace_back(std::pair<int64_t, int64_t>(shape_low.GetDim(idx), shape_upper.GetDim(idx)));
  }
  return res;
}

ExpandDimension::ExpandDimension() {}
ExpandDimension::~ExpandDimension() {}

int64_t ExpandDimension::GenerateReshapeType(const ge::Format &origin_format, const ge::Format &format,
                                             const size_t &origin_dim_size, const std::string &reshape_type) {
  GELOGD("Begin to generate integer reshape type, original format[%d], format[%d], dim size[%zu], reshape type[%s].",
         origin_format, format, origin_dim_size, reshape_type.c_str());
  int64_t ret_reshape_type = 0;
  size_t full_size = 0;
  if (!GetFormatFullSize(origin_format, full_size)) {
    return ret_reshape_type;
  }
  if (!IsNeedExpand(origin_format, format, origin_dim_size, full_size, reshape_type)) {
    return ret_reshape_type;
  }

  std::string valid_shape_type = reshape_type;
  if (!IsReshapeTypeValid(origin_format, origin_dim_size, reshape_type)) {
    if (!GetDefaultReshapeType(origin_format, origin_dim_size, valid_shape_type)) {
      return ret_reshape_type;
    }
    GELOGD("Reshape type[%s] is invalid, using default reshape type[%s]",
           reshape_type.c_str(), valid_shape_type.c_str());
  }

  if (origin_dim_size > valid_shape_type.length()) {
    GELOGW("The length of reshape type[%s] is longer than dim size[%zu]. Can not generate integer reshape type.",
           valid_shape_type.c_str(), origin_dim_size);
    return ret_reshape_type;
  }

  uint32_t format_key = GenerateFormatKey(origin_format);
  std::unordered_set<int32_t> dim_pos_set;
  for (const char &dim : valid_shape_type.substr(0, origin_dim_size)) {
    uint32_t axis_key = format_key | (static_cast<uint32_t>(dim) & 0xff);
    auto iter_axis_index = AXIS_INDEX_OF_FORMAT.find(axis_key);
    if (iter_axis_index != AXIS_INDEX_OF_FORMAT.end()) {
      dim_pos_set.emplace(iter_axis_index->second);
    }
  }

  for (size_t i = 0; i < full_size; i++) {
    if (dim_pos_set.count(i) == 0) {
      ret_reshape_type = ret_reshape_type | (1 << i);
    }
  }

  ret_reshape_type = ret_reshape_type | (full_size << kMaxReshapeTypeSize);
  GELOGD("Integer reshape type[%s] has been generated for original format[%d], dim size[%zu], reshape type[%s].",
         std::bitset<kBitSetDisplaySize>(ret_reshape_type).to_string().c_str(), origin_format, origin_dim_size,
         valid_shape_type.c_str());
  return ret_reshape_type;
}

bool ExpandDimension::IsNeedExpand(const ge::Format &origin_format, const ge::Format &format,
                                   const size_t &origin_dim_size, const size_t &full_size,
                                   const std::string &reshape_type) {
  if (origin_dim_size >= full_size) {
    return false;
  }
  if (reshape_type == RESHAPE_TYPE_FORBIDDEN) {
    return false;
  }
  if (kSupportedTransFormat.count(format) != 0) {
    return false;
  }
  if (origin_format == ge::FORMAT_ND && format == ge::FORMAT_FRACTAL_Z) {
    return false;
  }
  return true;
}

bool ExpandDimension::IsReshapeTypeValid(const ge::Format &origin_format, const size_t &origin_dim_size,
                                         const std::string &reshape_type) {
  if (reshape_type.empty()) {
    return origin_dim_size == 0;
  }
  int32_t pos = -1;
  uint32_t format_key = GenerateFormatKey(origin_format);
  uint32_t axis_key = 0;
  for (const char &dim : reshape_type) {
    axis_key = format_key | (static_cast<uint32_t>(dim) & 0xff);
    auto iter = AXIS_INDEX_OF_FORMAT.find(axis_key);
    if (iter == AXIS_INDEX_OF_FORMAT.end()) {
      return false;
    }
    if (iter->second > pos) {
      pos = iter->second;
    } else {
      return false;
    }
  }
  return true;
}

bool ExpandDimension::GetDefaultReshapeType(const ge::Format &origin_format, const size_t &origin_dim_size,
                                            std::string &reshape_type) {
  int32_t default_key = GenerateReshapeTypeKey(origin_format, origin_dim_size);
  auto iter = DEFAULT_RESHAPE_TYPE.find(default_key);
  if (iter == DEFAULT_RESHAPE_TYPE.end()) {
    GELOGW("Dim size %zu is invalid, default reshape type is not found.", origin_dim_size);
    return false;
  }

  reshape_type = iter->second;
  return true;
}

void ExpandDimension::ExpandDims(const int64_t &reshape_type, ge::GeShape &shape) {
  GELOGD("Begin to expand dims, reshape type[%ld], shape[%s].", reshape_type, shape.ToString().c_str());
  gert::Shape inner_shape;
  GeShapeToRtShape(shape, inner_shape);
  ExpandDims(reshape_type, inner_shape);
  RtShapeToGeShape(inner_shape, shape);
  GELOGD("After expanding dims, shape[%s].", shape.ToString().c_str());
}

void ExpandDimension::ExpandDims(const int64_t &reshape_type, const ge::GeShape &origin_shape, ge::GeShape &shape) {
  GELOGD("Begin to expand dims, reshape type[%ld], origin shape[%s].", reshape_type, origin_shape.ToString().c_str());
  gert::Shape inner_ori_shape;
  GeShapeToRtShape(origin_shape, inner_ori_shape);
  gert::Shape inner_shape;
  GeShapeToRtShape(shape, inner_shape);
  ExpandDims(reshape_type, inner_ori_shape, inner_shape);
  RtShapeToGeShape(inner_shape, shape);
  GELOGD("After expanding dims, shape[%s].", shape.ToString().c_str());
}

void ExpandDimension::ExpandDims(const int64_t &reshape_type, gert::Shape &shape) {
  if (reshape_type == 0) {
    return;
  }
  gert::ExpandDimsType expand_dims_type(reshape_type);
  expand_dims_type.Expand(shape);
}

void ExpandDimension::ExpandDims(const int64_t &reshape_type, const gert::Shape &origin_shape, gert::Shape &shape) {
  if (reshape_type == 0) {
    return;
  }
  gert::ExpandDimsType expand_dims_type(reshape_type);
  expand_dims_type.Expand(origin_shape, shape);
}

bool ExpandDimension::GetFormatFullSize(const ge::Format &format, size_t &full_size) {
  auto iter = FULL_SIZE_OF_FORMAT.find(format);
  if (iter == FULL_SIZE_OF_FORMAT.end()) {
    return false;
  }
  full_size = iter->second;
  return true;
}

int32_t ExpandDimension::GetAxisIndexByName(char ch, const ge::Format &format) {
  uint32_t format_key = GenerateFormatKey(format);
  uint32_t axis_key = 0;
  axis_key = format_key | (static_cast<uint32_t>(ch) & 0xff);
  auto iter = AXIS_INDEX_OF_FORMAT.find(axis_key);
  if (iter == AXIS_INDEX_OF_FORMAT.end()) {
    return -1;
  }
  return iter->second;
}
int64_t ExpandDimension::GetReshapeAxicValue(const int64_t &reshape_type_mask,
                                             const ge::GeShape &shape, int32_t axis_index) {
  GELOGD("axis_index = %d.", axis_index);
  if (axis_index == -1) {
    return -1;
  }
  gert::ExpandDimsType expand_dims_type(reshape_type_mask);
  if (!expand_dims_type.IsExpandIndex(axis_index)) {
    GELOGD("axis_index isorigindim idx = %d.", axis_index);
  }
  return shape.GetDim(static_cast<size_t>(axis_index));
}
int64_t ExpandDimension::GetReshapeAxicValueByName(const int64_t &reshape_type_mask, char ch,
                                                   const ge::GeShape &shape, const ge::Format &format) {
  auto idx = GetAxisIndexByName(ch, format);
  return GetReshapeAxicValue(reshape_type_mask, shape, idx);
}
} // namespace transformer
