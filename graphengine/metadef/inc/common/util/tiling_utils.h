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

#ifndef METADEF_CXX_INC_COMMON_UTIL_TILING_UTILS_H_
#define METADEF_CXX_INC_COMMON_UTIL_TILING_UTILS_H_
#include <cstdint>

namespace optiling {
union Fp32 {
  uint32_t u;
  float f;
};

inline uint16_t FloatToUint16(const float &value) {
  const Fp32 f32infty = {255U << 23};
  const Fp32 f16infty = {31U << 23};
  const Fp32 magic = {15U << 23};
  constexpr uint32_t sign_mask = 0x80000000U;
  constexpr uint32_t round_mask = ~0xFFFU;
  constexpr uint32_t round_max = 0x7FFFU;
  constexpr uint32_t dst_addr = 0x7C00U;
  constexpr uint32_t right_shift_13 = 13;
  constexpr uint32_t right_shift_16 = 16;

  Fp32 temp;
  uint16_t out;
  temp.f = value;
  uint32_t sign = temp.u & sign_mask;
  temp.u ^= sign;

  if (temp.u >= f32infty.u) {
    out = (temp.u > f32infty.u) ? round_max : dst_addr;
  } else {
    temp.u &= round_mask;
    temp.f *= magic.f;
    temp.u -= round_mask;
    if (temp.u > f16infty.u) {
      temp.u = f16infty.u;
    }
    out = uint16_t(temp.u >> right_shift_13);
  }

  out = uint16_t(out | (sign >> right_shift_16));
  return out;
}
}  // namespace optiling
#endif