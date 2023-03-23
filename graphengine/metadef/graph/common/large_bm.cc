/**
 * Copyright 2020-2021 Huawei Technologies Co., Ltd
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
#include "common/large_bm.h"
#include "graph/debug/ge_log.h"

namespace ge {
constexpr size_t kBitsEachValue = 64UL;

LargeBitmap::LargeBitmap(const size_t &size)
    : size_(size), bits_((size + kBitsEachValue - 1UL) / kBitsEachValue, 0UL) {}

bool LargeBitmap::operator==(const LargeBitmap &another_bm) const {
  return bits_ == another_bm.bits_;
}

bool LargeBitmap::operator!=(const LargeBitmap &another_bm) const {
  return bits_ != another_bm.bits_;
}

void LargeBitmap::SetValues(const uint64_t &value) {
  std::fill(bits_.begin(), bits_.end(), value);
}

void LargeBitmap::SetBit(const size_t &index) {
  if (index < size_) {
    bits_[index / kBitsEachValue] |= 1UL << (index % kBitsEachValue);
  } else {
    GE_LOGE("index %zu is not valid. Total size is %zu", index, size_);
    return;
  }
}

bool LargeBitmap::GetBit(const size_t &index) const {
  if (index < size_) {
    return static_cast<bool>(bits_[index / kBitsEachValue] & (1UL << (index % kBitsEachValue)));
  } else {
    GE_LOGE("index %zu is not valid. Total size is %zu", index, size_);
    return false;
  }
}

void LargeBitmap::Or(const LargeBitmap &another_bm) {
  size_t index = 0UL;
  const size_t another_size = another_bm.bits_.size();
  for (auto &bit : bits_) {
    if (index >= another_size) {
      return;
    }
    bit |= another_bm.bits_[index];
    ++index;
  }
}

void LargeBitmap::And(const LargeBitmap &another_bm) {
  size_t index = 0UL;
  const size_t another_size = another_bm.bits_.size();
  for (auto &bit : bits_) {
    if (index >= another_size) {
      return;
    }
    bit &= another_bm.bits_[index];
    ++index;
  }
}
}
