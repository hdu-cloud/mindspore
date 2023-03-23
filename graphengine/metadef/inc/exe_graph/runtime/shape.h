/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2021. All rights reserved.
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
#ifndef METADEF_CXX_INC_EXE_GRAPH_SHAPE_H_
#define METADEF_CXX_INC_EXE_GRAPH_SHAPE_H_

#include <array>
#include <vector>
#include <iostream>
#include <cstring>
#include <type_traits>
#include <limits>
#include "graph/utils/math_util.h"

namespace gert {
struct Shape {
 public:
  static constexpr size_t kMaxDimNum = 8;
  static constexpr int64_t kInvalidDimValue = std::numeric_limits<int64_t>::min();

 public:
  /**
   * 默认构造一个shape，默认构造的shape实例中，dim_num长度为0
   */
  Shape() : dim_num_(0), dims_{0} {}

  /**
   * 通过dims值构造shape，例如：Shape({8,3,224,224})创建一个Shape实例，有4个维度，每个维度的值分别是8,3,224,224
   * @param dims shape的所有dim值
   */
  Shape(std::initializer_list<int64_t> args) : dim_num_(0), dims_{0} {
    if (args.size() > kMaxDimNum) {
      return;
    }
    dim_num_ = args.size();
    size_t i = 0;
    for (auto arg : args) {
      dims_[i++] = arg;
    }
  }

  /**
   * 判断与另外一个shape对象是否相等，如果两个shape的dim num并且dim num内每个dim的值都相等，那么认为两个shape相等
   * @param rht 另一个Shape对象
   * @return true/false
   */
  bool operator==(const Shape &rht) const {
    if (this->dim_num_ != rht.dim_num_) {
      return false;
    }
    for (size_t i = 0; i < this->dim_num_; i++) {
      if (this->dims_[i] != rht.dims_[i]) {
        return false;
      }
    }
    return true;
  }

  /**
   * 判断与另一个Shape对象是否不等
   * @param rht 另一个Shape对象
   * @return true/false
   */
  bool operator!=(const Shape &rht) const {
    return !(*this == rht);
  }

  /**
   * 获取shape size，所谓shape size，是指shape中有多少个元素
   * @return shape-size，在发生溢出时，返回`kInvalidDimValue`
   */
  int64_t GetShapeSize() const {
    int64_t shape_size = 1;
    for (size_t i = 0; i < dim_num_; ++i) {
      if (ge::MulOverflow(shape_size, dims_[i], shape_size)) {
        return kInvalidDimValue;
      }
    }
    return shape_size;
  }

  /**
   * 判断本shape是否为标量，所谓标量，是指GetDimNum()为0
   * @return true/false
   */
  bool IsScalar() const {
    return dim_num_ == 0L;
  }

  /**
   * 设置shape为标量
   * @param none
   */
  void SetScalar() {
    dim_num_ = 0;
  }

  /**
   * 获取dim num
   * @return
   */
  size_t GetDimNum() const {
    return dim_num_;
  }

  /**
   * 设置dim num
   * @param dim_num
   */
  void SetDimNum(size_t dim_num) {
    this->dim_num_ = dim_num;
  }

  /**
   * 获取dim值
   * @param idx dim的index，调用者需要保证index合法
   * @return dim值，在idx超出MaxDimNum时，返回`kInvalidDimValue`
   */
  int64_t GetDim(size_t idx) const {
    if (idx >= kMaxDimNum) {
      return kInvalidDimValue;
    }
    return dims_[idx];
  }

  /**
   * 获取dim值
   * @param idx dim的index，调用者需要保证index合法
   * @return dim值，行为未定义
   */
  const int64_t &operator[](size_t idx) const {
    return dims_[idx];
  }

  /**
   * 获取dim值
   * @param idx dim的index，调用者需要保证index合法
   * @return dim值，在idx超出MaxDimNum时，行为未定义
   */
  int64_t &operator[](size_t idx) {
    return dims_[idx];
  }

  /**
   * 设置dim值
   * @param idx dim的index，调用者需要保证index合法
   * @return
   */
  void SetDim(size_t idx, int64_t dim_value) {
    if (idx >= kMaxDimNum) {
      return;
    }
    dims_[idx] = dim_value;
    this->dim_num_ = (this->dim_num_ < idx) ? idx : this->dim_num_;
  }

  /**
   * 向后扩展一个dim值，如果扩展的dim数量超出Shape的最大限制，那么本函数不做任何事情
   * @param 扩展的dim值
   * @return this引用
   */
  Shape& AppendDim(int64_t value) {
    if (this->dim_num_ >= kMaxDimNum) {
      return *this;
    }
    dims_[this->dim_num_++] = value;
    return *this;
  }

 private:
  size_t dim_num_;
  int64_t dims_[kMaxDimNum];
};
static_assert(std::is_standard_layout<Shape>::value, "The class Shape must be a POD");
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_SHAPE_H_
