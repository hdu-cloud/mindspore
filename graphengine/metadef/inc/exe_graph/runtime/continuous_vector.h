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

#ifndef METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTINUOUS_VECTOR_H_
#define METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTINUOUS_VECTOR_H_
#include <cstddef>
#include <cstdint>
#include <memory>
#include <type_traits>
#include "graph/ge_error_codes.h"
#include "graph/utils/math_util.h"
namespace gert {
class ContinuousVector {
 public:
  /**
   * 创建一个ContinuousVector实例，ContinuousVector不支持动态扩容
   * @tparam T 实例中包含的元素类型
   * @param capacity 实例的最大容量
   * @param total_size 本实例的总长度
   * @return 指向本实例的指针
   */
  template<typename T>
  static std::unique_ptr<uint8_t[]> Create(size_t capacity, size_t &total_size) {
    if (ge::MulOverflow(capacity, sizeof(T), total_size)) {
      return nullptr;
    }
    if (ge::AddOverflow(total_size, sizeof(ContinuousVector), total_size)) {
      return nullptr;
    }
    auto holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]);
    if (holder == nullptr) {
      return nullptr;
    }
    reinterpret_cast<ContinuousVector *>(holder.get())->Init(capacity);
    return holder;
  }
  /**
   * 创建一个ContinuousVector实例，ContinuousVector不支持动态扩容
   * @tparam T 实例中包含的元素类型
   * @param capacity 实例的最大容量
   * @return 指向本实例的指针
   */
  template<typename T>
  static std::unique_ptr<uint8_t[]> Create(size_t capacity) {
    size_t total_size;
    return Create<T>(capacity, total_size);
  }
  /**
   * 使用最大容量初始化本实例
   * @param capacity 最大容量
   */
  void Init(size_t capacity) {
    capacity_ = capacity;
    size_ = 0;
  }
  /**
   * 获取当前保存的元素个数
   * @return 当前保存的元素个数
   */
  size_t GetSize() const {
    return size_;
  }
  /**
   * 设置当前保存的元素个数
   * @param size 当前保存的元素个数
   * @return 成功时返回ge::GRAPH_SUCCESS
   */
  ge::graphStatus SetSize(size_t size) {
    if (size > capacity_) {
      GELOGE(ge::PARAM_INVALID, "Failed to set size for ContinuousVector, size(%zu) > cap(%zu)", size, capacity_);
      return ge::GRAPH_FAILED;
    }
    size_ = size;
    return ge::GRAPH_SUCCESS;
  }
  /**
   * 获取最大可保存的元素个数
   * @return 最大可保存的元素个数
   */
  size_t GetCapacity() const {
    return capacity_;
  }
  /**
   * 获取首个元素的指针地址，[GetData(), GetData() + GetSize()) 中的数据即为当前容器中保存的数据
   * @return 首个元素的指针地址
   */
  const void *GetData() const {
    return elements;
  }
  /**
   * 获取首个元素的指针地址，[GetData(), GetData() + GetSize()) 中的数据即为当前容器中保存的数据
   * @return 首个元素的指针地址
   */
  void *MutableData() {
    return elements;
  }

 private:
  size_t capacity_;
  size_t size_;
  uint8_t elements[8];
};
static_assert(std::is_standard_layout<ContinuousVector>::value, "The ContinuousVector must be a POD");

template<typename T>
class TypedContinuousVector : private ContinuousVector {
 public:
  using ContinuousVector::GetCapacity;
  using ContinuousVector::GetSize;
  using ContinuousVector::SetSize;
  /**
   * 获取首个元素的指针地址，[GetData(), GetData() + GetSize()) 中的数据即为当前容器中保存的数据
   * @return 首个元素的指针地址
   */
  T *MutableData() {
    return reinterpret_cast<T *>(ContinuousVector::MutableData());
  }
  /**
   * 获取首个元素的指针地址，[GetData(), GetData() + GetSize()) 中的数据即为当前容器中保存的数据
   * @return 首个元素的指针地址
   */
  const T *GetData() const {
    return reinterpret_cast<const T *>(ContinuousVector::GetData());
  }
};
}  // namespace gert
#endif  // METADEF_CXX_INC_EXE_GRAPH_RUNTIME_CONTINUOUS_VECTOR_H_
