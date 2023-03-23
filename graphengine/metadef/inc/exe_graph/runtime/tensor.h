/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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
#ifndef METADEF_CXX_INC_EXE_GRAPH_TENSOR_H_
#define METADEF_CXX_INC_EXE_GRAPH_TENSOR_H_

#include "graph/ge_error_codes.h"
#include "storage_shape.h"
#include "storage_format.h"
#include "tensor_data.h"

namespace gert {
using TensorAddress = void *;            ///< Tensor地址
using ConstTensorAddress = void *const;  ///< Tensor地址

class Tensor {
 public:
  Tensor() = default;
  Tensor(const StorageShape &storage_shape, const StorageFormat &storage_format, TensorPlacement placement,
         ge::DataType data_type, TensorAddress addr)
      : storage_shape_(storage_shape), storage_format_(storage_format), data_type_(data_type),
        tensor_data_(addr, nullptr, ge::GetSizeInBytes(GetShapeSize(), data_type_), placement) {
      (void) reserved_;
  }
  /**
   * 获取shape size，所谓shape size是指本shape中包含的element数量
   * @return shape size
   */
  int64_t GetShapeSize() const {
    return storage_shape_.GetStorageShape().GetShapeSize();
  }
  /**
   * 获取Tensor的数据地址
   * @tparam T 数据类型
   * @return 数据地址
   */
  template<class T>
  const T *GetData() const {
    return static_cast<const T *>(GetAddr());
  }
  /**
   * 获取Tensor的数据地址
   * @tparam T 数据类型
   * @return 数据地址
   */
  template<class T>
  T *GetData() {
    return static_cast<T *>(GetAddr());
  }
  /**
   * 设置Tensor数据
   * @param data 数据
   */
  void SetData(TensorData &&data) {
    tensor_data_ = std::move(data);
  }
  /**
   * 获取Tensor的数据地址
   * @return 数据地址
   */
  const void *GetAddr() const {
    if (tensor_data_.GetPlacement() == kFollowing) {
      return reinterpret_cast<const void *>(reinterpret_cast<const uint8_t *>(this) + sizeof(*this));
    } else {
      return tensor_data_.GetAddr();
    }
  }
  /**
   * 获取Tensor的数据地址
   * @return 数据地址
   */
  void *GetAddr() {
    if (tensor_data_.GetPlacement() == kFollowing) {
      return reinterpret_cast<void *>(reinterpret_cast<uint8_t *>(this) + sizeof(*this));
    } else {
      return tensor_data_.GetAddr();
    }
  }
  /**
   * 获取Tensor的内存大小
   * @return 内存大小
   */
  size_t GetSize() const {
    return tensor_data_.GetSize();
  }

  /**
   * 设置Tensor的内存大小
   * @param Tensor的内存大小
   */
  void SetSize(size_t size) {
    tensor_data_.SetSize(size);
  }

  /**
   * 获取Tensor的data type
   * @return data type
   */
  ge::DataType GetDataType() const {
    return data_type_;
  }
  /**
   * 设置Tensor的data type
   * @param data_type data type
   */
  void SetDataType(ge::DataType data_type) {
    data_type_ = data_type;
  }
  /**
   * 创建一个Tensor，tensor数据在Tensor对象后面连续排布
   * @param shape_size 元素个数
   * @param dt 数据类型
   * @param total_size 创建出的Tensor在内存中的长度
   * @return 创建出的Tensor指针
   */
  static std::unique_ptr<uint8_t[]> CreateFollowing(int64_t shape_size, ge::DataType dt, size_t &total_size) {
    total_size = ge::GetSizeInBytes(shape_size, dt);
    if (ge::AddOverflow(total_size, sizeof(Tensor), total_size)) {
      return nullptr;
    }
    auto holder = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[total_size]);
    if (holder == nullptr) {
      return nullptr;
    }

    auto tensor = reinterpret_cast<Tensor *>(holder.get());
    new (holder.get()) Tensor({}, {}, kFollowing, dt, nullptr);
    tensor->tensor_data_ = TensorData(nullptr, nullptr, total_size, kFollowing);
    return holder;
  }
  /**
   * 获取运行shape
   * @return 只读的运行时shape引用
   */
  const Shape &GetStorageShape() const {
    return storage_shape_.GetStorageShape();
  }
  /**
   * 获取运行shape
   * @return 运行时shape的引用
   */
  Shape &MutableStorageShape() {
    return storage_shape_.MutableStorageShape();
  }
  /**
   * 获取原始shape
   * @return 只读的原始shape引用
   */
  const Shape &GetOriginShape() const {
    return storage_shape_.GetOriginShape();
  }
  /**
   * 获取原始shape
   * @return 原始shape引用
   */
  Shape &MutableOriginShape() {
    return storage_shape_.MutableOriginShape();
  }
  /**
   * 获取shape，包含运行和原始shape
   * @return 只读的shape引用
   */
  const StorageShape &GetShape() const {
    return storage_shape_;
  }
  /**
   * 获取shape，包含运行和原始shape
   * @return shape引用
   */
  StorageShape &GetShape() {
    return storage_shape_;
  }
  /**
   * 获取运行时format
   * @return 运行时format
   */
  ge::Format GetStorageFormat() const {
    return storage_format_.GetStorageFormat();
  }
  /**
   * 设置运行时format
   * @param storage_format 运行时format
   */
  void SetStorageFormat(ge::Format storage_format) {
    storage_format_.SetStorageFormat(storage_format);
  }
  /**
   * 获取原始format
   * @return 原始format
   */
  ge::Format GetOriginFormat() const {
    return storage_format_.GetOriginFormat();
  }
  /**
   * 设置原始format
   * @param origin_format 原始format
   */
  void SetOriginFormat(ge::Format origin_format) {
    storage_format_.SetOriginFormat(origin_format);
  }
  /**
   * 获取format，format包含运行时format和原始format
   * @return 只读的format引用
   */
  const StorageFormat &GetFormat() const {
    return storage_format_;
  }
  /**
   * 获取format，format包含运行时format和原始format
   * @return format引用
   */
  StorageFormat &MutableFormat() {
    return storage_format_;
  }
  /**
   * 获取补维规则
   * @return 补维规则
   */
  ExpandDimsType GetExpandDimsType() const {
    return storage_format_.GetExpandDimsType();
  }
  /**
   * 设置补维规则
   * @param expand_dims_type 补维规则
   */
  void SetExpandDimsType(ExpandDimsType expand_dims_type) {
    storage_format_.SetExpandDimsType(expand_dims_type);
  }
  /**
   * 获取tensor的placement
   * @return tensor的placement
   */
  TensorPlacement GetPlacement() const {
    return tensor_data_.GetPlacement();
  }
  /**
   * 设置tensor的placement
   * @param tensor的placement
   */
  void SetPlacement(TensorPlacement placement) {
    tensor_data_.SetPlacement(placement);
  }
  /**
   * 获取tensor data
   * @return 只读的tensor data引用
   */
  const TensorData &GetTensorData() const {
    return tensor_data_;
  }
  /**
   * 获取tensor data
   * @return 可写的tensor data引用
   */
  TensorData &MutableTensorData() {
    return tensor_data_;
  }

 private:
  StorageShape storage_shape_;
  StorageFormat storage_format_;
  uint8_t reserved_[4] = {0U}; // Reserved field, 4-byte aligned
  ge::DataType data_type_;
  TensorData tensor_data_;
};
static_assert(std::is_standard_layout<Tensor>::value, "The class Tensor must be a POD");
}  // namespace gert

#endif  // METADEF_CXX_INC_EXE_GRAPH_TENSOR_H_
