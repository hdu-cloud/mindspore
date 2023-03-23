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

#ifndef MINDSPORE_CORE_IR_MAP_TENSOR_H_
#define MINDSPORE_CORE_IR_MAP_TENSOR_H_

#include <tuple>
#include <memory>
#include <string>
#include <utility>
#include "ir/anf.h"
#include "ir/dtype.h"
#include "ir/tensor.h"
#include "ir/param_info.h"
#include "ir/scalar.h"
#include "utils/macros.h"
#include "utils/shape_utils.h"

namespace mindspore {
using DataLenPair = std::pair<void *, size_t>;
namespace tensor {
class MapTensor;
// Smart pointer for MapTensor.
using MapTensorPtr = std::shared_ptr<MapTensor>;
///
/// \brief MapTensor is a dynamic tensor with map like index functions.
///
class MS_CORE_API MapTensor final : public Tensor {
 public:
  struct ExportData {
    TensorPtr key_tensor;
    TensorPtr value_tensor;
    TensorPtr status_tensor;
  };

  enum class Status {
    kUnchanged = 0,
    kModified = 1,
    kErased = 2,
  };

  MapTensor() = default;

  /// \brief Create a empty MapTensor.
  ///
  /// \param[in] key_dtype [TypeId] The key data type id.
  /// \param[in] value_dtype [TypeId] The value data type id.
  /// \param[in] value_shape [TypeId] The value shape.
  /// \param[in] default_value [ValuePtr] The default value.
  /// \param[in] permit_filter_value [ValuePtr] The permit filter value.
  /// \param[in] evict_filter_value [ValuePtr] The evict filter value.
  MapTensor(TypeId key_dtype, TypeId value_dtype, const ShapeVector &value_shape, const ValuePtr &default_value,
            const ValuePtr &permit_filter_value = nullptr, const ValuePtr &evict_filter_value = nullptr)
      : key_dtype_(key_dtype), default_value_(default_value) {
    data_type_ = value_dtype;
    value_shape_ = value_shape;
    key_shape_ = {abstract::Shape::kShapeDimAny};
    shape_ = {abstract::Shape::kShapeDimAny};
    (void)shape_.insert(shape_.cend(), value_shape.cbegin(), value_shape.cend());
    ShapeVector key_shape = {abstract::Shape::kShapeDimAny};
    key_tensor_ = std::make_shared<Tensor>(key_dtype, key_shape);
    value_tensor_ = std::make_shared<Tensor>(value_dtype, value_shape);
    status_tensor_ = std::make_shared<Tensor>(kNumberTypeUInt8, key_shape);
    permit_filter_value_ = (permit_filter_value == nullptr) ? std::make_shared<Int32Imm>(1) : permit_filter_value;
    evict_filter_value_ = (evict_filter_value == nullptr) ? std::make_shared<Int32Imm>(SIZE_MAX) : evict_filter_value;
  }

  /// \brief Create a new MapTensor.
  ///
  /// \param[in] key_tensor [Tensor] The key tensor.
  /// \param[in] value_tensor [Tensor] The value tensor.
  /// \param[in] status_tensor [Tensor] The status tensor.
  /// \param[in] default_value [ValuePtr] The default value.
  /// \param[in] permit_filter_value [ValuePtr] The permit filter value.
  /// \param[in] evict_filter_value [ValuePtr] The evict filter value.
  MapTensor(const TensorPtr &key_tensor, const TensorPtr &value_tensor, const TensorPtr &status_tensor,
            const ValuePtr &default_value, const ValuePtr &permit_filter_value = nullptr,
            const ValuePtr &evict_filter_value = nullptr)
      : default_value_(default_value) {
    key_dtype_ = key_tensor->data_type();
    data_type_ = value_tensor->data_type();
    value_shape_ = value_tensor->shape();
    key_shape_ = key_tensor->shape();
    shape_ = value_shape_;
    key_tensor_ = key_tensor;
    value_tensor_ = value_tensor;
    status_tensor_ = status_tensor;
    permit_filter_value_ = (permit_filter_value == nullptr) ? std::make_shared<Int32Imm>(1) : permit_filter_value;
    evict_filter_value_ = (evict_filter_value == nullptr) ? std::make_shared<Int32Imm>(SIZE_MAX) : evict_filter_value;
  }

  ~MapTensor() override = default;

  MS_DECLARE_PARENT(MapTensor, Tensor)

  std::size_t hash() const override;

  bool operator==(const Value &other) const override {
    if (this == &other) {
      return true;
    }
    if (!other.isa<MapTensor>()) {
      return false;
    }
    auto other_ = static_cast<const MapTensor &>(other);
    return *this == other_;
  }

  bool operator==(const MapTensor &other) const;

  TypeId key_dtype() const { return key_dtype_; }

  TypeId value_dtype() const { return data_type_; }

  size_t size() const { return size_; }

  const ShapeVector &value_shape() const { return value_shape_; }

  const ValuePtr &default_value() const { return default_value_; }

  const ValuePtr &permit_filter_value() const { return permit_filter_value_; }

  const ValuePtr &evict_filter_value() const { return evict_filter_value_; }

  TypePtr KeyDtype() const { return TypeIdToType(key_dtype_); }

  TypePtr ValueDtype() const { return TypeIdToType(data_type_); }

  abstract::AbstractBasePtr ToAbstract() override;

  std::string ToString() const override;

  /// \brief Get or create values.
  ///
  /// \param[in] key_tensor [Tensor] The key tensor.
  /// \param[in] insert_default_value [bool] The flag of insert default_value.
  /// \return The value tensor according the key tensor, return default_value if key_tensor is not exist.
  TensorPtr Get(const TensorPtr &key_tensor, bool insert_default_value);

  /// \brief Put or insert key value pairs.
  ///
  /// \param[in] key_tensor [Tensor] The key tensor.
  /// \param[in] value_tensor [Tensor] The value tensor.
  void Put(const TensorPtr &key_tensor, const TensorPtr &value_tensor);

  /// \brief Remove items with the given keys.
  ///
  /// \param[in] key_tensor [Tensor] The key tensor.
  void Erase(const TensorPtr &key_tensor);

  /// \brief Update MapTensor from exported data.
  ///
  /// \param[in] data [ExportData] The data.
  void Update(const ExportData &data);

  /// \brief Exported MapTensor data.
  ///
  /// \param[in] full [bool] True for full export, false for incremental export.
  /// \return The exported data.
  ExportData Export(bool full = false);

  /// \brief Exported MapTensor data from device.
  ///
  /// \param[in] full [bool] True for full export, false for incremental export.
  /// \return The exported data.
  ExportData ExportDataFromDevice(const DeviceSyncPtr &device_sync);

  /// \brief Get three tensor length from device data with tensor shape and type.
  ///
  /// \param[in] data_size [size_t] The size of device data.
  /// \return The length of tensor data.
  std::tuple<DataLenPair, DataLenPair, DataLenPair> GetTensorDataLen(size_t data_size);

  /// \brief Get the key tensor of MapTensor data.
  ///
  /// \return The key tensor.
  const TensorPtr &key_tensor() const { return key_tensor_; }

  /// \brief Get the value tensor of MapTensor data.
  ///
  /// \return The value tensor.
  const TensorPtr &value_tensor() const { return value_tensor_; }

  /// \brief Get the status tensor of MapTensor data.
  ///
  /// \return The status tensor.
  const TensorPtr &status_tensor() const { return status_tensor_; }

  void set_key_tensor(const TensorPtr key_tensor) { key_tensor_ = key_tensor; }

  void set_value_tensor(const TensorPtr value_tensor) { value_tensor_ = value_tensor; }

  void set_status_tensor(const TensorPtr status_tensor) { status_tensor_ = status_tensor; }

 private:
  // Data type of the keys.
  TypeId key_dtype_;

  // The shape of keys.
  ShapeVector key_shape_;

  // Default value. should be a scalar as the initial value or a string as the initializer name.
  ValuePtr default_value_;

  // Permission threshold: When an element is accessed more than the threshold, it will be actually inserted into map.
  ValuePtr permit_filter_value_;

  //  If the elements in the map are not used or updated within the time interval indicated by the threshold,
  //  these elements will be removed from the map.
  ValuePtr evict_filter_value_;

  // The shape of values
  ShapeVector value_shape_;

  // The size of keys, shape_ is (size_, value_shape_).
  size_t size_;

  // Key tensor of data.
  TensorPtr key_tensor_;

  // Value tensor of data.
  TensorPtr value_tensor_;

  // Status tensor of data.
  TensorPtr status_tensor_;
};
}  // namespace tensor
}  // namespace mindspore
#endif  // MINDSPORE_CORE_IR_MAP_TENSOR_H_
