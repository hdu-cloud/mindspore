/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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
#include "include/api/types.h"
#include <fstream>
#include <numeric>
#include "securec/include/securec.h"
#include "mindspore/core/ir/api_tensor_impl.h"
#include "mindspore/core/utils/convert_utils_base.h"
#include "utils/file_utils.h"
#include "common/utils.h"
#include "mindspore/core/ir/tensor.h"
#include "runtime/device/device_address.h"
#include "extendrt/utils/tensor_utils.h"
#include "extendrt/utils/tensor_default_impl.h"

namespace mindspore {
class Buffer::Impl {
 public:
  Impl() : data_() {}
  ~Impl() = default;
  Impl(const void *data, size_t data_len) {
    if (data != nullptr) {
      (void)SetData(data, data_len);
    } else {
      ResizeData(data_len);
    }
  }

  const void *Data() const { return data_.data(); }
  void *MutableData() { return data_.data(); }
  size_t DataSize() const { return data_.size(); }

  bool ResizeData(size_t data_len) {
    data_.resize(data_len);
    return true;
  }

  bool SetData(const void *data, size_t data_len) {
    ResizeData(data_len);
    if (DataSize() != data_len) {
      MS_LOG(ERROR) << "Set data failed, tensor current data size " << DataSize() << " not match data len " << data_len;
      return false;
    }

    if (data == nullptr) {
      return data_len == 0;
    }

    if (MutableData() == nullptr) {
      MS_LOG(ERROR) << "Set data failed, data len " << data_len;
      return false;
    }

    auto ret = memcpy_s(MutableData(), DataSize(), data, data_len);
    if (ret != 0) {
      MS_LOG(ERROR) << "Set data memcpy_s failed, ret = " << ret;
      return false;
    }
    return true;
  }

 protected:
  std::vector<uint8_t> data_;
};

MSTensor *MSTensor::CreateTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                                 const void *data, size_t data_len) noexcept {
  std::string name_str = CharToString(name);
  try {
    std::shared_ptr<Impl> impl =
      std::make_shared<TensorDefaultImpl>(name_str, type, shape, data, data_len, false, false);
    MSTensor *ret = new MSTensor(impl);
    return ret;
  } catch (const std::bad_alloc &) {
    MS_LOG(ERROR) << "Malloc memory failed.";
    return nullptr;
  } catch (...) {
    MS_LOG(ERROR) << "Unknown error occurred.";
    return nullptr;
  }
}

MSTensor *MSTensor::CreateRefTensor(const std::vector<char> &name, enum DataType type,
                                    const std::vector<int64_t> &shape, const void *data, size_t data_len,
                                    bool own_data) noexcept {
  std::string name_str = CharToString(name);
  try {
    std::shared_ptr<Impl> impl =
      std::make_shared<TensorDefaultImpl>(name_str, type, shape, data, data_len, true, own_data);
    if (data_len < impl->DataSize()) {
      MS_LOG(ERROR) << "The size " << data_len << " of data cannot be less that the memory size required by the shape "
                    << shape << " and data type " << TypeIdToString(static_cast<enum TypeId>(type));
      return nullptr;
    }
    MSTensor *ret = new MSTensor(impl);
    return ret;
  } catch (const std::bad_alloc &) {
    MS_LOG(ERROR) << "Malloc memory failed.";
    return nullptr;
  } catch (...) {
    MS_LOG(ERROR) << "Unknown error occurred.";
    return nullptr;
  }
}

MSTensor MSTensor::CreateDeviceTensor(const std::vector<char> &name, enum DataType type,
                                      const std::vector<int64_t> &shape, void *data, size_t data_size) noexcept {
  std::string name_str = CharToString(name);
  try {
    auto impl = std::make_shared<TensorDefaultImpl>(name_str, type, shape);
    if (data_size < impl->DataSize()) {
      MS_LOG(ERROR) << "The size " << data_size << " of data cannot be less that the memory size required by the shape "
                    << shape << " and data type " << TypeIdToString(static_cast<enum TypeId>(type));
      return MSTensor(nullptr);
    }
    impl->SetDeviceData(data);
    return MSTensor(impl);
  } catch (const std::bad_alloc &) {
    MS_LOG(ERROR) << "Malloc memory failed.";
    return MSTensor(nullptr);
  } catch (...) {
    MS_LOG(ERROR) << "Unknown error occurred.";
    return MSTensor(nullptr);
  }
}

MSTensor *MSTensor::CreateTensorFromFile(const std::vector<char> &file, enum DataType type,
                                         const std::vector<int64_t> &shape) noexcept {
  std::string file_str = CharToString(file);

  try {
    auto realpath = FileUtils::GetRealPath(file_str.c_str());
    if (!realpath.has_value()) {
      MS_LOG(ERROR) << "Get real path failed, path=" << file_str;
      return nullptr;
    }

    // Read image file
    auto file_path = realpath.value();
    if (file_path.empty()) {
      MS_LOG(ERROR) << "Can not find any input file.";
      return nullptr;
    }

    std::ifstream ifs(file_path, std::ios::in | std::ios::binary);
    if (!ifs.good()) {
      MS_LOG(ERROR) << "File: " + file_path + " does not exist.";
      return nullptr;
    }
    if (!ifs.is_open()) {
      MS_LOG(ERROR) << "File: " + file_path + " open failed.";
      return nullptr;
    }

    auto &io_seekg1 = ifs.seekg(0, std::ios::end);
    if (!io_seekg1.good() || io_seekg1.fail() || io_seekg1.bad()) {
      ifs.close();
      MS_LOG(ERROR) << "Failed to seekg file: " + file_path;
      return nullptr;
    }

    size_t size = static_cast<size_t>(ifs.tellg());
    std::vector<int64_t> tensor_shape;
    tensor_shape = shape.empty() ? std::vector<int64_t>{static_cast<int64_t>(size)} : shape;
    MSTensor *ret = new MSTensor(file_path, type, tensor_shape, nullptr, size);

    auto &io_seekg2 = ifs.seekg(0, std::ios::beg);
    if (!io_seekg2.good() || io_seekg2.fail() || io_seekg2.bad()) {
      ifs.close();
      MS_LOG(ERROR) << "Failed to seekg file: " + file_path;
      return nullptr;
    }

    std::map<enum DataType, size_t> TypeByte = {
      {DataType::kTypeUnknown, 0},       {DataType::kObjectTypeString, 0},  {DataType::kNumberTypeBool, 1},
      {DataType::kNumberTypeInt8, 1},    {DataType::kNumberTypeInt16, 2},   {DataType::kNumberTypeInt32, 4},
      {DataType::kNumberTypeInt64, 8},   {DataType::kNumberTypeUInt8, 1},   {DataType::kNumberTypeUInt16, 2},
      {DataType::kNumberTypeUInt32, 4},  {DataType::kNumberTypeUInt64, 8},  {DataType::kNumberTypeFloat16, 2},
      {DataType::kNumberTypeFloat32, 4}, {DataType::kNumberTypeFloat64, 8},
    };

    if (LongToSize(ret->ElementNum()) * TypeByte[type] != size) {
      ifs.close();
      MS_LOG(ERROR) << "Tensor data size: " << LongToSize(ret->ElementNum()) * TypeByte[type]
                    << " not match input data length: " << size;
      return nullptr;
    }

    auto &io_read = ifs.read(reinterpret_cast<char *>(ret->MutableData()), static_cast<std::streamsize>(size));
    if (!io_read.good() || io_read.fail() || io_read.bad()) {
      ifs.close();
      MS_LOG(ERROR) << "Failed to read file: " + file_path;
      return nullptr;
    }
    ifs.close();

    return ret;
  } catch (const std::bad_alloc &) {
    MS_LOG(ERROR) << "Malloc memory failed.";
    return nullptr;
  } catch (...) {
    MS_LOG(ERROR) << "Unknown error occurred.";
    return nullptr;
  }
}

MSTensor *MSTensor::CharStringsToTensor(const std::vector<char> &name, const std::vector<std::vector<char>> &str) {
  size_t tensor_num = str.size();
  std::vector<int32_t> offset(tensor_num + 1);
  const size_t extra_offset_num = 2;
  offset[0] = static_cast<int32_t>(sizeof(int32_t) * (tensor_num + extra_offset_num));
  for (size_t i = 0; i < tensor_num; i++) {
    offset[i + 1] = offset[i] + str[i].size();
  }
  std::vector<int64_t> shape = {offset[tensor_num]};
  auto tensor = CreateTensor(name, DataType::kObjectTypeString, shape, nullptr, offset[tensor_num]);
  if (tensor == nullptr) {
    MS_LOG(ERROR) << "create tensor failed.";
    return nullptr;
  }
  void *data = tensor->MutableData();
  int32_t *string_info = reinterpret_cast<int32_t *>(data);
  if (string_info == nullptr) {
    MS_LOG(ERROR) << "tensor data is nullptr.";
    DestroyTensorPtr(tensor);
    return nullptr;
  }
  char *string_data = reinterpret_cast<char *>(data);
  string_info[0] = static_cast<int32_t>(tensor_num);
  for (size_t i = 0; i <= tensor_num; i++) {
    string_info[i + 1] = offset[i];
  }
  for (size_t i = 0; i < tensor_num; i++) {
    memcpy_s(string_data + offset[i], str[i].size(), str[i].data(), str[i].size());
  }
  return tensor;
}

std::vector<std::vector<char>> MSTensor::TensorToStringChars(const MSTensor &tensor) {
  if (tensor == nullptr || tensor.DataType() != DataType::kObjectTypeString || tensor.DataSize() < 4) {
    MS_LOG(ERROR) << "Invalid tensor.";
    return {};
  }

  std::vector<std::vector<char>> strings;
  auto host_data = tensor.Data();
  const int32_t *data = reinterpret_cast<const int32_t *>(host_data.get());
  int32_t str_num = data[0];
  if (str_num == 0) {
    return {};
  }
  if (str_num < 0) {
    MS_LOG(ERROR) << "str num " << str_num << " cannot be negative.";
    return {};
  }

  if (tensor.DataSize() < (str_num + 1) * sizeof(int32_t)) {
    MS_LOG(ERROR) << "Invalid tensor data size " << tensor.DataSize() << ", need "
                  << IntToSize(str_num + 1) * sizeof(int32_t) << " at least for " << str_num << " strings.";
    return {};
  }
  for (size_t i = 0; i < static_cast<size_t>(str_num); ++i) {
    strings.push_back({});
    auto &str = strings[i];
    int32_t str_len;
    int32_t offset = data[i + 1];
    if (i + 1 != static_cast<size_t>(str_num)) {
      str_len = data[i + 1 + 1] - offset;
    } else {
      str_len = tensor.DataSize() - offset;
    }

    if (str_len == 0) {
      continue;
    }

    if (str_len < 0) {
      MS_LOG(ERROR) << "str " << i << " len " << str_len << " cannot be negative.";
      return {};
    }

    str.resize(str_len);
    const uint8_t *cur_data = reinterpret_cast<const uint8_t *>(data) + offset;
    auto ret = memcpy_s(reinterpret_cast<void *>(str.data()), str.size(), cur_data, str_len);
    if (ret != 0) {
      MS_LOG(ERROR) << "memcpy_s failed, ret = " << ret;
      return {};
    }
  }

  return strings;
}

void MSTensor::DestroyTensorPtr(MSTensor *tensor) noexcept {
  if (tensor != nullptr) {
    delete tensor;
  }
}

MSTensor::MSTensor() : impl_(std::make_shared<TensorDefaultImpl>()) {}
MSTensor::MSTensor(std::nullptr_t) : impl_(nullptr) {}
MSTensor::MSTensor(const std::shared_ptr<Impl> &impl) : impl_(impl) { MS_EXCEPTION_IF_NULL(impl); }
MSTensor::MSTensor(const std::vector<char> &name, enum DataType type, const std::vector<int64_t> &shape,
                   const void *data, size_t data_len)
    : impl_(std::make_shared<TensorDefaultImpl>(CharToString(name), type, shape, data, data_len, false, false)) {}
MSTensor::~MSTensor() = default;

bool MSTensor::operator==(std::nullptr_t) const { return impl_ == nullptr; }

bool MSTensor::operator!=(std::nullptr_t) const { return impl_ != nullptr; }

bool MSTensor::operator==(const MSTensor &tensor) const { return impl_ == tensor.impl_; }

bool MSTensor::operator!=(const MSTensor &tensor) const { return impl_ != tensor.impl_; }

MSTensor *MSTensor::Clone() const {
  MS_EXCEPTION_IF_NULL(impl_);
  try {
    MSTensor *ret = new MSTensor();
    ret->impl_ = impl_->Clone();
    return ret;
  } catch (const std::bad_alloc &) {
    MS_LOG(ERROR) << "Malloc memory failed.";
    return nullptr;
  } catch (...) {
    MS_LOG(ERROR) << "Unknown error occurred.";
    return nullptr;
  }
}

std::vector<char> MSTensor::CharName() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return StringToChar(impl_->Name());
}

enum DataType MSTensor::DataType() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->DataType();
}

const std::vector<int64_t> &MSTensor::Shape() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Shape();
}

int64_t MSTensor::ElementNum() const {
  MS_EXCEPTION_IF_NULL(impl_);
  const auto &shape = impl_->Shape();
  if (shape.empty()) {
    // element number of scalar is 1
    return 1;
  }
  return std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int64_t>());
}

std::shared_ptr<const void> MSTensor::Data() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Data();
}

void *MSTensor::MutableData() {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->MutableData();
}

size_t MSTensor::DataSize() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->DataSize();
}

bool MSTensor::IsDevice() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->IsDevice();
}

void MSTensor::SetShape(const std::vector<int64_t> &shape) {
  MS_EXCEPTION_IF_NULL(impl_);
  std::static_pointer_cast<MutableTensorImpl>(impl_)->SetShape(shape);
}

void MSTensor::SetDataType(enum DataType data_type) {
  MS_EXCEPTION_IF_NULL(impl_);
  std::static_pointer_cast<MutableTensorImpl>(impl_)->SetDataType(data_type);
}

void MSTensor::SetTensorName(const std::vector<char> &tensor_name) {
  MS_EXCEPTION_IF_NULL(impl_);
  std::static_pointer_cast<MutableTensorImpl>(impl_)->SetName(CharToString(tensor_name));
}

void MSTensor::SetAllocator(std::shared_ptr<Allocator> allocator) {
  MS_EXCEPTION_IF_NULL(impl_);
  std::static_pointer_cast<MutableTensorImpl>(impl_)->SetAllocator(allocator);
}

std::shared_ptr<Allocator> MSTensor::allocator() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return std::static_pointer_cast<MutableTensorImpl>(impl_)->GetAllocator();
}

void MSTensor::SetFormat(mindspore::Format format) {
  MS_EXCEPTION_IF_NULL(impl_);
  std::static_pointer_cast<MutableTensorImpl>(impl_)->SetFormat(format);
}

mindspore::Format MSTensor::format() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return std::static_pointer_cast<MutableTensorImpl>(impl_)->Format();
}

void MSTensor::SetData(void *data, bool own_data) {
  MS_EXCEPTION_IF_NULL(impl_);
  std::static_pointer_cast<MutableTensorImpl>(impl_)->SetData(data, own_data);
}

void MSTensor::SetDeviceData(void *data) {
  MS_EXCEPTION_IF_NULL(impl_);
  std::static_pointer_cast<MutableTensorImpl>(impl_)->SetDeviceData(data);
}

void *MSTensor::GetDeviceData() {
  MS_EXCEPTION_IF_NULL(impl_);
  return std::static_pointer_cast<MutableTensorImpl>(impl_)->GetDeviceData();
}

std::vector<QuantParam> MSTensor::QuantParams() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return std::static_pointer_cast<MutableTensorImpl>(impl_)->GetQuantParams();
}

void MSTensor::SetQuantParams(std::vector<QuantParam> quant_param) {
  MS_EXCEPTION_IF_NULL(impl_);
  std::static_pointer_cast<MutableTensorImpl>(impl_)->SetQuantParams(quant_param);
}

Buffer::Buffer() : impl_(std::make_shared<Impl>()) {}
Buffer::Buffer(const void *data, size_t data_len) : impl_(std::make_shared<Impl>(data, data_len)) {}
Buffer::~Buffer() = default;

Buffer Buffer::Clone() const {
  MS_EXCEPTION_IF_NULL(impl_);
  Buffer ret;
  ret.impl_ = std::make_shared<Impl>(*impl_);
  return ret;
}

const void *Buffer::Data() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->Data();
}

void *Buffer::MutableData() {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->MutableData();
}

size_t Buffer::DataSize() const {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->DataSize();
}

bool Buffer::ResizeData(size_t data_len) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->ResizeData(data_len);
}

bool Buffer::SetData(const void *data, size_t data_len) {
  MS_EXCEPTION_IF_NULL(impl_);
  return impl_->SetData(data, data_len);
}

std::vector<char> CharVersion() { return {}; }
}  // namespace mindspore
