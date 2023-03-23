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

#include "kernel/kernel.h"

#include <functional>
#include <algorithm>
#include <stack>
#include "utils/ms_context.h"
#include "utils/anf_utils.h"
#include "runtime/device/ms_device_shape_transfer.h"
#include "backend/common/session/anf_runtime_algorithm.h"
#include "include/common/utils/anfalgo.h"
#include "backend/common/optimizer/helper.h"
#include "kernel/common_utils.h"
using AnfAlgo = mindspore::session::AnfRuntimeAlgorithm;
namespace mindspore {
namespace kernel {
constexpr int64_t kInvalidShape = -2;

string KernelTensor::GetAbstractName() const {
  if (tensor_info_.abstract_base == nullptr) {
    return "null(no abstract base)";
  }
  return tensor_info_.abstract_base->ToString();
}

bool KernelTensor::IsDynamicShape() const {
  auto shape = this->GetShapeVector();
  return std::any_of(shape.cbegin(), shape.cend(), [](auto i) { return i < 0; });
}

size_t KernelTensor::GetSizeInBytes() const {
  auto unit_size = GetTypeByte(TypeIdToType(GetDtype()));
  auto shapes = this->GetShapeVector();
  if (shapes.size() == 0) {
    return unit_size;
  }

  auto cur_size = unit_size;
  for (const auto val : shapes) {
    if (val < 0) {
      MS_LOG_EXCEPTION << "Invalid shape value " << val << " for calculating size. Abstract name: " << GetAbstractName()
                       << ". Please contact MindSpore support.";
    }
    if (val == 0) {
      MS_LOG_WARNING << "One dim of the shape is 0. Abstract name: " << GetAbstractName() << ".";
    }
    cur_size *= val;
  }

  return cur_size;
}

TypeId KernelTensor::GetDtype() const {
  if (tensor_info_.abstract_base == nullptr) {
    return TypeId::kTypeUnknown;
  }

  auto type_ptr = tensor_info_.abstract_base->BuildType();
  if (type_ptr == nullptr || !type_ptr->isa<TensorType>()) {
    return TypeId::kTypeUnknown;
  }

  auto tensor_ptr = type_ptr->cast<TensorTypePtr>();
  auto elem = tensor_ptr->element();
  if (elem == nullptr) {
    return TypeId::kTypeUnknown;
  }
  return elem->type_id();
}

ShapeVector KernelTensor::GetShapeVector() const {
  auto base_shape_ptr = GetBaseShape();
  if (base_shape_ptr == nullptr || !base_shape_ptr->isa<abstract::Shape>()) {
    return {};
  }
  auto shape = base_shape_ptr->cast<abstract::ShapePtr>()->shape();
  return shape;
}

ShapeVector KernelTensor::GetMaxShape() const {
  auto base_shape_ptr = GetBaseShape();
  if (base_shape_ptr == nullptr || !base_shape_ptr->isa<abstract::Shape>()) {
    return {};
  }

  return base_shape_ptr->cast<abstract::ShapePtr>()->max_shape();
}

std::vector<TypeId> KernelTensor::GetListOrTupleDtype() const {
  if (tensor_info_.abstract_base == nullptr) {
    return {TypeId::kTypeUnknown};
  }

  auto type_ptr = tensor_info_.abstract_base->BuildType();
  if (type_ptr == nullptr || !type_ptr->isa<List>() || !type_ptr->isa<Tuple>()) {
    return {TypeId::kTypeUnknown};
  }

  std::vector<TypeId> types;
  if (type_ptr->isa<List>()) {
    auto tuple_ptr = type_ptr->cast<TuplePtr>();
    auto elements = tuple_ptr->elements();
    (void)std::transform(elements.begin(), elements.end(), std::back_inserter(types),
                         [](const TypePtr &t) { return t->type_id(); });
  } else if (type_ptr->isa<Tuple>()) {
    auto tuple_ptr = type_ptr->cast<TuplePtr>();
    auto elements = tuple_ptr->elements();
    (void)std::transform(elements.begin(), elements.end(), std::back_inserter(types),
                         [](const TypePtr &t) { return t->type_id(); });
  } else {
    types.push_back(TypeId::kTypeUnknown);
  }

  return types;
}

ShapeArray KernelTensor::GetListOrTupleShapeVector() const {
  auto base_shape_ptr = GetBaseShape();
  // ListShape or TupleShape is inherited from SequenceShape.
  if (base_shape_ptr == nullptr || !base_shape_ptr->isa<abstract::SequenceShape>()) {
    return {};
  }
  auto sequence_shape_ptr = base_shape_ptr->cast<abstract::SequenceShapePtr>();
  auto base_shape_list = sequence_shape_ptr->shape();
  std::vector<std::vector<int64_t>> shape_vector_list;
  for (auto base_shape : base_shape_list) {
    if (base_shape == nullptr || !base_shape->isa<abstract::Shape>()) {
      return {};
    }
    auto tmp_shape = base_shape->cast<abstract::ShapePtr>()->shape();
    shape_vector_list.push_back(tmp_shape);
  }

  return shape_vector_list;
}

void KernelTensor::SetDtype(const TypePtr &dtype) {
  if (tensor_info_.abstract_base == nullptr) {
    return;
  }
  tensor_info_.abstract_base->set_type(dtype);
}

void KernelTensor::SetShapeVector(const std::vector<int64_t> &shape) {
  if (tensor_info_.abstract_base == nullptr) {
    return;
  }
  tensor_info_.abstract_base->set_shape(std::make_shared<abstract::Shape>(shape));
}

abstract::BaseShapePtr KernelTensor::GetBaseShape() const {
  if (tensor_info_.abstract_base == nullptr) {
    return nullptr;
  }
  return tensor_info_.abstract_base->BuildShape();
}

void KernelTensor::SetBaseShape(const abstract::BaseShapePtr &base_shape) {
  if (tensor_info_.abstract_base == nullptr) {
    return;
  }
  tensor_info_.abstract_base->set_shape(base_shape);
}

const std::vector<int64_t> &KernelTensor::GetDeviceShapeAdaptively() const {
  return tensor_info_.device_shape_adaptively;
}

void KernelTensor::SetDeviceShapeAdaptively(const std::vector<int64_t> &device_shape_adaptively) {
  tensor_info_.device_shape_adaptively = device_shape_adaptively;
}

int KernelMod::Resize(const BaseOperatorPtr &base_operator, const std::vector<KernelTensorPtr> &inputs,
                      const std::vector<KernelTensorPtr> &outputs,
                      const std::map<uint32_t, tensor::TensorPtr> & /* inputsOnHost */) {
  auto ret = KRET_OK;
  workspace_size_list_.clear();
  input_size_list_.clear();
  input_shapes_.clear();
  for (auto &input : inputs) {
    size_t tensor_size = 0;
    size_t type_size = GetTypeByte(TypeIdToType(input->GetDtype()));
    MS_EXCEPTION_IF_NULL(input);
    auto shape = input->GetShapeVector();
    if (!IsValidShape(shape)) {
      // early stop if any input shape contains -1/-2, which means input shape is dynamic
      return KRET_UNKNOWN_SHAPE;
    } else {
      tensor_size =
        shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
      tensor_size = std::max(tensor_size, type_size);
    }
    (void)input_size_list_.emplace_back(tensor_size);
    input_shapes_.emplace_back(shape);
  }
  output_shapes_.clear();
  output_size_list_.clear();
  for (auto &output : outputs) {
    size_t tensor_size = 0;
    size_t type_size = GetTypeByte(TypeIdToType(output->GetDtype()));
    MS_EXCEPTION_IF_NULL(output);
    auto shape = output->GetShapeVector();
    if (!IsValidShape(shape)) {
      // Note:
      // If output shape is unknown, the op is a compute-depended op and max_shape should not be empty,
      // and the output_size_list_ can be set by max_shape
      auto max_shape = output->GetMaxShape();
      if (max_shape.empty()) {
        auto primitive = base_operator->GetPrim();
        MS_ERROR_IF_NULL(primitive);
        MS_LOG(DEBUG) << "For " << primitive->name()
                      << ", the max_shape should not be empty when input shape is known.";
        ret = KRET_UNKNOWN_OUT_SHAPE;
      } else {
        tensor_size = SizeOf(max_shape) * type_size;
        ret = KRET_UNKNOWN_OUT_SHAPE;
      }
    } else {
      tensor_size =
        shape.empty() ? type_size : std::accumulate(shape.begin(), shape.end(), type_size, std::multiplies<size_t>());
      tensor_size = std::max(tensor_size, type_size);
    }
    (void)output_size_list_.emplace_back(tensor_size);
    output_shapes_.emplace_back(shape);
  }
  return static_cast<int>(ret);
}

std::vector<int64_t> GetIntValueFromData(void *const data_c, const TypeId &type_id, size_t data_size,
                                         const size_t input_index, const std::string &kernel_name) {
  std::vector<int64_t> tensor_value;
  MS_EXCEPTION_IF_NULL(data_c);
  if (type_id == kNumberTypeInt32) {
    auto tensor_data = reinterpret_cast<int32_t *>(data_c);
    MS_EXCEPTION_IF_NULL(tensor_data);
    tensor_value.assign(tensor_data, tensor_data + data_size / sizeof(int32_t));
  } else if (type_id == kNumberTypeInt64) {
    auto tensor_data = reinterpret_cast<int64_t *>(data_c);
    MS_EXCEPTION_IF_NULL(tensor_data);
    tensor_value.assign(tensor_data, tensor_data + data_size / sizeof(int64_t));
  } else {
    MS_EXCEPTION(TypeError) << "For '" << kernel_name << "', the " << input_index
                            << "th input must be a Tensor[Int64] or Tensor[Int32] type, but got "
                            << TypeIdLabel(type_id);
  }
  return tensor_value;
}

std::optional<std::vector<int64_t>> TryGetIntValueFromInputs(const std::vector<KernelTensorPtr> &inputs,
                                                             const size_t input_index, const std::string &kernel_name,
                                                             bool data_from_host) {
  if (inputs.size() <= input_index) {
    MS_LOG(DEBUG) << "For '" << kernel_name << "', inputs size is " << inputs.size() << ", but require " << input_index;
    return std::nullopt;
  }

  AddressPtr data{nullptr};
  if (data_from_host) {
    data = inputs[input_index]->GetHostData();
  } else {
    data = inputs[input_index]->GetData();
  }

  // The value of dynamic attr can only be obtained after the InferOp() is executed.
  if (data == nullptr || data->addr == nullptr) {
    MS_LOG(DEBUG) << "For '" << kernel_name << "', fail to find the " << input_index << "th input's data.";
    return std::nullopt;
  }

  const auto &data_format = inputs[input_index]->GetFormat();
  if (data_format != mindspore::Format::DEFAULT_FORMAT && data_format != mindspore::Format::NCHW) {
    MS_LOG(EXCEPTION) << "For '" << kernel_name << "',  the format of the " << input_index
                      << "th input currently should be the default format and does not support " << data_format;
  }

  return GetIntValueFromData(data->addr, inputs[input_index]->GetDtype(), data->size, input_index, kernel_name);
}

bool TryGetIntValue(const CNodePtr &kernel_node, const size_t input_index, std::vector<int64_t> *attr_value,
                    bool data_from_host) {
  auto args = GetArgsFromCNode(kernel_node);
  if (args == nullptr) {
    return false;
  }
  auto op_name = common::AnfAlgo::GetCNodeName(kernel_node);
  auto res = TryGetIntValueFromInputs(args->inputs, input_index, op_name, data_from_host);
  if (!res.has_value()) {
    return false;
  }
  *attr_value = res.value();
  return true;
}
}  // namespace kernel
}  // namespace mindspore
