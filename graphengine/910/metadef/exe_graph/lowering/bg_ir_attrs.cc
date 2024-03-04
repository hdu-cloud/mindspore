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
#include "exe_graph/lowering/bg_ir_attrs.h"

#include <cstring>
#include <securec.h>
#include "common/ge_common/debug/ge_log.h"
#include "graph/utils/math_util.h"
#include "graph/def_types.h"
#include "external/graph/types.h"
#include "common/checker.h"
#include "graph/debug/ge_util.h"

#include "exe_graph/runtime/runtime_attrs.h"
#include "exe_graph/runtime/continuous_vector.h"
#include "exe_graph/runtime/tensor.h"
#include "exe_graph/runtime/runtime_attrs_def.h"

namespace gert {
namespace bg {
namespace {
void GeShapeToGertShape(const ge::GeShape &ge_shape, gert::Shape &gert_shape) {
  gert_shape.SetDimNum(ge_shape.GetDimNum());
  for (size_t i = 0; i < ge_shape.GetDimNum(); ++i) {
    gert_shape.SetDim(i, ge_shape.GetDim(i));
  }
}
template<typename T, typename std::enable_if<std::is_fundamental<T>::value, int>::type = 0>
bool AppendFundAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto val = attr.Get<T>();
  GE_ASSERT_NOTNULL(val);
  std::vector<uint8_t> runtime_attr(sizeof(*val));
  GE_ASSERT_EOK(memcpy_s(runtime_attr.data(), sizeof(*val), val, sizeof(*val)));
  attrs.emplace_back(std::move(runtime_attr));
  return true;
}
bool AppendStrAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto str = attr.Get<std::string>();
  GE_ASSERT_NOTNULL(str);
  std::vector<uint8_t> runtime_attr(str->size() + 1);
  GE_ASSERT_EOK(strcpy_s(ge::PtrToPtr<uint8_t, ge::char_t>(runtime_attr.data()), str->size() + 1, str->c_str()));
  attrs.emplace_back(std::move(runtime_attr));
  return true;
}
template<typename T, typename std::enable_if<std::is_fundamental<T>::value, int>::type = 0>
bool AppendVectorAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto val = attr.Get<std::vector<T>>();
  GE_ASSERT_NOTNULL(val);
  size_t total_size;
  if (ge::MulOverflow(val->size(), sizeof(T), total_size)) {
    return false;
  }
  if (ge::AddOverflow(total_size, sizeof(ContinuousVector), total_size)) {
    return false;
  }

  std::vector<uint8_t> buf(total_size);
  auto cv = new (buf.data()) ContinuousVector();
  GE_ASSERT_NOTNULL(cv);
  cv->Init(val->size());
  cv->SetSize(val->size());

  size_t copy_size = val->size() * sizeof(T);
  if (!val->empty()) {
    GE_ASSERT_EOK(memcpy_s(cv->MutableData(), cv->GetCapacity() * sizeof(T), val->data(), copy_size));
  }
  attrs.emplace_back(std::move(buf));
  return true;
}

template<typename T, typename std::enable_if<std::is_fundamental<T>::value, int>::type = 0>
bool AppendVectorVectorAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto vector_vector_list = attr.Get<std::vector<std::vector<T>>>();
  GE_ASSERT_NOTNULL(vector_vector_list);

  size_t total_size = ContinuousVectorVector::GetOverHeadLength(vector_vector_list->size());
  for (const auto &inner_vec : *vector_vector_list) {
    size_t inner_vec_length = 0U;
    if (ge::MulOverflow(inner_vec.size(), sizeof(T), inner_vec_length)) {
      return false;
    }
    if (ge::AddOverflow(inner_vec_length, sizeof(ContinuousVector), inner_vec_length)) {
      return false;
    }
    if (ge::AddOverflow(total_size, inner_vec_length, total_size)) {
      return false;
    }
  }
  std::vector<uint8_t> buf(total_size);
  auto cvv = new (buf.data()) ContinuousVectorVector();
  GE_ASSERT_NOTNULL(cvv);
  cvv->Init(vector_vector_list->size());

  for (const auto &inner_list : *vector_vector_list) {
    auto cv = cvv->Add<T>(inner_list.size());
    GE_ASSERT_NOTNULL(cv);
    if (!inner_list.empty()) {
      const size_t copy_size = inner_list.size() * sizeof(T);
      GE_ASSERT_EOK(memcpy_s(cv->MutableData(), cv->GetCapacity() * sizeof(T), inner_list.data(), copy_size));
    }
  }

  attrs.emplace_back(std::move(buf));
  return true;
}
size_t GetGeTensorSize(const ge::GeTensor &tensor) {
  auto dt = tensor.GetTensorDesc().GetDataType();
  if (dt == ge::DT_STRING) {
    return tensor.GetData().GetSize();
  }
  auto shape_size = tensor.GetTensorDesc().GetShape().GetShapeSize();
  return static_cast<size_t>(ge::GetSizeInBytes(shape_size, dt));
}
bool AppendTensorAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto val = attr.Get<ge::GeTensor>();
  GE_ASSERT_NOTNULL(val);
  auto &tensor_desc = val->GetTensorDesc();
  auto shape_size = tensor_desc.GetShape().GetShapeSize();
  if (shape_size < 0) {
    GELOGE(ge::PARAM_INVALID, "Failed to append tensor attr, shape size less than 0");
    return false;
  }
  size_t total_size;
  size_t tensor_size = GetGeTensorSize(*val);
  auto tensor_holder = Tensor::CreateFollowing(val->GetTensorDesc().GetDataType(), tensor_size, total_size);
  GE_ASSERT_NOTNULL(tensor_holder);
  auto tensor = ge::PtrToPtr<uint8_t, Tensor>(tensor_holder.get());
  GeShapeToGertShape(tensor_desc.GetShape(), tensor->MutableStorageShape());
  GeShapeToGertShape(tensor_desc.GetOriginShape(), tensor->MutableOriginShape());
  tensor->SetOriginFormat(tensor_desc.GetOriginFormat());
  tensor->SetStorageFormat(tensor_desc.GetFormat());
  if (total_size < sizeof(Tensor)) {
    GELOGE(ge::PARAM_INVALID, "total_size[%zu] < size of Tensor[%zu]", total_size, sizeof(Tensor));
    return false;
  }
  const auto copy_len = total_size - sizeof(Tensor);
  if (copy_len != 0U) {
    GE_CHECK_GE(val->GetData().size(), total_size - sizeof(Tensor));
    const auto ret_copy = ge::GeMemcpy(tensor->GetData<uint8_t>(), total_size - sizeof(Tensor),
        val->GetData().GetData(), total_size - sizeof(Tensor));
    GE_ASSERT_TRUE((ret_copy == ge::SUCCESS), "memcpy_s failed, copy size is %zu", (total_size - sizeof(Tensor)));
  }

  std::vector<uint8_t> buf(total_size);
  const auto ret = ge::GeMemcpy(buf.data(), total_size, tensor_holder.get(), total_size);
  GE_ASSERT_TRUE((ret == ge::SUCCESS), "memcpy_s failed, copy size is %zu", total_size);
  attrs.emplace_back(std::move(buf));
  return true;
}
bool AppendDataTypeAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto val = attr.Get<ge::DataType>();
  GE_ASSERT_NOTNULL(val);
  std::vector<uint8_t> runtime_attr(sizeof(*val));
  GE_ASSERT_EOK(memcpy_s(runtime_attr.data(), sizeof(*val), val, sizeof(*val)));
  attrs.emplace_back(std::move(runtime_attr));
  return true;
}
bool AppendVectorDataTypeAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto val = attr.Get<std::vector<ge::DataType>>();
  GE_ASSERT_NOTNULL(val);
  size_t total_size = 0U;
  GE_ASSERT_TRUE(!ge::MulOverflow(val->size(), sizeof(ge::DataType), total_size),
                 "Mul overflow vec size %zu, elem size %zu.",
                 val->size(),
                 sizeof(ge::DataType));
  GE_ASSERT_TRUE(!ge::AddOverflow(total_size, sizeof(ContinuousVector), total_size),
                 "Add overflow total size %zu, size of vec %zu.",
                 total_size,
                 sizeof(ContinuousVector));

  std::vector<uint8_t> buf(total_size);
  auto cv = new (buf.data()) ContinuousVector();
  GE_ASSERT_NOTNULL(cv);
  cv->Init(val->size());
  cv->SetSize(val->size());

  size_t copy_size = val->size() * sizeof(ge::DataType);
  if (!val->empty()) {
    GE_ASSERT_EOK(memcpy_s(cv->MutableData(), cv->GetCapacity() * sizeof(ge::DataType), val->data(), copy_size));
  }
  attrs.emplace_back(std::move(buf));
  return true;
}
bool AppendVectorStrAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  auto val = attr.Get<std::vector<std::string>>();
  GE_ASSERT_NOTNULL(val);

  size_t total_str_size = 0U;
  for (auto i = 0U; i < (*val).size(); ++i) {
    const auto ele_str_size = (*val)[i].size() + 1;
    if (ge::AddOverflow(total_str_size, ele_str_size, total_str_size)) {
      GELOGW("Add over flow ele str size %zu, total_str_size %zu.", ele_str_size, total_str_size);
      return false;
    }
  }
  size_t total_size = 0U;
  if (ge::AddOverflow(total_str_size, sizeof(ContinuousVector), total_size)) {
    GELOGW("Add over flow ContinuousVector size %zu, total_str_size %zu.", sizeof(ContinuousVector), total_str_size);
    return false;
  }

  std::vector<uint8_t> buf(total_size);
  auto cv = new (buf.data()) ContinuousVector();
  GE_ASSERT_NOTNULL(cv);
  cv->Init(val->size());
  cv->SetSize(val->size());
  size_t offset = 0U;
  for (auto i = 0U; i < val->size(); ++i) {
    const auto ele_str_size = (*val)[i].size() + 1U;
    GE_ASSERT_EOK(strcpy_s(ge::PtrToPtr<uint8_t, char>(ge::PtrToPtr<void, uint8_t>(cv->MutableData()) + offset),
                           total_str_size,
                           (*val)[i].c_str()));
    offset += ele_str_size;
  }
  attrs.emplace_back(std::move(buf));
  return true;
}

bool AppendAttr(const ge::AnyValue &attr, std::vector<std::vector<uint8_t>> &attrs) {
  switch (attr.GetValueType()) {
    case ge::AnyValue::VT_FLOAT:
      return AppendFundAttr<float>(attr, attrs);
    case ge::AnyValue::VT_BOOL:
      return AppendFundAttr<bool>(attr, attrs);
    case ge::AnyValue::VT_INT:
      return AppendFundAttr<int64_t>(attr, attrs);
    case ge::AnyValue::VT_DATA_TYPE:
      return AppendDataTypeAttr(attr, attrs);
    case ge::AnyValue::VT_STRING:
      return AppendStrAttr(attr, attrs);
    case ge::AnyValue::VT_TENSOR:
      return AppendTensorAttr(attr, attrs);
    case ge::AnyValue::VT_LIST_FLOAT:
      return AppendVectorAttr<float>(attr, attrs);
    case ge::AnyValue::VT_LIST_INT:
      return AppendVectorAttr<int64_t>(attr, attrs);
    case ge::AnyValue::VT_LIST_DATA_TYPE:
      return AppendVectorDataTypeAttr(attr, attrs);
    case ge::AnyValue::VT_LIST_STRING:
      return AppendVectorStrAttr(attr, attrs);
    case ge::AnyValue::VT_LIST_LIST_FLOAT:
      return AppendVectorVectorAttr<float>(attr, attrs);
    case ge::AnyValue::VT_LIST_LIST_INT:
      return AppendVectorVectorAttr<int64_t>(attr, attrs);
    default:
      GELOGE(ge::FAILED, "Does not support the attr type now, attr type %d", attr.GetValueType());
      return false;
  }
}
bool GetAllIrAttrs(const ge::NodePtr &node, std::vector<std::vector<uint8_t>> &runtime_attrs) {
  auto all_attrs = ge::AttrUtils::GetAllAttrs(node->GetOpDesc());
  const auto &ir_attr_names = node->GetOpDesc()->GetIrAttrNames();
  for (const auto &attr_name : ir_attr_names) {
    const std::map<std::string, ge::AnyValue>::const_iterator &iter = all_attrs.find(attr_name);
    if (iter == all_attrs.cend()) {
      runtime_attrs.clear();
      GELOGI("Can not find the IR attr %s from node %s(%s), clear all attrs",
             attr_name.c_str(), node->GetName().c_str(), node->GetType().c_str());
      return true;
    }
    GE_ASSERT_TRUE(AppendAttr(iter->second, runtime_attrs));
  }
  return true;
}
std::unique_ptr<uint8_t[]> CreateAttrBuffer(const std::vector<std::vector<uint8_t>> &attrs, size_t &total_size) {
  total_size = sizeof(RuntimeAttrsDef);
  size_t offset_size = 0U;
  if (ge::MulOverflow(sizeof(size_t), attrs.size(), offset_size)) {
    GELOGE(ge::FAILED, "Failed to create attr buffer, total size overflow, attrs size may invalid %zu", attrs.size());
    return nullptr;
  }
  if (ge::AddOverflow(total_size, offset_size, total_size)) {
    GELOGE(ge::FAILED, "Failed to create attr buffer, total size overflow, attrs offset may invalid %zu", offset_size);
    return nullptr;
  }
  for (const auto &attr : attrs) {
    if (ge::AddOverflow(total_size, attr.size(), total_size)) {
      GELOGE(ge::FAILED,
             "Failed to create attr buffer, total size overflow, attr size may invalid %zu, current total size %zu",
             attr.size(), total_size);
      return nullptr;
    }
  }
  auto attr_holder = ge::ComGraphMakeUnique<uint8_t[]>(total_size);
  GE_ASSERT_NOTNULL(attr_holder);
  auto attr_def = ge::PtrToPtr<uint8_t, RuntimeAttrsDef>(attr_holder.get());
  attr_def->attr_num = attrs.size();
  memset(attr_def->reserved_, 0, sizeof(attr_def->reserved_));
  size_t current_offset = sizeof(RuntimeAttrsDef) + sizeof(size_t) * attr_def->attr_num;
  auto attr_pos = attr_holder.get();
  for (size_t i = 0; i < attrs.size(); ++i) {
    attr_def->offset[i] = current_offset;
    const auto ret = ge::GeMemcpy(attr_pos + current_offset, total_size - current_offset,
        attrs[i].data(), attrs[i].size());
    GE_ASSERT_TRUE((ret == ge::SUCCESS), "memcpy_s failed, copy size is %zu, dst size is %zu",
        attrs[i].size(), total_size - current_offset);
    current_offset += attrs[i].size();
  }
  return attr_holder;
}
}  // namespace
std::unique_ptr<uint8_t[]> CreateAttrBuffer(const ge::NodePtr &node, size_t &size) {
  return CreateAttrBuffer(node, {}, size);
}

std::unique_ptr<uint8_t[]> CreateAttrBuffer(const ge::NodePtr &node,
                                            const std::vector<ge::AnyValue> &runtime_attrs_list,
                                            size_t &size) {
  std::vector<std::vector<uint8_t>> runtime_attrs;
  GE_ASSERT_TRUE(GetAllIrAttrs(node, runtime_attrs));
  for (auto &runtime_attr : runtime_attrs_list) {
    AppendAttr(runtime_attr, runtime_attrs);
  }
  return CreateAttrBuffer(runtime_attrs, size);
}

std::unique_ptr<uint8_t[]> CreateAttrBufferWithoutIr(const ge::NodePtr &node,
                                                     const std::vector<ge::AnyValue> &runtime_attrs_list,
                                                     size_t &size) {
  (void)node;
  std::vector<std::vector<uint8_t>> runtime_attrs;
  for (auto &runtime_attr : runtime_attrs_list) {
    AppendAttr(runtime_attr, runtime_attrs);
  }
  return CreateAttrBuffer(runtime_attrs, size);
}
}  // namespace bg
}  // namespace gert
