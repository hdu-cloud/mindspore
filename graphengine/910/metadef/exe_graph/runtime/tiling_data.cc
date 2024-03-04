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
#include "exe_graph/runtime/tiling_data.h"
#include "common/util/tiling_utils.h"
#include "graph/def_types.h"
#include "common/ge_common/util.h"

namespace gert {
namespace {
using AppendAttrFunc = std::function<ge::graphStatus(TilingData *, const RuntimeAttrs *, const size_t)>;

template<typename T>
ge::graphStatus CheckOverFlow(const size_t attr_size, const size_t tiling_data_size, const size_t capacity) {
  size_t append_size;
  if (ge::MulOverflow(sizeof(T), attr_size, append_size)) {
    GELOGE(ge::GRAPH_FAILED, "Mul over flow. Attr size is %zu", attr_size);
    return ge::GRAPH_FAILED;
  }
  size_t after_size;
  if (ge::AddOverflow(tiling_data_size, append_size, after_size)) {
    GELOGE(ge::GRAPH_FAILED, "Add over flow. Tiling data size is [%zu], append size is [%zu].", tiling_data_size,
           append_size);
    return ge::GRAPH_FAILED;
  }
  if (after_size > capacity) {
    GELOGE(ge::GRAPH_FAILED, "After size [%zu] is out of range, tiling data capacity is [%zu].", after_size, capacity);
    return ge::GRAPH_FAILED;
  }
  return ge::GRAPH_SUCCESS;
}

// get basic type attr and append
template<typename T>
ge::graphStatus AppendAttr(TilingData *tiling_data, const RuntimeAttrs *attrs, const size_t attr_index) {
  GE_CHECK_NOTNULL(tiling_data);
  const T *attr = attrs->GetAttrPointer<T>(attr_index);
  GE_CHECK_NOTNULL(attr);
  return tiling_data->Append<T>(*attr);
}

// get list attr and append
template<typename T>
ge::graphStatus AppendListAttr(TilingData *tiling_data, const RuntimeAttrs *attrs, const size_t attr_index) {
  GE_CHECK_NOTNULL(tiling_data);
  const ContinuousVector *attr = attrs->GetAttrPointer<ContinuousVector>(attr_index);
  GE_CHECK_NOTNULL(attr);
  return tiling_data->Append<T>(reinterpret_cast<const T *>(attr->GetData()), attr->GetSize());
}

// get basic type attr to convert and append
template<typename T1, typename T2>
ge::graphStatus AppendConvertedAttr(TilingData *tiling_data, const RuntimeAttrs *attrs, const size_t attr_index) {
  GE_CHECK_NOTNULL(tiling_data);
  const T1 *attr = attrs->GetAttrPointer<T1>(attr_index);
  GE_CHECK_NOTNULL(attr);
  if (!ge::IntegerChecker<T2>::Compat(*attr)) {
    GELOGE(ge::GRAPH_FAILED, "[Check][Param] attr[%zu] overflow, large than max dst type", attr_index);
  }
  T2 attr_data = static_cast<T2>(*attr);
  return tiling_data->Append<T2>(attr_data);
}

// get list attr to convert and append
template<typename T1, typename T2>
ge::graphStatus AppendConvertedListAttr(TilingData *tiling_data, const RuntimeAttrs *attrs, const size_t attr_index) {
  GE_CHECK_NOTNULL(tiling_data);
  const ContinuousVector *attr = attrs->GetAttrPointer<ContinuousVector>(attr_index);
  GE_CHECK_NOTNULL(attr);
  const auto ret = CheckOverFlow<T2>(attr->GetSize(), tiling_data->GetDataSize(), tiling_data->GetCapacity());
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }

  auto data_size = tiling_data->GetDataSize();
  const T1 *attr_data = reinterpret_cast<const T1 *>(attr->GetData());
  GE_CHECK_NOTNULL(attr_data);
  for (size_t i = 0UL; i < attr->GetSize(); ++i) {
    *reinterpret_cast<T2 *>(reinterpret_cast<uint8_t *>(tiling_data->GetData()) + data_size) =
        static_cast<T2>(attr_data[i]);
    data_size += sizeof(T2);
  }
  tiling_data->SetDataSize(data_size);
  return ge::GRAPH_SUCCESS;
}

// get char * attr to append
ge::graphStatus AppendStrAttr(TilingData *tiling_data, const RuntimeAttrs *attrs, const size_t attr_index) {
  GE_CHECK_NOTNULL(tiling_data);
  const char *attr = attrs->GetAttrPointer<char>(attr_index);
  GE_CHECK_NOTNULL(attr);
  return tiling_data->Append<char>(attr, strlen(attr));
}

// convert float32 attr to uint16 and append
ge::graphStatus AppendConvertedFpAttr(TilingData *tiling_data, const RuntimeAttrs *attrs, const size_t attr_index) {
  GE_CHECK_NOTNULL(tiling_data);
  const float *attr = attrs->GetAttrPointer<float>(attr_index);
  GE_CHECK_NOTNULL(attr);
  const uint16_t target_attr_val = optiling::FloatToUint16(*attr);
  return tiling_data->Append<uint16_t>(target_attr_val);
}

// convert list_float32 attr to list_uint16 and append
ge::graphStatus AppendConvertedListFpAttr(TilingData *tiling_data, const RuntimeAttrs *attrs, const size_t attr_index) {
  GE_CHECK_NOTNULL(tiling_data);
  const ContinuousVector *attr = attrs->GetAttrPointer<ContinuousVector>(attr_index);
  GE_CHECK_NOTNULL(attr);
  const auto ret = CheckOverFlow<uint16_t>(attr->GetSize(), tiling_data->GetDataSize(), tiling_data->GetCapacity());
  if (ret != ge::GRAPH_SUCCESS) {
    return ret;
  }

  auto data_size = tiling_data->GetDataSize();
  const float *attr_data = ge::PtrToPtr<const void, const float>(attr->GetData());
  GE_CHECK_NOTNULL(attr_data);
  for (size_t i = 0UL; i < attr->GetSize(); ++i) {
    *ge::PtrToPtr<uint8_t, uint16_t>(ge::PtrToPtr<void, uint8_t>(tiling_data->GetData()) + data_size) =
        optiling::FloatToUint16(attr_data[i]);
    data_size += sizeof(uint16_t);
  }
  tiling_data->SetDataSize(data_size);
  return ge::GRAPH_SUCCESS;
}

template<AttrDataType SRC, AttrDataType DST>
class AttrTable {
public:
  explicit AttrTable(const AppendAttrFunc default_val) {
    for (size_t i = 0UL; i < static_cast<size_t>(SRC); ++i) {
      for (size_t j = 0UL; j < static_cast<size_t>(DST); ++j) {
        elements[i][j] = default_val;
      }
    }
  }

  AppendAttrFunc Find(AttrDataType src, AttrDataType dst) const {
    if (src >= SRC || dst >= DST) {
      return nullptr;
    }
    return elements[static_cast<size_t>(src)][static_cast<size_t>(dst)];
  }

  AttrTable &Add(AttrDataType src, AttrDataType dst, AppendAttrFunc func) {
    elements[static_cast<size_t>(src)][static_cast<size_t>(dst)] = func;
    return *this;
  }

private:
  AppendAttrFunc elements[static_cast<size_t>(SRC)][static_cast<size_t>(DST)];
};

const auto kAttrTable =
    AttrTable<AttrDataType::kTypeEnd, AttrDataType::kTypeEnd>(nullptr)
        .Add(AttrDataType::kBool, AttrDataType::kBool, AppendAttr<bool>)
        .Add(AttrDataType::kInt64, AttrDataType::kInt32, AppendAttr<int32_t>)
        .Add(AttrDataType::kFloat32, AttrDataType::kFloat32, AppendAttr<float>)
        .Add(AttrDataType::kListInt64, AttrDataType::kListInt32, AppendConvertedListAttr<int64_t, int32_t>)
        .Add(AttrDataType::kListFloat32, AttrDataType::kListFloat32, AppendListAttr<float>)
        .Add(AttrDataType::kString, AttrDataType::kString, AppendStrAttr)
        .Add(AttrDataType::kInt64, AttrDataType::kUint32, AppendConvertedAttr<int64_t, uint32_t>)
        .Add(AttrDataType::kListInt64, AttrDataType::kListUint32, AppendConvertedListAttr<int64_t, uint32_t>)
        .Add(AttrDataType::kFloat32, AttrDataType::kFloat16, AppendConvertedFpAttr)
        .Add(AttrDataType::kListFloat32, AttrDataType::kListFloat16, AppendConvertedListFpAttr)
        .Add(AttrDataType::kFloat32, AttrDataType::kInt32, AppendConvertedAttr<float, int32_t>)
        .Add(AttrDataType::kListFloat32, AttrDataType::kListInt32, AppendConvertedListAttr<float, int32_t>)
        .Add(AttrDataType::kInt32, AttrDataType::kInt32, AppendAttr<int32_t>)
        .Add(AttrDataType::kListInt32, AttrDataType::kListInt32, AppendConvertedListAttr<int64_t, int32_t>);
}  // namespace

// src type and dst type are enum data
ge::graphStatus TilingData::AppendConvertedAttrVal(const RuntimeAttrs *attrs, const size_t attr_index,
                                                   const AttrDataType src_type, const AttrDataType dst_type) {
  GE_CHECK_NOTNULL(attrs);
  if (attr_index >= attrs->GetAttrNum()) {
    GELOGE(ge::GRAPH_FAILED, "Attr index is invalid, out of range %zu.", attrs->GetAttrNum());
    return ge::GRAPH_FAILED;
  }
  GELOGD("Begin to get attr index[%zu] data type[%d].", attr_index, static_cast<int32_t>(src_type));
  auto func = kAttrTable.Find(src_type, dst_type);
  if (func == nullptr) {
    GELOGE(ge::GRAPH_FAILED, "Get attr index[%zu] and transform from [%d] to [%d] is not supported.", attr_index,
           static_cast<int32_t>(src_type), static_cast<int32_t>(dst_type));
    return ge::GRAPH_FAILED;
  }
  return func(this, attrs, attr_index);
}
}  // namespace gert
