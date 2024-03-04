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

#include "register/tilingdata_base.h"
#include <cstring>
#include <securec.h>
#include "common/ge_common/debug/ge_log.h"
#include "graph/ascend_string.h"

namespace optiling {
std::vector<FieldInfo> TilingDef::GetFieldInfo() const {
  return field_info_;
}

const char *TilingDef::GetTilingClassName() const {
  return class_name_;
}

size_t TilingDef::GetDataSize() const {
  return data_size_;
}

void TilingDef::GeLogError(const std::string& str) const {
  GELOGE(ge::GRAPH_FAILED, "%s", str.c_str());
}

void TilingDef::SetDataPtr(void *ptr) {
  if (!inited_data_ptr && data_ptr_ != nullptr) {
    delete[] data_ptr_;
  }
  inited_data_ptr = true;
  data_ptr_ = (uint8_t*)ptr;
  for (auto &ptr : saveBufferPtr) {
    TilingDef* sub_ptr = (TilingDef *)ptr.first;
    size_t offset = ptr.second;
    uint8_t* struct_ptr = data_ptr_ + offset;
    sub_ptr->SetDataPtr(struct_ptr);
  }
}

void TilingDef::SaveToBuffer(void *pdata, size_t capacity) {
  if (inited_data_ptr) {
    GELOGI("TilingDef::SaveToBuffer, op %s, data had been saved.", class_name_);
    return;
  }
  // copy tilingdata to buffer without struct tiling data.
  auto mem_ret = memcpy_s(pdata, capacity, data_ptr_, data_size_);
  if (mem_ret != EOK) {
    GELOGE(ge::GRAPH_FAILED,
           "TilingDef::SaveToBuffer failed: memcpy_s return op [%s] [%d], capacity = [%zu], data_size_ = [%zu].",
           class_name_, mem_ret, capacity, data_size_);
  }
}

void TilingDef::CheckAlignAndGenPlaceHolder(const char *name, size_t typeSize) {
  if (data_size_ % typeSize == 0) {
    return;
  }
  size_t alignSize = typeSize - (data_size_ % typeSize);
  field_info_.emplace_back(FieldInfo("uint8_t", name, alignSize));
  data_size_ += alignSize;
  return;
}

void TilingDef::InitData() {
    GELOGI("TilingDef::InitData, op %s, init data size %d.", class_name_, data_size_);
    data_ptr_ = new (std::nothrow)uint8_t[data_size_]();
    if (data_ptr_ == nullptr) {
          GELOGE(ge::GRAPH_FAILED, "TilingDef::InitData failed: op %s, init data size %d.", class_name_, data_size_);
    }
    for (auto &ptr : saveBufferPtr) {
      TilingDef* sub_ptr = (TilingDef *)ptr.first;
      size_t offset = ptr.second;
      uint8_t* struct_ptr = data_ptr_ + offset;
      sub_ptr->SetDataPtr(struct_ptr);
    }
}

CTilingDataClassFactory &CTilingDataClassFactory::GetInstance()
{
  static CTilingDataClassFactory instance;
  return instance;
}

void CTilingDataClassFactory::RegisterTilingData(const char *op_type,
                                                 const TilingDataConstructor constructor) {
  instance_.emplace(op_type, constructor);
  GELOGI("RegisterTilingData: op_type:%s, constructor:%p, registered count:%zu", op_type, constructor,
         instance_.size());
}

std::shared_ptr<TilingDef> CTilingDataClassFactory::CreateTilingDataInstance(const char *op_type) {
  const auto it = instance_.find(op_type);
  if (it == instance_.end()) {
    GELOGW("CreateTilingDataInstance: cannot find op_type:%s.", op_type);
    return nullptr;
  }
  const TilingDataConstructor constructor = it->second;

  if (constructor == nullptr) {
    GELOGW("CreateTilingDataInstance: constructor is nullptr.");
    return nullptr;
  }

  return (*constructor)();
}
}  // end of namespace optiling