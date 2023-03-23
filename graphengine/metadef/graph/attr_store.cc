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

#include "graph/attr_store.h"

namespace ge {
const AttrId constInvalidAttrId = GetAttrId(0xffffffffU, 0U);

AnyValue *AttrStore::GetOrCreateAnyValue(const AttrId attr_id) const {
  return const_cast<AnyValue *>(GetAnyValue(attr_id));
}
AnyValue *AttrStore::MutableAnyValue(const AttrId attr_id) const noexcept {
  return const_cast<AnyValue *>(GetAnyValue(attr_id));
}
const AnyValue *AttrStore::GetAnyValue(const AttrId attr_id) const noexcept {
  const auto attr_type = GetAttrType(attr_id);
  if (attr_type == static_cast<uint32_t>(AttrType::kAttrPredefinedInIr)) {
    return pre_defined_attrs_.GetAnyValue(GetSubAttrId(attr_id));
  } else if (attr_type == static_cast<uint32_t>(AttrType::kAttrGeneral)) {
    return nullptr;  // general不支持
  } else {
    // empty
  }
  return nullptr;
}
AttrStore AttrStore::Create(const size_t pre_defined_attr_count) {
  AttrStore as;
  as.pre_defined_attrs_.Resize(pre_defined_attr_count);
  return as;
}
const AnyValue *AttrStore::GetAnyValue(const std::string &name) const noexcept {
  const auto id = GetIdByName(name);
  if (id != constInvalidAttrId) {
    return pre_defined_attrs_.GetAnyValue(GetSubAttrId(id));
  }

  const AnyValue *const av = general_attrs_.GetAnyValue(name);
  if (av != nullptr) {
    return av;
  }

  return nullptr;
}
AnyValue *AttrStore::MutableAnyValue(const std::string &name) const noexcept {
  return const_cast<AnyValue *>(GetAnyValue(name));
}
AnyValue *AttrStore::GetOrCreateAnyValue(const std::string &name) {
  const auto id = GetIdByName(name);
  if (id != constInvalidAttrId) {
    return pre_defined_attrs_.GetOrCreateAnyValue(GetSubAttrId(id));
  }
  return general_attrs_.GetOrCreateAnyValue(name);
}
AttrId AttrStore::GetIdByName(const std::string &name) const noexcept {
  const auto iter = names_to_id_.find(name);
  if (iter == names_to_id_.end()) {
    return constInvalidAttrId;
  }
  return iter->second;
}
void AttrStore::SetNameAndId(std::string name, const AttrId id) {
  names_to_id_[std::move(name)] = id;
}
bool AttrStore::Exists(const AttrId attr_id) const noexcept {
  return GetAnyValue(attr_id) != nullptr;
}
bool AttrStore::Exists(const std::string &name) const noexcept {
  return GetAnyValue(name) != nullptr;
}
bool AttrStore::Delete(const std::string &name) {
  const auto iter = names_to_id_.find(name);
  if (iter != names_to_id_.end()) {
    const auto sub_id = GetSubAttrId(iter->second);
    (void)names_to_id_.erase(iter);
    return pre_defined_attrs_.Delete(sub_id);
  }
  return general_attrs_.Delete(name);
}
std::set<std::string> AttrStore::GetAllAttrNames() const {
  std::set<std::string> names;
  for (const auto &iter : names_to_id_) {
    (void)names.insert(iter.first);
  }
  general_attrs_.GetAllNames(names);
  return names;
}
std::map<std::string, AnyValue> AttrStore::GetAllAttrs() const {
  std::map<std::string, AnyValue> attrs;
  for (const auto &iter : names_to_id_) {
    const auto av = pre_defined_attrs_.GetAnyValue(GetSubAttrId(iter.second));
    if (av == nullptr) {
      // error
      continue;
    }
    if (av->IsEmpty()) {
      continue;
    }
    attrs[iter.first] = *av;
  }
  general_attrs_.GetAllAttrs(attrs);
  return attrs;
}
void AttrStore::Swap(AttrStore &other) {
  pre_defined_attrs_.Swap(other.pre_defined_attrs_);
  general_attrs_.Swap(other.general_attrs_);
}


void AttrStore::PreDefinedAttrStore::Resize(const size_t s) {
  attrs_.resize(s);
}
bool AttrStore::PreDefinedAttrStore::Exists(const AttrSubId index) const noexcept {
  if (index >= attrs_.size()) {
    return false;
  }
  return !attrs_[static_cast<size_t>(index)].IsEmpty();
}
bool AttrStore::PreDefinedAttrStore::Delete(const AttrSubId index) {
  if (!Exists(index)) {
    return false;
  }
  attrs_[static_cast<size_t>(index)].Clear();
  return true;
}
AnyValue *AttrStore::PreDefinedAttrStore::GetOrCreateAnyValue(const AttrSubId index) const {
  return const_cast<AnyValue *>(GetAnyValue(index));
}
AnyValue *AttrStore::PreDefinedAttrStore::MutableAnyValue(const AttrSubId index) const noexcept {
  return const_cast<AnyValue *>(GetAnyValue(index));
}
const AnyValue *AttrStore::PreDefinedAttrStore::GetAnyValue(const AttrSubId index) const noexcept {
  if (index >= attrs_.size()) {
    return nullptr;
  }
  return &attrs_[static_cast<size_t>(index)];
}
void AttrStore::PreDefinedAttrStore::Swap(AttrStore::PreDefinedAttrStore &other) {
  attrs_.swap(other.attrs_);
}
bool AttrStore::CustomDefinedAttrStore::Exists(const std::string &name) const noexcept {
  return attrs_.count(name) > 0UL;
}
bool AttrStore::CustomDefinedAttrStore::Delete(const std::string &name) {
  return attrs_.erase(name) == 1UL;
}
AnyValue *AttrStore::CustomDefinedAttrStore::GetOrCreateAnyValue(const std::string &name) {
  return &attrs_[name];
}
AnyValue *AttrStore::CustomDefinedAttrStore::MutableAnyValue(const std::string &name) const noexcept {
  return const_cast<AnyValue *>(GetAnyValue(name));
}
const AnyValue *AttrStore::CustomDefinedAttrStore::GetAnyValue(const std::string &name) const noexcept {
  const auto iter = attrs_.find(name);
  if (iter != attrs_.end()) {
    return &iter->second;
  } else {
    return nullptr;
  }
}
void AttrStore::CustomDefinedAttrStore::GetAllNames(std::set<std::string> &names) const {
  for (const auto &iter : attrs_) {
    (void)names.insert(iter.first);
  }
}
void AttrStore::CustomDefinedAttrStore::GetAllAttrs(std::map<std::string, AnyValue> &names_to_attr) const {
  for (const auto &iter : attrs_) {
    names_to_attr[iter.first] = iter.second;
  }
}
void AttrStore::CustomDefinedAttrStore::Swap(AttrStore::CustomDefinedAttrStore &other) {
  attrs_.swap(other.attrs_);
}
bool AttrStore::SetAnyValueByName(const std::string &name, const AnyValue &value) {
  const auto av = GetOrCreateAnyValue(name);
  if (av == nullptr) {
    return false;
  }
  *av = value;
  return true;
}
}  // namespace ge