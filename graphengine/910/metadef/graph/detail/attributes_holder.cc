/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#include "graph/detail/attributes_holder.h"

#include "graph/debug/ge_log.h"
#include "graph/debug/ge_util.h"
#include "graph/ge_attr_value.h"
#include "proto/ge_ir.pb.h"

namespace ge {
void AttrHolder::CopyAttrsFrom(const AttrHolder &holder) {
  MutableAttrMap() = holder.GetAttrMap();
}
void AttrHolder::CopyFrom(const AttrHolder &holder) {
  required_attrs_and_type_ = holder.required_attrs_and_type_;
  ext_attrs_ = holder.ext_attrs_;
}

graphStatus AttrHolder::SetAttr(const std::string &name, const AnyValue &value) {
  if (value.IsEmpty()) {
    REPORT_INNER_ERROR("E18888", "param value is empty, check invalid, key of the attr:%s", name.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] value is empty, key of the attr is %s", name.c_str());
    return GRAPH_FAILED;
  }
  if (!MutableAttrMap().SetAnyValueByName(name, value)) {
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
graphStatus AttrHolder::TrySetAttr(const std::string &name, const AnyValue &value) {
  if (value.IsEmpty()) {
    REPORT_INNER_ERROR("E18888", "param value is empty, check invalid, key of the attr:%s", name.c_str());
    GELOGE(GRAPH_FAILED, "[Check][Param] value is empty, key of the attr is %s", name.c_str());
    return GRAPH_FAILED;
  }
  if (MutableAttrMap().Exists(name)) {
    GELOGW("attr %s already existed, skip update", name.c_str());
  } else {
    if (!MutableAttrMap().SetAnyValueByName(name, value)) {
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}
graphStatus AttrHolder::AddRequiredAttr(const std::string &name) {
  return AddRequiredAttrWithType(name, "");
}

graphStatus AttrHolder::AddRequiredAttrWithType(const std::string &name, const std::string &type) {
  if (HasAttr(name)) {
    return GRAPH_FAILED;
  }
  required_attrs_and_type_.emplace(name, type);
  return GRAPH_SUCCESS;
}

graphStatus AttrHolder::GetAttr(const std::string &name, AnyValue &value) const {
  const auto av = GetAttrMap().GetAnyValue(name);
  if (av == nullptr) {
    return GRAPH_FAILED;
  }
  value = *av;
  return GRAPH_SUCCESS;
}

bool AttrHolder::HasAttr(const std::string &name) const {
  if (GetAttrMap().Exists(name)) {
    return true;
  }
  return required_attrs_and_type_.find(name) != required_attrs_and_type_.end();
}

graphStatus AttrHolder::DelAttr(const std::string &name) {
  return MutableAttrMap().Delete(name) ? GRAPH_SUCCESS : GRAPH_FAILED;
}

const std::map<std::string, AnyValue> AttrHolder::GetAllAttrs() const {
  return GetAttrMap().GetAllAttrs();
}

const std::set<std::string> AttrHolder::GetAllAttrNames() const {
  return GetAttrMap().GetAllAttrNames();
}

template <>
void GeIrProtoHelper<proto::AttrDef>::InitDefault() {
  std::shared_ptr<proto::AttrDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::AttrDef>();
  if (proto_owner == nullptr) {
    REPORT_CALL_ERROR("E18888", "create AttrDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][AttrDef] proto::AttrDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::TensorDef>::InitDefault() {
  std::shared_ptr<proto::TensorDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::TensorDef>();
  if (proto_owner == nullptr) {
    REPORT_CALL_ERROR("E18888", "create TensorDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][TensorDef] proto::TensorDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::TensorDescriptor>::InitDefault() {
  std::shared_ptr<proto::TensorDescriptor> proto_owner;
  proto_owner = ComGraphMakeShared<proto::TensorDescriptor>();
  if (proto_owner == nullptr) {
    REPORT_CALL_ERROR("E18888", "create TensorDescriptor failed.");
    GELOGE(GRAPH_FAILED, "[Create][TensorDescriptor] proto::TensorDescriptor make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::ShapeDef>::InitDefault() {
  std::shared_ptr<proto::ShapeDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::ShapeDef>();
  if (proto_owner == nullptr) {
    REPORT_CALL_ERROR("E18888", "create ShapeDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][ShapeDef] proto::ShapeDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::NamedAttrs>::InitDefault() {
  std::shared_ptr<proto::NamedAttrs> proto_owner;
  proto_owner = ComGraphMakeShared<proto::NamedAttrs>();
  if (proto_owner == nullptr) {
    REPORT_CALL_ERROR("E18888", "create NamedAttrs failed.");
    GELOGE(GRAPH_FAILED, "[Create][NamedAttrs] proto::NamedAttrs make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::ModelDef>::InitDefault() {
  std::shared_ptr<proto::ModelDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::ModelDef>();
  if (proto_owner == nullptr) {
    REPORT_CALL_ERROR("E18888", "create ModelDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][ModelDef] proto::ModelDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::OpDef>::InitDefault() {
  std::shared_ptr<proto::OpDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::OpDef>();
  if (proto_owner == nullptr) {
    REPORT_CALL_ERROR("E18888", "create OpDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][OpDef] proto::OpDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}

template <>
void GeIrProtoHelper<proto::GraphDef>::InitDefault() {
  std::shared_ptr<proto::GraphDef> proto_owner;
  proto_owner = ComGraphMakeShared<proto::GraphDef>();
  if (proto_owner == nullptr) {
    REPORT_CALL_ERROR("E18888", "create GraphDef failed.");
    GELOGE(GRAPH_FAILED, "[Create][GraphDef] proto::GraphDef make shared failed");
    return;
  }
  protoMsg_ = proto_owner.get();
  protoOwner_ = proto_owner;
}
}  // namespace ge
