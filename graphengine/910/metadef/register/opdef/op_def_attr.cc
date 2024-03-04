/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include <vector>
#include <string>
#include <sstream>
#include "register/op_def.h"
#include "op_def_impl.h"

namespace ops {
OpAttrDef::OpAttrDef(const char *name) : impl_(new(std::nothrow) OpAttrDefImpl) {
  this->impl_->name = name;
}

OpAttrDef::OpAttrDef(const OpAttrDef &attr_def) : impl_(new(std::nothrow) OpAttrDefImpl) {
  this->impl_->name = attr_def.impl_->name;
  this->impl_->data_type = attr_def.impl_->data_type;
  this->impl_->required = attr_def.impl_->required;
  this->impl_->bool_value = attr_def.impl_->bool_value;
  this->impl_->float_value = attr_def.impl_->float_value;
  this->impl_->int_value = attr_def.impl_->int_value;
  this->impl_->str_value = attr_def.impl_->str_value;
  this->impl_->list_bool = attr_def.impl_->list_bool;
  this->impl_->list_float = attr_def.impl_->list_float;
  this->impl_->list_int = attr_def.impl_->list_int;
  this->impl_->list_list_int = attr_def.impl_->list_list_int;
}

OpAttrDef::~OpAttrDef() = default;

OpAttrDef &OpAttrDef::operator=(const OpAttrDef &attr_def) {
  if (this != &attr_def) {
    *this->impl_ = *attr_def.impl_;
  }
  return *this;
}

bool OpAttrDef::operator==(const OpAttrDef &attr_def) const {
  if (this->impl_->name == attr_def.impl_->name) {
    return true;
  }
  return false;
}

OpAttrDef &OpAttrDef::AttrType(Option attr_type) {
  if (attr_type == Option::OPTIONAL) {
    this->impl_->required = false;
  }
  return *this;
}

OpAttrDef &OpAttrDef::Bool(void) {
  this->impl_->data_type = AttrDataType::ATTR_DT_BOOL;
  return *this;
}

OpAttrDef &OpAttrDef::Bool(bool value) {
  this->impl_->bool_value = value;
  return this->Bool();
}

OpAttrDef &OpAttrDef::Float(void) {
  this->impl_->data_type = AttrDataType::ATTR_DT_FLOAT;
  return *this;
}

OpAttrDef &OpAttrDef::Float(float value) {
  this->impl_->float_value = value;
  return this->Float();
}

OpAttrDef &OpAttrDef::Int(void) {
  this->impl_->data_type = AttrDataType::ATTR_DT_INT;
  return *this;
}

OpAttrDef &OpAttrDef::Int(int64_t value) {
  this->impl_->int_value = value;
  return this->Int();
}

OpAttrDef &OpAttrDef::String(void) {
  this->impl_->data_type = AttrDataType::ATTR_DT_STR;
  return *this;
}

OpAttrDef &OpAttrDef::String(const char *value) {
  this->impl_->str_value = value;
  return this->String();
}

OpAttrDef &OpAttrDef::ListBool(void) {
  this->impl_->data_type = AttrDataType::ATTR_DT_LIST_BOOL;
  return *this;
}

OpAttrDef &OpAttrDef::ListBool(std::vector<bool> value) {
  this->impl_->list_bool = value;
  return this->ListBool();
}

OpAttrDef &OpAttrDef::ListFloat(void) {
  this->impl_->data_type = AttrDataType::ATTR_DT_LIST_FLOAT;
  return *this;
}

OpAttrDef &OpAttrDef::ListFloat(std::vector<float> value) {
  this->impl_->list_float = value;
  return this->ListFloat();
}

OpAttrDef &OpAttrDef::ListInt(void) {
  this->impl_->data_type = AttrDataType::ATTR_DT_LIST_INT;
  return *this;
}

OpAttrDef &OpAttrDef::ListInt(std::vector<int64_t> value) {
  this->impl_->list_int = value;
  return this->ListInt();
}

OpAttrDef &OpAttrDef::ListListInt(void) {
  this->impl_->data_type = AttrDataType::ATTR_DT_LIST_LIST_INT;
  return *this;
}

OpAttrDef &OpAttrDef::ListListInt(std::vector<std::vector<int64_t>> value) {
  this->impl_->list_list_int = value;
  return this->ListListInt();
}

ge::AscendString &OpAttrDef::GetName(void) const {
  return this->impl_->name;
}

bool OpAttrDef::IsRequired(void) {
  return this->impl_->required;
}

ge::AscendString &OpAttrDef::GetCfgDataType(void) const {
  static ge::AscendString dtype_names[] = {"bool",     "float",     "int",     "str",
                                           "listBool", "listFloat", "listInt", "listListInt"};
  return dtype_names[static_cast<size_t>(this->impl_->data_type)];
}

ge::AscendString &OpAttrDef::GetProtoDataType(void) const {
  static ge::AscendString dtype_names[] = {"Bool",     "Float",     "Int",     "String",
                                           "ListBool", "ListFloat", "ListInt", "ListListInt"};
  return dtype_names[static_cast<size_t>(this->impl_->data_type)];
}

template<class T>
std::string GetListStr(std::vector<T> list, const char *brac, void (*pfSout)(std::stringstream &s, T v)) {
  std::string str = "";
  std::stringstream sstream;
  if (brac == nullptr || brac[0] == '\0' || brac[1] == '\0') {
    return str.c_str();
  }
  sstream << brac[0];
  for (auto v : list) {
    pfSout(sstream, v);
  }
  str += sstream.str();
  if (list.size() > 0) {
    str.resize(str.size() - 1);
  }
  str += brac[1];
  return str;
}

ge::AscendString &OpAttrDef::GetAttrDefaultVal(const char *brac) {
  std::stringstream sstream;
  std::vector<std::string> strList;

  if (this->impl_->data_type == AttrDataType::ATTR_DT_BOOL) {
    sstream << (this->impl_->bool_value ? "true" : "false");
    this->impl_->value = sstream.str().c_str();
  } else if (this->impl_->data_type == AttrDataType::ATTR_DT_FLOAT) {
    sstream << this->impl_->float_value;
    this->impl_->value = sstream.str().c_str();
  } else if (this->impl_->data_type == AttrDataType::ATTR_DT_INT) {
    sstream << this->impl_->int_value;
    this->impl_->value = sstream.str().c_str();
  } else if (this->impl_->data_type == AttrDataType::ATTR_DT_STR) {
    this->impl_->value = this->impl_->str_value;
  } else if (this->impl_->data_type == AttrDataType::ATTR_DT_LIST_BOOL) {
    this->impl_->value = GetListStr<bool>(this->impl_->list_bool, brac, [](std::stringstream &s, bool v) {
                           s << (v ? "true" : "false") << ",";
                         }).c_str();
  } else if (this->impl_->data_type == AttrDataType::ATTR_DT_LIST_FLOAT) {
    this->impl_->value =
        GetListStr<float>(this->impl_->list_float, brac, [](std::stringstream &s, float v) { s << v << ","; }).c_str();
  } else if (this->impl_->data_type == AttrDataType::ATTR_DT_LIST_INT) {
    this->impl_->value = GetListStr<int64_t>(this->impl_->list_int, brac, [](std::stringstream &s, int64_t v) {
                           s << v << ",";
                         }).c_str();
  } else if (this->impl_->data_type == AttrDataType::ATTR_DT_LIST_LIST_INT) {
    for (auto listInt : this->impl_->list_list_int) {
      strList.emplace_back(GetListStr<int64_t>(listInt, brac, [](std::stringstream &s, int64_t v) { s << v << ","; }));
    }
    this->impl_->value =
        GetListStr<std::string>(strList, brac, [](std::stringstream &s, std::string v) { s << v << ","; }).c_str();
  } else {
    this->impl_->value = "";
  }
  return this->impl_->value;
}
}  // namespace ops
