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
#include "register/op_def.h"
#include "op_def_impl.h"

namespace ops {
OpParamDef::OpParamDef(const char *name) : impl_(new(std::nothrow) OpParamDefImpl) {
  this->impl_->name = name;
}

OpParamDef::OpParamDef(const OpParamDef &def) : impl_(new(std::nothrow) OpParamDefImpl) {
  this->impl_->name = def.impl_->name;
  this->impl_->param_type = def.impl_->param_type;
  this->impl_->types = def.impl_->types;
  this->impl_->formats = def.impl_->formats;
  this->impl_->need_compile = def.impl_->need_compile;
  this->impl_->reshape_type = def.impl_->reshape_type;
  this->impl_->value_depend = def.impl_->value_depend;
  this->impl_->unknown_shape_formats = def.impl_->unknown_shape_formats;
}

OpParamDef &OpParamDef::operator=(const OpParamDef &def) {
  if (this != &def) {
    *this->impl_ = *def.impl_;
  }
  return *this;
}

void OpParamDef::MergeParam(const OpParamDef &def) {
  this->impl_->param_type = def.impl_->param_type;
  if (def.impl_->types.size() > 0) {
    this->impl_->types = def.impl_->types;
  }
  if (def.impl_->formats.size() > 0) {
    this->impl_->formats = def.impl_->formats;
  }
  if (def.impl_->need_compile.GetLength() > 0) {
    this->impl_->need_compile = def.impl_->need_compile;
  }
  if (def.impl_->reshape_type.GetLength() > 0) {
    this->impl_->reshape_type = def.impl_->reshape_type;
  }
  if (def.impl_->value_depend.GetLength() > 0) {
    this->impl_->value_depend = def.impl_->value_depend;
  }
  if (def.impl_->unknown_shape_formats.size() > 0) {
    this->impl_->unknown_shape_formats = def.impl_->unknown_shape_formats;
  }
}

OpParamDef::~OpParamDef() = default;

bool OpParamDef::operator==(const OpParamDef &def) const {
  if (this->impl_->name == def.impl_->name) {
    return true;
  }
  return false;
}

OpParamDef &OpParamDef::ParamType(Option param_type) {
  this->impl_->param_type = param_type;
  return *this;
}

OpParamDef &OpParamDef::DataType(std::vector<ge::DataType> types) {
  this->impl_->types = types;
  return *this;
}

OpParamDef &OpParamDef::Format(std::vector<ge::Format> formats) {
  this->impl_->formats = formats;
  return *this;
}

OpParamDef &OpParamDef::UnknownShapeFormat(std::vector<ge::Format> formats) {
  this->impl_->unknown_shape_formats = formats;
  return *this;
}

OpParamDef &OpParamDef::ValueDepend(Option value_depend) {
  if (value_depend == Option::REQUIRED) {
    this->impl_->value_depend = "required";
  } else if (value_depend == Option::OPTIONAL) {
    this->impl_->value_depend = "optional";
  } else {
    this->impl_->value_depend = "";
  }
  return *this;
}

ge::AscendString &OpParamDef::GetParamName(void) {
  return this->impl_->name;
}
Option OpParamDef::GetParamType(void) {
  return this->impl_->param_type;
}
std::vector<ge::DataType> &OpParamDef::GetDataTypes(void) {
  return this->impl_->types;
}
std::vector<ge::Format> &OpParamDef::GetFormats(void) {
  return this->impl_->formats;
}
std::vector<ge::Format> &OpParamDef::GetUnknownShapeFormats(void) {
  return this->impl_->unknown_shape_formats;
}
ge::AscendString &OpParamDef::GetValueDepend(void) {
  return this->impl_->value_depend;
}

OpParamDef &OpParamTrunk::Input(const char *name) {
  return this->ParamGetOrCreate(name, false);
}

OpParamDef &OpParamTrunk::Output(const char *name) {
  return this->ParamGetOrCreate(name, true);
}

OpParamDef &OpParamTrunk::ParamGetOrCreate(const char *name, bool is_output) {
  OpParamDef *param;
  if (this->ParamFind(name, is_output, &param) == ItemFindStatus::ITEM_FIND) {
    return *param;
  } else {
    OpParamDef addParam(name);
    return this->ParamAdd(addParam, is_output);
  }
}

ItemFindStatus OpParamTrunk::ParamFind(const char *name, bool is_output, OpParamDef **param) {
  std::vector<OpParamDef> *paramList;

  if (is_output) {
    paramList = &(this->outputs_);
  } else {
    paramList = &(this->inputs_);
  }
  for (auto it = paramList->begin(); it != paramList->end(); it++) {
    if (it->GetParamName() == name) {
      *param = &(*it);
      return ItemFindStatus::ITEM_FIND;
    }
  }
  return ItemFindStatus::ITEM_NOEXIST;
}

OpParamDef &OpParamTrunk::ParamAdd(OpParamDef &param, bool is_output) {
  if (is_output) {
    this->outputs_.emplace_back(param);
    return this->outputs_.back();
  } else {
    this->inputs_.emplace_back(param);
    return this->inputs_.back();
  }
}

std::vector<OpParamDef> &OpParamTrunk::GetInputs(void) {
  return this->inputs_;
}

std::vector<OpParamDef> &OpParamTrunk::GetOutputs(void) {
  return this->outputs_;
}
}  // namespace ops
