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
#include "op_def_impl.h"
#include "common/ge_common/debug/ge_log.h"
#include "register/op_def.h"

namespace ops {
OpDef::OpDef(const char *type) : impl_(new(std::nothrow) OpDefImpl) {
  this->impl_->op_type = type;
}

OpDef::OpDef(const OpDef &op_def) : impl_(new(std::nothrow) OpDefImpl) {
  this->impl_->op_type = op_def.impl_->op_type;
  this->impl_->op_params = op_def.impl_->op_params;
  this->impl_->attrs = op_def.impl_->attrs;
  this->impl_->op_aicore = op_def.impl_->op_aicore;
  this->impl_->has_workspace = op_def.impl_->has_workspace;
  this->impl_->infer_shape = op_def.impl_->infer_shape;
  this->impl_->infer_shape_range = op_def.impl_->infer_shape_range;
  this->impl_->infer_data_type = op_def.impl_->infer_data_type;
}

OpDef::~OpDef() = default;

OpDef &OpDef::operator=(const OpDef &op_def) {
  if (this != &op_def) {
    *this->impl_ = *op_def.impl_;
  }
  return *this;
}

OpParamDef &OpDef::Input(const char *name) {
  return this->impl_->op_params.Input(name);
}

OpParamDef &OpDef::Output(const char *name) {
  return this->impl_->op_params.Output(name);
}

OpAttrDef &OpDef::Attr(const char *name) {
  return this->GetOrCreateAttr(name);
}

ItemFindStatus OpDef::FindAttr(const char *name, OpAttrDef **attr) {
  std::vector<OpAttrDef> *attrList = &this->impl_->attrs;
  for (auto it = attrList->begin(); it != attrList->end(); it++) {
    if (ge::AscendString(it->GetName()) == ge::AscendString(name)) {
      *attr = &(*it);
      return ItemFindStatus::ITEM_FIND;
    }
  }
  return ItemFindStatus::ITEM_NOEXIST;
}

OpAttrDef &OpDef::AddAttr(OpAttrDef &attr) {
  this->impl_->attrs.emplace_back(attr);
  return this->impl_->attrs.back();
}

OpAttrDef &OpDef::GetOrCreateAttr(const char *name) {
  OpAttrDef *pAttr;
  if (this->FindAttr(name, &pAttr) == ItemFindStatus::ITEM_FIND) {
    return *pAttr;
  } else {
    OpAttrDef attr(name);
    return this->AddAttr(attr);
  }
}

std::vector<OpAttrDef> &OpDef::GetAttrs(void) {
  return this->impl_->attrs;
}

OpDef &OpDef::SetInferShape(gert::OpImplKernelRegistry::InferShapeKernelFunc func) {
  this->impl_->infer_shape = func;
  return *this;
}

OpDef &OpDef::SetInferShapeRange(gert::OpImplKernelRegistry::InferShapeRangeKernelFunc func) {
  this->impl_->infer_shape_range = func;
  return *this;
}

OpDef &OpDef::SetInferDataType(gert::OpImplKernelRegistry::InferDataTypeKernelFunc func) {
  this->impl_->infer_data_type = func;
  return *this;
}

gert::OpImplKernelRegistry::InferShapeKernelFunc &OpDef::GetInferShape(void) {
  return this->impl_->infer_shape;
}
gert::OpImplKernelRegistry::InferShapeRangeKernelFunc &OpDef::GetInferShapeRange(void) {
  return this->impl_->infer_shape_range;
}
gert::OpImplKernelRegistry::InferDataTypeKernelFunc &OpDef::GetInferDataType(void) {
  return this->impl_->infer_data_type;
}
ge::AscendString &OpDef::GetOpType(void) {
  return this->impl_->op_type;
}
std::vector<OpParamDef> &OpDef::GetInputs(void) {
  return this->impl_->op_params.GetInputs();
}

std::vector<OpParamDef> &OpDef::GetOutputs(void) {
  return this->impl_->op_params.GetOutputs();
}

void OpDef::MergeParam(std::vector<OpParamDef> &merge, std::vector<OpParamDef> &aicore_params) const {
  for (auto &aicoreParam : aicore_params) {
    bool find = false;
    for (auto &mergeParam : merge) {
      if (mergeParam == aicoreParam) {
        mergeParam.MergeParam(aicoreParam);
        find = true;
        break;
      }
    }
    if (!find) {
      merge.emplace_back(aicoreParam);
    }
  }
}

void OpDef::CheckParam(std::vector<OpParamDef> &params) const {
  for (auto &param : params) {
    if (param.GetFormats().size() != 0) {
      if (param.GetDataTypes().size() == 0) {
        continue;
      }
      if (param.GetDataTypes().size() != param.GetFormats().size()) {
        GELOGE(ge::PARAM_INVALID, "dtype is not align with format, %ld != %ld", param.GetDataTypes().size(),
               param.GetFormats().size());
        return;
      }
    } else {
      std::vector<ge::Format> formats(param.GetDataTypes().size(), ge::FORMAT_ND);
      param.Format(formats);
    }
  }
}

std::vector<OpParamDef> OpDef::GetMergeInputs(OpAICoreConfig &aicore_config) {
  std::vector<OpParamDef> merge = this->GetInputs();
  MergeParam(merge, aicore_config.GetInputs());
  CheckParam(merge);
  return merge;
}

std::vector<OpParamDef> OpDef::GetMergeOutputs(OpAICoreConfig &aicore_config) {
  std::vector<OpParamDef> merge = this->GetOutputs();
  MergeParam(merge, aicore_config.GetOutputs());
  CheckParam(merge);
  return merge;
}

void OpDef::SetWorkspaceFlag(bool flag) {
  this->impl_->has_workspace = flag;
}

bool OpDef::GetWorkspaceFlag(void) {
  return this->impl_->has_workspace;
}

OpAICoreDef &OpDef::AICore(void) {
  return this->impl_->op_aicore;
}
}  // namespace ops
