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

#ifndef OP_DEF_IMPL_H
#define OP_DEF_IMPL_H

#include "register/op_def.h"
#include "register/op_impl_registry.h"
#include "register/op_check.h"
#include "graph/operator_reg.h"

namespace ops {
class OpParamDefImpl {
public:
  ge::AscendString name;
  Option param_type = Option::REQUIRED;
  std::vector<ge::DataType> types;
  std::vector<ge::Format> formats;
  ge::AscendString need_compile = "";
  ge::AscendString reshape_type = "";
  ge::AscendString value_depend = "";
  std::vector<ge::Format> unknown_shape_formats;
};

class OpParamTrunk {
public:
  OpParamDef &Input(const char *name);
  OpParamDef &Output(const char *name);
  std::vector<OpParamDef> &GetInputs(void);
  std::vector<OpParamDef> &GetOutputs(void);

private:
  ItemFindStatus ParamFind(const char *name, bool is_output, OpParamDef **param);
  OpParamDef &ParamAdd(OpParamDef &param, bool is_output);
  OpParamDef &ParamGetOrCreate(const char *name, bool is_output);
  std::vector<OpParamDef> inputs_;
  std::vector<OpParamDef> outputs_;
};

class OpAttrDefImpl {
public:
  ge::AscendString name;
  AttrDataType data_type = AttrDataType::ATTR_DT_BOOL;
  bool required = true;
  bool bool_value = false;
  float float_value = 0;
  int64_t int_value = 0;
  ge::AscendString str_value = "";
  std::vector<bool> list_bool = {};
  std::vector<float> list_float = {};
  std::vector<int64_t> list_int = {};
  std::vector<std::vector<int64_t>> list_list_int = {};
  ge::AscendString value = "";
};

class OpAICoreConfigImpl {
public:
  OpParamTrunk op_params;
  std::vector<ge::AscendString> cfg_keys;
  std::map<ge::AscendString, ge::AscendString> cfg_info;
};

class OpAICoreDefImpl {
public:
  gert::OpImplKernelRegistry::TilingKernelFunc tiling_func = nullptr;
  gert::OpImplKernelRegistry::TilingParseFunc tiling_parse = nullptr;
  gert::OpImplKernelRegistry::CompileInfoCreatorFunc ci_creator = nullptr;
  gert::OpImplKernelRegistry::CompileInfoDeleterFunc ci_deleter = nullptr;
  optiling::OP_CHECK_FUNC op_chk_support = nullptr;
  optiling::OP_CHECK_FUNC op_sel_format = nullptr;
  optiling::OP_CHECK_FUNC op_get_support = nullptr;
  optiling::OP_CHECK_FUNC op_get_spec = nullptr;
  optiling::PARAM_GENERALIZE_FUNC op_generlize_func = nullptr;
  std::map<ge::AscendString, OpAICoreConfig> aicore_configs = {};
};

class OpDefImpl {
public:
  gert::OpImplKernelRegistry::InferShapeKernelFunc infer_shape = nullptr;
  gert::OpImplKernelRegistry::InferShapeRangeKernelFunc infer_shape_range = nullptr;
  gert::OpImplKernelRegistry::InferDataTypeKernelFunc infer_data_type = nullptr;
  OpParamTrunk op_params;
  std::vector<OpAttrDef> attrs;
  OpAICoreDef op_aicore;
  ge::AscendString op_type;
  bool has_workspace = true;
};
}  // namespace ops

#endif
