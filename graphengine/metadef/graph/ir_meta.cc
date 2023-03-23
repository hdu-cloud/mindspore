/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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

#include "ir_meta.h"
#include "inc/common/util/trace_manager/trace_manager.h"
#include "graph/utils/ge_ir_utils.h"

namespace ge {
void IRMetaData::AppendIrAttrName(std::string name) {
  ir_attr_names_.emplace_back(std::move(name));
}
const std::vector<std::string> &IRMetaData::GetIrAttrNames() const {
  return ir_attr_names_;
}
void IRMetaData::AppendIrInput(std::string name, IrInputType input_type) {
  ir_inputs_.emplace_back(std::move(name), input_type);
}
const std::vector<std::pair<std::string, IrInputType>> &IRMetaData::GetIrInputs() const {
  return ir_inputs_;
}
graphStatus IRMetaData::AddRegisterInputName(const std::string &name) {
  if (find(register_input_name_.begin(), register_input_name_.end(), name) == register_input_name_.end()) {
    register_input_name_.push_back(name);
  }

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(),
                   op_name_, "register_input_name", "", "", name);
  return GRAPH_SUCCESS;
}

vector<std::string> IRMetaData::GetRegisterInputName() const {
  return register_input_name_;
}

bool IRMetaData::IsOptionalInput(const std::string &name) const {
  return optional_input_names_.find(name) != optional_input_names_.end();
}

graphStatus IRMetaData::AddRegisterOutputName(const std::string &name) {
  if (find(register_output_name_.begin(), register_output_name_.end(), name) == register_output_name_.end()) {
    register_output_name_.push_back(name);
  }

  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "add", TraceManager::GetOutGraphName(),
                   op_name_, "register_output_name", "", "", name);
  return GRAPH_SUCCESS;
}

vector<std::string> IRMetaData::GetRegisterOutputName() const {
  return register_output_name_;
}

void IRMetaData::RegisterSubgraphIrName(const std::string &name, const SubgraphType type) {
  subgraph_ir_names_to_type_[name] = type;
}

const std::map<std::string, SubgraphType> &IRMetaData::GetSubgraphIrNames() const {
  return subgraph_ir_names_to_type_;
}

SubgraphType IRMetaData::GetSubgraphTypeByIrName(const std::string &name) const {
  const auto iter = subgraph_ir_names_to_type_.find(name);
  if (iter == subgraph_ir_names_to_type_.end()) {
    return kSubgraphTypeEnd;
  }
  return iter->second;
}

IRDataTypeSymbolStore &IRMetaData::MutableIRDataTypeSymbolStore() {
  return dtype_symbol_store_;
}

const IRDataTypeSymbolStore &IRMetaData::GetIRDataTypeSymbolStore() const {
  return dtype_symbol_store_;
}

graphStatus IRMetaData::AddRegisterOptionalInputName(const string &name) {
  optional_input_names_.insert(name);
  return GRAPH_SUCCESS;
}

graphStatus IRMetaData::VerifyIR() const {
  auto ret = VerifyDataTypeSymbol();
  if ((ret != GRAPH_SUCCESS) && (ret != OP_WITHOUT_IR_DATATYPE_INFER_RULE)) {
    return ret;
  }
  return GRAPH_SUCCESS;
}

bool IRMetaData::operator==(const IRMetaData &other) const {
  return IsEqual(this->optional_input_names_, other.optional_input_names_,
                 "OpDesc.ir_meta.optional_input_names_");
}

IRMetaData::IRMetaData(const IRMetaData &other) {
  ir_attr_names_ = other.GetIrAttrNames();
  ir_inputs_ = other.GetIrInputs();
  subgraph_ir_names_to_type_ = other.GetSubgraphIrNames();
}

IRMetaData &IRMetaData::operator=(const IRMetaData &other) {
  ir_attr_names_ = other.GetIrAttrNames();
  ir_inputs_ = other.GetIrInputs();
  subgraph_ir_names_to_type_ = other.GetSubgraphIrNames();
  return *this;
}

std::set<std::string> IRMetaData::GetOptionalInputName() const {
  return optional_input_names_;
}

IrInputType IRMetaData::GetIrInputType(const string &name) const {
  for (const auto &name_2_type : ir_inputs_) {
    if (name == name_2_type.first) {
      return name_2_type.second;
    }
  }
  return kIrInputTypeEnd;
}

void IRMetaData::AppendIrOutput(std::string name, IrOutputType output_type) {
  ir_outputs_.emplace_back(std::move(name), output_type);
}

const std::vector<std::pair<std::string, IrOutputType>> &IRMetaData::GetIrOutputs() const {
  return ir_outputs_;
}

bool IRMetaData::IsOutputSymbolValid(const std::string &output_symbol) const {
  // output is infer by attr
  const auto iter = std::find(ir_attr_names_.begin(), ir_attr_names_.end(), output_symbol);
  if (iter != ir_attr_names_.end()) {
    return true;
  }

  // output is infer by input
  const auto &input_name = dtype_symbol_store_.GetInputNameFromDataTypeSymbol(output_symbol);
  if (!input_name.empty()) {
    return true;
  }

  // fix output datatype
  const auto &dtype_validator = dtype_symbol_store_.GetSymbolValidator(output_symbol);
  if (dtype_validator.IsValidSymbol() && dtype_validator.IsFixedRange()) {
    return true;
  }
  return false;
}

graphStatus IRMetaData::VerifyDataTypeSymbol() const {
  for (const auto &output_2_type : ir_outputs_) {
    const auto &output_name = output_2_type.first;
    const auto &output_symbol = dtype_symbol_store_.GetOutputDataTypeSymbol(output_name);
    if (output_symbol.empty()) {
      // no output symbol, may has user-defined infer rule, skip infer rule verify
      return OP_WITHOUT_IR_DATATYPE_INFER_RULE;
    }

    if (IsOutputSymbolValid(output_symbol)) {
      continue;
    }

    GELOGE(GRAPH_INVALID_IR_DEF,
           "Output %s data type symbol %s is not pre-defined. Please check IR.",
           output_name.c_str(),
           output_symbol.c_str());
    return GRAPH_INVALID_IR_DEF;
  }

  // check input type with datatype symbol type
  for (const auto &input_2_type : GetIrInputs()) {
    const auto &ir_input = input_2_type.first;
    const auto &ir_input_type = input_2_type.second;
    if ((ir_input_type == kIrInputRequired) || (ir_input_type == kIrInputOptional)) {
      const auto &dtype_symbol = dtype_symbol_store_.GetInputDataTypeSymbol(ir_input);
      const auto &dtype_symbol_validator = dtype_symbol_store_.GetSymbolValidator(dtype_symbol);
      if (dtype_symbol_validator.symbol_type == DataTypeSymbolType::kListTensorType) {
        GELOGE(GRAPH_INVALID_IR_DEF, "Op %s input %s is fix input, its datatype symbol should not in list type.",
               op_name_.c_str(), ir_input.c_str());
        return GRAPH_INVALID_IR_DEF;
      }
    }
  }
  return GRAPH_SUCCESS;
}
} // namespace ge