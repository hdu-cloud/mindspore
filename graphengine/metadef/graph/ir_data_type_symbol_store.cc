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

#include "ir_data_type_symbol_store.h"
#include "framework/common/debug/ge_log.h"
#include "framework/common/string_util.h"
namespace ge {
namespace {
bool IsValidDataTypeSymbol(const std::string &symbol) {
  return (symbol.find("(") == std::string::npos);
}
} // namespace

static const DataTypeSymbol& InvalidDataTypeSymbol() {
  const static DataTypeSymbol kGlobalInvalidDataTypeSymbol;
  return kGlobalInvalidDataTypeSymbol;
}

graphStatus IRDataTypeSymbolStore::RegisterDataTypeSymbol(const std::string &symbol,
                                                          const TensorType &type_range) {
  auto iter = symbols_to_validator.find(symbol);
  if (iter != symbols_to_validator.end()) {
    GELOGE(GRAPH_PARAM_INVALID, "Datatype Symbol %s is register duplicated.", symbol.c_str());
    return GRAPH_PARAM_INVALID;
  }
  symbols_to_validator[symbol] =
      DataTypeSymbol(symbol, DataTypeSymbolType::kTensorType, type_range);
  return GRAPH_SUCCESS;
}

graphStatus IRDataTypeSymbolStore::RegisterDataTypeSymbol(const std::string &symbol,
                                                          const ListTensorType &type_range) {
  auto iter = symbols_to_validator.find(symbol);
  if (iter != symbols_to_validator.end()) {
    GELOGE(GRAPH_PARAM_INVALID, "Datatype Symbol %s is register duplicated.", symbol.c_str());
    return GRAPH_PARAM_INVALID;
  }
  symbols_to_validator[symbol] =
      DataTypeSymbol(symbol, DataTypeSymbolType::kListTensorType, type_range.tensor_type);
  return GRAPH_SUCCESS;
}

graphStatus IRDataTypeSymbolStore::AddInputName2DataTypeSymbol(const std::string &input_name,
                                                               const std::string &symbol) {
  if (!IsValidDataTypeSymbol(symbol)) {
    return GRAPH_SUCCESS;
  }
  // user define would be like INPUT(x, "T")
  // after Micro, got "T" as symbol, should trim double quotation mark
  auto normalized_symbol = StringUtils::ReplaceAll(symbol, "\"", "");
  ir_inputs_2_symbol_[input_name] = normalized_symbol;
  symbols_2_inputs_[normalized_symbol].insert(input_name);
  return GRAPH_SUCCESS;
}

graphStatus IRDataTypeSymbolStore::AddOutputName2DataTypeSymbol(const std::string &output_name,
                                                                const std::string &symbol) {
  if (!IsValidDataTypeSymbol(symbol)) {
    return GRAPH_SUCCESS;
  }
  // user define would be like OUTPUT(x, "T")
  // after Micro, got "T" as symbol, should trim double quotation mark
  auto normalized_symbol = StringUtils::ReplaceAll(symbol, "\"", "");
  ir_outputs_2_symbol_[output_name] = normalized_symbol;
  return GRAPH_SUCCESS;
}

const DataTypeSymbol& IRDataTypeSymbolStore::GetSymbolValidator(const std::string &symbol) const {
  auto iter = symbols_to_validator.find(symbol);
  if (iter == symbols_to_validator.end()) {
    return InvalidDataTypeSymbol();
  }
  return iter->second;
}

std::string IRDataTypeSymbolStore::GetInputDataTypeSymbol(const std::string &input_name) const {
  auto iter = ir_inputs_2_symbol_.find(input_name);
  return iter->second;
}
std::string IRDataTypeSymbolStore::GetOutputDataTypeSymbol(const std::string &output_name) const {
  auto iter = ir_outputs_2_symbol_.find(output_name);
  if (iter == ir_outputs_2_symbol_.end()) {
    return "";
  }
  return iter->second;
}

std::string IRDataTypeSymbolStore::GetInputNameFromDataTypeSymbol(const std::string &symbol) const {
  auto iter = symbols_2_inputs_.find(symbol);
  if (iter == symbols_2_inputs_.end() || iter->second.empty()) {
    return std::string();
  }
  return *(iter->second.begin());
}

const InputNamesSharedDTypeSymbol &IRDataTypeSymbolStore::GetAllInputNamesShareDataTypeSymbol() const {
  return symbols_2_inputs_;
}
} // namespace ge