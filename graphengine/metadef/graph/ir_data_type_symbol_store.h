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

#ifndef METADEF_CXX_GRAPH_IR_DATA_TYPE_SYMBOL_STORE_H_
#define METADEF_CXX_GRAPH_IR_DATA_TYPE_SYMBOL_STORE_H_

#include <string>
#include <set>
#include <unordered_map>
#include "graph/types.h"
#include "graph/ge_error_codes.h"
#include "graph/tensor_type_impl.h"

namespace ge {
enum class DataTypeSymbolType {
  kTensorType,
  kListTensorType,
  kInvalidTensorType
};
using InputNamesSharedDTypeSymbol = std::unordered_map<std::string, std::set<std::string>>;
struct DataTypeSymbol {
  std::string symbol;
  DataTypeSymbolType symbol_type = DataTypeSymbolType::kInvalidTensorType;
  TensorType type_range = TensorType(DT_MAX);
  DataTypeSymbol() = default;
  DataTypeSymbol(const std::string &symbol, const DataTypeSymbolType symbol_type, const TensorType &type_range)
      : symbol(symbol), symbol_type(symbol_type), type_range(type_range) {};
  bool IsDataTypeInRange(const DataType &data_type) const {
    return type_range.tensor_type_impl_->IsDataTypeInRange(data_type);
  }
  bool IsValidSymbol() const {
    return (symbol_type != DataTypeSymbolType::kInvalidTensorType);
  }
  bool IsFixedRange() const {
    return (type_range.tensor_type_impl_->GetMutableDateTypeSet().size() == 1);
  }
};

/**
 * @brief 建立IR上输入输出和data type symbol的映射
 */
class IRDataTypeSymbolStore {
 public:
  IRDataTypeSymbolStore() = default;
  ~IRDataTypeSymbolStore() = default;
  graphStatus RegisterDataTypeSymbol(const std::string &symbol, const TensorType &type_range);
  graphStatus RegisterDataTypeSymbol(const std::string &symbol, const ListTensorType &type_range);

  graphStatus AddInputName2DataTypeSymbol(const std::string &input_name, const std::string &symbol);
  graphStatus AddOutputName2DataTypeSymbol(const std::string &output_name, const std::string &symbol);

  const DataTypeSymbol &GetSymbolValidator(const std::string &symbol) const;
  std::string GetInputDataTypeSymbol(const std::string &input_name) const;
  std::string GetOutputDataTypeSymbol(const std::string &output_name) const;

  std::string GetInputNameFromDataTypeSymbol(const std::string &symbol) const;
  const InputNamesSharedDTypeSymbol &GetAllInputNamesShareDataTypeSymbol() const;

 private:
  std::unordered_map<std::string, DataTypeSymbol> symbols_to_validator;
  std::unordered_map<std::string, std::string> ir_inputs_2_symbol_;
  std::unordered_map<std::string, std::set<std::string>> symbols_2_inputs_;
  std::unordered_map<std::string, std::string> ir_outputs_2_symbol_;
};
} // namespace ge
#endif // METADEF_CXX_GRAPH_IR_DATA_TYPE_SYMBOL_STORE_H_
