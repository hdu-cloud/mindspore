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

#ifndef METADEF_CXX_GRAPH_IR_META_H_
#define METADEF_CXX_GRAPH_IR_META_H_

#include <string>
#include <vector>
#include "inc/graph/ascend_limits.h"
#include "inc/graph/small_vector.h"
#include "inc/graph/op_desc.h"
#include "graph/ir_data_type_symbol_store.h"

namespace ge {
/**
 *  IR信息
 */
class IRMetaData {
 public:
  explicit IRMetaData(const std::string &op_name) : op_name_(op_name) {};
  IRMetaData() = default;
  IRMetaData(const IRMetaData &other);
  void SetOpName(const std::string &op_name) {
    op_name_ = op_name;
  }
  void AppendIrInput(std::string name, IrInputType input_type);
  const std::vector<std::pair<std::string, IrInputType>> &GetIrInputs() const;
  IrInputType GetIrInputType(const std::string &name) const;

  void AppendIrOutput(std::string name, IrOutputType output_type);
  const std::vector<std::pair<std::string, IrOutputType>> &GetIrOutputs() const;

  graphStatus AddRegisterInputName(const std::string &name);
  std::vector<std::string> GetRegisterInputName() const;

  graphStatus AddRegisterOptionalInputName(const std::string &name);
  std::set<std::string> GetOptionalInputName() const;
  bool IsOptionalInput(const std::string &name) const;

  graphStatus AddRegisterOutputName(const std::string &name);
  std::vector<std::string> GetRegisterOutputName() const;

  void AppendIrAttrName(std::string name);
  const std::vector<std::string> &GetIrAttrNames() const;

  void RegisterSubgraphIrName(const std::string &name, const SubgraphType type);
  const std::map<std::string, SubgraphType> &GetSubgraphIrNames() const;
  SubgraphType GetSubgraphTypeByIrName(const std::string &name) const;

  IRDataTypeSymbolStore &MutableIRDataTypeSymbolStore();
  const IRDataTypeSymbolStore &GetIRDataTypeSymbolStore() const;

  graphStatus VerifyIR() const;
  graphStatus VerifyDataTypeSymbol() const;

  bool operator==(const IRMetaData &other) const;
  IRMetaData &operator=(const IRMetaData &other);

 private:
  bool IsOutputSymbolValid(const std::string &output_symbol) const;
  std::string op_name_;
  std::vector<std::pair<std::string, IrInputType>> ir_inputs_;
  std::vector<std::pair<std::string, IrOutputType>> ir_outputs_;
  std::vector<std::string> register_input_name_; // todo need to deprecate
  std::set<std::string> optional_input_names_; // todo need to deprecate
  std::vector<std::string> register_output_name_;
  std::vector<std::string> ir_attr_names_;
  // subgraph ir names to type, for a `if` operator:
  // then_branch: static
  // else_branch: static
  // or for a `case` op:
  // branches: dynamic
  std::map<std::string, SubgraphType> subgraph_ir_names_to_type_;
  IRDataTypeSymbolStore dtype_symbol_store_;
};

class OpMetadata {
 public:
  using SmallIntVector = SmallVector<int64_t, static_cast<size_t>(kDefaultMaxInputNum)>;
  OpMetadata() = default;
  ~OpMetadata() = default;
  OpMetadata(std::string name, std::string type) : name_(std::move(name)), type_(std::move(type)), ir_meta_(name) {}
  int64_t GetId() const {return id_;}
  int64_t GetStreamId() const {return stream_id_;}
  const std::vector<std::string> &GetInputNames() const {return input_names_;}
  const std::vector<std::string> &GetSrcNames() const {return src_names_;}
  const std::vector<int64_t> &GetSrcIndexes() const {return src_indexes_;}
  const std::vector<std::string> &GetDstNames() const {return dst_names_;}
  const std::vector<int64_t> &GetDstIndexes() const {return dst_indexes_;}
  const std::vector<int64_t> &GetInputOffsets() const {return input_offsets_;}
  const std::vector<int64_t> &GetOutputOffsets() const {return output_offsets_;}
  const std::vector<bool> &GetIsInputConsts() const {return is_input_consts_;}
  const std::vector<std::string> &GetSubgraphNames() const {return subgraph_names_;}
  void AddSubGraphName(const std::string &name) {subgraph_names_.push_back(name);}
  void ClearSubgraphNames() { subgraph_names_.clear(); }
  void SetOpName(const std::string &name) {
    name_ = std::move(name);
    ir_meta_.SetOpName(name);
  }

 private:
  friend class OpDescImpl;
  std::string name_;
  std::string type_;
  std::vector<std::string> inputs_;
  bool has_out_attr_{false};
  int64_t id_{0};
  int64_t stream_id_{0};
  std::vector<std::string> input_names_;
  std::vector<std::string> src_names_;
  std::vector<int64_t> src_indexes_;
  std::vector<std::string> dst_names_;
  std::vector<int64_t> dst_indexes_;
  std::vector<int64_t> input_offsets_;
  std::vector<int64_t> output_offsets_;
  SmallIntVector workspaces;
  SmallIntVector workspace_bytes_list_;
  std::vector<bool> is_input_consts_;
  std::vector<std::string> subgraph_names_;
  IRMetaData ir_meta_;
};
} // namespace ge
#endif // METADEF_CXX_GRAPH_IR_META_H_
