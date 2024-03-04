/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2022. All rights reserved.
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
#include "common/ge_common/debug/ge_log.h"
#include "common/ge_common/string_util.h"
#include "common/util/mem_utils.h"
#include "common/checker.h"
#include "graph/utils/type_utils.h"
#include "graph/utils/op_desc_utils.h"
namespace ge {
namespace {
graphStatus UpdateOpOuputListDtype(const OpDescPtr &op, const size_t &start, const size_t &end,
                                   const std::vector<DataType> &dtypes) {
  GE_ASSERT((end - start) == dtypes.size(), "Size mismatch when update %s output [%zu, %zu) with %zu dtypes",
            op->GetName().c_str(), start, end, dtypes.size());
  for (size_t i = start; i < end; i++) {
    auto desc = op->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(desc);
    GELOGI("Update op %s output %s:%zu with dtype %s", op->GetType().c_str(), op->GetName().c_str(), i,
           TypeUtils::DataTypeToSerialString(dtypes[i - start]).c_str());
    desc->SetDataType(dtypes[i - start]);
  }
  return GRAPH_SUCCESS;
}

graphStatus UpdateOpOuputDtype(const OpDescPtr &op, const size_t &start, const size_t &end, const DataType &dtype) {
  for (size_t i = start; i < end; i++) {
    auto desc = op->MutableOutputDesc(i);
    GE_ASSERT_NOTNULL(desc);
    GELOGI("Update op %s output %s:%zu with dtype %s", op->GetType().c_str(), op->GetName().c_str(), i,
           TypeUtils::DataTypeToSerialString(dtype).c_str());
    desc->SetDataType(dtype);
  }
  return GRAPH_SUCCESS;
}

const char *ToString(const IrInputType &type) {
  if (type == kIrInputRequired) {
    return "Required";
  }
  if (type == kIrInputOptional) {
    return "Optional";
  }
  if (type == kIrInputDynamic) {
    return "Dynamic";
  }
  return "Unknown";
}

const char *ToString(const IrOutputType &type) {
  if (type == kIrOutputRequired) {
    return "Required";
  }
  if (type == kIrOutputDynamic) {
    return "Dynamic";
  }
  return "Unknown";
}

std::string DebugString(const TensorType &types) {
  std::string s = "[";
  for (auto &dtype : types.tensor_type_impl_->GetMutableDateTypeSet()) {
    s += TypeUtils::DataTypeToSerialString(dtype);
    s += ",";
  }
  s += "]";
  return s;
}

std::string DebugString(const std::vector<const char *> &syms) {
  std::string s = "[";
  for (auto &sym : syms) {
    s += sym;
    s += ",";
  }
  s += "]";
  return s;
}

std::string RemoveQuotes(const std::string &str) {
  std::string result = str;
  if (!result.empty() && result.front() == '\"') {
    result.erase(result.begin());
  }

  if (!result.empty() && result.back() == '\"') {
    result.pop_back();
  }
  return result;
}
}  // namespace

graphStatus IRDataTypeSymbolStore::InferDtype(const OpDescPtr &op) const {
  GE_ASSERT_NOTNULL(op);
  GELOGD("Start infer output dtype for op %s by syms", op->GetName().c_str());
  std::map<SymDtype *, TypeOrTypes> cached;  // 缓存每个Sym的求值结果，避免重复求值
  for (auto &item : named_syms_) {
    auto *sym = item.second;
    GE_ASSERT_GRAPH_SUCCESS(sym->Eval(*op, cached[sym]), "Failed eval sym %s of op %s", sym->DebugString().c_str(),
                            op->GetType().c_str());
    GELOGD("Succeed eval and checking sym %s", sym->DebugString().c_str());
  }

  std::map<size_t, std::pair<size_t, size_t>> ir_output_2_range;
  GE_ASSERT_GRAPH_SUCCESS(ge::GetIrOutputDescRange(op, ir_output_2_range));
  GE_ASSERT(ir_output_2_range.size() == op->GetIrOutputs().size(), "Failed get output instance info of %s %s",
            op->GetName().c_str(), op->GetType().c_str());

  GE_ASSERT(output_syms_.size() == op->GetIrOutputs().size(), "Op %s %s has %zu ir outputs, but %zu output syms",
            op->GetName().c_str(), op->GetType().c_str(), op->GetIrOutputs().size(), output_syms_.size());
  // 对全部输出表达式进行求值，并更新到op上
  for (size_t i = 0U; i < output_syms_.size(); i++) {
    auto *sym = output_syms_[i];
    GE_ASSERT_NOTNULL(sym);
    if (sym->IsLegacy()) {
      GELOGW("Trying infer legacy output %s(%s) of %s(%s)", sym->Id().c_str(), sym->DebugString().c_str(),
             op->GetName().c_str(), op->GetType().c_str());
      return GRAPH_FAILED;
    }

    TypeOrTypes type_or_types;
    auto cached_iter = cached.find(sym);
    if (cached_iter != cached.end()) {
      type_or_types = cached_iter->second;
    } else {
      GE_ASSERT_GRAPH_SUCCESS(sym->Eval(*op, type_or_types));
    }

    auto &output_range = ir_output_2_range[i];
    size_t start = output_range.first;
    size_t end = output_range.first + output_range.second;

    if (type_or_types.IsListType()) {  // ListType表示输出为动态输出，并且每个输出的类型可以不同
      GE_ASSERT(output_name_and_types_[i].second == kIrOutputDynamic, "Op %s %s output %s bind to list-type sym %s",
                op->GetType().c_str(), ToString(output_name_and_types_[i].second),
                output_name_and_types_[i].first.c_str(), sym->Id().c_str());
      GE_ASSERT_GRAPH_SUCCESS(UpdateOpOuputListDtype(op, start, end, type_or_types.UnsafeGetTypes()));
    } else {
      GE_ASSERT_GRAPH_SUCCESS(UpdateOpOuputDtype(op, start, end, type_or_types.UnsafeGetType()));
    }
  }

  return GRAPH_SUCCESS;
}

// 创建一个命名Sym，用于符号出现早于IR输入的情况
SymDtype *IRDataTypeSymbolStore::GetOrCreateSymbol(const std::string &origin_sym_id) {
  std::string sym_id = RemoveQuotes(origin_sym_id);
  for (auto &sym : syms_) {
    if (sym->Id() == sym_id) {
      return sym.get();
    }
  }
  GELOGI("Create new symbol %s", sym_id.c_str());
  auto sym = MakeShared<SymDtype>(sym_id);
  GE_ASSERT_NOTNULL(sym, "Failed create symbol %s", sym_id.c_str());
  syms_.emplace_back(sym);
  return syms_.back().get();
}

SymDtype *IRDataTypeSymbolStore::SetInputSymbol(const std::string &ir_input, IrInputType input_type,
                                                const std::string &sym_id) {
  GELOGI("Create symbol %s for %s input %s", sym_id.c_str(), ToString(input_type), ir_input.c_str());
  auto *sym = GetOrCreateSymbol(sym_id);
  GE_ASSERT_NOTNULL(sym);
  sym->BindIrInput(ir_input, input_type, num_ir_inputs++);
  return sym;
}

// 调用DATATYPE声明时，绑定Sym的取值范围
SymDtype *IRDataTypeSymbolStore::DeclareSymbol(const std::string &sym_id, const TensorType &types) {
  GELOGI("Bind symbol %s with allowed dtypes %s", sym_id.c_str(), DebugString(types).c_str());
  auto *sym = GetOrCreateSymbol(sym_id);
  GE_ASSERT_NOTNULL(sym);
  (void) named_syms_.emplace(sym_id, sym);
  sym->BindAllowedDtypes(types);
  return sym;
}

SymDtype *IRDataTypeSymbolStore::DeclareSymbol(const std::string &sym_id, const ListTensorType &types) {
  GELOGI("Bind symbol %s with allowed list-dtypes %s", sym_id.c_str(), DebugString(types.tensor_type).c_str());
  auto *sym = GetOrCreateSymbol(sym_id);
  GE_ASSERT_NOTNULL(sym);
  (void) named_syms_.emplace(sym_id, sym);
  sym->BindAllowedDtypes(types);
  return sym;
}

SymDtype *IRDataTypeSymbolStore::DeclareSymbol(const std::string &sym_id, const Promote &types) {
  GELOGI("Bind symbol %s with rule promote %s", sym_id.c_str(), DebugString(types.Syms()).c_str());
  std::vector<SymDtype *> syms;
  for (auto &id : types.Syms()) {
    GE_ASSERT(id != sym_id, "Trying promote symbol %s with itself", sym_id.c_str());
    auto *sym = GetOrCreateSymbol(id);
    GE_ASSERT_NOTNULL(sym);
    syms.emplace_back(sym);
  }

  auto *sym = GetOrCreateSymbol(sym_id);
  GE_ASSERT_NOTNULL(sym);
  (void) named_syms_.emplace(sym_id, sym);
  sym->BindExpression(MakeShared<PromotionSymDtypeExpression>(syms));
  return sym;
}

// 创建输出的Symbol表达式，用于支持类型推导
SymDtype *IRDataTypeSymbolStore::SetOutputSymbol(const std::string &ir_output, IrOutputType output_type,
                                                 const std::string &sym_id) {
  GELOGI("Create symbol expression %s for %s output %s", sym_id.c_str(), ToString(output_type), ir_output.c_str());
  auto *sym = GetOrCreateSymbol(sym_id);
  GE_ASSERT_NOTNULL(sym);

  output_syms_.emplace_back(sym);
  output_name_and_types_.emplace_back(ir_output, output_type);
  return sym;
}
}  // namespace ge