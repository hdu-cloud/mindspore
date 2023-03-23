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

#ifndef MINDSPORE_CORE_EXPANDER_EMITTER_H_
#define MINDSPORE_CORE_EXPANDER_EMITTER_H_
#include <memory>
#include <vector>
#include <string>
#include <utility>
#include <map>
#include <tuple>
#include "ir/func_graph.h"
#include "ops/core_ops.h"
#include "include/common/utils/utils.h"
#include "expander/node.h"
#include "expander/infer.h"

namespace mindspore {
namespace expander {
class MS_CORE_API Emitter {
 public:
  Emitter(const FuncGraphPtr &func_graph, const ExpanderInferPtr &infer, const ScopePtr &scope = nullptr)
      : func_graph_(func_graph), infer_(infer), scope_(scope) {
    MS_EXCEPTION_IF_NULL(infer);
  }

  /// \brief Emit a primitive CNode
  NodePtr Emit(const std::string &op_name, const NodePtrList &inputs, const DAttr &attrs = {}) const;

  /// \brief Emit a ValueNode
  NodePtr EmitValue(const ValuePtr &value) const;

  NodePtr MakeTuple(const NodePtrList &inputs) const { return Emit(prim::kMakeTuple, inputs); }
  NodePtr TupleGetItem(const NodePtr &input, size_t i) const {
    return Emit(prim::kTupleGetItem, {input, Value(static_cast<int64_t>(i))});
  }

  NodePtr Cast(const NodePtr &node, const TypePtr &type) const;
  NodePtr Cast(const NodePtr &node, TypeId type_id) const { return Cast(node, TypeIdToType(type_id)); }

  NodePtr Reshape(const NodePtr &node, const ShapeVector &shape) const;
  NodePtr ExpandDims(const NodePtr &node, int64_t axis) const { return Emit(kExpandDimsOpName, {node, Value(axis)}); }
  NodePtr Abs(const NodePtr &node) const { return Emit(prim::kAbs, {node}); }
  NodePtr Neg(const NodePtr &node) const { return Emit(prim::kNeg, {node}); }
  NodePtr Reciprocal(const NodePtr &node) const { return Emit(prim::kReciprocal, {node}); }
  NodePtr Square(const NodePtr &node) const { return Emit(prim::kSquare, {node}); }
  NodePtr Sign(const NodePtr &node) const { return Emit(prim::kPrimSign->name(), {node}); }
  NodePtr Exp(const NodePtr &x) const;
  NodePtr Log(const NodePtr &x) const;
  NodePtr Transpose(const NodePtr &node, const ShapeVector &perm) const;
  NodePtr Tile(const NodePtr &node, const ShapeVector &multiples) const {
    bool is_all_one = std::all_of(multiples.begin(), multiples.end(), [](int64_t shp) { return shp == 1; });
    if (is_all_one && node->shape().size() >= multiples.size()) {
      return node;
    }
    return Emit(kTileOpName, {node, Value(multiples)});
  }
  NodePtr Concat(const NodePtrList &inputs, int64_t axis) const {
    return Emit(kConcatOpName, {MakeTuple(inputs)}, {{kAttrAxis, MakeValue(axis)}});
  }

  NodePtr Add(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit(prim::kAdd, lhs, rhs); }
  NodePtr Sub(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit(prim::kSub, lhs, rhs); }
  NodePtr Mul(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit(prim::kMul, lhs, rhs); }
  NodePtr Div(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit(kDivOpName, lhs, rhs); }
  NodePtr RealDiv(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit(prim::kRealDiv, lhs, rhs); }
  NodePtr Mod(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit("Mod", lhs, rhs); }
  NodePtr Pow(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit(kPowOpName, lhs, rhs); }
  NodePtr MatMul(const NodePtr &a, const NodePtr &b, bool transpose_a = false, bool transpose_b = false) const;
  NodePtr BatchMatMul(const NodePtr &a, const NodePtr &b, bool transpose_a = false, bool transpose_b = false) const;
  NodePtr Maximum(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit(kMaximumOpName, lhs, rhs); }
  NodePtr Minimum(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit(kMinimumOpName, lhs, rhs); }
  NodePtr FloorDiv(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit("FloorDiv", lhs, rhs); }
  NodePtr FloorMod(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit("FloorMod", lhs, rhs); }
  NodePtr DivNoNan(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit("DivNoNan", lhs, rhs); }
  NodePtr MulNoNan(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit("MulNoNan", lhs, rhs); }
  NodePtr Xdivy(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit("Xdivy", lhs, rhs); }
  NodePtr Xlogy(const NodePtr &lhs, const NodePtr &rhs) const { return UnifyDtypeAndEmit("Xlogy", lhs, rhs); }

  NodePtr Select(const NodePtr &cond, const NodePtr &lhs, const NodePtr &rhs) const {
    auto [a, b] = UnifyDtype2(lhs, rhs);
    return Emit(kSelectOpName, {cond, a, b});
  }
  NodePtr Less(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type = nullptr) const {
    return CmpOpWithCast(kLessOpName, lhs, rhs, dst_type);
  }
  NodePtr LessEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type = nullptr) const {
    return CmpOpWithCast(kLessEqualOpName, lhs, rhs, dst_type);
  }
  NodePtr Greater(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type = nullptr) const {
    return CmpOpWithCast(kGreaterOpName, lhs, rhs, dst_type);
  }
  NodePtr GreaterEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type = nullptr) const {
    return CmpOpWithCast(kGreaterEqualOpName, lhs, rhs, dst_type);
  }
  NodePtr Equal(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type = nullptr) const {
    return CmpOpWithCast(kEqualOpName, lhs, rhs, dst_type);
  }
  NodePtr NotEqual(const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type = nullptr) const {
    return CmpOpWithCast("NotEqual", lhs, rhs, dst_type);
  }
  NodePtr LogicalAnd(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("LogicalAnd", {lhs, rhs}); }
  NodePtr LogicalOr(const NodePtr &lhs, const NodePtr &rhs) const { return Emit("LogicalOr", {lhs, rhs}); }
  std::pair<bool, ShapeVector> NeedReduce(const ShapeVector &shape, const std::vector<int64_t> &axis,
                                          bool keep_dim) const;
  NodePtr ReduceSum(const NodePtr &x, const ShapeVector &axis = {}, bool keep_dims = false) const;

  NodePtr ZerosLike(const NodePtr &node) const;
  NodePtr Fill(double value, const ShapeVector &shape, TypeId data_type) const;
  NodePtr Fill(int64_t value, const ShapeVector &shape, TypeId data_type) const;

  /// \brief Emit a value node
  template <typename T>
  NodePtr Value(const T &value) const {
    return EmitValue(MakeValue(value));
  }

  /// \brief Emit a Tensor node.
  template <typename T>
  NodePtr Tensor(T data, TypePtr type_ptr = nullptr) const {
    auto tensor_ptr = std::make_shared<tensor::Tensor>(data, type_ptr);
    return EmitValue(tensor_ptr);
  }

  /// \brief Emit a tensor node.
  NodePtr Tensor(TypeId data_type, const ShapeVector &shape, void *data, TypeId src_data_type) const {
    auto tensor_ptr = std::make_shared<tensor::Tensor>(data_type, shape, data, src_data_type);
    return EmitValue(tensor_ptr);
  }

  ExpanderInferPtr infer() const { return infer_; }

 protected:
  NodePtr NewNode(const AnfNodePtr &anfnode) const { return std::make_shared<Node>(anfnode, this); }
  NodePtr CmpOpWithCast(const std::string &op, const NodePtr &lhs, const NodePtr &rhs, const TypePtr &dst_type) const {
    auto node = UnifyDtypeAndEmit(op, lhs, rhs);
    return dst_type == nullptr ? node : Cast(node, dst_type);
  }
  std::tuple<NodePtr, NodePtr> UnifyDtype2(const NodePtr &lhs, const NodePtr &rhs) const;
  NodePtr UnifyDtypeAndEmit(const std::string &op, const NodePtr &a, const NodePtr &b, const DAttr &attrs = {}) const {
    auto [lhs, rhs] = UnifyDtype2(a, b);
    return Emit(op, {lhs, rhs}, attrs);
  }

  FuncGraphPtr func_graph_;
  ExpanderInferPtr infer_{nullptr};
  ScopePtr scope_{nullptr};
  inline static const std::map<TypeId, size_t> type_map_ = {
    {kNumberTypeBool, 1},    {kNumberTypeInt8, 2},    {kNumberTypeUInt8, 3},
    {kNumberTypeInt16, 4},   {kNumberTypeInt32, 5},   {kNumberTypeInt64, 6},
    {kNumberTypeFloat16, 7}, {kNumberTypeFloat32, 8}, {kNumberTypeFloat64, 9}};
};
using EmitterPtr = std::shared_ptr<Emitter>;

MS_CORE_API NodePtr operator+(const NodePtr &lhs, const NodePtr &rhs);
MS_CORE_API NodePtr operator-(const NodePtr &lhs, const NodePtr &rhs);
MS_CORE_API NodePtr operator*(const NodePtr &lhs, const NodePtr &rhs);
MS_CORE_API NodePtr operator/(const NodePtr &lhs, const NodePtr &rhs);
MS_CORE_API NodePtr operator-(const NodePtr &node);
}  // namespace expander
}  // namespace mindspore
#endif  // MINDSPORE_CORE_EXPANDER_EMITTER_H_
