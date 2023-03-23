/**
 * Copyright 2021 Huawei Technologies Co., Ltd
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

#define USE_DEPRECATED_API
#include "tools/optimizer/fusion/multi_head_attention_fusion.h"
#include <functional>
#include <utility>
#include <vector>
#include "tools/optimizer/common/gllo_utils.h"
#include "nnacl/op_base.h"
#include "ops/tuple_get_item.h"
#include "tools/common/tensor_util.h"

namespace mindspore::opt {
namespace {
const auto &p1 = std::placeholders::_1;
const size_t kWeightShapeSize = 2;
const int kAttentionOutputs = 3;
}  // namespace

bool MultiHeadAttentionFusion::Init() const {
  input_q_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_q_ != nullptr, false);
  input_k_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_k_ != nullptr, false);
  input_v_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(input_v_ != nullptr, false);

  weight_q_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(weight_q_ != nullptr, false);
  weight_k_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(weight_k_ != nullptr, false);
  weight_v_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(weight_v_ != nullptr, false);
  weight_o_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(weight_o_ != nullptr, false);

  bias_q_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(bias_q_ != nullptr, false);
  bias_k_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(bias_k_ != nullptr, false);
  bias_v_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(bias_v_ != nullptr, false);
  bias_o_ = std::make_shared<CondVar>(IsParamNode);
  MS_CHECK_TRUE_RET(bias_o_ != nullptr, false);

  mask_ = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(mask_ != nullptr, false);

  reshape_k_ = std::make_shared<Var>("reshape_k");
  MS_CHECK_TRUE_RET(reshape_k_ != nullptr, false);
  reshape_v_ = std::make_shared<Var>("reshape_v");
  MS_CHECK_TRUE_RET(reshape_v_ != nullptr, false);
  reshape_axis_ = std::make_shared<Var>("reshape_axis");
  MS_CHECK_TRUE_RET(reshape_axis_ != nullptr, false);
  v_transpose_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose), "v_transpose");
  MS_CHECK_TRUE_RET(v_transpose_ != nullptr, false);
  k_transpose_ = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose), "k_transpose");
  MS_CHECK_TRUE_RET(k_transpose_ != nullptr, false);
  return true;
}

namespace {
VectorRef DefineMask(const BaseRef &mask_input) {
  auto is_expand_dims = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimExpandDims));
  MS_CHECK_TRUE_RET(is_expand_dims != nullptr, {});
  auto var1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto expand_dims = VectorRef({is_expand_dims, mask_input, var1});
  auto is_sub = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSubFusion));
  MS_CHECK_TRUE_RET(is_sub != nullptr, {});
  auto var2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var2 != nullptr, {});
  auto sub = VectorRef({is_sub, var2, expand_dims});
  auto is_mul = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMulFusion));
  MS_CHECK_TRUE_RET(is_mul != nullptr, {});
  auto var3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var3 != nullptr, {});
  return VectorRef({is_mul, sub, var3});
}

STATUS GetAxis(const BaseRef &n, std::vector<int> *axes) {
  MS_ASSERT(axes != nullptr);
  if (utils::isa<ValueNodePtr>(n)) {
    auto axes_value_node = utils::cast<ValueNodePtr>(n);
    *axes = CastToInt(axes_value_node->value());
    return lite::RET_OK;
  } else {
    MS_LOG(ERROR) << "GetAxis supports only value node";
  }
  return lite::RET_ERROR;
}
}  // namespace

VectorRef MultiHeadAttentionFusion::DefineEmbedding(const BaseRef &input, const BaseRef &weight, const BaseRef &bias,
                                                    const BaseRef &axis, const BaseRef &transpose_var, bool test_div,
                                                    bool transpose) const {
  auto is_matmul = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "e-matmul");
  MS_CHECK_TRUE_RET(is_matmul != nullptr, {});
  auto dense = VectorRef({is_matmul, input, weight, bias});
  auto is_reshape = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "e-reshape");
  MS_CHECK_TRUE_RET(is_reshape != nullptr, {});
  auto reshape = VectorRef({is_reshape, dense, axis});
  auto var2 = std::make_shared<Var>();
  VectorRef conn;
  if (transpose) {
    conn = VectorRef({transpose_var, reshape, var2});
  } else {
    conn = reshape;
  }
  if (test_div) {
    auto is_div = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimRealDiv), "e-div");
    MS_CHECK_TRUE_RET(is_div != nullptr, {});
    auto var3 = std::make_shared<Var>();
    MS_CHECK_TRUE_RET(var3 != nullptr, {});
    auto div = VectorRef({is_div, conn, var3});
    return div;
  }
  return conn;
}

VectorRef MultiHeadAttentionFusion::DefineMPWithMaskPattern(bool cross, bool mask) const {
  VectorRef k_embedding, v_embedding;
  auto q_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(q_transpose != nullptr, {});
  auto q_embedding = DefineEmbedding(input_q_, weight_q_, bias_q_, reshape_axis_, q_transpose, true);
  MS_CHECK_TRUE_RET(!q_embedding.empty(), {});
  if (!cross) {
    k_embedding = DefineEmbedding(input_k_, weight_k_, bias_k_, reshape_axis_, k_transpose_, true);
    MS_CHECK_TRUE_RET(!k_embedding.empty(), {});
    v_embedding = DefineEmbedding(input_v_, weight_v_, bias_v_, reshape_axis_, v_transpose_);
    MS_CHECK_TRUE_RET(!v_embedding.empty(), {});
  } else {
    k_embedding = DefineEmbedding(input_k_, weight_k_, bias_k_, reshape_axis_, k_transpose_, true);
    MS_CHECK_TRUE_RET(!k_embedding.empty(), {});
    v_embedding = DefineEmbedding(input_v_, weight_v_, bias_v_, reshape_axis_, v_transpose_);
    MS_CHECK_TRUE_RET(!v_embedding.empty(), {});
  }
  auto is_matmul1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion));
  MS_CHECK_TRUE_RET(is_matmul1 != nullptr, {});
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto matmul1 = VectorRef({is_matmul1, q_embedding, k_embedding});
  auto var1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  VectorRef reshape1;
  if (mask) {
    auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion));
    MS_CHECK_TRUE_RET(is_add != nullptr, {});
    auto mask = DefineMask(mask_);
    MS_CHECK_TRUE_RET(!mask.empty(), {});
    auto add = VectorRef({is_add, mask, matmul1});
    reshape1 = VectorRef({is_reshape1, add, var1});
  } else {
    reshape1 = VectorRef({is_reshape1, matmul1, var1});
  }
  auto is_softmax = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSoftmax));
  MS_CHECK_TRUE_RET(is_softmax != nullptr, {});
  auto softmax = VectorRef({is_softmax, reshape1});
  auto is_reshape2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape2 != nullptr, {});
  auto var2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var2 != nullptr, {});
  auto reshape2 = VectorRef({is_reshape2, softmax, var2});
  auto is_matmul2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion));
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  auto matmul2 = VectorRef({is_matmul2, reshape2, v_embedding});
  auto is_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto var3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var3 != nullptr, {});
  auto transpose = VectorRef({is_transpose, matmul2, var3});
  auto is_reshape3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape));
  MS_CHECK_TRUE_RET(is_reshape3 != nullptr, {});
  auto var4 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var4 != nullptr, {});
  auto reshape3 = VectorRef({is_reshape3, transpose, var4});
  auto is_matmul3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion));
  MS_CHECK_TRUE_RET(is_matmul3 != nullptr, {});
  auto matmul3 = VectorRef({is_matmul3, reshape3, weight_o_, bias_o_});
  return matmul3;
}

VectorRef MultiHeadAttentionFusion::DefineMPWithMaskPatternT5(bool cross) const {
  VectorRef k_embedding, v_embedding;
  auto q_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose), "q_transpose");
  MS_CHECK_TRUE_RET(q_transpose != nullptr, {});
  auto q_embedding = DefineEmbedding(input_q_, weight_q_, bias_q_, reshape_axis_, q_transpose, true, false);
  MS_CHECK_TRUE_RET(!q_embedding.empty(), {});
  if (!cross) {
    k_embedding = DefineEmbedding(input_k_, weight_k_, bias_k_, reshape_axis_, k_transpose_, true);
    MS_CHECK_TRUE_RET(!k_embedding.empty(), {});
    v_embedding = DefineEmbedding(input_v_, weight_v_, bias_v_, reshape_axis_, v_transpose_);
    MS_CHECK_TRUE_RET(!v_embedding.empty(), {});
  } else {
    k_embedding = DefineEmbedding(input_k_, weight_k_, bias_k_, reshape_axis_, k_transpose_, true);
    MS_CHECK_TRUE_RET(!k_embedding.empty(), {});
    v_embedding = DefineEmbedding(input_v_, weight_v_, bias_v_, reshape_axis_, v_transpose_);
    MS_CHECK_TRUE_RET(!v_embedding.empty(), {});
  }
  auto is_matmul1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "matmul1");
  MS_CHECK_TRUE_RET(is_matmul1 != nullptr, {});
  auto is_reshape1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape1");
  MS_CHECK_TRUE_RET(is_reshape1 != nullptr, {});
  auto matmul1 = VectorRef({is_matmul1, q_embedding, k_embedding});
  auto var1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion), "add");
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto mask = DefineMask(mask_);
  MS_CHECK_TRUE_RET(!mask.empty(), {});
  auto add = VectorRef({is_add, mask, matmul1});
  auto reshape1 = VectorRef({is_reshape1, add, var1});
  auto is_softmax = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSoftmax), "softmax");
  MS_CHECK_TRUE_RET(is_softmax != nullptr, {});
  auto softmax = VectorRef({is_softmax, reshape1});
  auto is_reshape2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape2");
  MS_CHECK_TRUE_RET(is_reshape2 != nullptr, {});
  auto var2 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var2 != nullptr, {});
  auto reshape2 = VectorRef({is_reshape2, softmax, var2});
  auto is_matmul2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "matmul2");
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  auto matmul2 = VectorRef({is_matmul2, reshape2, v_embedding});
  auto is_reshape3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "reshape3");
  MS_CHECK_TRUE_RET(is_reshape3 != nullptr, {});
  auto var4 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var4 != nullptr, {});
  auto reshape3 = VectorRef({is_reshape3, matmul2, var4});
  auto is_matmul3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "matmul");
  MS_CHECK_TRUE_RET(is_matmul3 != nullptr, {});
  auto matmul3 = VectorRef({is_matmul3, reshape3, weight_o_, bias_o_});
  return matmul3;
}

VectorRef MultiHeadAttentionFusion::DefineMPWithMaskPatternPA(bool cross) const {
  VectorRef k_embedding, v_embedding;
  auto q_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(q_transpose != nullptr, {});
  auto q_embedding = DefineEmbedding(input_q_, weight_q_, bias_q_, reshape_axis_, q_transpose, true);
  MS_CHECK_TRUE_RET(!q_embedding.empty(), {});
  if (!cross) {
    k_embedding = DefineEmbedding(input_k_, weight_k_, bias_k_, reshape_axis_, k_transpose_, true);
    MS_CHECK_TRUE_RET(!k_embedding.empty(), {});
    v_embedding = DefineEmbedding(input_v_, weight_v_, bias_v_, reshape_axis_, v_transpose_);
    MS_CHECK_TRUE_RET(!v_embedding.empty(), {});
  } else {
    k_embedding = DefineEmbedding(input_k_, weight_k_, bias_k_, reshape_axis_, k_transpose_, true);
    MS_CHECK_TRUE_RET(!k_embedding.empty(), {});
    v_embedding = DefineEmbedding(input_v_, weight_v_, bias_v_, reshape_axis_, v_transpose_);
    MS_CHECK_TRUE_RET(!v_embedding.empty(), {});
  }
  auto is_matmul1 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion));
  MS_CHECK_TRUE_RET(is_matmul1 != nullptr, {});
  auto matmul1 = VectorRef({is_matmul1, q_embedding, k_embedding});
  auto var1 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var1 != nullptr, {});
  auto is_add = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimAddFusion));
  MS_CHECK_TRUE_RET(is_add != nullptr, {});
  auto mask = DefineMask(mask_);
  MS_CHECK_TRUE_RET(!mask.empty(), {});
  auto add = VectorRef({is_add, mask, matmul1});
  auto is_softmax = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimSoftmax));
  MS_CHECK_TRUE_RET(is_softmax != nullptr, {});
  auto softmax = VectorRef({is_softmax, add});
  auto is_matmul2 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion));
  MS_CHECK_TRUE_RET(is_matmul2 != nullptr, {});
  auto matmul2 = VectorRef({is_matmul2, softmax, v_embedding});
  auto is_transpose = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimTranspose));
  MS_CHECK_TRUE_RET(is_transpose != nullptr, {});
  auto var3 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var3 != nullptr, {});
  auto transpose = VectorRef({is_transpose, matmul2, var3});
  auto is_reshape3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimReshape), "is_reshape3");
  MS_CHECK_TRUE_RET(is_reshape3 != nullptr, {});
  auto var4 = std::make_shared<Var>();
  MS_CHECK_TRUE_RET(var4 != nullptr, {});
  auto reshape3 = VectorRef({is_reshape3, transpose, var4});
  auto is_matmul3 = std::make_shared<CondVar>(std::bind(IsOpType, p1, prim::kPrimMatMulFusion), "is_matmul3");
  MS_CHECK_TRUE_RET(is_matmul3 != nullptr, {});
  auto matmul3 = VectorRef({is_matmul3, reshape3, weight_o_, bias_o_});
  return matmul3;
}

namespace {
STATUS TransposeMatrix(std::shared_ptr<tensor::Tensor> src, std::shared_ptr<tensor::Tensor> dst) {
  MS_CHECK_TRUE_RET(src->shape().size() == C2NUM, RET_ERROR);
  MS_CHECK_TRUE_RET(dst->shape().size() == C2NUM, RET_ERROR);
  int rows = src->shape().at(0);
  int cols = src->shape().at(1);
  auto src_ptr = reinterpret_cast<float *>(src->data_c());
  auto dst_ptr = reinterpret_cast<float *>(dst->data_c());
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      auto val = src_ptr[r * cols + c];
      dst_ptr[c * rows + r] = val;
    }
  }
  return RET_OK;
}

std::shared_ptr<tensor::Tensor> ConcatTensors(const std::vector<std::shared_ptr<tensor::Tensor>> &tensors,
                                              bool transpose = false) {
  const std::vector<int64_t> &base_shape = tensors.at(0)->shape();
  auto base_shape_size = base_shape.size();
  auto base_data_type = tensors.at(0)->data_type();
  auto res =
    std::all_of(tensors.begin() + 1, tensors.end(),
                [&base_shape_size, &base_shape, &base_data_type](const std::shared_ptr<tensor::Tensor> &tensor) {
                  if (tensor->shape().size() != base_shape_size) {
                    return false;
                  }
                  auto &shape = tensor->shape();
                  for (std::size_t i = 1; i < shape.size(); ++i) {
                    if (shape.at(i) != base_shape.at(i)) return false;
                  }
                  if (tensor->data_type() != base_data_type) {
                    return false;
                  }
                  return true;
                });
  MS_CHECK_TRUE_RET(res, nullptr);
  // calculate shape
  std::vector<int64_t> new_shape;
  auto sum = std::accumulate(tensors.begin(), tensors.end(), 0,
                             [](int sum, const tensor::TensorPtr &tensor) { return sum + tensor->shape().at(0); });
  new_shape.push_back(sum);
  for (std::size_t i = 1; i < base_shape_size; ++i) {
    new_shape.push_back(base_shape.at(i));
  }

  // calculate data
  auto concat_tensor = std::make_shared<tensor::Tensor>(base_data_type, new_shape);
  MS_CHECK_TRUE_RET(concat_tensor != nullptr, nullptr);
  std::size_t offset = 0;
  for (const auto &tensor : tensors) {
    void *ptr = reinterpret_cast<uint8_t *>(concat_tensor->data_c()) + offset;
    memcpy_s(ptr, concat_tensor->Size() - offset, tensor->data_c(), tensor->Size());
    offset += tensor->Size();
  }
  if (transpose) {
    std::vector<int64_t> tshape = {new_shape[1], new_shape[0]};
    auto transposed_tensor = std::make_shared<tensor::Tensor>(base_data_type, tshape);
    auto status = TransposeMatrix(concat_tensor, transposed_tensor);
    MS_CHECK_TRUE_RET(status == RET_OK, nullptr);
    return transposed_tensor;
  }
  return concat_tensor;
}
}  // namespace

std::unordered_map<std::string, VectorRef> MultiHeadAttentionFusion::DefinePatterns() const {
  std::unordered_map<std::string, VectorRef> patterns;
  if (!Init()) {
    MS_LOG(ERROR) << "initial member failed.";
    return patterns;
  }
  patterns[kMPAWithMaskPatternName] = DefineMPWithMaskPattern();
  patterns[kMPAXWithMaskPatternName] = DefineMPWithMaskPattern(true);
  patterns[kMPAPatternName] = DefineMPWithMaskPattern(false, false);
  patterns[kMPAXPatternName] = DefineMPWithMaskPattern(true, false);
  patterns[kMPAWithMaskPatternNamePA] = DefineMPWithMaskPatternPA();
  patterns[kMPAXWithMaskPatternNamePA] = DefineMPWithMaskPatternPA(true);
  patterns[kMPAWithMaskPatternNameT5] = DefineMPWithMaskPatternT5();
  patterns[kMPAXWithMaskPatternNameT5] = DefineMPWithMaskPatternT5(true);
  return patterns;
}

bool MultiHeadAttentionFusion::CheckPattern(const EquivPtr &equiv, int *head_num, int *head_size) const {
  MS_ASSERT(equiv != nullptr);
  MS_ASSERT(head_num != nullptr);
  MS_ASSERT(head_size != nullptr);
  std::vector<int> reshape_axes;
  if (GetAxis((*equiv)[reshape_axis_], &reshape_axes) != lite::RET_OK) {
    return false;
  }
  if (reshape_axes.size() != C4NUM) {
    return false;
  }
  *head_num = reshape_axes.at(C2NUM);
  *head_size = reshape_axes.at(C3NUM);
  return true;
}

AnfNodePtr MultiHeadAttentionFusion::Process(const std::string &pattern_name, const mindspore::FuncGraphPtr &func_graph,
                                             const mindspore::AnfNodePtr &node,
                                             const mindspore::EquivPtr &equiv) const {
  if (func_graph == nullptr || node == nullptr || equiv == nullptr) {
    return nullptr;
  }
  if ((pattern_name == kMPAWithMaskPatternName) || (pattern_name == kMPAWithMaskPatternNamePA) ||
      (pattern_name == kMPAWithMaskPatternNameT5)) {
    return CreateMaskedMultiHeadAttentionNode(func_graph, equiv, node->fullname_with_scope());
  } else if ((pattern_name == kMPAXWithMaskPatternName) || (pattern_name == kMPAXWithMaskPatternNamePA) ||
             (pattern_name == kMPAXWithMaskPatternNameT5)) {
    return CreateMaskedMultiHeadAttentionNode(func_graph, equiv, node->fullname_with_scope(), true);
  } else if (pattern_name == kMPAPatternName) {
    return CreateMaskedMultiHeadAttentionNode(func_graph, equiv, node->fullname_with_scope(), false, false);
  } else if (pattern_name == kMPAXPatternName) {
    return CreateMaskedMultiHeadAttentionNode(func_graph, equiv, node->fullname_with_scope(), true, false);
  }

  { return nullptr; }
}

STATUS GetIntParameterData(const ParameterPtr &param_ptr, std::vector<int> *result) {
  if (param_ptr == nullptr || !param_ptr->has_default()) {
    MS_LOG(DEBUG) << "param not have default";
    return RET_ERROR;
  }
  auto default_param = param_ptr->default_param();
  if (default_param == nullptr || !utils::isa<tensor::TensorPtr>(default_param)) {
    MS_LOG(DEBUG) << "tensor_info is not tensor::TensorPtr";
    return RET_ERROR;
  }
  auto default_param_ptr = utils::cast<tensor::TensorPtr>(default_param);
  if (default_param_ptr->data_type() != kNumberTypeInt32 && default_param_ptr->data_type() != kNumberTypeInt) {
    MS_LOG(DEBUG) << "default param is not int";
    return RET_ERROR;
  }
  auto ptr = reinterpret_cast<int *>(default_param_ptr->data_c());
  int64_t shape_size =
    std::accumulate(default_param_ptr->shape().begin(), default_param_ptr->shape().end(), 1, std::multiplies<>());
  for (int64_t i = 0; i < shape_size; i++) {
    result->emplace_back(ptr[i]);
  }
  return RET_OK;
}

std::shared_ptr<ops::Attention> MultiHeadAttentionFusion::BuildAttentionPrim(const EquivPtr &equiv) const {
  MS_ASSERT(equiv != nullptr);
  auto attention_prim = std::make_shared<ops::Attention>();
  if (attention_prim == nullptr) {
    MS_LOG(ERROR) << "Build attention primitive failed.";
    return attention_prim;
  }
  if (!utils::isa<ParameterPtr>((*equiv)[reshape_k_])) {
    MS_LOG(ERROR) << "Reshape k is not a parameter";
    return nullptr;
  }

  if (!utils::isa<ParameterPtr>((*equiv)[reshape_v_])) {
    MS_LOG(ERROR) << "Reshape v is not a parameter";
    return nullptr;
  }

  auto reshape_k = utils::cast<ParameterPtr>((*equiv)[reshape_k_]);
  std::vector<int> shape_k;
  if (RET_OK != GetIntParameterData(reshape_k, &shape_k)) {
    MS_LOG(ERROR) << "Get reshape k data failed";
    return nullptr;
  }

  auto reshape_v = utils::cast<ParameterPtr>((*equiv)[reshape_v_]);
  std::vector<int> shape_v;
  if (RET_OK != GetIntParameterData(reshape_v, &shape_v)) {
    MS_LOG(ERROR) << "Get reshape k data failed";
    return nullptr;
  }
  if (shape_k.size() < kWeightShapeSize || shape_v.size() < kWeightShapeSize ||
      shape_k.at(shape_k.size() - kWeightShapeSize) != shape_v.at(shape_v.size() - kWeightShapeSize)) {
    MS_LOG(ERROR) << "Shape k or shape v is invalid.";
    return nullptr;
  }
  return attention_prim;
}

STATUS MultiHeadAttentionFusion::AdjustOtherGetItems(const FuncGraphPtr &func_graph, const CNodePtr &attention,
                                                     int index, const AnfNodePtr &node) const {
  auto manager = func_graph->manager();
  if (manager == nullptr) {
    MS_LOG(ERROR) << "manager is nullptr";
    return RET_ERROR;
  }
  auto transpose_users = manager->node_users()[node];
  auto user_node = transpose_users.front();
  if (!CheckPrimitiveType(user_node.first, prim::kPrimTranspose)) {
    MS_LOG(ERROR) << " missing transpose node for branch " << index << std::endl;
    return RET_ERROR;
  }
  // connect get item to it
  transpose_users = manager->node_users()[user_node.first];
  auto get_item = CreateOutputGetItem(func_graph, attention, index);
  MS_ASSERT(get_item != nullptr);
  if (transpose_users.size() == 1) {
    auto &snode = transpose_users.front();
    manager->SetEdge(snode.first, snode.second, get_item);
  } else {
    for (auto &snode : transpose_users) {
      if (CheckPrimitiveType(snode.first, prim::kPrimMakeTuple)) {
        manager->SetEdge(snode.first, snode.second, get_item);
        break;
      }
    }
  }
  return RET_OK;
}

CNodePtr MultiHeadAttentionFusion::CreateOutputGetItem(const FuncGraphPtr &func_graph, const CNodePtr &node,
                                                       const int item_index) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(node != nullptr);
  auto tuple_get_item_prim = std::make_shared<ops::TupleGetItem>();
  auto get_item_value = NewValueNode(MakeValue<int>(item_index));
  if (tuple_get_item_prim == nullptr || get_item_value == nullptr) {
    MS_LOG(ERROR) << "NewValueNode is nullptr";
    return nullptr;
  }
  auto tuple_get_item_prim_c = tuple_get_item_prim->GetPrim();
  MS_ASSERT(tuple_get_item_prim_c != nullptr);
  CNodePtr get_item_cnode = func_graph->NewCNode(tuple_get_item_prim_c, {node, get_item_value});
  MS_CHECK_TRUE_RET(get_item_cnode != nullptr, nullptr);
  auto abstract = lite::CreateTensorAbstract({}, kNumberTypeFloat32);
  if (abstract == nullptr) {
    MS_LOG(ERROR) << "Create tensor abstract failed";
    return nullptr;
  }
  get_item_cnode->set_abstract(abstract);
  get_item_cnode->set_fullname_with_scope(node->fullname_with_scope() + "_output_getitem_" +
                                          std::to_string(item_index));
  return get_item_cnode;
}

STATUS MultiHeadAttentionFusion::SetAbstractTuple(const CNodePtr &cnode, const int output_num) const {
  MS_ASSERT(cnode != nullptr);
  AbstractBasePtrList abstract_list;
  for (int i = 0; i < output_num; ++i) {
    auto abstract = lite::CreateTensorAbstract({}, kNumberTypeFloat32);
    if (abstract == nullptr) {
      MS_LOG(ERROR) << "Create tensor abstract failed";
      return RET_ERROR;
    }
    abstract_list.emplace_back(abstract);
  }
  auto abstract_tuple = std::make_shared<abstract::AbstractTuple>(abstract_list);
  if (abstract_tuple == nullptr) {
    MS_LOG(ERROR) << "create abstract_tuple failed";
    return RET_ERROR;
  }
  cnode->set_abstract(abstract_tuple);
  return RET_OK;
}

STATUS MultiHeadAttentionFusion::RemoveRedundantInput(const FuncGraphPtr &func_graph,
                                                      const std::vector<AnfNodePtr> &redundant) const {
  for (auto &node : redundant) {
    func_graph->DropNode(node);
  }
  return RET_OK;
}

std::shared_ptr<ops::Attention> MultiHeadAttentionFusion::CreatePrim(const EquivPtr &equiv, bool cross) const {
  auto attention_prim = std::make_shared<ops::Attention>();
  if (attention_prim == nullptr) {
    MS_LOG(ERROR) << "Build attention primitive failed.";
    return nullptr;
  }
  int head_num = 0;
  int head_size = 0;
  if (!CheckPattern(equiv, &head_num, &head_size)) {
    return nullptr;
  }
  attention_prim->Init(head_num, head_size, cross);
  return attention_prim;
}

CNodePtr MultiHeadAttentionFusion::MakeGetTuple(const FuncGraphPtr &func_graph, const CNodePtr &new_node,
                                                const AnfNodePtr &knode, const AnfNodePtr &vnode) const {
  auto get_item_node = CreateOutputGetItem(func_graph, new_node, 0);
  if (get_item_node == nullptr) {
    MS_LOG(ERROR) << "create attention output get_item node failed";
    return nullptr;
  }
  if (knode != nullptr) {
    auto status = AdjustOtherGetItems(func_graph, new_node, 1, knode);
    MS_CHECK_TRUE_RET(status == RET_OK, nullptr);
  }
  if (vnode != nullptr) {
    auto status = AdjustOtherGetItems(func_graph, new_node, 2, vnode);
    MS_CHECK_TRUE_RET(status == RET_OK, nullptr);
  }
  return get_item_node;
}

CNodePtr MultiHeadAttentionFusion::CreateMaskedMultiHeadAttentionNode(const FuncGraphPtr &func_graph,
                                                                      const EquivPtr &equiv, const string &base_name,
                                                                      bool cross, bool mask) const {
  MS_ASSERT(func_graph != nullptr);
  MS_ASSERT(equiv != nullptr);
  std::vector<AnfNodePtr> redundant;
  auto attention_prim = CreatePrim(equiv, cross);
  MS_CHECK_TRUE_RET(attention_prim != nullptr, nullptr);
  auto attention_prim_c = attention_prim->GetPrim();
  MS_CHECK_TRUE_RET(attention_prim_c != nullptr, nullptr);
  auto value_node = NewValueNode(attention_prim_c);
  MS_CHECK_TRUE_RET(value_node != nullptr, nullptr);

  auto input_q = utils::cast<AnfNodePtr>((*equiv)[input_q_]);
  auto input_k = utils::cast<AnfNodePtr>((*equiv)[input_k_]);
  auto input_v = utils::cast<AnfNodePtr>((*equiv)[input_v_]);
  AnfNodePtr input_mask;
  auto weight_q = utils::cast<AnfNodePtr>((*equiv)[weight_q_]);
  redundant.push_back(weight_q);
  auto weight_k = utils::cast<AnfNodePtr>((*equiv)[weight_k_]);
  auto weight_v = utils::cast<AnfNodePtr>((*equiv)[weight_v_]);
  redundant.push_back(weight_k);
  redundant.push_back(weight_v);
  auto weight_o = utils::cast<AnfNodePtr>((*equiv)[weight_o_]);
  auto bias_q = utils::cast<AnfNodePtr>((*equiv)[bias_q_]);
  if (!cross) {
    redundant.push_back(bias_q);
  }
  auto bias_k = utils::cast<AnfNodePtr>((*equiv)[bias_k_]);
  auto bias_v = utils::cast<AnfNodePtr>((*equiv)[bias_v_]);
  redundant.push_back(bias_k);
  redundant.push_back(bias_v);
  auto bias_o = utils::cast<AnfNodePtr>((*equiv)[bias_o_]);
  auto knode = utils::cast<AnfNodePtr>((*equiv)[k_transpose_]);
  auto vnode = utils::cast<AnfNodePtr>((*equiv)[v_transpose_]);
  if (mask) {
    input_mask = utils::cast<AnfNodePtr>((*equiv)[mask_]);
  }
  std::shared_ptr<tensor::Tensor> weight_q_tensor = GetTensorInfo(weight_q);
  std::shared_ptr<tensor::Tensor> weight_k_tensor = GetTensorInfo(weight_k);
  std::shared_ptr<tensor::Tensor> weight_v_tensor = GetTensorInfo(weight_v);
  std::shared_ptr<tensor::Tensor> bias_q_tensor = GetTensorInfo(bias_q);
  std::shared_ptr<tensor::Tensor> bias_k_tensor = GetTensorInfo(bias_k);
  std::shared_ptr<tensor::Tensor> bias_v_tensor = GetTensorInfo(bias_v);
  tensor::TensorPtr c_weights;
  tensor::TensorPtr q_weight_t;
  if (cross) {
    c_weights = ConcatTensors({weight_k_tensor, weight_v_tensor}, true);
    q_weight_t = ConcatTensors({weight_q_tensor}, true);
  } else {
    c_weights = ConcatTensors({weight_q_tensor, weight_k_tensor, weight_v_tensor}, true);
  }
  auto c_bias = ConcatTensors({bias_q_tensor, bias_k_tensor, bias_v_tensor});
  // convert tensors to params
  auto c_weight_param = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(c_weight_param != nullptr, nullptr);
  if (lite::InitParameterFromTensorInfo(c_weight_param, c_weights) != lite::RET_OK) {
    MS_LOG(ERROR) << "Init parameter from tensor info failed.";
    return nullptr;
  }
  c_weight_param->set_name(base_name + "/weight_qkv");
  auto c_bias_param = func_graph->add_parameter();
  MS_CHECK_TRUE_RET(c_bias_param != nullptr, nullptr);
  if (lite::InitParameterFromTensorInfo(c_bias_param, c_bias) != lite::RET_OK) {
    MS_LOG(ERROR) << "Init parameter from tensor info failed.";
    return nullptr;
  }
  c_bias_param->set_name(base_name + "/bias_qkv");
  ParameterPtr q_weight_param;
  if (cross) {
    q_weight_param = func_graph->add_parameter();
    MS_CHECK_TRUE_RET(q_weight_param != nullptr, nullptr);
    if (lite::InitParameterFromTensorInfo(q_weight_param, q_weight_t) != lite::RET_OK) {
      MS_LOG(ERROR) << "Init parameter from tensor info failed.";
      return nullptr;
    }
  }
  std::vector<AnfNodePtr> new_node_inputs;
  if (cross) {
    new_node_inputs = {value_node,     input_q,  input_k,      input_v, q_weight_param,
                       c_weight_param, weight_o, c_bias_param, bias_o};
  } else {
    new_node_inputs = {value_node, input_q, input_k, input_v, c_weight_param, weight_o, c_bias_param, bias_o};
  }
  if (mask) {
    new_node_inputs.push_back(input_mask);
  }
  auto new_node = func_graph->NewCNode(new_node_inputs);
  MS_CHECK_TRUE_RET(new_node != nullptr, nullptr);
  auto status = SetAbstractTuple(new_node, kAttentionOutputs);
  if (status != RET_OK) {
    return nullptr;
  }
  new_node->set_fullname_with_scope(base_name + "/attention");
  auto get_item_node = MakeGetTuple(func_graph, new_node, knode, vnode);
  RemoveRedundantInput(func_graph, redundant);
  return get_item_node;
}
}  // namespace mindspore::opt
