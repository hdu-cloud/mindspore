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

#include "ops/gru_v2.h"

#include <functional>
#include <iostream>
#include <map>
#include <unordered_map>
#include <string>

#include "abstract/ops/primitive_infer_map.h"
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "ops/primitive_c.h"
#include "mindapi/src/helper.h"

namespace mindspore {
namespace ops {
namespace {
constexpr size_t kGRUV2InputMinDim = 1;
constexpr size_t kGRUV2InputSize = 3;
constexpr size_t kGRUV2HSize = 3;
constexpr size_t kGRUV2SeqLenSize = 1;
constexpr int64_t kGRUV2InputNum = 4;
constexpr auto kRealNumLayers = "real_num_layers";
constexpr auto kRealHiddenSize = "real_hidden_size";

std::unordered_map<std::string, int64_t> GetAttrMap(const PrimitivePtr &primitive) {
  std::unordered_map<std::string, int64_t> attr_map;
  auto input_size_ptr = primitive->GetAttr(kInputSize);
  MS_EXCEPTION_IF_NULL(input_size_ptr);
  int64_t input_size = GetValue<int64_t>(input_size_ptr);
  attr_map[kInputSize] = input_size;

  auto hidden_size_ptr = primitive->GetAttr(kHiddenSize);
  MS_EXCEPTION_IF_NULL(hidden_size_ptr);
  int64_t hidden_size = GetValue<int64_t>(hidden_size_ptr);
  attr_map[kHiddenSize] = hidden_size;

  auto num_layers_ptr = primitive->GetAttr(kNumLayers);
  MS_EXCEPTION_IF_NULL(num_layers_ptr);
  int64_t num_layers = GetValue<int64_t>(num_layers_ptr);
  auto bidirectional_ptr = primitive->GetAttr(kBidirectional);
  MS_EXCEPTION_IF_NULL(bidirectional_ptr);
  bool bidirectional = GetValue<bool>(bidirectional_ptr);
  auto real_hidden_size = bidirectional ? hidden_size * 2 : hidden_size;
  auto real_num_layers = bidirectional ? num_layers * 2 : num_layers;
  attr_map[kRealNumLayers] = real_num_layers;
  attr_map[kRealHiddenSize] = real_hidden_size;
  return attr_map;
}

abstract::TupleShapePtr GRUV2InferShape(const PrimitivePtr &primitive, const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto op_name = primitive->name();
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kGRUV2InputNum, op_name);
  auto attr_map = GetAttrMap(primitive);
  auto x_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex0]->BuildShape());
  auto h_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex1]->BuildShape());
  auto w_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex2]->BuildShape());
  auto seq_lengths_shape_map = CheckAndConvertUtils::ConvertShapePtrToShapeMap(input_args[kInputIndex3]->BuildShape());
  auto x_shape = x_shape_map[kShape];
  auto h_shape = h_shape_map[kShape];
  auto seq_lengths_shape = seq_lengths_shape_map[kShape];
  (void)CheckAndConvertUtils::CheckInteger("input dims", SizeToLong(x_shape.size()), kEqual, kGRUV2InputSize, op_name);
  (void)CheckAndConvertUtils::CheckInteger("h dims", SizeToLong(h_shape.size()), kEqual, kGRUV2HSize, op_name);
  (void)CheckAndConvertUtils::CheckInteger("seq_lengths dims", SizeToLong(seq_lengths_shape.size()), kEqual,
                                           kGRUV2SeqLenSize, op_name);
  auto max_seq_lengths = x_shape[0];
  auto batch_size = x_shape[1];
  auto input_size = attr_map[kInputSize];
  auto hidden_size = attr_map[kHiddenSize];
  auto real_num_layers = attr_map[kRealNumLayers];
  auto real_hidden_size = attr_map[kRealHiddenSize];
  if (h_shape[kInputIndex1] != batch_size || seq_lengths_shape[kInputIndex0] != batch_size) {
    MS_LOG(EXCEPTION) << "For dynamic rnn, the batch_size should be the same between input, h, and seq_lengths.";
  }

  if (x_shape[kInputIndex2] != input_size) {
    MS_LOG(EXCEPTION) << "For dynamic rnn, the input_shape[2] should equal to input_size.";
  }

  if (h_shape[kInputIndex0] != real_num_layers) {
    MS_LOG(EXCEPTION) << "For dynamic rnn, the h_shape[0] should equal to num_directions * num_layers.";
  }

  if (h_shape[kInputIndex2] != hidden_size) {
    MS_LOG(EXCEPTION) << "For dynamic rnn, the h_shape[2] should equal to hidden_size.";
  }
  ShapeVector reserve_shape = {1, 1};
  auto reserve_shape_ptr = std::make_shared<abstract::Shape>(reserve_shape);
  ShapeVector state_shape = {1, 1};
  auto state_shape_ptr = std::make_shared<abstract::Shape>(state_shape);
  ShapeVector output_shape = {max_seq_lengths, batch_size, real_hidden_size};
  ShapeVector hn_shape = {real_num_layers, batch_size, hidden_size};

  bool is_dynamic_shp = !x_shape_map[kMaxShape].empty();
  if (!w_shape_map[kMaxShape].empty()) {
    MS_LOG(EXCEPTION) << "For GRUV2, the weight cannot be dynaimic shape.";
  }
  if (is_dynamic_shp) {
    auto x_max_shape = x_shape_map[kMaxShape];
    auto x_min_shape = x_shape_map[kMinShape];
    auto h_max_shape = h_shape_map[kMaxShape];
    auto seq_max_shape = seq_lengths_shape_map[kMaxShape];
    if (h_max_shape.empty() || seq_max_shape.empty()) {
      MS_LOG(EXCEPTION) << "For GRUV2, only the batch size can be dynamic shape, so that the input_shape[1], "
                           "h_shape[1], and seq_lengths_shape[1] should be dynamic in dynamic shape mode.";
    }
    // x_shape: (seq_len, batch_size, input_size)
    if (x_shape[kInputIndex0] == abstract::Shape::kShapeDimAny ||
        x_shape[kInputIndex2] == abstract::Shape::kShapeDimAny) {
      MS_LOG(EXCEPTION) << "For GRUV2, only the batch size can be dynamic shape.";
    }

    ShapeVector min_shape = {max_seq_lengths, x_min_shape[1], real_num_layers};
    ShapeVector max_shape = {max_seq_lengths, x_max_shape[1], real_num_layers};
    auto output_shape_ptr = std::make_shared<abstract::Shape>(output_shape, min_shape, max_shape);
    ShapeVector hn_min_shape = {real_num_layers, x_min_shape[1], hidden_size};
    ShapeVector hn_max_shape = {real_num_layers, x_max_shape[1], hidden_size};
    auto hn_shape_ptr = std::make_shared<abstract::Shape>(hn_shape, hn_min_shape, hn_max_shape);
    return std::make_shared<abstract::TupleShape>(
      std::vector<abstract::BaseShapePtr>{output_shape_ptr, hn_shape_ptr, reserve_shape_ptr, state_shape_ptr});
  }

  auto output_shape_ptr = std::make_shared<abstract::Shape>(output_shape);
  auto hn_shape_ptr = std::make_shared<abstract::Shape>(hn_shape);
  return std::make_shared<abstract::TupleShape>(
    std::vector<abstract::BaseShapePtr>{output_shape_ptr, hn_shape_ptr, reserve_shape_ptr, state_shape_ptr});
}

TuplePtr GRUV2InferType(const PrimitivePtr &prim, const std::vector<AbstractBasePtr> &input_args) {
  const std::set valid_types = {kFloat16, kFloat32};
  auto op_name = prim->name();
  (void)CheckAndConvertUtils::CheckTensorTypeValid("seq_lengths", input_args[3]->BuildType(), {kInt32}, op_name);
  std::map<std::string, TypePtr> types;
  (void)types.emplace("input", input_args[kInputIndex0]->BuildType());
  (void)types.emplace("h", input_args[kInputIndex1]->BuildType());
  (void)types.emplace("w", input_args[kInputIndex2]->BuildType());
  auto type = CheckAndConvertUtils::CheckTensorTypeSame(types, valid_types, op_name);
  return std::make_shared<Tuple>(std::vector<TypePtr>{type, type, type, type});
}
}  // namespace

AbstractBasePtr GRUV2Infer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kGRUV2InputNum, primitive->name());
  auto infer_shape = GRUV2InferShape(primitive, input_args);
  auto infer_type = GRUV2InferType(primitive, input_args);
  return abstract::MakeAbstract(infer_shape, infer_type);
}

MIND_API_OPERATOR_IMPL(GRUV2, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(GRUV2, prim::kPrimGRUV2, GRUV2Infer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
