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

#include <string>
#include <algorithm>
#include <memory>
#include <map>
#include <set>
#include <vector>
#include "ops/op_utils.h"
#include "utils/check_convert_utils.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/sparse_concat.h"

namespace mindspore {
namespace ops {
using mindspore::abstract::AbstractTensor;
using mindspore::abstract::AbstractTuple;
namespace {
constexpr size_t kFirstInput = 0;
constexpr size_t kSpInputIndicesStart = 0;
constexpr size_t kSpInputValuesStart = 1;
constexpr size_t kSpInputShapesStart = 2;
constexpr auto kConcatDim = "concat_dim";

inline void CheckSparseConcatShape(const ShapeVector &input_shape, const size_t &expected_dim,
                                   const std::string &arg_name, const std::string &prim_name) {
  if (!IsDynamicRank(input_shape) && input_shape.size() != expected_dim) {
    MS_EXCEPTION(mindspore::ValueError) << "For " << prim_name << ", " << arg_name << " must be a " << expected_dim
                                        << "-dimension, but got a " << input_shape.size()
                                        << "-dimension in SparseConcat.";
  }
}

inline bool CheckSparseConcatShapeValue(const ShapeVector &indices_shape, const ShapeVector &values_shape,
                                        const ShapeVector &shapes_shape, const std::string &prim_name) {
  auto is_dynamic = IsDynamic(indices_shape) || IsDynamic(values_shape) || IsDynamic(shapes_shape);
  if (!is_dynamic) {
    if (indices_shape[1] != shapes_shape[0]) {
      MS_EXCEPTION(mindspore::ValueError) << "For " << prim_name << ", the indices shape rank is " << indices_shape[1]
                                          << ", but the shape rank is " << shapes_shape[0] << ".";
    }
    if (indices_shape[0] != values_shape[0]) {
      MS_EXCEPTION(mindspore::ValueError)
        << "For " << prim_name << ", the indices element number is " << indices_shape[0]
        << ", but the value element number is " << values_shape[0] << ".";
    }
  }
  return is_dynamic;
}
}  // namespace

std::vector<TypePtr> SparseConcatInferType(const PrimitivePtr &primitive,
                                           const std::vector<AbstractBasePtr> &input_args) {
  auto prim_name = primitive->name();
  if (!input_args[kSpInputIndicesStart]->isa<abstract::AbstractTuple>() &&
      !input_args[kSpInputIndicesStart]->isa<abstract::AbstractList>()) {
    MS_EXCEPTION(mindspore::ValueError) << "For " << prim_name
                                        << ", the sp_input must be a list or tuple of sparse tensor. but got: "
                                        << input_args[kSpInputIndicesStart]->ToString() << ".";
  }
  auto inputs_indices = input_args[kSpInputIndicesStart]->isa<abstract::AbstractTuple>()
                          ? input_args[kSpInputIndicesStart]->cast<abstract::AbstractTuplePtr>()->elements()
                          : input_args[kSpInputIndicesStart]->cast<abstract::AbstractListPtr>()->elements();

  if (!input_args[kSpInputValuesStart]->isa<abstract::AbstractTuple>() &&
      !input_args[kSpInputValuesStart]->isa<abstract::AbstractList>()) {
    MS_EXCEPTION(mindspore::ValueError) << "For " << prim_name
                                        << ", the sp_input must be a list or tuple of sparse tensor. but got: "
                                        << input_args[kSpInputValuesStart]->ToString() << ".";
  }
  auto inputs_values = input_args[kSpInputValuesStart]->isa<abstract::AbstractTuple>()
                         ? input_args[kSpInputValuesStart]->cast<abstract::AbstractTuplePtr>()->elements()
                         : input_args[kSpInputValuesStart]->cast<abstract::AbstractListPtr>()->elements();

  if (!input_args[kSpInputShapesStart]->isa<abstract::AbstractTuple>() &&
      !input_args[kSpInputShapesStart]->isa<abstract::AbstractList>()) {
    MS_EXCEPTION(mindspore::ValueError) << "For " << prim_name
                                        << ", the sp_input must be a list or tuple of sparse tensor. but got: "
                                        << input_args[kSpInputShapesStart]->ToString() << ".";
  }
  auto inputs_shapes = input_args[kSpInputShapesStart]->isa<abstract::AbstractTuple>()
                         ? input_args[kSpInputShapesStart]->cast<abstract::AbstractTuplePtr>()->elements()
                         : input_args[kSpInputShapesStart]->cast<abstract::AbstractListPtr>()->elements();
  std::map<std::string, TypePtr> values_types;
  if ((inputs_indices.size() != inputs_values.size()) || (inputs_indices.size() != inputs_shapes.size())) {
    MS_EXCEPTION(mindspore::ValueError) << "For " << prim_name
                                        << ", the sp_input is not a COO tensor, the COO tensor indices number is "
                                        << inputs_indices.size() << " but values number is " << inputs_values.size()
                                        << " and shape number is " << inputs_shapes.size() << ".";
  }
  for (unsigned int i = 0; i < inputs_indices.size(); i++) {
    std::string elementi = "values" + std::to_string(i);
    auto ind_type = inputs_indices[i]->BuildType();
    auto sha_type = inputs_shapes[i]->BuildType();
    (void)values_types.emplace(elementi, inputs_values[i]->BuildType());
    (void)CheckAndConvertUtils::CheckTensorTypeValid("indices" + std::to_string(i), ind_type, {kInt64}, prim_name);
    (void)CheckAndConvertUtils::CheckTensorTypeValid("shapes" + std::to_string(i), sha_type, {kInt64, kInt32},
                                                     prim_name);
  }
  (void)CheckAndConvertUtils::CheckTensorTypeSame(values_types, common_valid_types_with_complex_and_bool, prim_name);
  std::vector<TypePtr> out_type = {};
  out_type.push_back(inputs_indices[kFirstInput]->BuildType());
  out_type.push_back(inputs_values[kFirstInput]->BuildType());
  out_type.push_back(inputs_shapes[kFirstInput]->BuildType());
  return out_type;
}

std::vector<abstract::ShapePtr> SparseConcatInferShape(const PrimitivePtr &primitive,
                                                       const std::vector<AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  auto prim_name = primitive->name();
  auto inputs_indices = input_args[kSpInputIndicesStart]->isa<abstract::AbstractTuple>()
                          ? input_args[kSpInputIndicesStart]->cast<abstract::AbstractTuplePtr>()->elements()
                          : input_args[kSpInputIndicesStart]->cast<abstract::AbstractListPtr>()->elements();
  auto inputs_values = input_args[kSpInputValuesStart]->isa<abstract::AbstractTuple>()
                         ? input_args[kSpInputValuesStart]->cast<abstract::AbstractTuplePtr>()->elements()
                         : input_args[kSpInputValuesStart]->cast<abstract::AbstractListPtr>()->elements();
  auto inputs_shapes = input_args[kSpInputShapesStart]->isa<abstract::AbstractTuple>()
                         ? input_args[kSpInputShapesStart]->cast<abstract::AbstractTuplePtr>()->elements()
                         : input_args[kSpInputShapesStart]->cast<abstract::AbstractListPtr>()->elements();
  int64_t kNumOne = 1;
  size_t indices_expect_rank = 2;
  size_t values_expect_rank = 1;
  size_t shapes_expect_rank = 1;
  int64_t ConcatNum = SizeToLong(inputs_indices.size());
  (void)primitive->AddAttr("N", MakeValue(ConcatNum));

  (void)CheckAndConvertUtils::CheckInteger("indices' num", ConcatNum, kGreaterThan, kNumOne, prim_name);
  (void)CheckAndConvertUtils::CheckInteger("indices' num and values' num", ConcatNum, kEqual,
                                           SizeToLong(inputs_values.size()), prim_name);
  (void)CheckAndConvertUtils::CheckInteger("indices' num and shapes' num", ConcatNum, kEqual,
                                           SizeToLong(inputs_shapes.size()), prim_name);

  auto indices_element0_shape =
    CheckAndConvertUtils::ConvertShapePtrToShapeMap(inputs_indices[0]->BuildShape())[kShape];
  auto values_element0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(inputs_values[0]->BuildShape())[kShape];
  auto shapes_element0_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(inputs_shapes[0]->BuildShape())[kShape];

  CheckSparseConcatShape(indices_element0_shape, indices_expect_rank, "indices shape", prim_name);
  CheckSparseConcatShape(values_element0_shape, values_expect_rank, "values shape", prim_name);
  CheckSparseConcatShape(shapes_element0_shape, shapes_expect_rank, "shape shape", prim_name);
  CheckSparseConcatShapeValue(indices_element0_shape, values_element0_shape, shapes_element0_shape, prim_name);

  if (IsDynamicRank(indices_element0_shape)) {
    abstract::ShapePtr y_indices_shape_ptr = std::make_shared<mindspore::abstract::Shape>(ShapeVector{-1, -1});
    abstract::ShapePtr y_values_shape_ptr = std::make_shared<mindspore::abstract::Shape>(ShapeVector{-1});
    abstract::ShapePtr y_shape_shape_ptr = std::make_shared<mindspore::abstract::Shape>(ShapeVector{-1});
    return std::vector<abstract::ShapePtr>{y_indices_shape_ptr, y_values_shape_ptr, y_shape_shape_ptr};
  }

  std::vector<int64_t> out_indices_shape = {};
  out_indices_shape.push_back(0);
  out_indices_shape.push_back(indices_element0_shape[1]);
  std::vector<int64_t> out_values_shape = {0};
  auto out_shape_shape = shapes_element0_shape;
  bool is_dynamic = false;
  for (int64_t i = 0; i < ConcatNum; i++) {
    auto indices_element_shape =
      CheckAndConvertUtils::ConvertShapePtrToShapeMap(inputs_indices[i]->BuildShape())[kShape];
    auto values_element_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(inputs_values[i]->BuildShape())[kShape];
    auto shapes_element_shape = CheckAndConvertUtils::ConvertShapePtrToShapeMap(inputs_shapes[i]->BuildShape())[kShape];
    is_dynamic = is_dynamic || CheckSparseConcatShapeValue(indices_element_shape, values_element_shape,
                                                           shapes_element_shape, prim_name);
    if (is_dynamic) {
      break;
    }
    out_indices_shape[0] += indices_element_shape[0];
    out_values_shape[0] += values_element_shape[0];
    if ((out_indices_shape[1] != indices_element_shape[1]) || (out_shape_shape != shapes_element_shape)) {
      MS_EXCEPTION(mindspore::ValueError)
        << "For " << prim_name << ", indices or shape rank is not fit. The No.0 indice shape rank is "
        << out_indices_shape[1] << ", dense shape rank is " << out_shape_shape << ". The No." << i
        << " indices number is " << indices_element_shape[1] << " dense shape rank is " << shapes_element_shape << ".";
    }
  }

  if (is_dynamic) {
    out_indices_shape[0] = -1;
    out_values_shape[0] = -1;
  }

  std::vector<abstract::ShapePtr> out_shape = {};
  out_shape.push_back(std::make_shared<mindspore::abstract::Shape>(out_indices_shape));
  out_shape.push_back(std::make_shared<mindspore::abstract::Shape>(out_values_shape));
  out_shape.push_back(std::make_shared<mindspore::abstract::Shape>(out_shape_shape));
  return out_shape;
}

AbstractBasePtr SparseConcatInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                                  const std::vector<abstract::AbstractBasePtr> &input_args) {
  MS_EXCEPTION_IF_NULL(primitive);
  const auto prim_name = primitive->name();
  for (auto item : input_args) {
    MS_EXCEPTION_IF_NULL(item);
  }
  const int64_t kInputNum = 3;
  CheckAndConvertUtils::CheckInputArgs(input_args, kEqual, kInputNum, prim_name);
  auto infer_types = SparseConcatInferType(primitive, input_args);
  auto infer_shapes = SparseConcatInferShape(primitive, input_args);
  auto out_indices_abstract = abstract::MakeAbstract(infer_shapes[0], infer_types[0]);
  auto out_values_abstract = abstract::MakeAbstract(infer_shapes[1], infer_types[1]);
  auto out_shape_abstract = abstract::MakeAbstract(infer_shapes[2], infer_types[2]);

  AbstractBasePtrList ret = {out_indices_abstract, out_values_abstract, out_shape_abstract};
  return std::make_shared<AbstractTuple>(ret);
}

void SparseConcat::Init(int64_t concat_dim) { this->set_concat_dim(concat_dim); }

void SparseConcat::set_concat_dim(const int64_t &concat_dim) {
  (void)this->AddAttr(kConcatDim, api::MakeValue(concat_dim));
}

int64_t SparseConcat::get_concat_dim() const {
  auto value_ptr = GetAttr(kConcatDim);
  return GetValue<int64_t>(value_ptr);
}

MIND_API_OPERATOR_IMPL(SparseConcat, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(SparseConcat, prim::kPrimSparseConcat, SparseConcatInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
