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

#include "ops/csr_mv.h"

#include "abstract/dshape.h"
#include "abstract/param_validator.h"
#include "abstract/ops/primitive_infer_map.h"
#include "mindapi/src/helper.h"
#include "ops/op_utils.h"
#include "utils/anf_utils.h"
#include "utils/check_convert_utils.h"
#include "utils/tensor_construct_utils.h"

namespace mindspore {
namespace ops {
using abstract::AbstractTensor;
using abstract::AbstractTuple;
AbstractBasePtr CSRMVInfer(const abstract::AnalysisEnginePtr &, const PrimitivePtr &primitive,
                           const std::vector<AbstractBasePtr> &input_args) {
  constexpr auto kCSRMVInputsNum = 5;
  constexpr auto kCSRMVShapeSize = 2;
  const std::string op_name = primitive->name();
  CheckArgsSize(op_name, input_args, kCSRMVInputsNum);
  auto indptr = abstract::CheckArg<AbstractTensor>(op_name, input_args, 0);
  auto indices = abstract::CheckArg<AbstractTensor>(op_name, input_args, 1);
  auto values = abstract::CheckArg<AbstractTensor>(op_name, input_args, 2);
  auto shape = abstract::CheckArg<AbstractTuple>(op_name, input_args, 3);
  auto dense = abstract::CheckArg<AbstractTensor>(op_name, input_args, 4);
  MS_EXCEPTION_IF_NULL(indptr);
  MS_EXCEPTION_IF_NULL(indices);
  MS_EXCEPTION_IF_NULL(values);
  MS_EXCEPTION_IF_NULL(shape);
  MS_EXCEPTION_IF_NULL(dense);

  CheckSparseIndicesDtypeInt32(indptr->element()->BuildType(), "Indptr");
  CheckSparseIndicesDtypeInt32(indices->element()->BuildType(), "Indices");

  ShapeVector sparse_shape = ConvertToShapeVector(shape);
  ShapeVector dense_shape = dense->shape()->shape();
  if (sparse_shape.size() != kCSRMVShapeSize || dense_shape.size() != kCSRMVShapeSize) {
    MS_EXCEPTION(ValueError) << "Currently, only support " << kCSRMVShapeSize << "-D inputs! "
                             << "But csr tensor has " << sparse_shape.size() << " dimensions, "
                             << "and dense tensor has " << dense_shape.size() << " dimension(s). ";
  }
  if (dense_shape[kIndexZero] != sparse_shape[kIndexOne] || dense_shape[kIndexOne] != 1) {
    MS_EXCEPTION(ValueError) << "The dense_vector's shape should be (" << sparse_shape[kIndexOne] << ", 1)"
                             << ", but its current shape is: "
                             << "(" << dense_shape[kIndexZero] << ", " << dense_shape[kIndexOne] << ").";
  }

  ShapeVector out_shape = {sparse_shape[kIndexZero], dense_shape[kIndexOne]};
  auto ret = std::make_shared<AbstractTensor>(values->element()->BuildType(), out_shape);
  // SetAttr
  auto nnz_vec = indices->shape()->shape();
  auto csr_avg_rows = nnz_vec[kIndexZero] / dense_shape[kIndexZero];
  primitive->set_attr(kCSRAvgRows, MakeValue(csr_avg_rows));
  primitive->set_attr(kIsCSR, MakeValue(true));
  return ret;
}
MIND_API_OPERATOR_IMPL(CSRMV, BaseOperator);
REGISTER_PRIMITIVE_EVAL_IMPL(CSRMV, prim::kPrimCSRMV, CSRMVInfer, nullptr, true);
}  // namespace ops
}  // namespace mindspore
