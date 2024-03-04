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

#include "graph/utils/op_desc_utils_ex.h"

#include "common/ge_common/util.h"
#include "common/util/trace_manager/trace_manager.h"
#include "graph/operator_impl.h"
#include "graph/operator_factory_impl.h"
#include "graph/common_error_codes.h"
#include "graph/ge_context.h"
#include "graph/ir_definitions_recover.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/transformer_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "common/util/mem_utils.h"
#include "common/checker.h"

namespace ge {
graphStatus OpDescUtilsEx::CallInferFuncV2Inner(const OpDescPtr &op_desc, Operator &op) {
  const auto call_infer_data_type = OperatorFactoryImpl::GetInferDataTypeFunc();
  const auto call_infer_shape_v2 = OperatorFactoryImpl::GetInferShapeV2Func();
  const auto call_infer_shape_range = OperatorFactoryImpl::GetInferShapeRangeFunc();
  if ((call_infer_data_type == nullptr) || (call_infer_shape_v2 == nullptr) || (call_infer_shape_range == nullptr)) {
    GELOGW("infer func v2 has not been initialized");
    return GRAPH_PARAM_INVALID;
  }
  if (op_desc->GetIrInputs().empty() && op_desc->GetIrOutputs().empty() && op_desc->GetAllOutputsDescSize() != 0U) {
    GE_CHK_STATUS_RET(RecoverOpDescIrDefinition(op_desc), "Failed recover ir def for %s %s", op_desc->GetNamePtr(),
                      op_desc->GetTypePtr());
  }
  NodeShapeTransUtils transformer(op_desc);
  GE_CHK_BOOL_RET_STATUS(transformer.Init(), GRAPH_FAILED, "Failed to init transformer for %s", op_desc->GetNamePtr());
  GE_CHK_BOOL_RET_STATUS(transformer.CatchFormatAndShape(), GRAPH_FAILED,
                         "Failed to catch format and shape for %s", op_desc->GetNamePtr());
  GE_CHK_STATUS_RET_NOLOG(call_infer_data_type(op_desc));
  GE_CHK_STATUS_RET_NOLOG(call_infer_shape_v2(op, op_desc));
  GE_CHK_STATUS_RET_NOLOG(call_infer_shape_range(op, op_desc));
  GE_CHK_BOOL_RET_STATUS(transformer.UpdateFormatAndShape(), GRAPH_FAILED,
                         "Failed to update format and shape for %s", op_desc->GetNamePtr());
  return GRAPH_SUCCESS;
}

graphStatus OpDescUtilsEx::CallInferFuncV2(const OpDescPtr &op_desc, Operator &op) {
  const auto ret_v2 = CallInferFuncV2Inner(op_desc, op);
  if (ret_v2 != GRAPH_SUCCESS) {
    GELOGW("[Call][InferFuncV2] failed, op %s ret_v2[%u]", op_desc->GetName().c_str(), ret_v2);
    // compatible with V1 processing by upper layer
    return GRAPH_PARAM_INVALID;
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDescUtilsEx::CallInferFuncV1(const OpDescPtr &op_desc, Operator &op) {
  NodeShapeTransUtils transformer(op_desc);
  const auto is_init_success = transformer.Init();
  if (!is_init_success) {
    GELOGE(GRAPH_FAILED, "[Call][Init] for transformer failed");
    return GRAPH_FAILED;
  }
  if (!transformer.CatchFormatAndShape()) {
    GELOGE(GRAPH_FAILED, "[Call][CatchFormatAndShape] for transformer failed!");
    return GRAPH_FAILED;
  }
  graphStatus graph_status = GRAPH_SUCCESS;
  {
    const auto &node_ptr = NodeUtilsEx::GetNodeFromOperator(op);
    const bool empty_name = (node_ptr == nullptr) || (node_ptr->GetOwnerComputeGraph() == nullptr);
    const auto &graph_name = empty_name ? std::string("")
                                        : node_ptr->GetOwnerComputeGraph()->GetName();
    TraceOwnerGuard guard("OP", op_desc->GetName() + ":infershape", graph_name);
    auto infer_func = op_desc->GetInferFunc();
    graph_status = infer_func(op);
  }
  if ((graph_status != GRAPH_SUCCESS) &&
      (graph_status != GRAPH_NODE_NEED_REPASS)) {
    GELOGE(GRAPH_FAILED, "[Call][InferFunc] for %s failed. ret:%u", op_desc->GetName().c_str(), graph_status);
    return GRAPH_FAILED;
  }
  if (!transformer.UpdateFormatAndShape()) {
    GELOGE(GRAPH_FAILED, "[Call][UpdateFormatAndShape] for transformer failed!");
    return GRAPH_FAILED;
  }
  return graph_status;
}

graphStatus OpDescUtilsEx::CallInferFunc(const OpDescPtr &op_desc, Operator &op) {
  GE_CHECK_NOTNULL(op_desc, ", Op is null for Infer Shape.");
  auto infer_func = op_desc->GetInferFunc();
  if (infer_func == nullptr) {
    infer_func = OperatorFactoryImpl::GetInferShapeFunc(op_desc->GetType());
  }
  // priority of use infer func v1
  // when v2 func is ready, remove v1 func, it will automatically follow the V2 process
  if (infer_func != nullptr) {
    op_desc->AddInferFunc(infer_func);
    return CallInferFuncV1(op_desc, op);
  } else {
    return CallInferFuncV2(op_desc, op);
  }
}

graphStatus OpDescUtilsEx::CallInferFormatFunc(const OpDescPtr &op_desc, Operator &op) {
  GE_CHECK_NOTNULL(op_desc, ", Op is null for Infer Format.");
  auto infer_format_func = op_desc->GetInferFormatFunc();
  if (infer_format_func != nullptr) {
    return static_cast<graphStatus>(infer_format_func(op));
  }
  infer_format_func = OperatorFactoryImpl::GetInferFormatFunc(op_desc->GetType());
  if (infer_format_func == nullptr) {
    return op_desc->DefaultInferFormat();
  }
  op_desc->AddInferFormatFunc(infer_format_func);
  return infer_format_func(op);
}

graphStatus OpDescUtilsEx::CallInferValueRangeFunc(const OpDescPtr &op_desc, Operator &op) {
  GE_CHECK_NOTNULL(op_desc, ", Op is null for Infer ValueRange.");
  auto infer_value_range_func = op_desc->GetInferValueRangeFunc();
  if (infer_value_range_func != nullptr) {
    return static_cast<graphStatus>(infer_value_range_func(op));
  }

  const InferValueRangePara infer_value_range_param = OperatorFactoryImpl::GetInferValueRangePara(op_desc->GetType());
  if (!infer_value_range_param.is_initialized) {
    REPORT_CALL_ERROR("E18888", "Node %s does not register func to infer value range.", op_desc->GetName().c_str());
    GELOGE(GRAPH_PARAM_INVALID, "Node %s does not register func to infer value range.", op_desc->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }

  infer_value_range_func = infer_value_range_param.infer_value_func;
  if (infer_value_range_func == nullptr) {
    REPORT_CALL_ERROR("E18888", "Value range infer func of node %s has been registered, but infer func is nullptr.",
                      op_desc->GetName().c_str());
    GELOGE(GRAPH_PARAM_INVALID, "Value range infer func of node %s has been registered, but infer func is nullptr.",
           op_desc->GetName().c_str());
    return GRAPH_PARAM_INVALID;
  }
  op_desc->AddInferValueRangeFunc(infer_value_range_func);
  return infer_value_range_func(op);
}

graphStatus OpDescUtilsEx::OpVerify(const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc, ", Op is null for Infer Verify.");
  auto verify_func = op_desc->GetVerifyFunc();
  if (verify_func == nullptr) {
    verify_func = OperatorFactoryImpl::GetVerifyFunc(op_desc->GetType());
  }
  if (verify_func != nullptr) {
    Operator op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
    const graphStatus ret = static_cast<graphStatus>(verify_func(op));
    op_desc->AddVerifierFunc(verify_func);
    op.BreakConnect();
    return ret;
  }
  return GRAPH_SUCCESS;
}

graphStatus OpDescUtilsEx::InferShapeAndType(const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc, ", Op is null for Infer Shape.");
  auto infer_func = op_desc->GetInferFunc();
  if (infer_func == nullptr) {
    infer_func = OperatorFactoryImpl::GetInferShapeFunc(op_desc->GetType());
    if (infer_func == nullptr) {
      GELOGW("[InferShape][Check] %s does not have infer_func.", op_desc->GetName().c_str());
      /// The infer_func has not been added for each operator in the current operator information library.
      /// No infer_func added operator skips the call
      /// and directly uses the shape information passed down by the upper framework
      return GRAPH_SUCCESS;
    }
  }
  Operator op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  const graphStatus ret = static_cast<graphStatus>(infer_func(op));
  op_desc->AddInferFunc(infer_func);
  op.BreakConnect();
  return ret;
}

graphStatus OpDescUtilsEx::InferDataSlice(const OpDescPtr &op_desc) {
  GE_CHECK_NOTNULL(op_desc, ", Op is null for Infer Slice.");
  auto infer_data_slice_func = op_desc->GetInferDataSliceFunc();
  if (infer_data_slice_func == nullptr) {
    infer_data_slice_func = OperatorFactoryImpl::GetInferDataSliceFunc(op_desc->GetType());
    if (infer_data_slice_func == nullptr) {
      GELOGW("[InferDataSlice][Check] %s does not have infer data slice func.", op_desc->GetName().c_str());
      return NO_DEPENDENCE_FUNC;
    }
  }
  Operator op = OpDescUtils::CreateOperatorFromOpDesc(op_desc);
  const graphStatus ret = static_cast<graphStatus>(infer_data_slice_func(op));
  op_desc->AddInferDataSliceFunc(infer_data_slice_func);
  op.BreakConnect();
  return ret;
}

void OpDescUtilsEx::SetType(OpDescPtr &op_desc, const std::string &type) {
  // If the type changes, IR related variables should be modified accordingly
  auto op = OperatorFactory::CreateOperator("tmp", type.c_str());
  op.BreakConnect();

  op_desc->SetType(type);
  op_desc->SetIrRelated(OpDescUtils::GetOpDescFromOperator(op));
  TRACE_GEN_RECORD(TraceManager::GetTraceHeader(), "modify", TraceManager::GetOutGraphName(),
                   op_desc->GetName(), "type", "", "", type);
}


void OpDescUtilsEx::UpdateShapeAndDType(const GeTensorDescPtr &src, const GeTensorDescPtr &dst) {
  dst->SetOriginShape(src->GetOriginShape());
  dst->SetShape(src->GetShape());
  dst->SetDataType(src->GetDataType());
  dst->SetOriginDataType(src->GetOriginDataType());
  std::vector<std::pair<int64_t, int64_t>> src_shape_range;
  src->GetShapeRange(src_shape_range);
  dst->SetShapeRange(src_shape_range);
  dst->SetOriginShapeRange(src_shape_range);
  ge::TensorUtils::SetRealDimCnt(*dst, static_cast<uint32_t>(src->GetShape().GetDims().size()));
}
} // namespace ge