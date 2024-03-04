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

#include "graph/utils/graph_utils_ex.h"

#include "common/ge_common/util.h"
#include "common/util/trace_manager/trace_manager.h"
#include "graph/format_refiner.h"
#include "graph/operator_impl.h"
#include "graph/common_error_codes.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/utils/node_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/op_desc_utils_ex.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/transformer_utils.h"
#include "graph/utils/node_utils_ex.h"
#include "common/util/mem_utils.h"
#include "graph/utils/op_type_utils.h"

namespace ge {
graphStatus GraphUtilsEx::Verify(const ComputeGraphPtr &graph) {
  GE_CHECK_NOTNULL(graph, ", Graph is null for Infer Shape.");
  const bool is_unknown_graph = graph->GetGraphUnknownFlag();
  for (const auto &node_ptr : graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node_ptr);
    GE_CHECK_NOTNULL(node_ptr->GetOpDesc());
    if (is_unknown_graph) {
      continue;
    }
    if (node_ptr->GetOpDesc()->CommonVerify() != GRAPH_SUCCESS) {
      REPORT_CALL_ERROR("E18888", "Verifying %s failed.", node_ptr->GetName().c_str());
      GELOGE(FAILED, "[Call][CommonVerify] Verifying %s failed.", node_ptr->GetName().c_str());
      return GRAPH_FAILED;
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus GraphUtilsEx::InferOriginFormat(const ComputeGraphPtr &graph) {
  return FormatRefiner::InferOrigineFormat(graph);
}

graphStatus GraphUtilsEx::InferShapeInNeed(const ComputeGraphPtr &graph) {
  GE_LOGW_IF(graph->TopologicalSorting() != GRAPH_SUCCESS, "Verify failed.");
  for (const auto &node_ptr : graph->GetAllNodes()) {
    GE_CHECK_NOTNULL(node_ptr);
    const auto op_desc = node_ptr->GetOpDesc();
    bool is_need_infer = false;
    (void)AttrUtils::GetBool(op_desc, NEED_INFER, is_need_infer);
    if (is_need_infer) {
      if (NodeUtilsEx::Verify(node_ptr) != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E18888", "Verifying %s failed.", node_ptr->GetName().c_str());
        GELOGE(FAILED, "[Call][Verify] Verifying %s failed.", node_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }

      const graphStatus status = NodeUtilsEx::InferShapeAndType(node_ptr);
      if ((!OpTypeUtils::IsDataNode(node_ptr->GetType())) && (status == GRAPH_PARAM_INVALID)) {
        GELOGI("Op %s does not have the IMPLEMT_INFERFUNC definition, "
               "and subsequent operators no longer perform shape inference.",
               node_ptr->GetName().c_str());
        break;
      }
      if (status != GRAPH_SUCCESS) {
        REPORT_CALL_ERROR("E18888", "Inferring %s failed.", node_ptr->GetName().c_str());
        GELOGE(FAILED, "[Call][InferShapeAndType] Inferring %s failed.", node_ptr->GetName().c_str());
        return GRAPH_FAILED;
      }

      for (const auto &out_anchor : node_ptr->GetAllOutDataAnchors()) {
        GE_CHECK_NOTNULL(out_anchor->GetOwnerNode()->GetOpDesc());
        auto output_tensor =
            out_anchor->GetOwnerNode()->GetOpDesc()->MutableOutputDesc(static_cast<uint32_t>(out_anchor->GetIdx()));
        GE_CHECK_NOTNULL(output_tensor);
        TensorUtils::SetRealDimCnt(*(output_tensor.get()),
                                   static_cast<uint32_t>(output_tensor->GetShape().GetDims().size()));

        for (const auto &peer_anchor : out_anchor->GetPeerInDataAnchors()) {
          const auto peer_in_tensor_desc =
              peer_anchor->GetOwnerNode()->GetOpDesc()->MutableInputDesc(static_cast<uint32_t>(peer_anchor->GetIdx()));
          GE_CHECK_NOTNULL(peer_in_tensor_desc);
          OpDescUtilsEx::UpdateShapeAndDType(output_tensor, peer_in_tensor_desc);
        }
      }
    }
  }
  return GRAPH_SUCCESS;
}

graphStatus GraphUtilsEx::CopyGraph(const Graph &src_graph, Graph &dst_graph) {
  std::string graph_name;
  AscendString ascend_name;
  if (dst_graph.GetName(ascend_name) == GRAPH_SUCCESS) {
    graph_name = std::string((ascend_name.GetString() != nullptr) ? ascend_name.GetString() : "");
  }
  if (graph_name.empty() && (src_graph.GetName(ascend_name) == GRAPH_SUCCESS)) {
    graph_name = std::string((ascend_name.GetString() != nullptr) ? ascend_name.GetString() : "");
  }

  ComputeGraphPtr new_compute_graph = MakeShared<ComputeGraph>(graph_name);
  GE_CHECK_NOTNULL(new_compute_graph);
  const ComputeGraphPtr src_compute_graph = GraphUtilsEx::GetComputeGraph(src_graph);
  GE_CHECK_NOTNULL(src_compute_graph);
  if (src_compute_graph->GetParentGraph() != nullptr) {
    GELOGE(GRAPH_FAILED, "[Check][RootGraph] Only support copy root graph, current graph name:%s, "
                         "parent graph name:%s.", src_compute_graph->GetName().c_str(),
           src_compute_graph->GetParentGraph()->GetName().c_str());
    return GRAPH_FAILED;
  }
  const int32_t depth = 0;
  std::map<ConstNodePtr, NodePtr> node_old_2_new;
  std::map<ConstOpDescPtr, OpDescPtr> op_desc_old_2_new;
  graphStatus ret = GraphUtils::CopyComputeGraph(src_compute_graph, new_compute_graph,
                                                 node_old_2_new, op_desc_old_2_new, depth);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][Graph] failed, ret:%d.", ret);
    return GRAPH_FAILED;
  }
  Graph tmp_graph = GraphUtilsEx::CreateGraphFromComputeGraph(new_compute_graph);
  ret = GraphUtilsEx::CopyGraphImpl(src_graph, tmp_graph, node_old_2_new, op_desc_old_2_new);
  if (ret != GRAPH_SUCCESS) {
    GELOGE(GRAPH_FAILED, "[Copy][GraphImpl] failed, ret:%d.", ret);
    return GRAPH_FAILED;
  }
  std::swap(dst_graph, tmp_graph);
  return GRAPH_SUCCESS;
}
} // namespace ge

