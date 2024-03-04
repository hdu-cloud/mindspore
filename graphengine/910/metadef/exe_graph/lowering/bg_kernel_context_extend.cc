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
#include "exe_graph/lowering/bg_kernel_context_extend.h"

#include "common/ge_common/debug/ge_log.h"
#include "common/checker.h"

#include "exe_graph/lowering/bg_ir_attrs.h"
#include "exe_graph/runtime/context_extend.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_util.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/math_util.h"

namespace gert {
namespace bg {
namespace {
ge::graphStatus InitIOInstanceInfo(const ge::NodePtr &node, ComputeNodeInfo &compute_node_info) {
  auto in_ir_index_to_instance_index_pair_map
    = ge::OpDescUtils::GetInputIrIndexes2InstanceIndexesPairMap(node->GetOpDesc());
  if (in_ir_index_to_instance_index_pair_map.empty()) {
    GELOGI("node [%s(%s)] ir_index_to_instance_index_pair_map is empty",
           node->GetName().c_str(), node->GetType().c_str());
    return ge::GRAPH_SUCCESS;
  }
  const auto &ir_inputs = node->GetOpDesc()->GetIrInputs();
  size_t input_index = 0;
  for (size_t i = 0; i < ir_inputs.size(); ++i) {
    auto ins_info = compute_node_info.MutableInputInstanceInfo(i);
    GE_ASSERT_NOTNULL(ins_info);
    size_t instance_num = in_ir_index_to_instance_index_pair_map[i].second;
    compute_node_info.MutableInputInstanceInfo(i)->SetInstantiationNum(instance_num);
    compute_node_info.MutableInputInstanceInfo(i)->SetInstanceStart(input_index);
    input_index += instance_num;
  }

  auto out_ir_index_to_instance_index_pair_map
      = ge::OpDescUtils::GetOutputIrIndexes2InstanceIndexesPairMap(node->GetOpDesc());
  if (out_ir_index_to_instance_index_pair_map.empty()) {
    GELOGI("node [%s(%s)] output ir_index_to_instance_index_pair_map is empty",
           node->GetName().c_str(), node->GetType().c_str());
    return ge::GRAPH_SUCCESS;
  }
  const auto &ir_outputs = node->GetOpDesc()->GetIrOutputs();
  size_t output_index = 0;
  for (size_t i = 0; i < ir_outputs.size(); ++i) {
    auto ins_info = compute_node_info.MutableOutputInstanceInfo(i);
    GE_ASSERT_NOTNULL(ins_info);
    size_t instance_num = out_ir_index_to_instance_index_pair_map[i].second;
    compute_node_info.MutableOutputInstanceInfo(i)->SetInstantiationNum(instance_num);
    compute_node_info.MutableOutputInstanceInfo(i)->SetInstanceStart(output_index);
    output_index += instance_num;
  }
  return ge::GRAPH_SUCCESS;
}

void SetCompileTimeTd(const ge::ConstGeTensorDescPtr &desc, CompileTimeTensorDesc &td) {
  td.SetDataType(desc->GetDataType());
  td.SetOriginFormat(desc->GetOriginFormat());
  td.SetStorageFormat(desc->GetFormat());
  int64_t reshape_type_mask = 0;
  if (ge::AttrUtils::GetInt(desc, ge::ATTR_NAME_RESHAPE_TYPE_MASK, reshape_type_mask)) {
    td.SetExpandDimsType(ExpandDimsType(reshape_type_mask));
  }
}

ge::graphStatus GetConnectedEdgeIndexesToAnchorIndexMap(const ge::NodePtr &node,
    std::map<size_t, size_t> &connected_edge_indexes_to_anchor_index) {
  size_t compute_node_index = 0U;
  for (const auto &anchor : node->GetAllInDataAnchors()) {
    GE_ASSERT_NOTNULL(anchor);
    if (anchor->GetPeerOutAnchor() == nullptr) {
      continue;
    }
    GE_ASSERT_NOTNULL(anchor->GetPeerOutAnchor()->GetOwnerNode());
    connected_edge_indexes_to_anchor_index[compute_node_index] = static_cast<size_t>(anchor->GetIdx());
    ++compute_node_index;
  }
  return ge::GRAPH_SUCCESS;
}

ge::graphStatus InitCompileTimeTD(const ge::NodePtr &node, ComputeNodeInfo &compute_node_info) {
  std::map<size_t, size_t> connected_edge_indexes_to_anchor_index;
  const auto ret = GetConnectedEdgeIndexesToAnchorIndexMap(node, connected_edge_indexes_to_anchor_index);
  if (ret != ge::GRAPH_SUCCESS) {
    GELOGE(ret, "get connected edge indexes to anchor index map failed. node:%s(%s)",
           node->GetName().c_str(), node->GetType().c_str());
    return ret;
  }
  GE_ASSERT_TRUE(connected_edge_indexes_to_anchor_index.size() == compute_node_info.GetInputsNum());
  for (size_t i = 0; i < compute_node_info.GetInputsNum(); ++i) {
    GE_ASSERT_TRUE(i < node->GetOpDesc()->GetAllInputsSize());
    const auto &desc_need_check = node->GetOpDesc()->GetInputDesc(connected_edge_indexes_to_anchor_index[i]);
    if (desc_need_check.IsValid() != ge::GRAPH_SUCCESS) {
      continue;
    }
    const auto desc = node->GetOpDesc()->GetInputDescPtr(connected_edge_indexes_to_anchor_index[i]);
    GE_ASSERT_NOTNULL(desc);
    auto td = compute_node_info.MutableInputTdInfo(i);
    GE_ASSERT_NOTNULL(td);
    SetCompileTimeTd(desc, *td);
  }

  for (size_t i = 0; i < node->GetAllOutDataAnchorsSize(); ++i) {
    auto desc = node->GetOpDesc()->GetOutputDescPtr(i);
    GE_ASSERT_NOTNULL(desc);
    auto td = compute_node_info.MutableOutputTdInfo(i);
    GE_ASSERT_NOTNULL(td);
    SetCompileTimeTd(desc, *td);
  }
  return ge::SUCCESS;
}
bool GetPrivateAttrsList(const ge::NodePtr &node, const gert::OpImplKernelRegistry::PrivateAttrList &private_attrs,
                         std::vector<ge::AnyValue> &runtime_attrs_list) {
  const auto &all_attrs = node->GetOpDesc()->GetAllAttrs();
  for (auto &private_attr : private_attrs) {
    auto &private_attr_name = private_attr.first;
    auto iter = all_attrs.find(private_attr_name.GetString());
    if (iter == all_attrs.end()) {
      if (!private_attr.second.IsEmpty()) {
        runtime_attrs_list.push_back(private_attr.second);
        continue;
      }
      GELOGE(ge::FAILED, "Can not find the private attr %s from node %s",
             private_attr_name.GetString(), node->GetName().c_str());
      return false;
    }
    runtime_attrs_list.push_back(iter->second);
  }
  return true;
}
std::unique_ptr<uint8_t[]> CreateComputeNodeInfoImpl(const std::unique_ptr<uint8_t[]> &attr_buf,
                                                     const size_t attr_size,
                                                     const ge::NodePtr &node,
                                                     BufferPool &buffer_pool,
                                                     size_t &total_size) {
  const auto ir_input_num = node->GetOpDesc()->GetIrInputs().size();
  const auto ir_output_num = node->GetOpDesc()->GetIrOutputs().size();
  const auto input_num = node->GetInDataNodesAndAnchors().size();
  const auto output_num = node->GetAllOutDataAnchorsSize();
  GELOGD("node: %s(%s), ir_input_num:%zu, ir_output_num:%zu, input_num:%zu, output_num:%u.", node->GetName().c_str(),
         node->GetType().c_str(), ir_input_num, ir_output_num, input_num, output_num);
  GE_ASSERT_SUCCESS(ComputeNodeInfo::CalcSize(ir_input_num, ir_output_num, input_num, output_num, total_size));
  GE_ASSERT_TRUE(!ge::AddOverflow(total_size, attr_size, total_size));
  auto compute_node_info_holder = ge::ComGraphMakeUnique<uint8_t[]>(total_size);
  GE_ASSERT_NOTNULL(compute_node_info_holder, "Create compute node info holder failed");

  auto node_name = buffer_pool.AddStr(node->GetName().c_str());
  auto node_type = buffer_pool.AddStr(node->GetType().c_str());
  auto compute_node_info = ge::PtrToPtr<uint8_t, ComputeNodeInfo>(compute_node_info_holder.get());
  compute_node_info->Init(ir_input_num, ir_output_num, input_num, output_num, attr_size,
                          ge::PtrToPtr<void, ge::char_t>(ge::ValueToPtr(node_name)),
                          ge::PtrToPtr<void, ge::char_t>(ge::ValueToPtr(node_type)));

  auto ret = InitIOInstanceInfo(node, *compute_node_info);
  GE_ASSERT_SUCCESS(ret, "Init input instance info for node:%s failed.", node->GetName().c_str());

  ret = InitCompileTimeTD(node, *compute_node_info);
  GE_ASSERT_SUCCESS(ret, "Init compile time tensor desc for node:%s failed.", node->GetName().c_str());

  auto attr = compute_node_info->MutableAttrs();
  const auto offset = ge::PtrToPtr<RuntimeAttrs, uint8_t>(attr) - compute_node_info_holder.get();
  if (static_cast<size_t>(offset) > total_size) {
    GELOGE(
        ge::FAILED,
        "Failed to create kernel context extend info, the offset of attr %zu beyond the total size of ExtendInfo %zu",
        offset, attr_size);
    return nullptr;
  }
  const auto outputs_ins_info_size = compute_node_info->GetIrOutputsNum() * sizeof(AnchorInstanceInfo);
  ret = ge::GeMemcpy(ge::PtrToPtr<RuntimeAttrs, uint8_t>(attr), (total_size - offset - outputs_ins_info_size),
                     attr_buf.get(), attr_size);
  GE_ASSERT_SUCCESS(ret, "memcpy_s failed, copy size is %zu, dst size is %zu", attr_size,
                    (total_size - offset - outputs_ins_info_size));
  GELOGI("Node %s, compute_node_info attr_size %zu, outputs_ins_info_size:%zu, offset:%zu, total_size:%zu.",
         node->GetNamePtr(), attr_size, outputs_ins_info_size, offset, total_size);
  return compute_node_info_holder;
}
}  // namespace

std::unique_ptr<uint8_t[]> CreateComputeNodeInfo(const ge::NodePtr &node, BufferPool &buffer_pool, size_t &total_size) {
  size_t attr_size;
  const auto attr_buf = CreateAttrBuffer(node, attr_size);
  GE_ASSERT_NOTNULL(attr_buf, "Create attr buffer for node: %s failed", node->GetName().c_str());
  return CreateComputeNodeInfoImpl(attr_buf, attr_size, node, buffer_pool, total_size);
}

std::unique_ptr<uint8_t[]> CreateComputeNodeInfo(const ge::NodePtr &node,
                                                 BufferPool &buffer_pool,
                                                 const gert::OpImplKernelRegistry::PrivateAttrList &private_attrs,
                                                 size_t &total_size) {
  std::vector<ge::AnyValue> runtime_attrs_list;
  GE_ASSERT_TRUE(GetPrivateAttrsList(node, private_attrs, runtime_attrs_list));
  size_t attr_size;
  const auto attr_buf = CreateAttrBuffer(node, runtime_attrs_list, attr_size);
  GE_ASSERT_NOTNULL(attr_buf, "Create attr buffer for node: %s failed", node->GetName().c_str());
  return CreateComputeNodeInfoImpl(attr_buf, attr_size, node, buffer_pool, total_size);
}
std::unique_ptr<uint8_t[]> CreateComputeNodeInfoWithoutIrAttr(const ge::NodePtr &node, BufferPool &buffer_pool,
    const gert::OpImplKernelRegistry::PrivateAttrList &private_attrs, size_t &total_size) {
  std::vector<ge::AnyValue> runtime_attrs_list;
  GE_ASSERT_TRUE(GetPrivateAttrsList(node, private_attrs, runtime_attrs_list));
  size_t attr_size;
  const auto attr_buf = CreateAttrBufferWithoutIr(node, runtime_attrs_list, attr_size);
  GE_ASSERT_NOTNULL(attr_buf, "Create attr buffer without ir for node: %s failed", node->GetName().c_str());
  return CreateComputeNodeInfoImpl(attr_buf, attr_size, node, buffer_pool, total_size);
}
std::unique_ptr<uint8_t[]> CreateComputeNodeInfo(const ge::NodePtr &node, BufferPool &buffer_pool) {
  size_t total_size;
  return CreateComputeNodeInfo(node, buffer_pool, total_size);
}
}  // namespace bg
}  // namespace gert
