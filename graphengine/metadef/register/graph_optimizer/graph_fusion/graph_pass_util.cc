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

#include "register/graph_optimizer/fusion_common/graph_pass_util.h"
#include "graph/debug/ge_log.h"
#include "register/graph_optimizer/fusion_common/fusion_turbo_utils.h"

#define REGISTER_MAKE_SHARED(exec_expr0, exec_expr1) \
  do {                                               \
    try {                                            \
      exec_expr0;                                    \
    } catch (...) {                                  \
      GELOGW("Make shared failed");                  \
      exec_expr1;                                    \
    }                                                \
  } while (0)

namespace fe {
namespace {
const std::string kPassName = "pass_name";
const char* kDumpGeGraph = "DUMP_GE_GRAPH";
const char* kBackWard = "_backward";
const char* kRecompute = "_recompute";
const char* kOptimizer = "_optimizer";
const std::array<string, 2> kBoolAttrNeedInherit = {kRecompute, kOptimizer};
}

void GraphPassUtil::SetOutputDescAttr(const uint32_t &origin_index, const uint32_t &fusion_index,
                                      const ge::NodePtr &origin_node, const ge::NodePtr &fusion_node) {
  if (origin_node == nullptr || fusion_node == nullptr) {
    return;
  }

  if (fusion_node->GetOpDesc() == nullptr) {
    return;
  }

  const ge::OpDescPtr origin_op_desc = origin_node->GetOpDesc();
  if (origin_op_desc == nullptr) {
    return;
  }

  auto origin_node_output_desc = origin_node->GetOpDesc()->GetOutputDescPtr(origin_index);
  if (origin_node_output_desc == nullptr) {
    return;
  }

  const ge::GeTensorDescPtr fusion_node_output_desc = fusion_node->GetOpDesc()->MutableOutputDesc(fusion_index);
  if (fusion_node_output_desc == nullptr) {
    return;
  }

  SetOutputDescAttr(origin_node_output_desc, static_cast<int64_t>(origin_index), origin_op_desc,
                    fusion_node_output_desc);
}

void GraphPassUtil::SetOutputDescAttr(ge::ConstGeTensorDescPtr &origin_tensor_desc, const int64_t origin_index,
                                      const ge::OpDescPtr &origin_op_desc,
                                      const ge::GeTensorDescPtr &target_tensor_desc) {
  if (origin_tensor_desc == nullptr || target_tensor_desc == nullptr || origin_op_desc == nullptr) {
    return;
  }

  // set origin name
  std::string original_name;
  if (!ge::AttrUtils::GetStr(origin_tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, original_name) ||
    original_name.empty()) {
    std::vector<std::string> original_names;
    if (ge::AttrUtils::GetListStr(origin_op_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names) &&
      !original_names.empty()) {
      original_name = original_names[0];
    } else {
      original_name = origin_op_desc->GetName();
    }
  }
  (void)ge::AttrUtils::SetStr(target_tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_NAME, original_name);

  // set origin output index
  int64_t origin_output_index = 0;
  if (ge::AttrUtils::GetInt(origin_tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_output_index)) {
    (void)ge::AttrUtils::SetInt(target_tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_output_index);
  } else {
    (void)ge::AttrUtils::SetInt(target_tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OUTPUT_INDEX, origin_index);
  }

  // set origin output data type
  const ge::DataType origin_data_type = GetDataDumpOriginDataType(origin_tensor_desc);
  if (origin_data_type != ge::DT_UNDEFINED) {
    SetDataDumpOriginDataType(origin_data_type, target_tensor_desc);
  } else {
    SetDataDumpOriginDataType(origin_tensor_desc->GetOriginDataType(), target_tensor_desc);
  }

  // set origin output format
  const ge::Format origin_format = GetDataDumpOriginFormat(origin_tensor_desc);
  if (origin_format != ge::FORMAT_RESERVED) {
    SetDataDumpOriginFormat(origin_format, target_tensor_desc);
  } else {
    SetDataDumpOriginFormat(origin_tensor_desc->GetOriginFormat(), target_tensor_desc);
  }
}

/** get origin format for data dump
 *
 * @param tensor_desc,usually is output_desc
 *
 * @return format of this tensor_desc
 */
ge::Format GraphPassUtil::GetDataDumpOriginFormat(const ge::GeTensorDescPtr &tensor_desc) {
  std::string origin_format_str;
  if (!ge::AttrUtils::GetStr(tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, origin_format_str)) {
    // Can not get the certificate and it's not set,return directly
    return ge::FORMAT_RESERVED;
  }
  if (origin_format_str == "RESERVED") {
    return ge::FORMAT_RESERVED;
  }
  return ge::TypeUtils::SerialStringToFormat(origin_format_str);
}

ge::Format GraphPassUtil::GetDataDumpOriginFormat(ge::ConstGeTensorDescPtr &tensor_desc) {
  std::string origin_format_str;
  if (!ge::AttrUtils::GetStr(tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, origin_format_str)) {
    // Can not get the certificate and it's not set,return directly
    return ge::FORMAT_RESERVED;
  }
  if (origin_format_str == "RESERVED") {
    return ge::FORMAT_RESERVED;
  }
  return ge::TypeUtils::SerialStringToFormat(origin_format_str);
}

/** set origin format for data dump
 *
 * @param origin format
 *
 * @param tensor_desc,usually is output_desc
 */
void GraphPassUtil::SetDataDumpOriginFormat(const ge::Format &origin_format,
                                            const ge::GeTensorDescPtr &tensor_desc) {
  std::string origin_format_str = "RESERVED";
  if (origin_format != ge::FORMAT_RESERVED) {
    origin_format_str = ge::TypeUtils::FormatToSerialString(origin_format);
  }
  (void)ge::AttrUtils::SetStr(tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_FORMAT, origin_format_str);
}

/** set origin datatype for data dump
 *
 * @param origin datatype
 *
 * @param tensor_desc,usually is output_desc
 */
void GraphPassUtil::SetDataDumpOriginDataType(const ge::DataType origin_data_type,
                                              const ge::GeTensorDescPtr &tensor_desc) {
  std::string origin_data_type_str = "RESERVED";
  if (origin_data_type != ge::DT_UNDEFINED) {
    origin_data_type_str = ge::TypeUtils::DataTypeToSerialString(origin_data_type);
  }
  (void)ge::AttrUtils::SetStr(tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, origin_data_type_str);
}

/** get origin datatype for data dump
 *
 * @param tensor_desc,usually is output_desc
 *
 * @return format of this tensor_desc
 */
ge::DataType GraphPassUtil::GetDataDumpOriginDataType(const ge::GeTensorDescPtr &tensor_desc) {
  std::string origin_data_type_str;
  if (!ge::AttrUtils::GetStr(tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, origin_data_type_str)) {
    return ge::DT_UNDEFINED;
  }
  if (origin_data_type_str == "RESERVED") {
    return ge::DT_UNDEFINED;
  }
  return ge::TypeUtils::SerialStringToDataType(origin_data_type_str);
}

ge::DataType GraphPassUtil::GetDataDumpOriginDataType(ge::ConstGeTensorDescPtr &tensor_desc) {
  std::string origin_data_type_str;
  if (!ge::AttrUtils::GetStr(tensor_desc, ge::ATTR_NAME_DATA_DUMP_ORIGIN_DATA_TYPE, origin_data_type_str)) {
    return ge::DT_UNDEFINED;
  }
  if (origin_data_type_str == "RESERVED") {
    return ge::DT_UNDEFINED;
  }
  return ge::TypeUtils::SerialStringToDataType(origin_data_type_str);
}

void GraphPassUtil::AddNodeFromOpTypeMap(const NodeMapInfoPtr &node_map_info, const ge::NodePtr &node_ptr) {
  if ((node_map_info == nullptr) || (node_ptr == nullptr)) {
    return;
  }
  const NodeTypeMapPtr node_type_map = node_map_info->node_type_map;
  std::string real_op_type = ge::NodeUtils::GetNodeType(*node_ptr);
  const auto iter = node_type_map->find(real_op_type);
  if (iter != node_type_map->end()) {
    iter->second[node_ptr->GetName()] = node_ptr;
  } else {
    (void)node_type_map->emplace(std::make_pair(real_op_type,
        std::map<std::string, ge::NodePtr>{{node_ptr->GetName(), node_ptr}}));
  }
}

Status GraphPassUtil::GetOpTypeMapToGraph(NodeMapInfoPtr &node_map_info, const ge::ComputeGraph &graph) {
  node_map_info = graph.TryGetExtAttr("NodeMapInfo", node_map_info);
  if (node_map_info == nullptr) {
    return FAILED;
  }
  return SUCCESS;
}

void GraphPassUtil::RecordPassnameAndOriginalNames(const std::vector<ge::NodePtr> &original_nodes,
                                                   std::vector<ge::NodePtr> &fus_nodes, const string &pass_name,
                                                   const std::vector<std::string> &origin_op_names) {
  for (auto &node : fus_nodes) {
    (void)StoreAndUpdataOriginFusionPassName(node->GetOpDesc(), original_nodes, pass_name);
    RecordOriginalOpNames(original_nodes, node->GetOpDesc(), pass_name, origin_op_names);
  }
}

Status GraphPassUtil::StoreAndUpdataOriginFusionPassName(const ge::OpDescPtr &op_desc,
                                                         const std::vector<ge::NodePtr> &original_nodes,
                                                         const std::string &pass_name) {
  std::vector<std::string> pass_names;
  std::vector<std::string> pass_names_tmp;
  if (op_desc == nullptr) {
    return FAILED;
  }
  for (const ge::NodePtr &original_node : original_nodes) {
    if ((original_node == nullptr)) {
      return FAILED;
    }
    const ge::OpDescPtr origin_op_desc_ptr = original_node->GetOpDesc();
    if (!ge::AttrUtils::GetListStr(origin_op_desc_ptr, kPassName, pass_names_tmp) || pass_names_tmp.empty()) {
      continue;
    }
    (void)pass_names.insert(pass_names.cend(), pass_names_tmp.cbegin(), pass_names_tmp.cend());
  }
  pass_names.push_back(pass_name);
  if (!ge::AttrUtils::SetListStr(op_desc, kPassName, pass_names)) {
    return FAILED;
  }
  return SUCCESS;
}

void GraphPassUtil::GetBackWardAttr(const std::vector<ge::NodePtr> &original_nodes,
                                    bool &backward, BackWardInheritMode inherit_mode) {
  if (inherit_mode == BackWardInheritMode::kInheritTrue) {
    backward = true;
    return;
  }

  if (inherit_mode != BackWardInheritMode::kDoNotInherit) {
    for (const auto &origin_node : original_nodes) {
      (void) ge::AttrUtils::GetBool(origin_node->GetOpDesc(), kBackWard, backward);
      if (!backward) {
        continue;
      }

      if (inherit_mode != BackWardInheritMode::kFusedNode) {
        break;
      }

      bool has_in_node_backward = false;
      for (const auto &in_node : origin_node->GetInNodes()) {
        (void) ge::AttrUtils::GetBool(in_node->GetOpDesc(), kBackWard, has_in_node_backward);
        if (has_in_node_backward) {
          return;
        }
      }

      if (!has_in_node_backward) {
        backward = false;
      }
    }
  }
}

void GraphPassUtil::InheritGraphRelatedAttr(const std::vector<ge::NodePtr> &original_nodes,
                                            const std::vector<ge::NodePtr> &fusion_nodes,
                                            BackWardInheritMode inherit_mode) {
  vector<bool> bool_attrs(kBoolAttrNeedInherit.size(), false);
  size_t i = 0;
  for (const auto &attr : kBoolAttrNeedInherit) {
    for (const auto &origin_node : original_nodes) {
      bool value = false;
      (void)ge::AttrUtils::GetBool(origin_node->GetOpDesc(), attr, value);
      if (value) {
        bool_attrs[i] = value;
        break;
      }
    }
    ++i;
  }

  bool backward = false;
  GetBackWardAttr(original_nodes, backward, inherit_mode);

  for (const auto &fusion_node : fusion_nodes) {
    const ge::OpDescPtr fusion_op = fusion_node->GetOpDesc();
    if (backward && !ge::AttrUtils::HasAttr(fusion_op, kBackWard)) {
      (void) ge::AttrUtils::SetBool(fusion_op, kBackWard, backward);
    }

    if (bool_attrs.size() != kBoolAttrNeedInherit.size()) {
      GELOGW("[Fusion][InheritAttr]Integer attributes size %zu is not correct, should be %zu.",
             bool_attrs.size(), kBoolAttrNeedInherit.size());
      return;
    }

    i = 0;
    for (const auto &attr : kBoolAttrNeedInherit) {
      if (bool_attrs[i] != 0 && !ge::AttrUtils::HasAttr(fusion_op, attr)) {
        (void) ge::AttrUtils::SetBool(fusion_op, attr, bool_attrs[i]);
      }
      ++i;
    }
  }
}

void GraphPassUtil::InheritAttrFromOriNodes(const std::vector<ge::NodePtr> &original_nodes,
                                            const std::vector<ge::NodePtr> &fusion_nodes,
                                            BackWardInheritMode inherit_mode) {
  std::string op_compile_strategy;
  for (const auto &origin_node : original_nodes) {
    if (ge::AttrUtils::GetStr(origin_node->GetOpDesc(), ge::ATTR_NAME_OP_COMPILE_STRATEGY, op_compile_strategy) &&
        !op_compile_strategy.empty()) {
      break;
    }
  }

  int64_t keep_dtype = 0;
  for (const auto &origin_node : original_nodes) {
    if (ge::AttrUtils::GetInt(origin_node->GetOpDesc(), ge::ATTR_NAME_KEEP_DTYPE, keep_dtype) &&
        keep_dtype != 0) {
      break;
    }
  }

  for (const auto &fusion_node : fusion_nodes) {
    const ge::OpDescPtr fusion_op = fusion_node->GetOpDesc();
    if (!op_compile_strategy.empty() && !ge::AttrUtils::HasAttr(fusion_op, ge::ATTR_NAME_OP_COMPILE_STRATEGY)) {
      (void) ge::AttrUtils::SetStr(fusion_op, ge::ATTR_NAME_OP_COMPILE_STRATEGY, op_compile_strategy);
    }

    if (keep_dtype != 0 && !ge::AttrUtils::HasAttr(fusion_op, ge::ATTR_NAME_KEEP_DTYPE)) {
      (void) ge::AttrUtils::SetInt(fusion_op, ge::ATTR_NAME_KEEP_DTYPE, keep_dtype);
    }
  }

  InheritGraphRelatedAttr(original_nodes, fusion_nodes, inherit_mode);
}

void GraphPassUtil::RecordOriginalOpNames(const std::vector<ge::NodePtr> &original_nodes,
                                          const ge::OpDescPtr &op_desc, const string &pass_name,
                                          const std::vector<std::string> &origin_op_names) {
  const char* dump_ge_graph = std::getenv(kDumpGeGraph);
  FUSION_TURBO_NOTNULL(dump_ge_graph,);

  // 1. get the original_names
  GELOGD("Start to record op[%s] origin op names after pass[%s]", op_desc->GetName().c_str(), pass_name.c_str());
  std::shared_ptr<UnorderedMapping> origin_op_names_map = nullptr;
  REGISTER_MAKE_SHARED(origin_op_names_map = std::make_shared<UnorderedMapping>(), return);
  std::vector<std::string> origin_op_names_vec;
  size_t index = 0;

  if (op_desc == nullptr) {
    GELOGD("op_desc is nullptr");
    return;
  }
  for (const ge::NodePtr &original_node : original_nodes) {
    if (original_node == nullptr) {
      return;
    }
    const ge::OpDescPtr origin_op_desc_ptr = original_node->GetOpDesc();
    if (origin_op_desc_ptr == nullptr) {
      return;
    }
    std::shared_ptr<UnorderedMapping> op_names_maps_tmp = nullptr;
    REGISTER_MAKE_SHARED(op_names_maps_tmp = std::make_shared<UnorderedMapping>(), return);
    op_names_maps_tmp = origin_op_desc_ptr->TryGetExtAttr(ge::ATTR_NAME_ORIGIN_OP_NAMES_MAP, op_names_maps_tmp);
    if ((op_names_maps_tmp != nullptr) && (!op_names_maps_tmp->empty())) {
      size_t op_names_index = 0;
      std::vector<std::string> pass_names;
      if (!ge::AttrUtils::GetListStr(origin_op_desc_ptr, kPassName, pass_names) || pass_names.empty()) {
        continue;
      }
      for (const auto &pass_name_tmp : pass_names) {
        if (op_names_maps_tmp->find(pass_name_tmp) == op_names_maps_tmp->cend()) {
          GELOGD("Not find pass_name[%s] in ATTR_NAME_ORIGIN_OP_NAMES_MAP", pass_name_tmp.c_str());
          continue;
        }
        (void)origin_op_names_map->insert(std::pair<std::string, std::vector<std::string>>(pass_name_tmp,
            (*op_names_maps_tmp)[pass_name_tmp]));
        // get last item of op_names_maps_tmp and push all origin_op_names into vector
        if (op_names_index == (pass_names.size() - 1)) {
          for (const auto &origin_op_names_tmp : (*op_names_maps_tmp)[pass_name_tmp]) {
            origin_op_names_vec.push_back(origin_op_names_tmp);
          }
        }
        ++op_names_index;
      }
    } else if (origin_op_names.empty()) {
      origin_op_names_vec.push_back(origin_op_desc_ptr->GetName().c_str());
    } else if (index < origin_op_names.size()) {
      origin_op_names_vec.push_back(origin_op_names.at(index));
    }
    ++index;
  }
  (void)origin_op_names_map->insert(std::pair<std::string, std::vector<std::string>>(pass_name, origin_op_names_vec));

  // 2. set the dump attr
  (void)op_desc->SetExtAttr(ge::ATTR_NAME_ORIGIN_OP_NAMES_MAP, origin_op_names_map);
}

void GraphPassUtil::RecordOriginalNames(const std::vector<ge::NodePtr> &original_nodes,
                                        const ge::NodePtr &node) {
  // 1. get the original_names
  std::vector<std::string> original_names;
  for (const ge::NodePtr &original_node : original_nodes) {
    if ((original_node == nullptr) || (original_node->GetOpDesc() == nullptr)) {
      return;
    }

    const ge::OpDescPtr origin_op_desc_ptr = original_node->GetOpDesc();
    std::vector<std::string> names_tmp;
    const bool is_has_attr =
        ge::AttrUtils::GetListStr(origin_op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, names_tmp) &&
        !names_tmp.empty();
    if (is_has_attr) {
      for (const auto &node_name : names_tmp) {
        if (!node_name.empty()) {
          original_names.push_back(node_name);
        }
      }
    } else {
      original_names.push_back(origin_op_desc_ptr->GetName());
    }
  }

  // 2. set the dump attr
  if ((node == nullptr) || (node->GetOpDesc() == nullptr)) {
    return;
  }
  const ge::OpDescPtr node_op_desc_ptr = node->GetOpDesc();
  (void)ge::AttrUtils::SetListStr(node_op_desc_ptr, ge::ATTR_NAME_DATA_DUMP_ORIGIN_OP_NAMES, original_names);
}

void GraphPassUtil::AddNodeToNodeTypeMap(const NodeTypeMapPtr &node_type_map, const std::string &op_type,
                                         const ge::NodePtr &node_ptr) {
  if ((node_type_map == nullptr) || (node_ptr == nullptr)) {
    return;
  }
  const auto iter = node_type_map->find(op_type);
  if (iter == node_type_map->end()) {
    (void)node_type_map->emplace(std::make_pair(op_type,
        std::map<std::string, ge::NodePtr>{{node_ptr->GetName(), node_ptr}}));
  } else {
    (void)iter->second.emplace(node_ptr->GetName(), node_ptr);
  }
}

void GraphPassUtil::RemoveNodeFromNodeTypeMap(NodeTypeMapPtr &node_type_map, const std::string &op_type,
                                              const ge::NodePtr &node_ptr) {
  if ((node_type_map == nullptr) || (node_ptr == nullptr)) {
    return;
  }
  const auto iter = node_type_map->find(op_type);
  if (iter != node_type_map->end()) {
    (void)iter->second.erase(node_ptr->GetName());
  }
}

void GraphPassUtil::GetNodesFromNodeTypeMap(NodeTypeMapPtr &node_type_map, const std::string &op_type,
                                            std::vector<ge::NodePtr> &nodes) {
  if (node_type_map == nullptr) {
    return;
  }

  const auto iter = node_type_map->find(op_type);
  if (iter == node_type_map->end()) {
    return;
  }
  if (iter->second.empty()) {
    return;
  }

  for (auto node_iter = iter->second.cbegin(); node_iter != iter->second.cend(); node_iter++) {
    nodes.push_back(node_iter->second);
  }
}
}