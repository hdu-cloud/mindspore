/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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

#include <unordered_set>
#include "graph/utils/op_type_utils.h"
#include "graph/debug/ge_op_types.h"
#include "graph/debug/ge_attr_define.h"
#include "graph/debug/ge_util.h"

namespace ge {
namespace {
const std::unordered_set<std::string> kDataOpSet = {DATA, REFDATA, AIPPDATA, ANN_DATA};
const std::unordered_set<std::string> kVariableOpSet = {VARIABLE, VARIABLEV2};
const std::unordered_set<std::string> kAssignOpSet = {
    ASSIGNADD, ASSIGN, ASSIGNSUB, ASSIGNADDVARIABLEOP, ASSIGNSUBVARIABLEOP, ASSIGNVARIABLEOP};
const std::unordered_set<std::string> kIdentityOpSet = {IDENTITY, READVARIABLEOP};
}  // namespace

/**
 * @brief 判断类型是否为DATA
 *        其中不包含QueueData, 该算子原型与其他Data不同，只有输出没有输出
 *        且在编译过程中有自由逻辑，不宜一起判断。
 *
 * @param type
 * @return true
 * @return false
 */
bool OpTypeUtils::IsDataNode(const std::string &type) {
  return (kDataOpSet.count(type) > 0);
}

bool OpTypeUtils::IsVariableNode(const std::string &type) {
  return (kVariableOpSet.count(type) > 0);
}

bool OpTypeUtils::IsVarLikeNode(const std::string &type) {
  return IsVariableNode(type) || (type == REFDATA);
}

bool OpTypeUtils::IsAssignLikeNode(const std::string &type) {
  return kAssignOpSet.count(type) > 0U;
}

bool OpTypeUtils::IsIdentityLikeNode(const std::string &type) {
  return kIdentityOpSet.count(type) > 0U;
}

/**
 * @brief get the Original Type of FrameworkOp
 *        其中不包含QueueData, 该算子原型与其他Data不同，只有输出没有输出
 *        且在编译过程中有自由逻辑，不宜一起判断。
 *
 * @param [in] node
 * @param [out] type
 * @return graphStatus
 */
graphStatus OpTypeUtils::GetOriginalType(const ge::NodePtr &node, std::string &type) {
  GE_CHECK_NOTNULL(node);
  type = node->GetType();
  GE_IF_BOOL_EXEC(type != FRAMEWORKOP, return GRAPH_SUCCESS);
  GE_CHECK_NOTNULL(node->GetOpDesc());
  const bool ret = ge::AttrUtils::GetStr(node->GetOpDesc(), ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE, type);
  if (!ret) {
    REPORT_INNER_ERROR("E19999", "Get Attr:%s fail from op:%s(%s)", ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE.c_str(),
                       node->GetName().c_str(), node->GetType().c_str());
    GELOGE(INTERNAL_ERROR, "[Get][Attr] %s fail from op:%s(%s)", ATTR_NAME_FRAMEWORK_ORIGINAL_TYPE.c_str(),
           node->GetName().c_str(), node->GetType().c_str());
    return INTERNAL_ERROR;
  }
  GELOGD("Get FrameWorkOp original type [%s]", type.c_str());
  return GRAPH_SUCCESS;
}
}  // namespace ge
