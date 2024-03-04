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

#ifndef FLOW_GRAPH_FLOW_ATTR_UTIL_H_
#define FLOW_GRAPH_FLOW_ATTR_UTIL_H_

#include <vector>
#include <map>
#include <cstdint>
#include "flow_graph/flow_attr.h"
#include "graph/utils/op_desc_utils.h"

namespace ge {
namespace dflow {
class FlowAttrUtil {
public:
  static graphStatus SetAttrsToTensorDesc(const std::vector<DataFlowInputAttr> &attrs, GeTensorDescPtr &tensor_desc);

private:
  static bool CheckAttrsIsSupport(const std::vector<DataFlowInputAttr> &attrs);
  static graphStatus SetCountBatchAttr(const void *const attr_value, GeTensorDescPtr &tensor_desc);
  static graphStatus SetTimeBatchAttr(const void *const attr_value, GeTensorDescPtr &tensor_desc);
  using SetAttrFunc = graphStatus (*)(const void *const attr_value, GeTensorDescPtr &input_tensor_desc);
  static const std::map<DataFlowAttrType, SetAttrFunc> set_attr_funcs_;
};
}  // namespace dflow
}  // namespace ge
#endif // FLOW_GRAPH_FLOW_ATTR_UTIL_H_