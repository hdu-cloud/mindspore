/**
 * Copyright 2019-2020 Huawei Technologies Co., Ltd
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

#ifndef UT_COMMON_GRAPH_TESTCASE_GE_GRAPH_GRAPH_BUILDER_UTILS_H_
#define UT_COMMON_GRAPH_TESTCASE_GE_GRAPH_GRAPH_BUILDER_UTILS_H_

#include <string>
#include <vector>

#include "graph/compute_graph.h"
#include "graph/graph.h"
#include "graph/node.h"

namespace ge {
namespace ut {
class GraphBuilder {
 public:
  explicit GraphBuilder(const std::string &name) { graph_ = std::make_shared<ComputeGraph>(name); }
  NodePtr AddNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt,
                  Format format = FORMAT_NCHW, DataType data_type = DT_FLOAT,
                  std::vector<int64_t> shape = {1, 1, 224, 224});
  NodePtr AddNDNode(const std::string &name, const std::string &type, int in_cnt, int out_cnt) {
    return AddNode(name, type, in_cnt, out_cnt, FORMAT_ND, DT_FLOAT, {1, 1, 224, 224});
  }
  void AddDataEdge(NodePtr &src_node, int src_idx, NodePtr &dst_node, int dst_idx);
  void AddControlEdge(NodePtr &src_node, NodePtr &dst_node);
  ComputeGraphPtr GetGraph() {
    graph_->TopologicalSorting();
    return graph_;
  }

 private:
  ComputeGraphPtr graph_;
};
}  // namespace ut
}  // namespace ge

#endif  // UT_COMMON_GRAPH_TESTCASE_GE_GRAPH_GRAPH_BUILDER_UTILS_H_
