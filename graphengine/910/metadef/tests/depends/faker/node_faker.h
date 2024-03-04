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

#ifndef AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_NODE_FAKER_H_
#define AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_NODE_FAKER_H_
#include "graph/node.h"
#include "graph/compute_graph.h"
namespace gert {
class ComputeNodeFaker {
 public:
  ComputeNodeFaker() : graph_(std::make_shared<ge::ComputeGraph>("TempGraph")) {}
  explicit ComputeNodeFaker(ge::ComputeGraphPtr graph) : graph_(std::move(graph)) {}
  ComputeNodeFaker &IoNum(size_t input_num, size_t output_num, ge::DataType data_type = ge::DT_FLOAT);
  ComputeNodeFaker &InputNames(std::vector<std::string> names);
  ComputeNodeFaker &OutputNames(std::vector<std::string> names);
  ComputeNodeFaker &NameAndType(std::string name, std::string type);
  template <typename T>
  ComputeNodeFaker &Attr(const char *key, const T &value) {
    attr_keys_to_value_[key] = ge::AnyValue::CreateFrom<T>(value);
    return *this;
  }
  ge::NodePtr Build();

 private:
  ge::ComputeGraphPtr graph_;
  std::string name_;
  std::string type_;
  std::vector<ge::GeTensorDesc> inputs_desc_;
  std::vector<ge::GeTensorDesc> outputs_desc_;
  std::vector<std::string> input_names_;
  std::vector<std::string> output_names_;
  std::map<std::string, ge::AnyValue> attr_keys_to_value_;
};
}  // namespace gert
#endif  // AIR_CXX_TESTS_FRAMEWORK_GE_RUNTIME_STUB_INCLUDE_FAKER_NODE_FAKER_H_
