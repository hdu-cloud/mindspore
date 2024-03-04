/**
 * Copyright 2023 Huawei Technologies Co., Ltd
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

#ifndef GE_MOCK_H
#define GE_MOCK_H
#include <iostream>
#include <memory>
#include <string>
#include <cstring>
#include <sstream>
#include "graph/tensor.h"
#include "graph/operator_reg.h"
#include "graph/operator.h"
#include "graph/utils/tensor_utils.h"
#include "graph/utils/tensor_adapter.h"

#include "external/ge/ge_api.h"

namespace ge {

Session::Session(const std::map<std::string, std::string> &options) {}
Session::~Session() {}

Status Session::RunGraph(uint32_t id, const std::vector<Tensor> &inputs, std::vector<Tensor> &outputs) {
  // for test!!! just copy inputs to outputs:
  for (auto it = inputs.begin(); it != inputs.end(); it++) {
    outputs.emplace_back(*it);
  }
  return ge::GRAPH_SUCCESS;
}

Status Session::AddGraph(uint32_t id, const Graph &graph) { return ge::GRAPH_SUCCESS; }

Status GEInitialize(const std::map<std::string, std::string> &options) { return ge::GRAPH_SUCCESS; }

Status GEFinalize() { return ge::GRAPH_SUCCESS; }

Status Graph::SaveToFile(const string &file_name) const { return ge::GRAPH_SUCCESS; }

Status Session::RunGraphAsync(uint32_t graph_id, const std::vector<ge::Tensor> &inputs, RunAsyncCallback callback) {
  return ge::GRAPH_SUCCESS;
}

Status Session::RunGraphAsync(uint32_t graph_id, const ContinuousTensorList &inputs, RunAsyncCallback callback) {
  return ge::GRAPH_SUCCESS;
}

Status Session::AddGraph(uint32_t graph_id, const Graph &graph, const std::map<std::string, std::string> &options) {
  return ge::GRAPH_SUCCESS;
}

Status Session::CompileGraph(uint32_t graph_id) { return ge::GRAPH_SUCCESS; }

CompiledGraphSummaryPtr Session::GetCompiledGraphSummary(uint32_t graph_id) { return nullptr; }

Status Session::UpdateGraphFeatureMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  return ge::GRAPH_SUCCESS;
}

Status Session::SetGraphConstMemoryBase(uint32_t graph_id, const void *const memory, size_t size) {
  return ge::GRAPH_SUCCESS;
}

Status Session::RemoveGraph(uint32_t graph_id) { return ge::GRAPH_SUCCESS; }

Status Session::RunGraphWithStreamAsync(uint32_t graph_id, void *stream, const std::vector<Tensor> &inputs,
                                        std::vector<Tensor> &outputs) {
  return ge::GRAPH_SUCCESS;
}

void Operator::RequiredAttrWithTypeRegister(const char_t *name, const char_t *type) {}
}  // namespace ge

#endif
