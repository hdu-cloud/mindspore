/**
 * Copyright 2020 Huawei Technologies Co., Ltd
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

#include <cstdint>
#include <utility>

#include "graph/utils/tensor_adapter.h"
#include "framework/common/debug/ge_log.h"
#include "graph/runtime_inference_context.h"

namespace ge {
void RuntimeInferenceContext::Release() {
  const std::lock_guard<std::mutex> lk(mu_);
  ge_tensors_.clear();
}

graphStatus RuntimeInferenceContext::SetTensor(int64_t node_id, int32_t output_id, GeTensorPtr tensor) {
  const std::lock_guard<std::mutex> lk(mu_);
  auto &output_ge_tensors = ge_tensors_[node_id];
  if (static_cast<size_t>(output_id) >= output_ge_tensors.size()) {
    const size_t output_tensor_size = static_cast<size_t>(output_id) + 1U;
    output_ge_tensors.resize(output_tensor_size);
  }

  GELOGD("Set tensor for node_id = %ld, output_id = %d", node_id, output_id);
  output_ge_tensors[static_cast<size_t>(output_id)] = std::move(tensor);

  return GRAPH_SUCCESS;
}

graphStatus RuntimeInferenceContext::GetTensor(const int64_t node_id, int32_t output_id, GeTensorPtr &tensor) const {
  if (output_id < 0) {
    REPORT_INNER_ERROR("E18888", "Invalid output index: %d", output_id);
    GELOGE(GRAPH_PARAM_INVALID, "[Check][Param] Invalid output index: %d", output_id);
    return GRAPH_PARAM_INVALID;
  }

  const std::lock_guard<std::mutex> lk(mu_);
  const auto iter = ge_tensors_.find(node_id);
  if (iter == ge_tensors_.end()) {
    GELOGW("Node not register. Id = %ld", node_id);
    return INTERNAL_ERROR;
  }

  auto &output_tensors = iter->second;
  if (static_cast<uint32_t>(output_id) >= output_tensors.size()) {
    GELOGW("The %d th output tensor for node id [%ld] has not been registered.", output_id, node_id);
    return GRAPH_FAILED;
  }

  GELOGD("Get ge tensor for node_id = %ld, output_id = %d", node_id, output_id);
  tensor = output_tensors[static_cast<size_t>(output_id)];
  if (tensor == nullptr) {
    GELOGW("The %d th output tensor registered for node id [%ld] is nullptr.", output_id, node_id);
    return GRAPH_FAILED;
  }
  return GRAPH_SUCCESS;
}
} // namespace ge