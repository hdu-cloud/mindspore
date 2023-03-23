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
#ifndef METADEF_CXX_RUNTIME_KERNEL_CONTEXT_BUILDER_H_
#define METADEF_CXX_RUNTIME_KERNEL_CONTEXT_BUILDER_H_

#include "graph/node.h"
#include "exe_graph/runtime/compute_node_info.h"
#include "exe_graph/runtime/kernel_context.h"
#include "exe_graph/lowering/buffer_pool.h"

namespace gert {
class KernelContextHolder {
public:
  KernelContextHolder() = default;
  KernelContextHolder(KernelContextHolder &&holder) {
    context_holder_ = std::move(holder.context_holder_);
    value_holder_ = std::move(holder.value_holder_);
    compute_node_extend_holder_ = std::move(holder.compute_node_extend_holder_);
    buffer_pool_ = holder.buffer_pool_;
    context_ = holder.context_;
  }

  KernelContextHolder &operator=(KernelContextHolder &&holder) {
    context_holder_ = std::move(holder.context_holder_);
    value_holder_ = std::move(holder.value_holder_);
    compute_node_extend_holder_ = std::move(holder.compute_node_extend_holder_);
    buffer_pool_ = holder.buffer_pool_;
    context_ = holder.context_;
    return *this;
  }

  ~KernelContextHolder() {
    for (auto &value : value_holder_) {
      value.Set(nullptr, nullptr);
    }
  }

  std::unique_ptr<uint8_t[]> context_holder_;
  std::vector<Chain> value_holder_;
  std::unique_ptr<uint8_t[]> compute_node_extend_holder_;
  bg::BufferPool buffer_pool_;
  KernelContext *context_;
};
class KernelRunContextBuilder {
public:
  KernelRunContextBuilder() = default;
  KernelRunContextBuilder &Inputs(std::vector<std::pair<void *, Chain::Deleter>> inputs) {
    inputs_ = std::move(inputs);
    return *this;
  }

  KernelRunContextBuilder &Inputs(std::vector<void *> inputs) {
    for (auto &input : inputs) {
      inputs_.emplace_back(input, nullptr);
    }
    return *this;
  }

  KernelRunContextBuilder &Outputs(std::vector<void *> outputs) {
    for (auto &output : outputs) {
      outputs_.emplace_back(output, nullptr);
    }
    return *this;
  }

  KernelRunContextBuilder &Outputs(std::vector<std::pair<void *, Chain::Deleter>> outputs) {
    outputs_ = std::move(outputs);
    return *this;
  }

  KernelContextHolder Build(ge::OpDescPtr &op_desc);

private:
  ge::NodePtr MakeNode(ge::OpDescPtr &op_desc);
private:
  ge::ComputeGraphPtr graph_;
  std::vector<std::pair<void *, Chain::Deleter>> inputs_;
  std::vector<std::pair<void *, Chain::Deleter>> outputs_;
};
}  // namespace gert
#endif
