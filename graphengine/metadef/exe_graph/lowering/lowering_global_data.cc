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

#include "exe_graph/lowering/lowering_global_data.h"
#include <memory>
#include "common/checker.h"
#include "graph/debug/ge_log.h"
#include "exe_graph/lowering/frame_selector.h"
namespace gert {
const bg::ValueHolderPtr &LoweringGlobalData::GetStream() const {
  return stream_;
}
LoweringGlobalData &LoweringGlobalData::SetStream(bg::ValueHolderPtr &&stream) {
  stream_ = std::move(stream);
  return *this;
}
const LoweringGlobalData::NodeCompileResult *LoweringGlobalData::FindCompiledResult(const ge::NodePtr &node) const {
  auto iter = node_name_to_compile_result_holders_.find(node->GetName());
  if (iter == node_name_to_compile_result_holders_.end()) {
    return nullptr;
  }
  return &iter->second;
}
LoweringGlobalData &LoweringGlobalData::AddCompiledResult(const ge::NodePtr &node,
                                                          LoweringGlobalData::NodeCompileResult compile_result) {
  node_name_to_compile_result_holders_[node->GetName()] = std::move(compile_result);
  return *this;
}

void *LoweringGlobalData::FindKnownSubgraphModel(const ge::NodePtr &node) const {
  const std::map<int64_t, void *>::const_iterator iter =
      node_ids_to_known_subgraph_models_.find(node->GetOpDesc()->GetId());
  if (iter == node_ids_to_known_subgraph_models_.cend()) {
    return nullptr;
  }
  return iter->second;
}

LoweringGlobalData &LoweringGlobalData::AddKnownSubgraphModel(const ge::NodePtr &node, void *const model) {
  node_ids_to_known_subgraph_models_[node->GetOpDesc()->GetId()] = model;
  return *this;
}

bg::ValueHolderPtr LoweringGlobalData::GetAllocator(AllocatorDesc desc) const {
  auto iter = placements_to_allocator_.find(desc);
  if (iter == placements_to_allocator_.end()) {
    return nullptr;
  }
  return iter->second;
}
LoweringGlobalData &LoweringGlobalData::SetAllocator(AllocatorDesc desc, bg::ValueHolderPtr allocator) {
  placements_to_allocator_[desc] = std::move(allocator);
  return *this;
}
LoweringGlobalData &LoweringGlobalData::SetExternalAllocator(bg::ValueHolderPtr &&allocator) {
  external_allocator_ = std::move(allocator);
  return *this;
}
bg::ValueHolderPtr LoweringGlobalData::GetOrCreateAllocator(AllocatorDesc desc) {
  const auto &iter = placements_to_allocator_.find(desc);
  if (iter == placements_to_allocator_.end()) {
    auto allocator = bg::FrameSelector::OnMainRoot([&]() -> std::vector<bg::ValueHolderPtr> {
      auto placement_holder = bg::ValueHolder::CreateConst(&desc.placement, sizeof(desc.placement));
      auto memory_type_holder = bg::ValueHolder::CreateConst(&desc.usage, sizeof(desc.usage));
      auto created_allocator = bg::ValueHolder::CreateSingleDataOutput("CreateAllocator",
                                                                       {placement_holder, memory_type_holder});
      auto selected_allocator = created_allocator;
      if (external_allocator_ != nullptr) {
        selected_allocator = bg::ValueHolder::CreateSingleDataOutput("SelectAllocator",
                                                                     {placement_holder, memory_type_holder,
                                                                      external_allocator_,
                                                                      created_allocator});
      }
      return {selected_allocator};
    });
    GE_ASSERT_EQ(allocator.size(), 1U);
    GE_ASSERT_NOTNULL(allocator[0]);
    SetAllocator(desc, allocator[0]);
    return allocator[0];
  }
  return iter->second;
}

uint64_t LoweringGlobalData::GetSessionId() {
  static std::atomic<uint64_t> global_session_id(0U);
  if (session_id_ == std::numeric_limits<uint64_t>::max()) {
    session_id_ = global_session_id++;
  }
  return session_id_;
}

bg::ValueHolderPtr LoweringGlobalData::GetOrCreateUniqueValueHolder(const std::string &name,
    const std::function<bg::ValueHolderPtr()> &builder) {
  const auto &iter = names_to_unique_value_holder_.find(name);
  if (iter == names_to_unique_value_holder_.end()) {
    auto holder = builder();
    return names_to_unique_value_holder_.emplace(name, holder).first->second;
  }
  return iter->second;
}
}  // namespace gert