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
#include "runtime/allocator.h"

namespace gert {
namespace {
const std::set<gert::TensorPlacement> kCurrentAllocatorSupportPlacement = {
    TensorPlacement::kOnDeviceHbm, TensorPlacement::kOnHost, TensorPlacement::kFollowing};
const std::set<gert::AllocatorUsage> kCurrentAllocatorSupportUsage = {
    AllocatorUsage::kAllocNodeOutput, AllocatorUsage::kAllocNodeWorkspace, AllocatorUsage::kAllocNodeShapeBuffer};

bool CurrentOnInitGraph() {
  ge::NodePtr subgraph_node = nullptr;
  auto current_graph = bg::ValueHolder::GetCurrentGraph();
  while ((current_graph != nullptr) && (current_graph->GetParentNode() != nullptr)) {
    subgraph_node = current_graph->GetParentNode();
    current_graph = subgraph_node->GetOwnerComputeGraph().get();
  }

  if (subgraph_node != nullptr) {
    return strcmp(subgraph_node->GetType().c_str(), GetExecuteGraphTypeStr(ExecuteGraphType::kInit)) == 0;
  } else {
    return false;
  }
}
/*
 * 此处用于判断是否能使用init图里的allocator:
 * 1.用户设置了always_external_allocator选项
 * 2.AllocatorDesc在我们当前支持的allocator范围内
 *
 * 为了兼容性考虑，当前只能支持现有的allocator，否则后续我们新增placement/useage时则会出错，用户老的版本加上我们新的软件会出错
 * */
bool CanUseInitAllocator(const bool always_external_allocator, const AllocatorDesc &desc) {
  if (!always_external_allocator) {
    return false;
  }
  if ((kCurrentAllocatorSupportPlacement.count(desc.placement) > 0U) &&
      (kCurrentAllocatorSupportUsage.count(desc.usage) > 0U)) {
    return true;
  } else {
    GELOGW("We don't support placement[%d] or usage[%d] current while always_external_allocator is true",
           static_cast<int32_t>(desc.placement), static_cast<int32_t>(desc.usage));
    return false;
  }
}
}  // namespace
const bg::ValueHolderPtr &LoweringGlobalData::GetStream() const {
  ExecuteGraphType graph_type = ExecuteGraphType::kMain;
  if (CurrentOnInitGraph()) {
    graph_type = ExecuteGraphType::kInit;
  }
  return streams_.holders[static_cast<size_t>(graph_type)];
}
LoweringGlobalData &LoweringGlobalData::SetStream(bg::ValueHolderPtr &&stream) {
  return SetStream(std::move(stream), ExecuteGraphType::kMain);
}
LoweringGlobalData &LoweringGlobalData::SetStream(bg::ValueHolderPtr &&stream, const ExecuteGraphType graph_type) {
  if (graph_type >= ExecuteGraphType::kNum) {
    return *this;
  }
  streams_.holders[static_cast<size_t>(graph_type)] = std::move(stream);
  return *this;
}
const LoweringGlobalData::NodeCompileResult *LoweringGlobalData::FindCompiledResult(const ge::NodePtr &node) const {
  const auto iter = node_name_to_compile_result_holders_.find(node->GetName());
  if (iter == node_name_to_compile_result_holders_.cend()) {
    return nullptr;
  }
  return &iter->second;
}
LoweringGlobalData &LoweringGlobalData::AddCompiledResult(const ge::NodePtr &node,
                                                          LoweringGlobalData::NodeCompileResult compile_result) {
  node_name_to_compile_result_holders_[node->GetName()] = std::move(compile_result);
  return *this;
}

void *LoweringGlobalData::GetGraphStaticCompiledModel(const std::string &graph_name) const {
  const auto iter = graph_to_static_models_.find(graph_name);
  if (iter == graph_to_static_models_.cend()) {
    return nullptr;
  }
  return iter->second;
}

LoweringGlobalData &LoweringGlobalData::AddStaticCompiledGraphModel(const std::string &graph_name, void *const model) {
  graph_to_static_models_[graph_name] = model;
  return *this;
}

bg::ValueHolderPtr LoweringGlobalData::GetAllocator(const AllocatorDesc &desc) const {
  if (CurrentOnInitGraph()) {
    return GetUniqueValueHolder(desc.GetKey() + "-Init");
  } else {
    return GetUniqueValueHolder(desc.GetKey());
  }
}
LoweringGlobalData &LoweringGlobalData::SetExternalAllocator(bg::ValueHolderPtr &&allocator) {
  return SetExternalAllocator(std::move(allocator), ExecuteGraphType::kMain);
}
LoweringGlobalData &LoweringGlobalData::SetExternalAllocator(bg::ValueHolderPtr &&allocator,
                                                             const ExecuteGraphType graph_type) {
  if (graph_type >= ExecuteGraphType::kNum) {
    return *this;
  }
  external_allocators_.holders[static_cast<size_t>(graph_type)] = std::move(allocator);
  return *this;
}

bg::ValueHolderPtr LoweringGlobalData::GetExternalAllocator(const bool from_init, const string &key,
                                                            const AllocatorDesc &desc) {
  bg::ValueHolderPtr init_selected_allocator = nullptr;
  auto init_out =
      bg::FrameSelector::OnInitRoot([&desc, &init_selected_allocator, this]() -> std::vector<bg::ValueHolderPtr> {
        auto placement_holder = bg::ValueHolder::CreateConst(&desc.placement, sizeof(desc.placement));
        auto memory_type_holder = bg::ValueHolder::CreateConst(&desc.usage, sizeof(desc.usage));
        init_selected_allocator = nullptr;
        if (external_allocators_.holders[static_cast<size_t>(ExecuteGraphType::kInit)] != nullptr) {
          init_selected_allocator = bg::ValueHolder::CreateSingleDataOutput(
              "GetAllocator",
              {placement_holder, memory_type_holder,
               external_allocators_.holders[static_cast<size_t>(ExecuteGraphType::kInit)]});
        } else {
          GELOGE(ge::PARAM_INVALID, "always_external_allocator option is true but external_allocators is nullptr!");
        }
        return {init_selected_allocator};
      });
  GE_ASSERT_EQ(init_out.size(), 1U);
  GE_ASSERT_NOTNULL(init_out[0]);

  auto allocator = bg::FrameSelector::OnMainRoot([&desc, &init_out, this]() -> std::vector<bg::ValueHolderPtr> {
    auto main_selected_allocator = init_out[0];
    auto placement_holder = bg::ValueHolder::CreateConst(&desc.placement, sizeof(desc.placement));
    auto memory_type_holder = bg::ValueHolder::CreateConst(&desc.usage, sizeof(desc.usage));
    if (external_allocators_.holders[static_cast<size_t>(ExecuteGraphType::kMain)] != nullptr) {
      main_selected_allocator = bg::ValueHolder::CreateSingleDataOutput(
          "SelectAllocator",
          {placement_holder, memory_type_holder,
           external_allocators_.holders[static_cast<size_t>(ExecuteGraphType::kMain)], init_out[0], GetStream()});
    }
    return {main_selected_allocator};
  });
  GE_ASSERT_EQ(allocator.size(), 1U);

  SetUniqueValueHolder(key + "-Init", init_selected_allocator);
  SetUniqueValueHolder(key, allocator[0]);
  if (from_init) {
    return init_selected_allocator;
  } else {
    return allocator[0];
  }
}

/* CanUseInitAllocator is true
 * +------------------------------------------------------------------+
 * |Main Graph                                                        |
 * |                 AllocMemory                                      |
 * |                     |                                            |
 * |                 (allocator)                                      |
 * |                     |                                            |
 * |                 InnerData                                        |
 * +------------------------------------------------------------------+
 * +------------------------------------------------------------------+
 * |Init Graph                                                        |
 * |                                                                  |
 * |                   InnerNetOutput                                 |
 * |                        ^                                         |
 * |                        |                                         |
 * |                    GetAllocator                                  |
 * |                   /     |      \                                 |
 * |  Const(placement) Const(usage) Data(Allocator)(-2) |             |
 * +------------------------------------------------------------------+
 */

/* CanUseInitAllocator is false
 * +------------------------------------------------------------------+
 * |Main Graph                                                        |
 * |                 (allocator)                                      |
 * |                     |                                            |
 * |     +------>  SelectAllocator  <-----+                           |
 * |     |           /       \            |                           |
 * | InnerData  InnerData   InnerData   Data(-2)                      |
 * +------------------------------------------------------------------+
 * +------------------------------------------------------------------+
 * |Init Graph                                                        |
 * |                                                                  |
 * |   +------+--->  InnerNetOutput    (allocator)                    |
 * |   |      |              ^            |                           |
 * |   |      |              |     SelectAllocator                    |
 * |   |      |              |    /         ^     \                   |
 * |   |      |     CreateAllocator         |   Data(Allocator)(-2)   |
 * |   |      |          /  \               |                         |
 * |   |  Const(placement)  Const(usage)    |                         |
 * |   |                         |          |                         |
 * |   +-------------------------+----------+                         |
 * +------------------------------------------------------------------+
 */
bg::ValueHolderPtr LoweringGlobalData::GetOrCreateAllocator(const AllocatorDesc desc) {
  const auto key = desc.GetKey();
  const auto init_key = key + "-Init";
  const auto from_init = CurrentOnInitGraph();

  bg::ValueHolderPtr allocator_holder;
  if (from_init) {
    allocator_holder = GetUniqueValueHolder(init_key);
  } else {
    allocator_holder = GetUniqueValueHolder(key);
  }

  if (allocator_holder != nullptr) {
    return allocator_holder;
  }
  /*
   * 用户设置always_external_allocator场景下，同时external_allocators_不为空的情况下，一定认为所有类型的allocator都创建好了，原因：
   * 1.不能考虑外置的external_allocators_中存在某些类型的allocator没有创建，之前为了保证正确性，必须在构图时根据placement跟usage
   * 创建一个CreateAllocator节点，在执行时创建兜底的allocator对象，但是allocator对象是需要浪费host内存资源，对单算子场景下，
   * 频繁创建导致host内存上升，因此设置了always_external_allocator的场景下不考虑某些类型的allocator没有创建
   *
   * 2.为什么这个地方不能判断满足当前placement+usage的allocator是否已经创建好了？这个地方还在构图，此时还是valueholder，还没有到初始化
   * 执行，因此无法感知用户是否完整创建了所有allocator，只有初始化图执行时才知道。
   *
   * 3.因此对于此场景，考虑在初始化图执行时做一个校验，用户设置了always_external_allocator的场景下，确保所有类型的allocator都创建好了
   *  因此, 在单算子场景下，需要无脑校验
   *
   * 4.为了兼容性考虑，当前只能支持现有的allocator，否则后续我们新增placement/useage时则会出错，用户老的版本加上我们新的软件会出错
   *
   * 5.always_external_allocator可以后续整改为always_use_init_allocator
   * */
  if (CanUseInitAllocator(lowering_option_.always_external_allocator, desc)) {
    return GetExternalAllocator(from_init, key, desc);
  } else {
    bg::ValueHolderPtr init_selected_allocator = nullptr;
    auto init_out = bg::FrameSelector::OnInitRoot([&desc, &init_selected_allocator,
                                                      this]() -> std::vector<bg::ValueHolderPtr> {
      auto placement_holder = bg::ValueHolder::CreateConst(&desc.placement, sizeof(desc.placement));
      auto memory_type_holder = bg::ValueHolder::CreateConst(&desc.usage, sizeof(desc.usage));
      auto created_allocator =
          bg::ValueHolder::CreateSingleDataOutput("CreateAllocator", {placement_holder, memory_type_holder});
      if (external_allocators_.holders[static_cast<size_t>(ExecuteGraphType::kInit)] != nullptr) {
        init_selected_allocator = bg::ValueHolder::CreateSingleDataOutput(
            "SelectAllocator",
            {placement_holder, memory_type_holder,
             external_allocators_.holders[static_cast<size_t>(ExecuteGraphType::kInit)],
             created_allocator, GetStream()});
      } else {
        init_selected_allocator = created_allocator;
      }

      return {created_allocator, placement_holder, memory_type_holder};
    });
    GE_ASSERT_EQ(init_out.size(), 3U);

    auto allocator = bg::FrameSelector::OnMainRoot([&init_out, this]() -> std::vector<bg::ValueHolderPtr> {
      auto main_selected_allocator = init_out[0];
      if (external_allocators_.holders[static_cast<size_t>(ExecuteGraphType::kMain)] != nullptr) {
        main_selected_allocator = bg::ValueHolder::CreateSingleDataOutput(
            "SelectAllocator",
            {init_out[1], init_out[2], external_allocators_.holders[static_cast<size_t>(ExecuteGraphType::kMain)],
             init_out[0], GetStream()});
      }
      return {main_selected_allocator};
    });
    GE_ASSERT_EQ(allocator.size(), 1U);

    SetUniqueValueHolder(key + "-Init", init_selected_allocator);
    SetUniqueValueHolder(key, allocator[0]);

    if (from_init) {
      return init_selected_allocator;
    } else {
      return allocator[0];
    }
  }
}

bg::ValueHolderPtr LoweringGlobalData::GetOrCreateUniqueValueHolder(
    const std::string &name, const std::function<bg::ValueHolderPtr()> &builder) {
  return GetOrCreateUniqueValueHolder(name, [&builder]() -> std::vector<bg::ValueHolderPtr> { return {builder()}; })[0];
}

std::vector<bg::ValueHolderPtr> LoweringGlobalData::GetOrCreateUniqueValueHolder(
    const std::string &name, const std::function<std::vector<bg::ValueHolderPtr>()> &builder) {
  const decltype(unique_name_to_value_holders_)::const_iterator &iter = unique_name_to_value_holders_.find(name);
  if (iter == unique_name_to_value_holders_.cend()) {
    auto holder = builder();
    return unique_name_to_value_holders_.emplace(name, holder).first->second;
  }
  return iter->second;
}
void LoweringGlobalData::SetUniqueValueHolder(const string &name, const bg::ValueHolderPtr &holder) {
  unique_name_to_value_holders_.emplace(name, std::vector<bg::ValueHolderPtr>{holder});
}
bg::ValueHolderPtr LoweringGlobalData::GetUniqueValueHolder(const string &name) const {
  const auto &iter = unique_name_to_value_holders_.find(name);
  if (iter == unique_name_to_value_holders_.cend()) {
    return nullptr;
  }
  return iter->second[0];
}

void LoweringGlobalData::SetValueHolders(const string &name, const bg::ValueHolderPtr &holder) {
  unique_name_to_value_holders_[name].emplace_back(holder);
}

size_t LoweringGlobalData::GetValueHoldersSize(const string &name) {
  const auto &iter = unique_name_to_value_holders_.find(name);
  if (iter == unique_name_to_value_holders_.cend()) {
    return 0U;
  }
  return iter->second.size();
}

void LoweringGlobalData::SetModelWeightSize(const size_t require_weight_size) {
  model_weight_size_ = require_weight_size;
}
size_t LoweringGlobalData::GetModelWeightSize() const {
  return model_weight_size_;
}

const LoweringOption &LoweringGlobalData::GetLoweringOption() const {
  return lowering_option_;
}
void LoweringGlobalData::SetLoweringOption(const LoweringOption &lowering_option) {
  lowering_option_ = lowering_option;
}
}  // namespace gert