/**
 * Copyright 2022 Huawei Technologies Co., Ltd
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
#include <gtest/gtest.h>
#include "exe_graph/runtime/tiling_context_builder.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "exe_graph/runtime/atomic_clean_tiling_context.h"
#include "exe_graph/lowering/value_holder.h"
#include "platform/platform_infos_def.h"
#include "common/ge_common/util.h"
#include "register/op_impl_registry.h"
#include "register/op_impl_registry_base.h"
#include "faker/node_faker.h"
#include "faker/space_registry_faker.h"
#include "graph/debug/ge_attr_define.h"
#include "common/checker.h"

namespace gert {
class TilingContextBuilderUT : public testing::Test {};
namespace {
IMPL_OP(DDIT02).InputsDataDependency({0, 2});

ge::Status AddDataNodeForAtomic(ge::ComputeGraphPtr &graph, ge::NodePtr &clean_node, size_t output_size) {
  // add data node for workspace
  auto workspace_data_op_desc = std::make_shared<ge::OpDesc>(clean_node->GetName() + "_Data_0", "Data");
  GE_CHECK_NOTNULL(workspace_data_op_desc);
  if (workspace_data_op_desc->AddOutputDesc(ge::GeTensorDesc()) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "workspace_data_op_desc add output desc failed");
    return ge::FAILED;
  }
  auto workspace_data_node = graph->AddNode(workspace_data_op_desc);
  GE_CHECK_NOTNULL(workspace_data_node);
  auto ret = ge::GraphUtils::AddEdge(workspace_data_node->GetOutDataAnchor(0), clean_node->GetInDataAnchor(0));
  if (ret != ge::SUCCESS) {
    GELOGE(ge::FAILED, "add edge between [%s] and [%s] failed", workspace_data_node->GetName().c_str(),
           clean_node->GetName().c_str());
    return ge::FAILED;
  }

  // add data node for output
  for (size_t i = 0U; i < output_size; ++i) {
    auto data_op_desc = std::make_shared<ge::OpDesc>(clean_node->GetName() + "_Data_" + std::to_string(i + 1), "Data");
    GE_CHECK_NOTNULL(data_op_desc);
    if (data_op_desc->AddOutputDesc(ge::GeTensorDesc()) != ge::SUCCESS) {
      GELOGE(ge::FAILED, "data_op_desc add output desc failed, i = %zu", i);
      return ge::FAILED;
    }
    auto data_node = graph->AddNode(data_op_desc);
    GE_CHECK_NOTNULL(data_node);
    ret = ge::GraphUtils::AddEdge(data_node->GetOutDataAnchor(0), clean_node->GetInDataAnchor(i + 1));
    if (ret != ge::SUCCESS) {
      GELOGE(ge::FAILED, "add edge between [%s] and [%s] failed", data_node->GetName().c_str(),
             clean_node->GetName().c_str());
      return ge::FAILED;
    }
  }
  return ge::SUCCESS;
}

ge::NodePtr BuildAtomicNode(ge::ComputeGraphPtr &graph) {
  std::vector<int64_t> workspace_indexes = {1,2};
  std::vector<int64_t> outputs_indexes = {0,2};

  auto atomic_op_desc = std::make_shared<ge::OpDesc>("AtomicClean", "DynamicAtomicAddrClean");

  atomic_op_desc->AppendIrInput("workspace", ge::kIrInputRequired);
  atomic_op_desc->AppendIrInput("output", ge::kIrInputDynamic);

  atomic_op_desc->AddInputDesc("workspace", ge::GeTensorDesc());
  for (size_t i = 0U; i < outputs_indexes.size(); ++i) {
    atomic_op_desc->AddInputDesc("output" + std::to_string(i + 1), ge::GeTensorDesc());
  }
  if (!ge::AttrUtils::SetListInt(atomic_op_desc, "WorkspaceIndexes", workspace_indexes)) {
    return nullptr;
  }
  auto clean_node = graph->AddNode(atomic_op_desc);
  if (clean_node == nullptr) {
    GELOGE(ge::FAILED, "add node failed");
    return nullptr;
  }
  if (AddDataNodeForAtomic(graph, clean_node, outputs_indexes.size()) != ge::SUCCESS) {
    GELOGE(ge::FAILED, "add data node for atomic clean node failed, outputs_indexes size = %zu",
           outputs_indexes.size());
    return nullptr;
  }
  return clean_node;
}
} // namespace

TEST_F(TilingContextBuilderUT, CompileInfoNullptr) {
  fe::PlatFormInfos platform_infos;
  auto builder = TilingContextBuilder();
  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT02").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node->shared_from_this());

  auto tiling_context_holder = builder
                               .CompileInfo(nullptr)
                               .PlatformInfo(reinterpret_cast<void *>(&platform_infos))
                               .Build(op);
  EXPECT_NE(tiling_context_holder.context_, nullptr);
}

TEST_F(TilingContextBuilderUT, PlatformInfoNullptr) {
  fe::PlatFormInfos platform_infos;
  auto builder = TilingContextBuilder();
  std::string op_compile_info_json = "{}";

  auto node = ComputeNodeFaker().NameAndType("Test", "DDIT02").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  ASSERT_NE(node, nullptr);
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node->shared_from_this());

  auto tiling_context_holder = builder
                               .CompileInfo(&op_compile_info_json)
                               .PlatformInfo(nullptr)
                               .Build(op);
  EXPECT_NE(tiling_context_holder.context_, nullptr);
}

TEST_F(TilingContextBuilderUT, BuildRTInputTensorsFailed) {
  auto node = ComputeNodeFaker().NameAndType("UbNode", "DDIT02").IoNum(3, 1).InputNames({"x", "y", "z"}).Build();
  node->GetOpDesc()->SetOpInferDepends({"x", "z"});

  auto graph = std::make_shared<ge::ComputeGraph>("ub_graph");
  auto data0 = ComputeNodeFaker(graph)
                   .NameAndType("Data0", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 0)
                   .IoNum(0, 1)
                   .Build();
  auto data1 = ComputeNodeFaker(graph)
                   .NameAndType("Data1", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 1)
                   .IoNum(0, 1)
                   .Build();
  auto data2 = ComputeNodeFaker(graph)
                   .NameAndType("Data2", "Data")
                   .Attr<int64_t>(ge::ATTR_NAME_PARENT_NODE_INDEX.c_str(), 2)
                   .IoNum(0, 1)
                   .Build();
  auto node2 = ComputeNodeFaker().NameAndType("UbNode2", "DDIT02").IoNum(1, 1).InputNames({"d"}).Build();
  ge::GraphUtils::AddEdge(data0->GetOutDataAnchor(0), node->GetInDataAnchor(0));
  ge::GraphUtils::AddEdge(data1->GetOutDataAnchor(0), node->GetInDataAnchor(1));
  ge::GraphUtils::AddEdge(data2->GetOutDataAnchor(0), node->GetInDataAnchor(2));
  graph->SetParentNode(node2);

  // construct op impl registry
  auto space_registry = std::make_shared<gert::OpImplSpaceRegistry>();
  auto registry_holder = std::make_shared<gert::OpImplRegistryHolder>();
  auto funcs = gert::OpImplRegistry::GetInstance().GetOpImpl("DDIT02");
  registry_holder->AddTypesToImpl("DDIT02", *funcs);
  space_registry->AddRegistry(registry_holder);
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(space_registry);

  auto tiling_data = gert::TilingData::CreateCap(1024);
  auto workspace_size = gert::ContinuousVector::Create<size_t>(16);
  std::string op_compile_info_json = "{}";
  fe::PlatFormInfos platform_infos;
  auto builder = TilingContextBuilder();
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node->shared_from_this());
  auto tiling_context_holder = builder
                               .CompileInfo(const_cast<char *>(op_compile_info_json.c_str()))
                               .PlatformInfo(reinterpret_cast<void *>(&platform_infos))
                               .TilingData(tiling_data.get())
                               .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
                               .SpaceRegistry(space_registry)
                               .Build(op);
  EXPECT_NE(tiling_context_holder.context_, nullptr);
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(nullptr);
}

// 值依赖场景，输入数据来自const
TEST_F(TilingContextBuilderUT, BuildWithInputConstSuccess) {
  auto tiling_data = gert::TilingData::CreateCap(1024);
  auto workspace_size = gert::ContinuousVector::Create<size_t>(16);
  std::string op_compile_info_json = "{}";
  fe::PlatFormInfos platform_infos;
  auto space_registry = DefaultOpImplSpaceRegistry::GetInstance().GetDefaultSpaceRegistry();
  auto builder = TilingContextBuilder();

  bg::ValueHolder::PopGraphFrame();
  (void)bg::ValueHolder::PushGraphFrame();
  auto foo = bg::ValueHolder::CreateVoid("Foo", {});
  EXPECT_EQ(foo->GetNode()->GetAllOutDataAnchorsSize(), 0);
  auto outputs = foo->AppendOutputs(2);
  EXPECT_EQ(outputs.size(), 2);
  EXPECT_EQ(foo->GetNode()->GetAllOutDataAnchorsSize(), 2);
  auto bar = bg::ValueHolder::CreateSingleDataOutput("Bar", outputs);
  EXPECT_NE(bar, nullptr);
  auto node = bar->GetNode();
  EXPECT_EQ(node->GetAllInDataAnchorsSize(), 2);
  ge::OpDescPtr op_desc = node->GetOpDesc();
  ge::GeTensorDesc tensor_desc(ge::GeShape({1}));
  op_desc->AddInputDesc("x", tensor_desc);
  op_desc->AddInputDesc("y", tensor_desc);
  op_desc->AddOutputDesc("z", tensor_desc);
  op_desc->MutableInputDesc(1)->SetDataType(ge::DT_INT32);
  op_desc->MutableInputDesc(1)->SetShape(ge::GeShape({1}));
  op_desc->MutableInputDesc(1)->SetOriginShape(ge::GeShape({1}));
  auto op = ge::OpDescUtils::CreateOperatorFromNode(node->shared_from_this());

  auto tiling_context_holder = builder
                               .CompileInfo(const_cast<char *>(op_compile_info_json.c_str()))
                               .PlatformInfo(reinterpret_cast<void *>(&platform_infos))
                               .TilingData(tiling_data.get())
                               .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
                               .SpaceRegistry(space_registry)
                               .Build(op);

  auto tiling_context = reinterpret_cast<TilingContext *>(tiling_context_holder.context_);
  // check content in context
  // 1.check input shape and tensor
  auto input_tensor1 = tiling_context->GetInputTensor(1);
  EXPECT_NE(input_tensor1, nullptr);
  EXPECT_EQ(input_tensor1->GetDataType(), ge::DT_INT32);
  EXPECT_EQ(input_tensor1->GetOriginShape().GetDim(0), 1);
  bg::ValueHolder::PopGraphFrame();
  DefaultOpImplSpaceRegistry::GetInstance().SetDefaultSpaceRegistry(nullptr);
}

TEST_F(TilingContextBuilderUT, BuildAtomicCompileInfoNullptr) {
  // build atomic clean node
  auto tmp_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  BuildAtomicNode(tmp_graph);
  auto op = ge::OpDescUtils::CreateOperatorFromNode(tmp_graph->FindNode("AtomicClean"));
  auto builder = AtomicTilingContextBuilder();
  auto tiling_context_holder = builder
      .CompileInfo(nullptr)
      .Build(op);
  auto context = reinterpret_cast<AtomicCleanTilingContext *>(tiling_context_holder.context_);
  EXPECT_NE(context, nullptr);
}

TEST_F(TilingContextBuilderUT, BuildAtomicTilingContextSuccess) {
  // build atomic clean node
  std::vector<int64_t> output_clean_sizes = {44, 55};
  auto tmp_graph = std::make_shared<ge::ComputeGraph>("tmp-graph");
  BuildAtomicNode(tmp_graph);

  auto tiling_data = gert::TilingData::CreateCap(1024);
  auto workspace_size = gert::ContinuousVector::Create<size_t>(16);

  std::string op_compile_info_json = "{}";
  auto clean_workspace_size = gert::ContinuousVector::Create<size_t>(16);
  auto clean_workspace_ptr = reinterpret_cast<gert::TypedContinuousVector<size_t> *>(clean_workspace_size.get());
  clean_workspace_ptr->SetSize(2);
  *(clean_workspace_ptr->MutableData()) = 22;
  *(clean_workspace_ptr->MutableData() + 1) = 33;

  auto op = ge::OpDescUtils::CreateOperatorFromNode(tmp_graph->FindNode("AtomicClean"));
  auto builder = AtomicTilingContextBuilder();
  auto tiling_context_holder = builder
      .CompileInfo(const_cast<char *>(op_compile_info_json.c_str()))
      .CleanWorkspaceSizes(reinterpret_cast<gert::ContinuousVector *>(clean_workspace_size.get()))
      .CleanOutputSizes(output_clean_sizes)
      .TilingData(tiling_data.get())
      .Workspace(reinterpret_cast<gert::ContinuousVector *>(workspace_size.get()))
      .Build(op);

  auto context = reinterpret_cast<AtomicCleanTilingContext *>(tiling_context_holder.context_);
  // check content in context
  auto clean_workspace_size_in_context = context->GetCleanWorkspaceSizes();
  EXPECT_EQ(clean_workspace_size_in_context->GetSize(), 2);
  auto ws_size_data = reinterpret_cast<const uint64_t*>(clean_workspace_size_in_context->GetData());
  EXPECT_EQ(ws_size_data[0], 22);
  EXPECT_EQ(ws_size_data[1], 33);

  EXPECT_EQ(context->GetCleanOutputSize(0), 44);
  EXPECT_EQ(context->GetCleanOutputSize(1), 55);
}
}  // namespace gert
