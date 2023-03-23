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
#include "plugin/device/ascend/optimizer/ge_optimization.h"

#include <string>
#include <memory>

#include "backend/common/optimizer/optimizer.h"
#include "include/common/utils/config_manager.h"
#include "common/graph_kernel/graph_kernel_flags.h"
#include "include/common/debug/anf_ir_dump.h"
#include "plugin/device/ascend/optimizer/ge/clip_by_norm_fission.h"
#include "plugin/device/ascend/optimizer/ge/lamb_fission.h"
#include "plugin/device/ascend/optimizer/ge/reduce_axis_update.h"
#include "plugin/device/ascend/optimizer/ge/convert_resize_nearest_neighbor_x_dtype.h"
#include "plugin/device/ascend/optimizer/ge/convert_attr_to_input.h"
#include "plugin/device/ascend/optimizer/ge/batchnorm_transform.h"
#include "plugin/device/ascend/optimizer/ge/dropout_for_ge.h"
#include "plugin/device/ascend/optimizer/ge/avg_pool_grad_for_ge.h"
#include "plugin/device/ascend/optimizer/ge/ge_specialized_prepare.h"
#include "plugin/device/ascend/optimizer/ge/ge_tensor_array.h"
#include "plugin/device/ascend/optimizer/ge/sparse_softmax_cross_entropy_with_logits_split.h"

namespace mindspore {
namespace opt {
void GeOptimization(const FuncGraphPtr &func_graph) {
  MS_LOG(INFO) << "GE optimization start, graph: " << func_graph->ToString() << ".";

#ifdef ENABLE_DUMP_IR
  auto context_ptr = MsContext::GetInstance();
  MS_EXCEPTION_IF_NULL(context_ptr);
  bool save_graphs = context_ptr->get_param<bool>(MS_CTX_SAVE_GRAPHS_FLAG);
  if (save_graphs) {
    std::string file_name = "hwopt_d_before_ge_optimization_graph_" + func_graph->ToString() + ".ir";
    DumpIR(file_name, func_graph);
  }
#endif

  auto optimizer = std::make_shared<opt::GraphOptimizer>();
  auto pm = std::make_shared<opt::PassManager>("ge_optimization_pm");
  pm->AddPass(std::make_shared<opt::ConvertAttrToInput>());
  pm->AddPass(std::make_shared<opt::ConvertResizeNearestNeighborXDtype>());
  pm->AddPass(std::make_shared<opt::BatchNormTransform>());
  auto env_train = common::GetEnv("MS_GE_TRAIN");
  if (env_train == "1") {
    pm->AddPass(std::make_shared<opt::SparseSoftmaxCrossEntropyWithLogitsSplitCond1>());
    pm->AddPass(std::make_shared<opt::SparseSoftmaxCrossEntropyWithLogitsSplitCond2>());
  } else {
    pm->AddPass(std::make_shared<opt::SparseSoftmaxCrossEntropyWithLogitsSplitInfer>());
  }
  pm->AddPass(std::make_shared<opt::AvgPoolGradForGE>());
  pm->AddPass(std::make_shared<opt::DropoutForGE>());
  pm->AddPass(std::make_shared<opt::DropoutGradForGE>());
  pm->AddPass(std::make_shared<opt::LambFissionGe>());
  pm->AddPass(std::make_shared<opt::ClipByNormFissionGe>());
  pm->AddPass(std::make_shared<opt::GeTensorArrayAddFlowCond1>());
  pm->AddPass(std::make_shared<opt::GeTensorArrayAddFlowCond2>());
  pm->AddPass(std::make_shared<opt::GeTensorArrayCastIndex>());
  pm->AddPass(std::make_shared<opt::GeTensorArrayPrepare>());
  pm->AddPass(std::make_shared<opt::ReduceAxisUpdate>());
  optimizer->AddPassManager(pm);

  (void)optimizer->Optimize(func_graph);

#ifdef ENABLE_DUMP_IR
  if (save_graphs) {
    std::string file_name = "hwopt_d_after_ge_optimization_graph_" + func_graph->ToString() + ".ir";
    DumpIR(file_name, func_graph);
  }
#endif

  MS_LOG(INFO) << "GE optimization end.";
}
}  // namespace opt
}  // namespace mindspore
