/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2023. All rights reserved.
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
#include "exe_graph/runtime/tiling_parse_context_builder.h"

#include "graph/compute_graph.h"
#include "graph/utils/op_desc_utils.h"
#include "graph/utils/graph_utils.h"
#include "graph/debug/ge_util.h"
#include "graph/def_types.h"
#include "common/checker.h"
#include "graph/debug/ge_util.h"

namespace gert {
TilingParseContextBuilder &TilingParseContextBuilder::CompileJson(const ge::char_t *compile_json) {
  compile_json_ = const_cast<ge::char_t *>(compile_json);
  return *this;
}

TilingParseContextBuilder &TilingParseContextBuilder::PlatformInfo(void *platform_info) {
  platform_info_ = platform_info;
  return *this;
}

TilingParseContextBuilder &TilingParseContextBuilder::CompileInfoCreatorFunc(
    OpImplKernelRegistry::CompileInfoCreatorFunc create_func) {
  create_func_ = create_func;
  return *this;
}

TilingParseContextBuilder &TilingParseContextBuilder::CompileInfoDeleterFunc(
    OpImplKernelRegistry::CompileInfoDeleterFunc delete_func) {
  delete_func_ = delete_func;
  return *this;
}

KernelContextHolder TilingParseContextBuilder::Build(const ge::Operator &op) {
  KernelContextHolder holder;
  if (compile_json_ == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Compile info is nullptr.");
    return holder;
  }
  if (platform_info_ == nullptr) {
    GELOGE(ge::GRAPH_PARAM_INVALID, "Platform info is nullptr.");
    return holder;
  }
  const auto op_desc = ge::OpDescUtils::GetOpDescFromOperator(op);
  GE_CHECK_NOTNULL_EXEC(op_desc, return holder);
  std::vector<std::pair<void *, gert::Chain::Deleter>> tiling_parse_outputs(1, std::make_pair(nullptr, nullptr));
  if (create_func_ != nullptr && delete_func_ != nullptr) {
    tiling_parse_outputs[0].first = create_func_();
    tiling_parse_outputs[0].second = delete_func_;
  }
  return gert::KernelRunContextBuilder()
      .Inputs({compile_json_})
      .Inputs({platform_info_})
      .Inputs({const_cast<ge::char_t *>(op_desc->GetType().c_str())})
      .Outputs(tiling_parse_outputs)
      .Build(op_desc);
}
}  // namespace gert
