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

#ifndef METADEF_CXX_EXE_GRAPH_LOWERING_VALUE_HOLDER_INNER_H_
#define METADEF_CXX_EXE_GRAPH_LOWERING_VALUE_HOLDER_INNER_H_
#include <deque>
#include "exe_graph/lowering/builtin_node_types.h"
#include "exe_graph/lowering/graph_frame.h"
namespace gert {
namespace bg {
void SetCurrentFrame(GraphFrame *frame);
GraphFrame *GetCurrentFrame();
std::deque<std::unique_ptr<GraphFrame>> &GetGraphFrames();
}  // namespace bg
}  // namespace gert
#endif  // METADEF_CXX_EXE_GRAPH_LOWERING_VALUE_HOLDER_INNER_H_
